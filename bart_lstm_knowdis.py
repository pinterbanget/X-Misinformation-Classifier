#%% Import Libraries
import numpy as np
import pandas as pd
import os
import re
import torch
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, BartModel
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler

import logging
from datetime import datetime

test_size = 0.25 # 0.25 is default

# Initialize logging
os.makedirs('logs/single_tower_bart_lstm', exist_ok=True)
os.makedirs('models/best_model', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/single_tower_bart_lstm/combined_bart_lstm_dt20250312_tsize_{test_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#%% Dataset Class for Combined Text Model
class CombinedTextDataset(Dataset):
    def __init__(self, tweet_texts, community_notes, labels, tokenizer, max_len, use_note_prob=1.0):
        """
        Dataset class for the combined text model.
        
        Args:
            tweet_texts (list): List of tweet tokens
            community_notes (list): List of community note tokens
            labels (list): List of labels
            tokenizer: Tokenizer to use
            max_len (int): Maximum length for tokenization
            use_note_prob (float): Probability of using community notes
        """
        self.tweet_texts = [' '.join(tokens) for tokens in tweet_texts]
        self.community_notes = [' '.join(tokens) for tokens in community_notes]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_note_prob = use_note_prob

    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, idx):
        tweet_text = self.tweet_texts[idx]
        comm_note = self.community_notes[idx]
        
        # Determine whether to use the community note based on probability
        use_note = random.random() < self.use_note_prob
        
        # If using note, combine tweet and note; otherwise use just tweet
        if use_note and comm_note.strip():
            combined_text = tweet_text + " " + comm_note
        else:
            combined_text = tweet_text
        
        # Handle empty text with a placeholder if completely empty
        if not combined_text.strip():
            combined_text = "[EMPTY]"
        
        # Encode combined text
        encoding = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

#%% BART-LSTM Model
class BARTLSTMClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-base')
        
        # bi-directional LSTM
        self.lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        # Adjusted classifier input size (128)
        self.classifier = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(0.3)
        self.layer_norm = torch.nn.LayerNorm(128)  # Match classifier input
        
    def forward(self, input_ids, attention_mask):
        # Get full BART output
        bart_output = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = bart_output.last_hidden_state
        
        # LSTM processing
        lstm_out, _ = self.lstm(sequence_output)
        
        # Simplified pooling (mean only)
        pooled = torch.mean(lstm_out, dim=1)  # (batch, 128)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        return self.classifier(pooled)

#%% Main Execution
if __name__ == '__main__':
    logger.info("Running Knowledge Distillation BART-LSTM model training...")
    
    # Configuration
    MAX_LEN = 192
    BATCH_SIZE = 60
    TEACHER_EPOCHS = 3
    STUDENT_EPOCHS = 3
    LEARNING_RATE = 2e-5
    TEMPERATURE = 2.0  # Softmax temperature for knowledge distillation
    
    # Process data
    with open("clean_data/20250314_100char.pkl", "rb") as f:
        dt_dict = pickle.load(f)
        X_tweet_preprocessed = dt_dict['X']
        X_notes_preprocessed = dt_dict['X_notes']
        y_processed = dt_dict['y']
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base',)
    teacher_model = BARTLSTMClassifier(num_classes=len(np.unique(y_processed)))
    student_model = BARTLSTMClassifier(num_classes=len(np.unique(y_processed)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Multi-GPU setup for both models
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_gpus}")
    if num_gpus > 1:
        logger.info(f"Using DataParallel with {num_gpus} GPUs")
        teacher_model = DataParallel(teacher_model)
        student_model = DataParallel(student_model)
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    effective_batch_size = BATCH_SIZE * max(1, num_gpus)
    
    # Split datasets into train (70%), validation (20%), and test (10%)
    # First split: separate out test set (90-10 split)
    X_tweet_temp, X_tweet_test, X_notes_temp, X_notes_test, y_temp, y_test = train_test_split(
        X_tweet_preprocessed, X_notes_preprocessed, y_processed, 
        test_size=0.1,  # 10% for test
        random_state=42
    )
    
    # Second split: divide remaining data into train and validation (78-22 split ≈ 70-20 of total)
    X_tweet_train, X_tweet_val, X_notes_train, X_notes_val, y_train, y_val = train_test_split(
        X_tweet_temp, X_notes_temp, y_temp,
        test_size=0.22,  # 20% of total (0.22 * 0.9 ≈ 0.2)
        random_state=42
    )
    
    # Create datasets for teacher (always using notes) and student (tweet-only)
    teacher_train_dataset = CombinedTextDataset(
        X_tweet_train, X_notes_train, y_train, 
        tokenizer, MAX_LEN, use_note_prob=1.0  # Always use notes for teacher
    )
    
    teacher_val_dataset = CombinedTextDataset(
        X_tweet_val, X_notes_val, y_val,
        tokenizer, MAX_LEN, use_note_prob=1.0  # Always use notes for teacher
    )
    
    student_train_dataset = CombinedTextDataset(
        X_tweet_train, X_notes_train, y_train, 
        tokenizer, MAX_LEN, use_note_prob=0.0  # Never use notes for student
    )
    
    student_val_dataset = CombinedTextDataset(
        X_tweet_val, X_notes_val, y_val,
        tokenizer, MAX_LEN, use_note_prob=0.0  # Never use notes for student
    )
    
    test_dataset = CombinedTextDataset(
        X_tweet_test, X_notes_test, y_test, 
        tokenizer, MAX_LEN, use_note_prob=0.0  # Test on tweets only
    )
    
    # Create data loaders
    teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=effective_batch_size, shuffle=True)
    teacher_val_loader = DataLoader(teacher_val_dataset, batch_size=effective_batch_size)
    student_train_loader = DataLoader(student_train_dataset, batch_size=effective_batch_size, shuffle=True)
    student_val_loader = DataLoader(student_val_dataset, batch_size=effective_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)
    
    # Training setup for teacher
    teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=LEARNING_RATE)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_processed), y=y_processed)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    kd_loss = torch.nn.KLDivLoss(reduction='batchmean')
    scaler = GradScaler('cuda')

    # Train teacher model
    logger.info("Training teacher model...")
    best_teacher_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(TEACHER_EPOCHS):
        teacher_model.train()
        total_loss = 0
        
        for batch in tqdm(teacher_train_loader, desc=f"Epoch {epoch+1}/{TEACHER_EPOCHS}"):
            teacher_optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast('cuda'):
                outputs = teacher_model(input_ids, attention_mask)
                loss = ce_loss(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(teacher_optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_teacher_loss = total_loss / len(teacher_train_loader)
        logger.info(f"Average teacher loss: {avg_teacher_loss:.4f}")
        
        # Validation phase
        teacher_model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(teacher_val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                with autocast('cuda'):
                    outputs = teacher_model(input_ids, attention_mask)
                    loss = ce_loss(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_val_loss = val_loss / len(teacher_val_loader)
        
        logger.info(f"Validation metrics:")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info("\n" + classification_report(all_labels, all_preds, digits=4, target_names=['Reliable', 'Misinformed']))
        
        # Check for improvement
        if val_f1 > best_teacher_f1:
            best_teacher_f1 = val_f1
            patience_counter = 0
            # Save the best model
            best_teacher_path = f'models/best_model/teacher_model_dt20250312_tsize_{test_size}_{datetime.now().strftime("%Y%m%d")}.pth'
            if isinstance(teacher_model, DataParallel):
                torch.save(teacher_model.module.state_dict(), best_teacher_path)
            else:
                torch.save(teacher_model.state_dict(), best_teacher_path)
            logger.info(f"New best teacher model saved with F1 score: {val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                logger.info("Early stopping triggered!")
                break
    
    # Load best teacher model for knowledge distillation
    teacher_model.load_state_dict(torch.load(best_teacher_path))
    teacher_model.eval()
    
    # Train student model with knowledge distillation
    logger.info("Training student model with knowledge distillation...")
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    best_student_f1 = 0
    patience_counter = 0
    
    for epoch in range(STUDENT_EPOCHS):
        student_model.train()
        total_loss = 0
        
        for batch in tqdm(student_train_loader, desc=f"Epoch {epoch+1}/{STUDENT_EPOCHS}"):
            student_optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast('cuda'):
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids, attention_mask)
                    teacher_probs = torch.nn.functional.softmax(teacher_logits / TEMPERATURE, dim=1)
                
                # Get student predictions
                student_logits = student_model(input_ids, attention_mask)
                student_probs = torch.nn.functional.log_softmax(student_logits / TEMPERATURE, dim=1)
                
                # Compute distillation loss
                distillation_loss = kd_loss(student_probs, teacher_probs) * (TEMPERATURE ** 2)
                # Compute standard cross-entropy loss
                ce_loss_val = ce_loss(student_logits, labels)
                # Combined loss (α=0.5 balances between distillation and true labels)
                loss = 0.5 * distillation_loss + 0.5 * ce_loss_val
            
            scaler.scale(loss).backward()
            scaler.step(student_optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_student_loss = total_loss / len(student_train_loader)
        logger.info(f"Average student loss: {avg_student_loss:.4f}")
        
        # Validation phase
        student_model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(student_val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                with autocast('cuda'):
                    outputs = student_model(input_ids, attention_mask)
                    loss = ce_loss(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_val_loss = val_loss / len(student_val_loader)
        
        logger.info(f"Validation metrics:")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info("\n" + classification_report(all_labels, all_preds, digits=4, target_names=['Reliable', 'Misinformed']))
        
        # Check for improvement
        if val_f1 > best_student_f1:
            best_student_f1 = val_f1
            patience_counter = 0
            # Save the best model
            best_student_path = f'models/best_model/student_model_dt20250312_tsize_{test_size}_{datetime.now().strftime("%Y%m%d")}.pth'
            if isinstance(student_model, DataParallel):
                torch.save(student_model.module.state_dict(), best_student_path)
            else:
                torch.save(student_model.state_dict(), best_student_path)
            logger.info(f"New best student model saved with F1 score: {val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                logger.info("Early stopping triggered!")
                break
    
    # Final evaluation of student model
    logger.info("Evaluating final student model on test set...")
    student_model.eval()

    test_loss = 0
    all_test_preds, all_test_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast('cuda'):
                outputs = student_model(
                    input_ids, 
                    attention_mask
                )
                loss = ce_loss(outputs, labels)
            
            test_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # Calculate test metrics
    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    avg_test_loss = test_loss / len(test_loader)

    logger.info(f"Test metrics:")
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info("\n" + classification_report(all_test_labels, all_test_preds, digits=4, target_names=['Reliable', 'Misinformed']))
    
    # Save final model regardless of performance
    final_student_path = f'models/student_model_final_dt20250312_tsize_{test_size}_{datetime.now().strftime("%Y%m%d")}.pth'
    if isinstance(student_model, DataParallel):
        torch.save(student_model.module.state_dict(), final_student_path)
    else:
        torch.save(student_model.state_dict(), final_student_path)
    logger.info("Training completed. Final student model saved.")