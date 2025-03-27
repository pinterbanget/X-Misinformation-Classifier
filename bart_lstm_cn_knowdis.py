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

dataset_name = "filtered_data.pkl"

# Initialize logging
os.makedirs('logs/single_tower_bart_lstm', exist_ok=True)
os.makedirs('models/best_model', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/single_tower_bart_lstm/combined_bart_lstm_dt20250312_tsize_{0.15}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, idx):
        tweet_text = self.tweet_texts[idx]
        comm_note = self.community_notes[idx]
        
        # Determine whether to use the community note based on probability
        use_note = random.random() < self.use_note_prob
        
        # Handle empty text with a placeholder
        if not tweet_text.strip():
            tweet_text = "[EMPTY]"
        if not comm_note.strip():
            comm_note = "[EMPTY]"
        
        # Combine text with separator
        combined_text = tweet_text + " [SEP] " + (comm_note if use_note else "[EMPTY]")
        
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
        
        # Convert labels to tensor based on type
        if isinstance(self.labels[idx], (np.ndarray, list)):
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label_tensor
        }

#%% BART-LSTM Model
class BARTLSTMClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-base')
        
        # Single LSTM layer
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
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Process through BART
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence = outputs.last_hidden_state
        
        # Process through LSTM
        lstm_out, _ = self.lstm(sequence)
        pooled = torch.mean(lstm_out, dim=1)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Classify
        return self.classifier(pooled)

#%% Main Execution
if __name__ == '__main__':
    logger.info("Running Knowledge Distillation BART-LSTM model training...")
    
    # Configuration
    MAX_LEN = 192
    BATCH_SIZE = 60
    TEACHER_EPOCHS = 50
    STUDENT_EPOCHS = 50
    LEARNING_RATE = 2e-5
    TEMPERATURE = 2.0  # Softmax temperature for knowledge distillation
    
    # Process data
    with open(dataset_name, "rb") as f:
        dt_dict = pickle.load(f)
        X_tweet_preprocessed = dt_dict['X']
        X_notes_preprocessed = dt_dict['X_notes']
        y_processed = dt_dict['y']
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base',)
    teacher_model = BARTLSTMClassifier(num_classes=len(np.unique(y_processed)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First split data to isolate test set completely (75-25)
    X_tweet_temp, X_tweet_test, X_notes_temp, X_notes_test, y_temp, y_test = train_test_split(
        X_tweet_preprocessed, X_notes_preprocessed, y_processed,
        test_size=0.25, random_state=42
    )
    
    # Split remaining data into train-val (60% split of the 25% ≈ 10-15 of total)
    X_tweet_train, X_tweet_val, X_notes_train, X_notes_val, y_train, y_val = train_test_split(
        X_tweet_temp, X_notes_temp, y_temp,
        test_size=0.60, random_state=42
    )
    
    # Create training dataset for teacher (with community notes)
    teacher_train_dataset = CombinedTextDataset(
        X_tweet_train, X_notes_train, y_train, 
        tokenizer, MAX_LEN, use_note_prob=1.0
    )
    
    teacher_val_dataset = CombinedTextDataset(
        X_tweet_val, X_notes_val, y_val,
        tokenizer, MAX_LEN, use_note_prob=1.0
    )
    
    teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    teacher_val_loader = DataLoader(teacher_val_dataset, batch_size=BATCH_SIZE)
    
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
    
    # Generate teacher predictions for train and validation sets
    logger.info("Generating teacher predictions for training and validation data...")
    X_student = X_tweet_train + X_tweet_val
    teacher_probabilities = []
    
    # Create dataset for teacher predictions
    prediction_dataset = CombinedTextDataset(
        X_student, [[] for _ in X_student], [0] * len(X_student),
        tokenizer, MAX_LEN, use_note_prob=0.0  # Only tweets for prediction
    )
    prediction_loader = DataLoader(prediction_dataset, batch_size=BATCH_SIZE)
    
    with torch.no_grad():
        for batch in tqdm(prediction_loader, desc="Generating teacher predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = teacher_model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(outputs / TEMPERATURE, dim=1)
            teacher_probabilities.extend(probs.cpu().numpy())
    
    # Split predictions back into train and validation sets
    train_probs = teacher_probabilities[:len(X_tweet_train)]
    val_probs = teacher_probabilities[len(X_tweet_train):]
    
    # Create datasets for student using teacher predictions
    student_train_dataset = CombinedTextDataset(
        X_tweet_train, [[] for _ in X_tweet_train], train_probs,
        tokenizer, MAX_LEN, use_note_prob=0.0
    )
    
    student_val_dataset = CombinedTextDataset(
        X_tweet_val, [[] for _ in X_tweet_val], val_probs,
        tokenizer, MAX_LEN, use_note_prob=0.0
    )
    
    student_test_dataset = CombinedTextDataset(
        X_tweet_test, [[] for _ in X_tweet_test], y_test,
        tokenizer, MAX_LEN, use_note_prob=0.0
    )
    
    # Create data loaders for student
    student_train_loader = DataLoader(student_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    student_val_loader = DataLoader(student_val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(student_test_dataset, batch_size=BATCH_SIZE)
    
    # Student training (tweet-only)
    logger.info("Training student model...")
    student_model = BARTLSTMClassifier(num_classes=2)
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    kd_loss = torch.nn.KLDivLoss(reduction='batchmean')
    scaler = GradScaler('cuda')

    best_student_f1 = 0
    patience = 5
    patience_counter = 0
    
    # Ensure model is on correct device
    student_model = student_model.to(device)
    
    for epoch in range(STUDENT_EPOCHS):
        student_model.train()
        total_loss = 0
        
        for batch in tqdm(student_train_loader, desc=f"Epoch {epoch+1}/{STUDENT_EPOCHS}"):
            student_optimizer.zero_grad()
            
            # Ensure all inputs are on the same device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_probs = batch['label'].to(device).float()
            
            # Print device information for debugging
            logger.debug(f"Device check - Model: {next(student_model.parameters()).device}, "
                      f"Input IDs: {input_ids.device}, "
                      f"Attention Mask: {attention_mask.device}, "
                      f"Teacher Probs: {teacher_probs.device}")
            
            with autocast('cuda'):
                student_logits = student_model(input_ids, attention_mask)
                student_probs = torch.nn.functional.log_softmax(student_logits / TEMPERATURE, dim=1)
                loss = kd_loss(student_probs, teacher_probs) * (TEMPERATURE ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(student_optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        # Validation
        student_model.eval()
        val_loss = 0
        val_preds = []
        val_teacher_preds = []
        
        with torch.no_grad():
            for batch in student_val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                teacher_probs = batch['label'].to(device).float()
                
                student_logits = student_model(input_ids, attention_mask)
                student_probs = torch.nn.functional.log_softmax(student_logits / TEMPERATURE, dim=1)
                
                # Get predicted classes
                student_pred = torch.argmax(student_logits, dim=1)
                teacher_pred = torch.argmax(teacher_probs, dim=1)
                
                # Calculate KL loss
                loss = kd_loss(student_probs, teacher_probs) * (TEMPERATURE ** 2)
                val_loss += loss.item()
                
                # Store predictions for F1 calculation
                val_preds.extend(student_pred.cpu().numpy())
                val_teacher_preds.extend(teacher_pred.cpu().numpy())
        
        avg_val_loss = val_loss / len(student_val_loader)
        val_f1 = f1_score(val_teacher_preds, val_preds, average='weighted')
        
        logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_student_f1:
            best_student_f1 = val_f1
            patience_counter = 0
            best_student_path = f'models/best_model/student_model_dt20250312_tsize_{test_size}_{datetime.now().strftime("%Y%m%d")}.pth'
            if isinstance(student_model, DataParallel):
                torch.save(student_model.module.state_dict(), best_student_path)
            else:
                torch.save(student_model.state_dict(), best_student_path)
            logger.info(f"New best student model saved with F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered!")
                break
    
    # Load best student model for final evaluation
    student_model.load_state_dict(torch.load(best_student_path))
    student_model.eval()
    
    # Final evaluation on test set
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = student_model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate and log final metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    logger.info("\nFinal Test Results:")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(test_labels, test_preds, digits=4))

#%% Inference Function
def predict_misleading(tweet_text, community_note=None, model_path=None, tokenizer_path='facebook/bart-base', device=None, use_note=False):
    """
    Make prediction on a single tweet and optional community note
    
    Args:
        tweet_text (list or str): Preprocessed tweet tokens or raw text
        community_note (list or str, optional): Preprocessed community note tokens or raw text
                                             If None or use_note=False, only tweet text is used
        model_path (str): Path to the saved model
        tokenizer_path (str): Path to the tokenizer
        device (str): Device to use ('cuda' or 'cpu')
        use_note (bool): Whether to use the community note in prediction
        
    Returns:
        int: Predicted label (0: Reliable, 1: Misinformed)
        float: Probability of being misinformation
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find the latest model if not specified
    if model_path is None:
        model_files = [f for f in os.listdir('models/best_model/') if f.startswith(f'student_model_dt20250312_tsize_{test_size}')]
        if not model_files:
            raise ValueError("No model files found in models/best_model/")
        model_path = f'models/best_model/{sorted(model_files)[-1]}'
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = BARTLSTMClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Prepare inputs
    if isinstance(tweet_text, list):
        tweet_text = ' '.join(tweet_text)
    
    # Prepare combined text based on whether to use note
    if use_note and community_note:
        if isinstance(community_note, list):
            community_note = ' '.join(community_note)
        combined_text = tweet_text + " " + community_note
    else:
        combined_text = tweet_text
    
    # Use placeholder for empty text
    if not combined_text.strip():
        combined_text = "[EMPTY]"
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=192,  # Match the training max_len
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Move to device
    input_ids = {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device)
    }
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**input_ids)
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_label].item()
    
    return pred_label, pred_prob