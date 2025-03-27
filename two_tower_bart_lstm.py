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
import time

import logging
from datetime import datetime

batch_sizes = 48  # 320 48GB vram, 48 12GB vram, 24 6G vram
choices = "default" # default will use test_sizes["default"] params

# 0.3 is default, 0.25 (75:10:15), 0.9 (10:10:80)
test_sizes = {
    "default": [0.25, 0.6], # 75:10:15 (train:val:test)
    "90": [0.90, 1/9],      # 10:10:80 (train:val:test)
    "80": [0.80, 1/8],      # 20:10:70 (train:val:test)
    "70": [0.70, 1/7],      # 30:10:60 (train:val:test)
    "60": [0.60, 1/6],      # 40:10:50 (train:val:test)
    "50": [0.50, 1/5],      # 50:10:40 (train:val:test)
    "40": [0.40, 1/4],      # 60:10:30 (train:val:test)
    "30": [0.30, 1/3],      # 70:10:20 (train:val:test)
    "20": [0.20, 1/5]       # 80:10:10 (train:val:test)
}
class_1_weights = 1.5 # giving more weight towards class 1
dataset_name = "dt20250314" # dt20250312 or dt20250314

model_name = f"two_tower_bart_{dataset_name}_100char_test_size_{test_sizes[choices][0]}_clweight_{class_1_weights}"
timestamp = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
timestamp_day = f'{datetime.now().strftime("%Y%m%d")}'

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/two_tower_bart_lstm/{model_name}_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#%% Dataset Class for Two-Tower Model with Augmentation
class TwoTowerDataset(Dataset):
    def __init__(self, tweet_texts, community_notes, labels, tokenizer, max_len, empty_note_prob=0.0):
        self.tweet_texts = [' '.join(tokens) for tokens in tweet_texts]
        self.community_notes = [' '.join(tokens) for tokens in community_notes]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.empty_note_prob = empty_note_prob  # Probability of replacing a note with empty string

    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, idx):
        tweet_text = self.tweet_texts[idx]
        comm_note = self.community_notes[idx]
        
        # Apply data augmentation - randomly replace community notes with empty
        if random.random() < self.empty_note_prob:
            # comm_note = "[EMPTY]"
            comm_note = tweet_text
        
        # Handle empty text with a placeholder if completely empty
        if not tweet_text.strip():
            tweet_text = "[EMPTY]"
        if not comm_note.strip():
            comm_note = "[EMPTY]"  # Use a placeholder for empty community notes
        
        # Encode tweet text
        tweet_encoding = self.tokenizer.encode_plus(
            tweet_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Encode community note
        note_encoding = self.tokenizer.encode_plus(
            comm_note,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'tweet_input_ids': tweet_encoding['input_ids'].flatten(),
            'tweet_attention_mask': tweet_encoding['attention_mask'].flatten(),
            'note_input_ids': note_encoding['input_ids'].flatten(),
            'note_attention_mask': note_encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

#%% Two-Tower BART-LSTM Model
class TwoTowerBARTClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Tweet Tower
        self.tweet_bart = BartModel.from_pretrained('models/bart-base')
        self.tweet_lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        self.tweet_layer_norm = torch.nn.LayerNorm(128)
        self.tweet_dropout = torch.nn.Dropout(0.3)
        
        # Community Note Tower
        self.note_bart = BartModel.from_pretrained('models/bart-base')
        self.note_lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        self.note_layer_norm = torch.nn.LayerNorm(128)
        self.note_dropout = torch.nn.Dropout(0.3)
        
        # Combined classifier
        # 128 from tweet tower + 128 from note tower
        self.fusion_layer = torch.nn.Linear(256, 128)
        self.fusion_act = torch.nn.ReLU()
        self.fusion_norm = torch.nn.LayerNorm(128)
        self.fusion_dropout = torch.nn.Dropout(0.3)
        
        # Final classifier
        self.classifier = torch.nn.Linear(128, num_classes)
        
    def forward(self, tweet_input_ids, tweet_attention_mask, note_input_ids, note_attention_mask):
        # Process tweet tower
        tweet_output = self.tweet_bart(
            input_ids=tweet_input_ids,
            attention_mask=tweet_attention_mask,
            return_dict=True
        )
        tweet_seq_output = tweet_output.last_hidden_state
        
        tweet_lstm_out, _ = self.tweet_lstm(tweet_seq_output)
        tweet_pooled = torch.mean(tweet_lstm_out, dim=1)  # (batch, 128)
        tweet_pooled = self.tweet_layer_norm(tweet_pooled)
        tweet_pooled = self.tweet_dropout(tweet_pooled)
        
        # Process community note tower
        note_output = self.note_bart(
            input_ids=note_input_ids,
            attention_mask=note_attention_mask,
            return_dict=True
        )
        note_seq_output = note_output.last_hidden_state
        
        note_lstm_out, _ = self.note_lstm(note_seq_output)
        note_pooled = torch.mean(note_lstm_out, dim=1)  # (batch, 128)
        note_pooled = self.note_layer_norm(note_pooled)
        note_pooled = self.note_dropout(note_pooled)
        
        # Combine towers
        combined = torch.cat([tweet_pooled, note_pooled], dim=1)  # (batch, 256)
        
        # Fusion layer
        fused = self.fusion_layer(combined)
        fused = self.fusion_act(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)
        
        # Final classification
        return self.classifier(fused)

#%% Main Execution
if __name__ == '__main__':
    
    logger.info("Running Two-Tower model training with data augmentation...")
    
    # Configuration
    MAX_LEN = 128
    BATCH_SIZE = batch_sizes
    EPOCHS = 100
    LEARNING_RATE = 2e-5
    
    # Data augmentation settings
    EMPTY_NOTE_PROB_TRAIN = 0.7  # 70% of training samples will have empty notes
    EMPTY_NOTE_PROB_VAL = 0.5   # 50% of validation samples will have empty notes
    
    # Process data
    if dataset_name == 'dt20250312':
        with open("clean_data/20250312.pkl", "rb") as f:
            dt_dict = pickle.load(f)
            X_tweet_preprocessed = dt_dict['X']  
            X_notes_preprocessed = dt_dict['X_notes'] 
            y_processed = dt_dict['y'] 
    elif dataset_name == 'dt20250314': 
        with open("clean_data/20250314_100char.pkl", "rb") as f:
            dt_dict = pickle.load(f)
            X_tweet_preprocessed = dt_dict['X']  
            X_notes_preprocessed = dt_dict['X_notes'] 
            y_processed = dt_dict['y']
    else:
        logger.info(f"No matching dataset. stopping..")
        exit()
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained('models/bart-base')
    model = TwoTowerBARTClassifier(num_classes=len(np.unique(y_processed)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_gpus}")
    
    # Multi-GPU setup
    if num_gpus > 1:
        logger.info(f"Using DataParallel with {num_gpus} GPUs")
        # Use DataParallel for multiple GPUs
        model = DataParallel(model)
        
    model = model.to(device)
    
     # Adjust effective batch size for multiple GPUs
    effective_batch_size = BATCH_SIZE * max(1, num_gpus)
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    logger.info("Initializing training process with data augmentation")
    logger.info(f"Using device: {device}")
    
    logger.info("Training configuration:\n"
            f"Using dataset: {dataset_name}\n"
            f"model name: {model_name}\n"
            f"MAX_LEN: {MAX_LEN}\n"
            f"BATCH_SIZE: {BATCH_SIZE}\n"
            f"EPOCHS: {EPOCHS}\n"
            f"LEARNING_RATE: {LEARNING_RATE}\n"
            f"EMPTY_NOTE_PROB_TRAIN: {EMPTY_NOTE_PROB_TRAIN}\n"
            f"EMPTY_NOTE_PROB_VAL: {EMPTY_NOTE_PROB_VAL}")
    
    # Split datasets
    X_tweet_train, X_tweet_temp, X_notes_train, X_notes_temp, y_train, y_temp = train_test_split(
        X_tweet_preprocessed, X_notes_preprocessed, y_processed, test_size=test_sizes[choices][0], random_state=42
    )
    # Further split temp into validation and test sets
    X_tweet_val, X_tweet_test, X_notes_val, X_notes_test, y_val, y_test = train_test_split(
        X_tweet_temp, X_notes_temp, y_temp, test_size=test_sizes[choices][1], random_state=42  # This gives ~15% test, 10% validation
    )
    
    # Create datasets with augmentation
    train_dataset = TwoTowerDataset(X_tweet_train, X_notes_train, y_train, tokenizer, MAX_LEN, 
                                   empty_note_prob=EMPTY_NOTE_PROB_TRAIN)
    val_dataset = TwoTowerDataset(X_tweet_val, X_notes_val, y_val, tokenizer, MAX_LEN, 
                                 empty_note_prob=EMPTY_NOTE_PROB_VAL)
    test_val_dataset = TwoTowerDataset(X_tweet_temp, X_tweet_temp, y_temp, tokenizer, MAX_LEN, 
                                 empty_note_prob=EMPTY_NOTE_PROB_VAL)
    
    # Test dataset always uses empty community notes to simulate real-world conditions
    test_dataset = TwoTowerDataset(
                                X_tweet_test, 
                                #  [['[EMPTY]'] for _ in range(len(X_tweet_test))],
                                X_tweet_test, 
                                y_test, tokenizer, MAX_LEN, empty_note_prob=1.0)
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
    test_val_loader = DataLoader(test_val_dataset, batch_size=effective_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)

    logger.info(f"Dataset splits: Train: {len(X_tweet_train)}, Validation: {len(X_tweet_val)}, Test: {len(X_tweet_test)}")
    
    # Initialize training parameters
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_processed), 
        y=y_processed
    )
    class_weights[1] *= class_1_weights
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda')  # Mixed precision
    
    training_metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'learning_rate': []
    }

    test_metrics = {
        'test_loss_empty': None,
        'test_accuracy_empty': None,
        'test_f1_empty': None,
        'test_inference_empty': None,
        'test_inference_with_notes': None,
        'test_loss_with_notes': None,
        'test_accuracy_with_notes': None,
        'test_f1_with_notes': None,
        'classification_report_empty': None,
        'classification_report_with_notes': None
    }
    
    # Training loop
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            mem_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            
            if batch_idx % 10 == 0:  # Reduce logging frequency
                logger.info(f"Processing batch {batch_idx} / {len(train_loader)}; "
                           f"memory: Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
            
            optimizer.zero_grad()
            
            tweet_input_ids = batch['tweet_input_ids'].to(device)
            tweet_attention_mask = batch['tweet_attention_mask'].to(device)
            note_input_ids = batch['note_input_ids'].to(device)
            note_attention_mask = batch['note_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast('cuda'):
                outputs = model(
                    tweet_input_ids, 
                    tweet_attention_mask,
                    note_input_ids,
                    note_attention_mask
                )
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                tweet_input_ids = batch['tweet_input_ids'].to(device)
                tweet_attention_mask = batch['tweet_attention_mask'].to(device)
                note_input_ids = batch['note_input_ids'].to(device)
                note_attention_mask = batch['note_attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                with autocast('cuda'):
                    outputs = model(
                        tweet_input_ids, 
                        tweet_attention_mask,
                        note_input_ids,
                        note_attention_mask
                    )
                    loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Validation metrics:")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info("\n" + classification_report(all_labels, all_preds, digits=4, target_names=['Reliable', 'Misinformed']))
        
        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save the best model
            model_path = f'models/best_model/{model_name}_{timestamp_day}.pth'
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with F1 score: {val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                logger.info("Early stopping triggered!")
                break
        
        # Update learning rate based on validation F1
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        training_metrics['epoch'].append(epoch + 1)
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(avg_val_loss)
        training_metrics['val_accuracy'].append(val_accuracy)
        training_metrics['val_f1'].append(val_f1)
        training_metrics['learning_rate'].append(current_lr)
        scheduler.step(val_f1)
    
    with open(f'metrics/two_tower_bart_lstm/{model_name}_training_metrics_{timestamp_day}.pkl', 'wb') as f:
        pickle.dump(training_metrics, f)
    logger.info(f"Training metrics saved to metrics/two_tower_bart_lstm/{model_name}_training_metrics_{timestamp_day}.pkl")

    logger.info("Evaluating best model on test set (with empty community notes)")
    # best_model_path = f'models/best_model/two_tower_bart_dt20250314_100char_test_size_0.25_clweight_1.5_20250315.pth'
    # best_model_path = f'models/best_model/two_tower_bart_dt20250314_100char_test_size_0.25_clweight_1.5_20250316.pth'  
    best_model_path = f'models/best_model/{model_name}_{timestamp_day}.pth'  
    state_dict = torch.load(best_model_path)
    
    if isinstance(model, DataParallel):
    # If loading into a DataParallel model, we need to add "module." prefix
        if not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
    else:
        # If loading into a non-DataParallel model, we need to remove "module." prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    test_loss = 0
    all_test_preds, all_test_labels = [], []

    logger.info("Calculating Inference Time.. starts now!")
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
        # for batch in tqdm(test_val_loader, desc="Test+Val Evaluation"):
            mem_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"memory: Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
            tweet_input_ids = batch['tweet_input_ids'].to(device)
            tweet_attention_mask = batch['tweet_attention_mask'].to(device)
            note_input_ids = batch['note_input_ids'].to(device)
            note_attention_mask = batch['note_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast('cuda'):
                outputs = model(
                    tweet_input_ids, 
                    tweet_attention_mask,
                    note_input_ids,
                    note_attention_mask
                )
                loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    inference_duration = end_time - start_time
    logger.info(f"Inference completed in {inference_duration:.4f} seconds.")
    
    # Calculate test metrics
    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    avg_test_loss = test_loss / len(test_loader)
    class_rep = classification_report(all_test_labels, all_test_preds, digits=4, target_names=['Reliable', 'Misinformed'])
    logger.info(f"Test metrics (with empty community notes):")
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info("\n" + class_rep)
    
    test_metrics['test_inference_empty'] = inference_duration
    test_metrics['test_loss_empty'] = avg_test_loss
    test_metrics['test_accuracy_empty'] = test_accuracy
    test_metrics['test_f1_empty'] = test_f1
    test_metrics['classification_report_empty'] = classification_report(all_test_labels, all_test_preds, digits=4, target_names=['Reliable', 'Misinformed'], output_dict=True)

    # Additional evaluation: Test the model with ACTUAL community notes
    logger.info("Evaluating best model on test set WITH ACTUAL community notes for comparison")
    
    # Create test dataset with actual community notes
    test_dataset_with_notes = TwoTowerDataset(X_tweet_test, X_notes_test, y_test, tokenizer, MAX_LEN, empty_note_prob=0.0)
    test_loader_with_notes = DataLoader(test_dataset_with_notes, batch_size=effective_batch_size)
    
    test_loss_with_notes = 0
    all_test_preds_with_notes, all_test_labels_with_notes = [], []

    
    logger.info("Calculating Inference Time.. starts now!")
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader_with_notes, desc="Test Evaluation (with notes)"):
            tweet_input_ids = batch['tweet_input_ids'].to(device)
            tweet_attention_mask = batch['tweet_attention_mask'].to(device)
            note_input_ids = batch['note_input_ids'].to(device)
            note_attention_mask = batch['note_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast('cuda'):
                outputs = model(
                    tweet_input_ids, 
                    tweet_attention_mask,
                    note_input_ids,
                    note_attention_mask
                )
                loss = loss_fn(outputs, labels)
            
            test_loss_with_notes += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_test_preds_with_notes.extend(preds.cpu().numpy())
            all_test_labels_with_notes.extend(labels.cpu().numpy())

    end_time = time.time()
    inference_duration = end_time - start_time
    logger.info(f"Inference completed in {inference_duration:.4f} seconds.")
    
    # Calculate test metrics with notes
    test_accuracy_with_notes = accuracy_score(all_test_labels_with_notes, all_test_preds_with_notes)
    test_f1_with_notes = f1_score(all_test_labels_with_notes, all_test_preds_with_notes, average='weighted')
    avg_test_loss_with_notes = test_loss_with_notes / len(test_loader_with_notes)

    logger.info(f"Test metrics (WITH ACTUAL community notes):")
    logger.info(f"Test Loss: {avg_test_loss_with_notes:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy_with_notes:.4f}")
    logger.info(f"Test F1 Score: {test_f1_with_notes:.4f}")
    logger.info("\n" + classification_report(all_test_labels_with_notes, all_test_preds_with_notes, digits=4,
                                           target_names=['Reliable', 'Misinformed']))
    
    test_metrics['test_loss_with_notes'] = avg_test_loss_with_notes
    test_metrics['test_inference_with_notes'] = inference_duration
    test_metrics['test_accuracy_with_notes'] = test_accuracy_with_notes
    test_metrics['test_f1_with_notes'] = test_f1_with_notes
    test_metrics['classification_report_with_notes'] = classification_report(all_test_labels_with_notes, all_test_preds_with_notes, digits=4, target_names=['Reliable', 'Misinformed'],
                                                                            output_dict=True)

    with open(f'metrics/two_tower_bart_lstm/{model_name}_test_metrics_{timestamp_day}.pkl', 'wb') as f:
        pickle.dump(test_metrics, f)
    logger.info(f"Test metrics saved to metrics/two_tower_bart_lstm/{model_name}_test_metrics_{timestamp_day}.pkl")
    
    # Save final model regardless of performance
    final_model_path = f'models/{model_name}_{timestamp_day}.pth'
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
        
    logger.info("Training completed. Final model saved.")