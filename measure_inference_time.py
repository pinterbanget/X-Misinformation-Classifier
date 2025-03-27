import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import pickle
from tqdm import tqdm
import logging
from datetime import datetime
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import BartModel
from sklearn.metrics import accuracy_score

# Import your model and dataset classes
from bart_lstm_cn import  CombinedTextDataset  # or from xlstm_cn import xLSTMClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/inference_timing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BARTLSTMClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-base')
        
        # bi-directional LSTM with correct sizes
        self.lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=128,  # This was 64 before, needs to be 128 to match saved model
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier matching the saved model architecture
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BART output
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence = outputs.last_hidden_state
        
        # Process through LSTM
        lstm_out, _ = self.lstm(sequence)
        pooled = torch.mean(lstm_out, dim=1)
        
        # Classify
        return self.classifier(pooled) 
    
    
def measure_inference_time(model, test_loader, device, num_runs=5):
    model.eval()
    batch_times = []
    total_samples = 0
    
    with torch.no_grad():
        for run_idx in range(num_runs):
            run_times = []
            for batch in tqdm(test_loader, desc=f"Run {run_idx+1}/{num_runs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = input_ids.size(0)
                
                # Warm-up run for first iteration
                if run_idx == 0:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids, attention_mask)
                    continue
                
                # Measure time
                start_time = time.perf_counter()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask)
                torch.cuda.synchronize()  # Wait for GPU
                end_time = time.perf_counter()
                
                run_times.append((end_time - start_time, batch_size))
                total_samples += batch_size
            
            if run_idx > 0:  # Skip warm-up run
                batch_times.append(run_times)
    
    # Calculate statistics
    all_times = []
    for run in batch_times:
        run_total_time = sum(t for t, _ in run)
        all_times.append(run_total_time)
    
    avg_time = np.mean(all_times)
    std_time = np.std(all_times)
    
    # Calculate throughput
    avg_throughput = total_samples / (len(batch_times) * avg_time)
    
    return {
        'average_time': avg_time,
        'std_time': std_time,
        'throughput': avg_throughput,
        'total_samples': total_samples,
        'num_runs': len(batch_times)
    }

def measure_complete_inference_time(model, X_tweet_test, tokenizer, max_len, device, batch_size=32):
    model.eval()
    total_samples = len(X_tweet_test)
    all_preds = []
    
    # Start timing from dataset creation
    start_time = time.perf_counter()
    
    # Create dataset
    test_dataset = CombinedTextDataset(
        X_tweet_test, 
        [[] for _ in X_tweet_test],  # Empty notes for inference
        [0] * len(X_tweet_test),  # Dummy labels
        tokenizer, 
        max_len, 
        use_note_prob=0.0
    )
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Complete Pipeline Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
    
    # End timing after all predictions are collected
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return {
        'total_time': total_time,
        'samples_per_second': total_samples / total_time,
        'total_samples': total_samples,
        'predictions': all_preds
    }

if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 32
    MAX_LEN = 192
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    with open("clean_data/20250314_100char.pkl", "rb") as f:
        dt_dict = pickle.load(f)
        X_tweet_preprocessed = dt_dict['X']
        X_notes_preprocessed = dt_dict['X_notes']
        y_processed = dt_dict['y']

    # First split data to isolate test set
    X_tweet_temp, X_tweet_test, X_notes_temp, X_notes_test, y_temp, y_test = train_test_split(
        X_tweet_preprocessed, X_notes_preprocessed, y_processed,
        test_size=0.15, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    model = BARTLSTMClassifier(num_classes=2)
    
    # Load student model
    model_path = 'models/best_model/student_model_dt20250312_tsize_0.3_20250316.pth'
    state_dict = torch.load(model_path)
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Measure complete inference time
    logger.info("Starting complete inference pipeline measurement...")
    results = measure_complete_inference_time(model, X_tweet_test, tokenizer, MAX_LEN, device, BATCH_SIZE)
    
    # Log results
    logger.info("\nComplete Pipeline Results:")
    logger.info(f"Total time: {results['total_time']:.4f} seconds")
    logger.info(f"Throughput: {results['samples_per_second']:.2f} samples/second")
    logger.info(f"Total samples processed: {results['total_samples']}")
    
    # Calculate accuracy if you want to verify predictions
    accuracy = accuracy_score(y_test, results['predictions'])
    logger.info(f"Accuracy on test set: {accuracy:.4f}")

