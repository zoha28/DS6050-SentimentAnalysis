# train_model.py
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load data
def load_data(dataset_name, data_dir='./processed_data'):
    train = pd.read_csv(f'{data_dir}/{dataset_name}_train.csv')
    val = pd.read_csv(f'{data_dir}/{dataset_name}_val.csv')
    test = pd.read_csv(f'{data_dir}/{dataset_name}_test.csv')
    
    print(f"Loaded {dataset_name}: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
    return train, val, test

# Create datasets
def create_datasets(train_df, val_df, test_df, tokenizer, text_col='text_normalized', label_col='sentiment', max_length=128):
    train_dataset = TweetDataset(train_df[text_col].values, train_df[label_col].values, tokenizer, max_length)
    val_dataset = TweetDataset(val_df[text_col].values, val_df[label_col].values, tokenizer, max_length)
    test_dataset = TweetDataset(test_df[text_col].values, test_df[label_col].values, tokenizer, max_length)
    return train_dataset, val_dataset, test_dataset

# Compute metrics
def calc_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }

# Plotting functions
def plot_training_history(log_history, model_name):
    train_loss = [x['loss'] for x in log_history if 'loss' in x]
    val_loss = [x['eval_loss'] for x in log_history if 'eval_loss' in x]
    val_accuracy = [x['eval_accuracy'] for x in log_history if 'eval_accuracy' in x]
    val_f1 = [x['eval_f1_macro'] for x in log_history if 'eval_f1_macro' in x]
    
    if not train_loss:
        print("No training history to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_train = range(1, len(train_loss) + 1)
    epochs_val = range(1, len(val_loss) + 1)
    
    ax1.plot(epochs_train, train_loss, marker='o', label='Train Loss')
    ax1.plot(epochs_val, val_loss, marker='s', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title(f'{model_name} - Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_val, val_accuracy, marker='o', label='Accuracy')
    ax2.plot(epochs_val, val_f1, marker='s', label='F1 (macro)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score'); ax2.set_title(f'{model_name} - Metrics'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.tight_layout()
    os.makedirs("./models", exist_ok=True)
    safe_title = title.replace(' ', '_').replace('-', '_')
    plt.savefig(f'./models/{safe_title}_confusion_matrix.png')
    plt.show()

# Train model
def train_model(model_name, dataset_name, num_labels, text_col='text_normalized', label_col='sentiment',
                batch_size=16, learning_rate=2e-5, epochs=5, fp16=True):
    
    print(f"Training: {model_name} on {dataset_name}")
    
    train_df, val_df, test_df = load_data(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        normalization=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    print(f"Model Name: {model_name}, Number of labels: {num_labels}, Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    train_dataset, val_dataset, test_dataset = create_datasets(train_df, val_df, test_df, tokenizer, text_col, label_col)
    
    save_name = f"{model_name.split('/')[-1]}_{dataset_name}"
    output_dir = f"./models/{save_name}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        seed=SEED,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        disable_tqdm=False,
        log_level='info',
        fp16=fp16,
        ddp_find_unused_parameters=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=calc_metrics
    )
    
    trainer.train()
    
    plot_training_history(trainer.state.log_history, save_name)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved")
    
    # Evaluate
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    if dataset_name == 'sentiment140':
        label_names = ['Negative', 'Positive']
    else:
        with open('./processed_data/ipc_label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            label_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))
    plot_confusion_matrix(test_labels, test_preds, label_names, f'{save_name} - Confusion Matrix')
    
    results = {'model': model_name, 'dataset': dataset_name, 'accuracy': float(test_accuracy), 'f1_macro': float(test_f1)}
    
    with open(f'{output_dir}/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/test_results.json")
    return model, results
