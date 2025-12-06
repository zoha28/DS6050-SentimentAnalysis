import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
import time
import wandb

warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Configuration for ablations. This sets the baseline from our original traing, and ABLATION_CONFIGS sets everythign we will need to test
BASELINE_CONFIG = {
    'learning_rate': 2e-5,
    'epochs': 5,
    'weight_decay': 0.01,
    'batch_size': 16,
    'model_name': 'roberta-base',
    'use_scheduler': False,
    'max_length': 128,
    'fp16': True
}

ABLATION_CONFIGS = {
    'learning_rate': {
        'variants': ['1e-5', '2e-5', '3e-5', 'scheduler'],
        'baseline': '2e-5'
    },
    'epochs': {
        'variants': [3, 5, 10],  # 15 can be added later if we need it
        'baseline': 5
    },
    'weight_decay': {
        'variants': [0.001, 0.01, 0.1],
        'baseline': 0.01
    },
    'domain_transfer': {
        'variants': ['from_scratch', 'pretrain_sentiment140'],
        'baseline': 'from_scratch'
    }
}

# Dataset Class, same as OG function
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

# I updated the load_date function to include the new upsampled/balanced datasets
def load_data(dataset_name, data_dir='./processed_data'):
    """Load train, val, test data for specified dataset."""
    if dataset_name in ['ipc', 'gazawar']:
        train = pd.read_csv("~/Deep_Learning/processed_data/train_ipc_oversampled.csv")
        val = pd.read_csv("~/Deep_Learning/processed_data/val_ipc_balanced.csv")
        test = pd.read_csv("~/Deep_Learning/processed_data/test_ipc_balanced.csv")
    elif dataset_name == 'sentiment140':
        train = pd.read_csv(f'{data_dir}/{dataset_name}_train.csv')
        val = pd.read_csv(f'{data_dir}/{dataset_name}_val.csv')
        test = pd.read_csv(f'{data_dir}/{dataset_name}_test.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {dataset_name}: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
    return train, val, test

def create_datasets(train_df, val_df, test_df, tokenizer, text_col='text_normalized', label_col='sentiment', max_length=128):
    """Create PyTorch datasets from dataframes."""
    train_dataset = TweetDataset(train_df[text_col].values, train_df[label_col].values, tokenizer, max_length)
    val_dataset = TweetDataset(val_df[text_col].values, val_df[label_col].values, tokenizer, max_length)
    test_dataset = TweetDataset(test_df[text_col].values, test_df[label_col].values, tokenizer, max_length)
    return train_dataset, val_dataset, test_dataset

def get_label_names(dataset_name):
    """Get label names for confusion matrix and classification report."""
    if dataset_name == 'sentiment140':
        return ['Negative', 'Positive']
    else:
        with open('./processed_data/ipc_label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            return [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]

# Added per-class metrics to 
def calc_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1_per_class = f1_score(labels, predictions, average=None)

    per_class_dict = {f"f1_class_{i}": f for i, f in enumerate(f1_per_class)}

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
        **per_class_dict
    }

# Plotting functions, same as before but now it saves the training_history plot
def plot_training_history(log_history, save_path):
    """Plot training and validation metrics over epochs."""
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
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_val, val_accuracy, marker='o', label='Accuracy')
    ax2.plot(epochs_val, val_f1, marker='s', label='F1 (macro)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# This allows AdamW optimizer to run, and overrider HF defaults so we can run ablations
class CustomTrainer(Trainer):
    """Custom Trainer to use explicit AdamW optimizer and optional scheduler."""
    
    def __init__(self, *args, learning_rate=2e-5, weight_decay=0.01, use_scheduler=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_lr = learning_rate
        self.custom_wd = weight_decay
        self.use_scheduler = use_scheduler
        
    def create_optimizer(self):
        """Create AdamW optimizer with custom parameters."""
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.custom_wd,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.custom_lr)
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create learning rate scheduler if use_scheduler is True."""
        if self.use_scheduler:
            if self.lr_scheduler is None:
                num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

# original train_model function, new changes are the addition of WandB and new metrics (micro f1, per class f1)
def train_model(
    model_name='roberta-base',
    dataset_name='gazawar',
    num_labels=None,
    text_col='text_normalized',
    label_col='sentiment',
    batch_size=16,
    learning_rate=2e-5,
    epochs=5,
    weight_decay=0.01,
    use_scheduler=False,
    fp16=True,
    pretrained_model_path=None,
    run_name=None,
    wandb_tags=None,
    save_model=True
):
    """
    Train a model with specified hyperparameters.
    
    Args:
        model_name: HuggingFace model identifier or path to local model
        dataset_name: 'sentiment140', 'gazawar', or 'ipc'
        num_labels: Number of output labels (auto-detected if None)
        pretrained_model_path: Path to pretrained checkpoint for fine-tuning
        run_name: Custom name for W&B run
        wandb_tags: List of tags for W&B
        save_model: Whether to save the trained model
    """
    
    # Load data
    train_df, val_df, test_df = load_data(dataset_name)
    if dataset_name in ['gazawar', 'ipc']:
        label_col = 'label'
    # Auto-detect number of labels
    if num_labels is None:
        num_labels = train_df[label_col].nunique()
    
    # Initialize W&B
    config = {
        'model_name': model_name,
        'dataset': dataset_name,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'use_scheduler': use_scheduler,
        'num_labels': num_labels,
        'seed': SEED,
        'pretrained_from': pretrained_model_path if pretrained_model_path else 'scratch'
    }
    
    wandb.init(
        project='ipc-sentiment-ablation',
        name=run_name,
        config=config,
        tags=wandb_tags if wandb_tags else [],
        reinit=True
    )
    
    print(f"Training: {model_name} on {dataset_name}")
    print(f"Run: {run_name}")
    print(f"Config: lr={learning_rate}, epochs={epochs}, wd={weight_decay}, scheduler={use_scheduler}")
    print(f"{'-'*80}\n")
    
    # Load tokenizer and model
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from: {pretrained_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=True, normalization=False)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path,
                                                                   num_labels=num_labels,
                                                                   ignore_mismatched_sizes=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, normalization=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=num_labels,
                                                                   ignore_mismatched_sizes=True)
    
    print(f"Model: {model_name}, Labels: {num_labels}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df, tokenizer, text_col, label_col
    )
    
    # Define directories
    SCRATCH_DIR = './models'
    RESULTS_DIR = os.path.expanduser('~/Deep_Learning/Results')
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    save_name = run_name if run_name else f"{model_name.split('/')[-1]}_{dataset_name}"
    output_dir = os.path.join(SCRATCH_DIR, save_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,  # Will be overridden by custom optimizer
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,  # Will be overridden by custom optimizer
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="wandb",
        seed=SEED,
        logging_dir=f'{output_dir}/logs',
        disable_tqdm=False,
        log_level='info',
        fp16=fp16,
        ddp_find_unused_parameters=False
    )
    
    # Create custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=calc_metrics,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_scheduler=use_scheduler
    )
    
    # Train
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {type(e).__name__} - {e}")
        wandb.finish(exit_code=1)
        return None, None
    
    # Save model
    if save_model:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    # Evaluate on test set
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_micro = f1_score(test_labels, test_preds, average='micro')
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    test_f1_per_class = f1_score(test_labels, test_preds, average=None)

    # Get label names
    label_names = get_label_names(dataset_name)
    
    # Print classification report
    print("TEST SET RESULTS")
    print("-"*80)
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))
    
    # Save results
    result_prefix = os.path.join(RESULTS_DIR, save_name)
    os.makedirs(result_prefix, exist_ok=True)
    
    # Plot confusion matrix and training history
    plot_confusion_matrix(test_labels, test_preds, label_names, 
                         f'{result_prefix}/{save_name}_confusion_matrix.png')
    plot_training_history(trainer.state.log_history, 
                         f'{result_prefix}/{save_name}_training_history.png')
    
    # Log plots to W&B
    wandb.log({
        "confusion_matrix": wandb.Image(f'{result_prefix}/{save_name}_confusion_matrix.png'),
        "training_history": wandb.Image(f'{result_prefix}/{save_name}_training_history.png')
    })
    
    # Compile results
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'run_name': run_name,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'use_scheduler': use_scheduler,
        'accuracy': float(test_accuracy),
        'f1_macro': float(test_f1_macro),
        'f1_micro': float(test_f1_micro),
        'f1_weighted': float(test_f1_weighted),
        'precision_macro': float(test_precision),
        'recall_macro': float(test_recall)
    }

    # Add per-class F1
    for i, f1 in enumerate(test_f1_per_class):
        key = f'f1_class_{i}'
        results[key] = float(f1)
        wandb.log({f"test_{key}": float(f1)})
        
    # Log final metrics to W&B
    wandb.log({
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_f1_weighted': test_f1_weighted,
        'test_precision_macro': test_precision,
        'test_recall_macro': test_recall
    })

    # Save results JSON
    results_path = f'{result_prefix}/test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    wandb.finish()
    
    return model, results

# This is the function for the fine-tuning model
def train_two_stage(
    stage1_dataset='sentiment140',
    stage2_dataset='gazawar',
    **shared_params
):
    """
    Two-stage training: pretrain on Sentiment140, then fine-tune on GazaWar.
    
    Args:
        stage1_dataset: Dataset for pretraining (default: sentiment140)
        stage2_dataset: Dataset for fine-tuning (default: gazawar)
        **shared_params: Hyperparameters to use for both stages
    """
    
    print("\n" + "="*80)
    print("STAGE 1: Pretraining on Sentiment140")
    print("="*80 + "\n")
    
    # Stage 1: Train on Sentiment140
    stage1_model, stage1_results = train_model(
        model_name=shared_params['model_name'],
        dataset_name=stage1_dataset,
        num_labels=2,
        batch_size=shared_params['batch_size'],
        learning_rate=shared_params['learning_rate'],
        epochs=shared_params['epochs'],
        weight_decay=shared_params['weight_decay'],
        use_scheduler=shared_params['use_scheduler'],
        fp16=shared_params['fp16'],
        run_name=f"roberta_{stage1_dataset}_stage1",
        wandb_tags=['ablation', 'domain_transfer', 'stage1', stage1_dataset],
        save_model=True
    )
    
    if stage1_model is None:
        print("Stage 1 training failed!")
        return None, None
    
    # Get path to stage 1 model
    stage1_model_path = f"./models/roberta_{stage1_dataset}_stage1"
    
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning on GazaWar")
    print("="*80 + "\n")
    
    # Stage 2: Fine-tune on GazaWar
    stage2_model, stage2_results = train_model(
        model_name=shared_params['model_name'],
        dataset_name=stage2_dataset,
        num_labels=6,
        batch_size=shared_params['batch_size'],
        learning_rate=shared_params['learning_rate'],
        epochs=shared_params['epochs'],
        weight_decay=shared_params['weight_decay'],
        use_scheduler=shared_params['use_scheduler'],
        fp16=shared_params['fp16'],
        pretrained_model_path=stage1_model_path,
        run_name=f"roberta_{stage2_dataset}_domain-transfer",
        wandb_tags=['ablation', 'domain_transfer', 'stage2', stage2_dataset],
        save_model=True
    )
    
    # Add stage 1 info to stage 2 results
    if stage2_results:
        stage2_results['stage1_dataset'] = stage1_dataset
        stage2_results['stage1_f1'] = stage1_results['f1_macro']
        stage2_results['stage1_accuracy'] = stage1_results['accuracy']
    
    return stage2_model, stage2_results

#Check if what we are running is the baseline model or an ablation variant
def is_baseline_config(ablation_type, variant):
    """Check if the given variant is the baseline for this ablation type."""
    if ablation_type not in ABLATION_CONFIGS:
        return False
    baseline = ABLATION_CONFIGS[ablation_type]['baseline']
    return str(variant) == str(baseline)
    
#fully new, runs the actual ablations
def run_ablations(
    ablation_type,
    variant=None,
    dataset_name='gazawar',
    pretrained_model_path=None,
    **override_params
):
    """
    Run a single ablation experiment.
    
    Args:
        ablation_type: 'learning_rate', 'epochs', 'weight_decay', 'domain_transfer', 'combined'
        variant: Specific variant to test (e.g., '1e-5', 3, 0.1, 'pretrain_sentiment140')
        dataset_name: 'sentiment140' or 'gazawar'/'ipc'
        pretrained_model_path: Path to pretrained model (for domain transfer or combined)
        **override_params: For 'combined' type, override specific hyperparameters
    
    Returns:
        model, results dictionary
    """
    
    # Start with baseline config
    config = BASELINE_CONFIG.copy()
    
    # Determine number of labels
    if dataset_name == 'sentiment140':
        num_labels = 2
    else:  # gazawar/ipc
        num_labels = 6
    
    # Generate run name and tags
    tags = ['ablation', ablation_type, dataset_name, config['model_name']]
    
    if ablation_type == 'combined':
        # Combined best model - use override_params
        config.update(override_params)
        run_name = f"roberta_{dataset_name}_combined-best"
        tags.append('combined')
        variant_str = 'combined'
        
    elif ablation_type == 'domain_transfer':
        if variant == 'pretrain_sentiment140':
            # Two-stage training
            return train_two_stage(dataset_name=dataset_name, **config)
        else:  # from_scratch
            run_name = f"roberta_{dataset_name}_from-scratch"
            variant_str = 'from_scratch'
            
    elif ablation_type == 'learning_rate':
        if variant == 'scheduler':
            config['use_scheduler'] = True
            config['learning_rate'] = 2e-5  # Base LR for scheduler
            variant_str = 'scheduler'
        else:
            config['learning_rate'] = float(variant)
            variant_str = variant
        run_name = f"roberta_{dataset_name}_lr-{variant_str}"
        tags.append(f"lr_{variant_str}")
        
    elif ablation_type == 'epochs':
        config['epochs'] = int(variant)
        variant_str = str(variant)
        run_name = f"roberta_{dataset_name}_epochs-{variant_str}"
        tags.append(f"epochs_{variant_str}")
        
    elif ablation_type == 'weight_decay':
        config['weight_decay'] = float(variant)
        variant_str = str(variant)
        run_name = f"roberta_{dataset_name}_wd-{variant_str}"
        tags.append(f"wd_{variant_str}")
        
    else:
        raise ValueError(f"Unknown ablation_type: {ablation_type}")
    
    # Add baseline tag if this is the baseline configuration
    if is_baseline_config(ablation_type, variant):
        tags.append('baseline')
    
    # Train model
    model, results = train_model(
        model_name=config['model_name'],
        dataset_name=dataset_name,
        num_labels=num_labels,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        weight_decay=config['weight_decay'],
        use_scheduler=config['use_scheduler'],
        fp16=config['fp16'],
        pretrained_model_path=pretrained_model_path,
        run_name=run_name,
        wandb_tags=tags
    )
    
    return model, results

if __name__ == "__main__":
    # Example usage
    print("This script should be run via run_ablation.py")
    print("Example: python run_ablation.py --ablation_type learning_rate --variant 1e-5 --dataset gazawar")
