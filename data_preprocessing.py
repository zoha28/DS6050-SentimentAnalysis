#### Imports and Setup

import pandas as pd
import numpy as np
import re
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)



#### Text Preprocessing Functions

def clean_tweet(text):
    """Remove HTML entities and control characters"""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'&[a-z]+;', '', text)  # Remove &amp; etc
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control chars
    
    return text.strip()

def normalize_tweet(text):
    # Normalize tweets following Nguyen et al. ( URLs -> HTTPURL, @mentions -> @USER,Emojis -> text descriptions

    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'http\S+|www\.\S+', 'HTTPURL', text)
    text = re.sub(r'@\w+', '@USER', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


### Dataset Loading Functions

def load_sentiment140(file_path, sample_size=400000):
    #Load dataset
    print("Load Sentiment140 dataset")
    
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    df = pd.read_csv(
        file_path,
        encoding='latin-1',
        header=None,
        names=columns,
        usecols=[0, 5],
        on_bad_lines='skip'
    )
    
    print(f"Loaded {len(df):,} tweets")
    
    # Sample balanced classes
    negative = df[df['target'] == 0].sample(n=sample_size//2, random_state=SEED)
    positive = df[df['target'] == 4].sample(n=sample_size//2, random_state=SEED)
    df = pd.concat([negative, positive]).reset_index(drop=True)
    
    # Create sentiment column
    df['sentiment'] = df['target'].map({0: 0, 4: 1})
    df = df[['text', 'sentiment']].copy()
    
    # Remove empty tweets
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    
    print(f"Sampled {len(df):,} tweets")
    print(f"  Negative: {(df['sentiment']==0).sum():,}")
    print(f"  Positive: {(df['sentiment']==1).sum():,}")
    
    return df

def load_gazawar(file_path):   
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} tweets")
    
    # Adjust column names if needed
    if 'tweet' in df.columns:
        df = df.rename(columns={'tweet': 'text'})
    if 'stance' in df.columns:
        df = df.rename(columns={'stance': 'label'})
    
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    return df


#### Preprocessing Pipeline

def preprocess_dataset(df, text_col='text'):
    #Apply cleaning and normalization
    
    df['text_clean'] = df[text_col].apply(clean_tweet)
    df['text_normalized'] = df['text_clean'].apply(normalize_tweet)
    
    print(f"Final: {len(df):,} tweets")
    print(f"Avg length: {df['text_length'].mean():.0f} chars")
    
    return df

def prepare_gazawar_labels(df):
    #Convert text labels to numeric IDs
    unique_labels = sorted(df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_id'] = df['label'].map(label_mapping)
    
    print("\nLabel mapping:")
    for label, idx in label_mapping.items():
        count = (df['label'] == label).sum()
        print(f"  {idx}: {label} ({count:,} tweets)")
    
    return df, label_mapping


#### Train/Val/Test Split

def split_data(df, label_col='sentiment', test_size=0.1, val_size=0.1):
    """
    Split data: 80% train, 10% val, 10% test (stratified)
    """
    print(f"\nSplitting data (80/10/10)...")
    
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df[label_col]
    )
    
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=SEED, stratify=train_val[label_col]
    )
    
    print(f"Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test


#### Visualization

def plot_distribution(df, label_col, title):
    """Plot label distribution"""
    plt.figure(figsize=(10, 6))
    df[label_col].value_counts().sort_index().plot(kind='bar', color='steelblue')
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_text_lengths(df, title):
    """Plot text length distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['text_length'], bins=50, color='steelblue', edgecolor='black')
    plt.axvline(df['text_length'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["text_length"].mean():.0f}')
    plt.title(title)
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_dataset(df, dataset_name, label_col='sentiment'):
    """Complete dataset analysis"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Analysis")
    print(f"{'='*60}")
    print(f"\nTotal: {len(df):,} tweets")
    print(f"\nText length stats:")
    print(df['text_length'].describe())
    print(f"\nLabel counts:")
    print(df[label_col].value_counts())
    
    plot_distribution(df, label_col, f'{dataset_name} - Label Distribution')
    plot_text_lengths(df, f'{dataset_name} - Text Lengths')


#### Save Functions

def save_splits(train, val, test, dataset_name, output_dir='./processed_data'):
    """Save splits to CSV"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train.to_csv(f'{output_dir}/{dataset_name}_train.csv', index=False)
    val.to_csv(f'{output_dir}/{dataset_name}_val.csv', index=False)
    test.to_csv(f'{output_dir}/{dataset_name}_test.csv', index=False)
    
    print(f"\nSaved to {output_dir}/")


#### Main Execution

if __name__ == "__main__":
    print("Data preprocessing")
    
    # Process Sentiment140
    print("\n Sentiment140")
    df_s140 = load_sentiment140('sentiment140.csv', sample_size=400000)
    df_s140 = preprocess_dataset(df_s140)
    analyze_dataset(df_s140, 'Sentiment140', label_col='sentiment')
    s140_train, s140_val, s140_test = split_data(df_s140, label_col='sentiment')
    save_splits(s140_train, s140_val, s140_test, 'sentiment140')
    
    # Process GazaWar
    print("\n\n GazaWar")
    df_gaza = load_gazawar('gazawar.csv')
    df_gaza = preprocess_dataset(df_gaza)
    df_gaza, label_mapping = prepare_gazawar_labels(df_gaza)
    
    import json
    with open('./processed_data/gazawar_label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    analyze_dataset(df_gaza, 'GazaWar', label_col='label_id')
    gaza_train, gaza_val, gaza_test = split_data(df_gaza, label_col='label_id')
    save_splits(gaza_train, gaza_val, gaza_test, 'gazawar')
    
    print("Preprocessing complete")

