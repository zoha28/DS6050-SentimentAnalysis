# run_training.py
from train_model import train_model
import pandas as pd

if __name__ == "__main__":
    all_results = []

    # -------------------------------
    # BERT-base on Sentiment140
    # -------------------------------
    print("\nBERT-base on Sentiment140")
    model, results = train_model(
        model_name='bert-base-uncased',
        dataset_name='sentiment140',
        num_labels=2,
        label_col='sentiment',
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        fp16=True
    )
    all_results.append(results)

    # -------------------------------
    # BERT-base on IPC
    # -------------------------------
    print("\nBERT-base on IPC")
    model, results = train_model(
        model_name='bert-base-uncased',
        dataset_name='ipc',
        num_labels=6,
        label_col='label_id',
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        fp16=True
    )
    all_results.append(results)

    # -------------------------------
    # RoBERTa-base on Sentiment140
    # -------------------------------
    print("\nRoBERTa-base on Sentiment140")
    model, results = train_model(
        model_name='roberta-base',
        dataset_name='sentiment140',
        num_labels=2,
        label_col='sentiment',
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        fp16=True
    )
    all_results.append(results)

    # -------------------------------
    # RoBERTa-base on IPC
    # -------------------------------
    print("\nRoBERTa-base on IPC")    
    model, results = train_model(
        model_name='roberta-base',
        dataset_name='ipc',
        num_labels=6,
        label_col='label_id',
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        fp16=True
    )
    all_results.append(results)

    # -------------------------------
    # BERTweet on Sentiment140
    # -------------------------------
    print("\nBERTweet on Sentiment140")
    model, results = train_model(
        model_name='vinai/bertweet-base',
        dataset_name='sentiment140',
        num_labels=2,
        label_col='sentiment',
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        fp16=True
    )
    all_results.append(results)

    # -------------------------------
    # BERTweet on IPC
    # -------------------------------
    print("\nBERTweet on IPC")
    model, results = train_model(
        model_name='vinai/bertweet-base',
        dataset_name='ipc',
        num_labels=6,
        label_col='label_id',
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        fp16=True
    )
    all_results.append(results)

    # -------------------------------
    # Compare final results
    # -------------------------------
    print("\nFinal results:")

    comparison_df = pd.DataFrame(all_results)
    print("\n", comparison_df.to_string(index=False))

    comparison_df.to_csv('./models/final_comparison.csv', index=False)

    print("\nAll models complete")
    print("Results saved to ./models/")

    # -------------------------------
    # Example: reload a trained model
    # -------------------------------
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # model = AutoModelForSequenceClassification.from_pretrained('./models/bert-base-uncased_sentiment140')
    # tokenizer = AutoTokenizer.from_pretrained('./models/bert-base-uncased_sentiment140')
