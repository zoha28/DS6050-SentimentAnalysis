# DS6050-SentimentAnalysis
Sentiment Analysis of the Israeli-Palestinian Conflict

## Overview
This repository contains the full experimental pipeline for a sentiment-classification study using transformer models on two Twitter datasets: Sentiment140 (1.6M tweets, binary sentiment) and GazaWar (7,369 tweets, six-way sentiment categories). The primary objective is to evaluate how well general-purpose sentiment models transfer to a domain-specific geopolitical dataset, and to characterize which modeling choices most influence performance.

The project includes three baseline transformer architectures—BERT, RoBERTa, and BERTweet—implemented in both standalone training scripts and an interactive jupyter notebook. Following this, the repository contains a modular ablation study framework that allows controlled variation of key training components, including learning rate, epoch counts, weight decay, and transfer-learning strategies.

All experiment configurations are logged to Weights & Biases (W&B), and results can be aggregated into summary tables for analysis. The repository provides complete preprocessing scripts, model training utilities, batch execution tools, and result aggregation modules to fully reproduce the experimental workflow.

## Repository Structure
-  Parent folder
    - Files for data pre-processing:
        - `data_preprocessing.py`
            - File for preprocessing our data, including soft normalization, removing personal information, and creating the train/test split.
        - `Sample_rebalancing_ipc.ipynb`
            - File for upsampling out data to deal with class imbalance. The upsampling technique used here was a custom technique based on a GeeksForGeeks tutorial.
    - Files for initial model testing, designed to compare the performance of BERT, RoBERTa, and BERTweet bases:
        - `model-evaluation.ipynb`
            -  Jupyter notebook that contains all the code to run all three models
        - `train_model.py`
            - Python file to run all three models, to be used in tandem with the below file.
            - This file support Parallel Data Processing and is therefore slightly more efficient than the jupyter notebook.
        - `train_all_model.py`
            - Calls the previous python file and tests all three models.
            - Can be ran at the CLI with the command ' '
    - Files for running the ablation studies, including transfer learning from Sentiment140 dataset to GazaWar dataset
        - `train_model_albation.py`
            - Main source code file that details how to run the ablations in a modular way.
            - Sets baseline model and potential values for ablations.
            - Logs each ablation run to WandB.
            - Updated to include AdamW Optimizer
        - `run_ablation.py`
            - Wrapper script that prepares and launches individual ablation runs.
            - Passes experiment settings into `train_model_ablation.py`
            - Potential options are all included in `batch_scripts.py`
        - `batch_scripts.py`
            - Contains CLI-ready command templates for running multiple ablations sequentially.
            - Users have the option to input individual commands at the CLI to run one ablation at a time.
            - Can be ran as a whole with the command `python batch_scripts.py --run_all`. Please note, running all ablations at once is very time and resource intensive, so it is recommended to run one at a time.
        - `aggregate_results.py`
            - All results are sent to the 'results' folder, and this file will aggregate the results into cleaner tables and outputs.
-  Subfolders
    - Processed data for training, testing, and validation
    - CSV files and images of results including training curves and confusion matrices
- `Requirements.txt`
    - This file details the necessary dependencies
    - It can be installed at the CLI with 'pip install -r requirements.txt'
