# DS6050-SentimentAnalysis
Sentiment Analysis of the Israeli-Palestinian Conflict

## Overview
This repository contains two implementations of sentiment analysis:
- A Python script (`train_all_models.py`) for command-line execution. The intention is to run base training on Sentiment140 and GazaWar datasets on all three models
- A Python script (`train_model.py`) for command-line execution. This option was used to just run one model at a time when there were issues with our cloud computing resources. Currently this is set to run Bert base.
- A Jupyter Notebook (`model-evaluation.ipynb`) for an interactive version of the training.

## Repository Structure
-  Parent folder
    - Options for model training (train_all_models.py, train_model.py, model-evaluation.ipynb)
    - Requirements text
    - Images of initial results
-  Subfolder
    - Processed data for training, testing, and validation
## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries:
  - pandas
  - scikit-learn
  - nltk (or other text preprocessing libraries)
  - matplotlib / seaborn (optional for visualization)
  - pytorch
  - torchvision
  - torchaudio
  - transformers
  - datasets
  - accelerate
  - numpy
  - seaborn

Install dependencies:
```bash
pip install -r requirements.txt
