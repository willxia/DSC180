
# DSC180 - Models for Creditworthiness 

Machine Learning for credit score calculation. Python Version 3.9.5

## Description

In the complex landscape of financial services, creditworthiness is vital for managing financial risks, ensuring economic stability, and making reasonable lending decisions. The FICO score, the standard measure of creditworthiness, is updated depending on the creditor’s reporting schedule, typically ranging from one month to forty-five days. As a result, the FICO score fails to capture the rapid changes in an individual’s financial behavior or the emergence of new financial risks. Our technique aims to refine this process and update the score within two weeks, thereby significantly enhancing the efficiency and accuracy of creditworthiness assessment. 

Although the final goal of this project is to predict a "CashScore", the current state only involves predicting the category of a transaction given its memo statement. This will later help us in building holistic features for each customer, which will lead us to predicting a "CashScore"

## Installation

Before you can run the project, you need to install the necessary Python packages. This project depends on several third-party libraries, which are listed below:

- `numpy`: A fundamental package for scientific computing with Python.
- `pandas`: An open-source data analysis and manipulation tool.
- `torch`: An open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
- `nltk`: A leading platform for building Python programs to work with human language data (Natural Language Toolkit).
- `transformers`: A library by Hugging Face offering state-of-the-art general-purpose architectures for Natural Language Processing (NLP).
- `scikit-learn`: A simple and efficient tools for predictive data analysis.

To install these packages, you can use `pip`, which is the package installer for Python. Feel free to do the following in a virtual environment. Run the following command in your terminal:

```bash
# Install required packages
pip install -r requirements.txt
```
Check if Python 3.9.5 is installed:

Open a terminal and run:

```bash
python3.9 --version
```

## Testing Models

To begin working with this project, you'll need to set up your environment and obtain the necessary files and models. Follow these steps to get everything up and running:

1. **Clone the Repository and Download Pre-trained Model**
   - Use `git clone` to copy the project notebook and associated files to your local machine.
   - Download the pre-trained BERT model by running the `saved_models/download_bert_model.py` script.
2. **Demo Script**
   - Now that you have the pre-trained models saved locally, you can test them out by executing the `prediction_demo.py` script.
   - Given a memo statement that you input, our models will predict which transaction category it should belong in.

### Retraining Models

Note: Depending on your machine and hardware, retraining models may take anywhere from 30 minutes to a few days.

1. **Dataset and Preprocessing**
   - In order to retrain our models, you will first need access to the datasets used. To access the datasets, unzip the `data\datasets.zip` file. You may request the password from wxia@ucsd.edu.
   - Next, we will preprocess the data. Executing `preprocessing.py` will create a new parquet file containing our cleaned data with the necessary features.
2. **TF-IDF Logistic Regression Model**
   - If you wish to retrain the TF-IDF Logistic Regression model, execute the `feature_and_modeling.py` file.
3. **BERT Classification Model**
   - If you wish to retrain the BERT model with your data or tweak the training process, execute the entire notebook found in `model_training/bert_classification.ipynb`. This will take you through the entire pipeline from data preprocessing to model training.


