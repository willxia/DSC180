import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split


import re
import nltk
from nltk.corpus import stopwords
import string
import os

inflows = pd.read_parquet('Transacation_inflows_with_date_3k.pqt')
outflows = pd.concat([
    pd.read_parquet('Transacation_outflows_with_date_3k_firsthalf.pqt'),
    pd.read_parquet('Transacation_outflows_with_date_3k_secondhalf.pqt')
])


def lower_strip_memo(df):
    df['memo'] = df['memo'].str.lower()
    df['memo'] = df['memo'].str.strip()


def remove_stop_memo(df):
    stop = stopwords.words('english')
    # remove stopwords in the list
    df['memo'] = df['memo'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop)]))


def remove_punctuation_memo(df):
    punctuation_to_keep = {'-', "'"}
    punctuation_to_remove = ''.join(
        set(string.punctuation) - punctuation_to_keep)

    # Escape punctuation characters that need to be escaped
    punctuation_to_remove = re.escape(punctuation_to_remove)

    # Remove specified punctuation and handle '-'
    df['memo'] = df['memo'].str.replace(
        f'[{punctuation_to_remove}-]', '', regex=True)

    # Replace underscores with spaces
    df['memo'] = df['memo'].str.replace('_', ' ')


def remove_x_memo(df):
    alphabet = set('abcdefghijklmnopqrstuvwxyz')

    def process_memo(memo):
        splits = memo.split(' ')
        results = [s for s in splits if not alphabet.intersection(set(s)) == set(
            'x') and s not in ['dates', 'date'] and s.count('x') < 3]
        return ' '.join(results)

    df['memo'] = df['memo'].apply(process_memo)


def clean_memo(df):
    lower_strip_memo(df)
    remove_stop_memo(df)
    remove_punctuation_memo(df)
    remove_x_memo(df)


clean_memo(outflows)

outflows = outflows.reset_index()

# save the cleaned data
outflows.to_parquet('outflows_cleaned.pqt')
