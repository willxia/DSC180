import os
import pandas as pd
from sklearn.model_selection import train_test_split

def export_dataframe(df, filename):
    df.to_csv(os.path.join('data', 'filename'), index=False)

def train_test_val_split(df):
    train, temp = train_test_split(df, test_size=0.4, random_state=42)
    test, val = train_test_split(temp, test_size=0.5, random_state=42)
    return train, test, val