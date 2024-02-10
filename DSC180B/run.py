#!/usr/bin/env python

import sys
import json

from features import get_data, get_categorical_features
from models import train_model, test_model
from utils import export_dataframe, train_test_val_split

def run():
    data = get_data()

    categorical_features_df = get_categorical_features(data)
    export_dataframe(categorical_features_df, 'categorical_features.csv')
    
    income_features_df = get_categorical_features(data)
    export_dataframe(income_features_df, 'income_features.csv')

    train, test, val = train_test_val_split(categorical_features_df)
    train_model(train)
    test_model(test)


if __name__ == '__main__':
    run()