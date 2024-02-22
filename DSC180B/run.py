
import sys
import json
import pandas as pd

from features import get_data, get_categorical_features
from models import train_models, test_models
from utils import export_dataframe, train_test_val_split


def run():
    """
    Main function to run the script.

    This function retrieves data, generates categorical and income features,
    exports them to CSV files, merges the features, splits the data into
    training, testing, and validation sets, trains a model, and tests the model.
    """
    data = get_data()

    categorical_features_df = get_categorical_features(data)
    export_dataframe(categorical_features_df, 'categorical_features.csv')

    income_features_df = get_categorical_features(data)
    export_dataframe(income_features_df, 'income_features.csv')

    features_df = categorical_features_df.merge(
        income_features_df,
        on='prism_consumer_id',
        how='left'
    )

    train, test, val = train_test_val_split(features_df)
    train_models(train)
    test_models(test)


if __name__ == '__main__':
    run()
