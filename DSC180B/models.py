import xgboost as xgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def create_models():
    """
    Creates machine learning models.

    Returns:
        list: A list of machine learning models.
    """
    # Define different models
    mod1 = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
    mod2 = GradientBoostingClassifier()
    mod3 = LogisticRegression()
    mod4 = SGDClassifier()
    # Store models in a list
    models = [mod1, mod2, mod3, mod4]
    return models

def train_models(train_df):
    """
    Trains machine learning models using the provided training dataset.

    Args:
        train_df (DataFrame): The training dataset.

    Returns:
        list: A list of trained machine learning models.
    """
    # Define different models
    mod1 = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
    mod2 = GradientBoostingClassifier()
    mod3 = LogisticRegression()
    mod4 = SGDClassifier()
    # Store models in a list
    models = [mod1, mod2, mod3, mod4]

    # Train each model using the training dataset
    for model in models:
        model.fit(train_df.drop['FPF_TARGET'], train_df['FPF_TARGET'])
    return models

def test_models(models, test_df):
    """
    Tests trained machine learning models using the provided testing dataset.

    Args:
        models (list): A list of trained machine learning models.
        test_df (DataFrame): The testing dataset.
    """
    # Iterate through each model
    for model in models:
        print(model)
        print('-----')
        # Make predictions using the testing dataset
        y_pred = model.predict(test_df.drop['FPF_TARGET'])
        # Print evaluation metrics
        print_metrics(test_df['FPF_TARGET'], y_pred) 
        print('\n')

def print_metrics(y_true, y_pred):
    """
    Prints evaluation metrics.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
    """
    print(f"""
Accuracy: {accuracy_score(y_true, y_pred)}
Precision: {precision_score(y_true, y_pred)}
Recall: {recall_score(y_true, y_pred)}
ROC-AUC: {roc_auc_score(y_true, y_pred)}
    """.strip())
