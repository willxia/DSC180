import xgboost as xgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def create_models():
    mod1 = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
    mod2 = GradientBoostingClassifier()
    mod3 = LogisticRegression()
    mod4 = SGDClassifier()
    models = [mod1, mod2, mod3, mod4]
    return models

def train_models(train_df, feature_cols):
    mod1 = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
    mod2 = GradientBoostingClassifier()
    mod3 = LogisticRegression()
    mod4 = SGDClassifier()
    models = [mod1, mod2, mod3, mod4]

    for model in models:
        model.fit(train_df[feature_cols], train_df['FPF_TARGET'])
    return models

def test_models(models, test_df, feature_cols):
    for model in models:
        print(model)
        print('-----')
        y_pred = model.predict(test_df[feature_cols])
        print_metrics(test_df['FPF_TARGET'], y_pred) 
        print('\n')

def print_metrics(y_true, y_pred):
    print(f"""
Accuracy: {accuracy_score(y_true, y_pred)}
Precision: {precision_score(y_true, y_pred)}
Recall: {recall_score(y_true, y_pred)}
ROC-AUC: {roc_auc_score(y_true, y_pred)}
    """.strip())
    
