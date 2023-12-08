# Might take upt o 15 minutes to run

from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import holidays
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


us_holidays = holidays.UnitedStates()

# Read in data
outflows = pd.read_parquet('outflows_cleaned.pqt')

outflows = outflows[outflows["category"] != outflows['memo']]
outflows = outflows.reset_index(drop=True)

# Feature Engineering
outflows['ordinal_date'] = outflows['posted_date'].apply(
    lambda x: x.toordinal())
outflows['amount_quartile'] = pd.qcut(outflows['amount'], 10, labels=False)
outflows['weekday'] = outflows['posted_date'].apply(
    lambda x: x.weekday() if x else None)
outflows['month'] = outflows['posted_date'].apply(
    lambda x: x.month if x else None)
outflows['isHoliday'] = outflows['posted_date'].apply(
    lambda x: 1 if x in us_holidays else 0)

df = outflows
data = df[['memo', 'category', 'ordinal_date',
           'amount_quartile', 'weekday', 'month', 'isHoliday']]

# TFIDF Only (No GridSearchCV)
# Define the transformations for each column
text_transformer = ('tfidf', TfidfVectorizer(stop_words='english'), 'memo')

preprocessor = ColumnTransformer(
    transformers=[
        text_transformer,
    ],
    remainder='passthrough'
)

pipeline_SGC = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
     alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

pipeline_Logistic = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=0))
])

pipeline_NB = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', MultinomialNB())
])

# Split the data into training and test sets, can't stratify because one category has only one sample
X_train, X_test, y_train, y_test = train_test_split(
    data[['memo']], data['category'], test_size=0.2, random_state=42)

# Fit the model
pipeline_SGC.fit(X_train, y_train)
pipeline_Logistic.fit(X_train, y_train)
pipeline_NB.fit(X_train, y_train)

# Evaluate the model
score_SGC = pipeline_SGC.score(X_test, y_test)
score_Logistic = pipeline_Logistic.score(X_test, y_test)
score_NB = pipeline_NB.score(X_test, y_test)

print(f'Score of SGDClassifier: {score_SGC}')
print(f'Score of LogisticRegression: {score_Logistic}')
print(f'Score of MultinomialNB: {score_NB}')

y_pred_SGC = pipeline_SGC.predict(X_test)
y_pred_Logistic = pipeline_Logistic.predict(X_test)
y_pred_NB = pipeline_NB.predict(X_test)

print(classification_report(y_test, y_pred_SGC))
print(classification_report(y_test, y_pred_Logistic))
print(classification_report(y_test, y_pred_NB))

# TFIDF + Other Features (No GridSearchCV)

# Define the transformations for each column
# cant do standardize for ordinal_date because Naive Bayes can't handle negative values, actually, it can't even handle continuous values from tfidf
# Also one hot encoding should let us use Bernoulli Naive Bayes, so overall this is expected to be bad
text_transformer = ('tfidf', TfidfVectorizer(stop_words='english'), 'memo')
standardize_transformer = ('normalize', MinMaxScaler(), ['ordinal_date'])
one_hot_transformer = ('one_hot', OneHotEncoder(), [
                       'amount_quartile', 'weekday', 'month'])

preprocessor = ColumnTransformer(
    transformers=[
        text_transformer,
        standardize_transformer,
        one_hot_transformer
    ],
    remainder='passthrough'
)

pipeline_SGC = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
     alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

pipeline_Logistic = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=0))
])

pipeline_NB = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', MultinomialNB())
])

# Split the data into training and test sets, can't stratify because one category has only one sample
X_train, X_test, y_train, y_test = train_test_split(data.drop(
    'category', axis=1), data['category'], test_size=0.2, random_state=42)

# Fit the model
pipeline_SGC.fit(X_train, y_train)
pipeline_Logistic.fit(X_train, y_train)
pipeline_NB.fit(X_train, y_train)

# Evaluate the model
score_SGC = pipeline_SGC.score(X_test, y_test)
score_Logistic = pipeline_Logistic.score(X_test, y_test)
score_NB = pipeline_NB.score(X_test, y_test)

print(f'Score of SGDClassifier: {score_SGC}')
print(f'Score of LogisticRegression: {score_Logistic}')
print(f'Score of MultinomialNB: {score_NB}')
