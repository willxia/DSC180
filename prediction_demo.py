from transformers import BertTokenizer, BertForSequenceClassification
import torch
import sklearn
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Tested working with sklearn 1.0.2 only
text = input("Enter your memo for transaction: \n")

filename = input("Select your model: Bert(1) or LogisticRegression(2) \n")
while filename != '1' and filename != '2':
    filename = input("Invalid input. Type 1 or 2: \n")


if filename == '2':
    filename = "logit_model.sav"

    loaded_model = pickle.load(open(filename, 'rb'))

    X = pd.DataFrame([text], columns=['memo'])
    print("Your transaction likely belongs to category:")
    print(loaded_model.predict(X)[0])

elif filename == '1':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        './bank_transaction_model')

    label_to_id = {
        'EDUCATION': 0,
        'FOOD_AND_BEVERAGES': 1,
        'GENERAL_MERCHANDISE': 2,
        'GROCERIES': 3,
        'MORTGAGE': 4,
        'OVERDRAFT': 5,
        'PETS': 6,
        'RENT': 7,
        'TRAVEL': 8
    }

    id_to_label = {val: key for key, val in label_to_id.items()}
    tokens = tokenizer(text, return_tensors='pt')

    outputs = model(**tokens)

    predicted_label = torch.argmax(outputs.logits).item()
    print("Your transaction likely belongs to category:")

    print(id_to_label[predicted_label])
