# -*- coding: utf-8 -*-
"""classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oaB2KFggh62WhyO1WMAjPEdu-QXz2tit
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

C = 10
n_splits = 5

"""## Load Data"""
datasets = pd.read_csv('telecom.csv')

"""# Data Preparation"""
datasets.columns = datasets.columns.str.lower().str.replace(' ', '_')
categorical_columns = datasets.dtypes[datasets.dtypes == "object"].index
for col in categorical_columns:
    datasets[col] = datasets[col].str.lower().str.replace(' ', '_')
datasets.totalcharges = pd.to_numeric(datasets.totalcharges, errors="coerce")
datasets.totalcharges = datasets.totalcharges.fillna(0)
datasets.churn = (datasets.churn == "yes").astype(int)
datasets_for_full_train, datasets_for_test = train_test_split(
    datasets, test_size=0.2, random_state=1)
datasets_for_train, datasets_for_val = train_test_split(
    datasets_for_full_train, test_size=0.25, random_state=1)
datasets_for_full_train = datasets_for_full_train.reset_index(drop=True)
datasets_for_train = datasets_for_train.reset_index(drop=True)
datasets_for_val = datasets_for_val.reset_index(drop=True)
datasets_for_test = datasets_for_test.reset_index(drop=True)
labels_for_full_train = datasets_for_full_train.churn.values
labels_for_train = datasets_for_train.churn.values
labels_for_val = datasets_for_val.churn.values
labels_for_test = datasets_for_test.churn.values
del datasets_for_train['churn']
del datasets_for_val['churn']
del datasets_for_test['churn']
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender', 'seniorcitizen', 'partner', 'dependents',
    'phoneservice', 'multiplelines', 'internetservice',
    'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
    'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
    'paymentmethod'
]
# train model


def train(df, labels, C=1.0):
    dicts = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    features = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(features, labels)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical+numerical].to_dict(orient='records')
    features = dv.transform(dicts)

    pred_labels = model.predict_proba(features)[:, 1]
    return pred_labels

# Final Model


scores = []
fold = 0
print('validation the model')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
# Validation
for train_index, val_index in kfold.split(datasets_for_full_train):
    train_datasets = datasets_for_full_train.iloc[train_index]
    val_datasets = datasets_for_full_train.iloc[val_index]

    labels_for_train = train_datasets.churn.values
    labels_for_val = val_datasets.churn.values

    dv, model = train(train_datasets, labels_for_train, C=C)
    pred_labels_for_val = predict(val_datasets, dv, model)

    auc = roc_auc_score(labels_for_val, pred_labels_for_val)
    scores.append(auc)
    print(f"AUC on {fold}")
    fold = fold + 1

print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))

print('training the final model')

dv, model = train(datasets_for_full_train,
                  datasets_for_full_train.churn.values, C=10)
pred_labels_for_test = predict(datasets_for_test, dv, model)
auc = roc_auc_score(labels_for_test, pred_labels_for_test)

print(f'final model {auc}')

print('saving the model')
# Saving the model
output_file = 'model_C=%s.bin' % C
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
