#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction import DictVectorizer

# Data preparation

# Load base dataset
df = pd.read_csv('Base.csv')

# Delete feature with same value for all instances
df.drop(columns=['device_fraud_count'], inplace=True)

# Classifying features as numerical, categorical or binary
numerical = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit', 'session_length_in_minutes', 'device_distinct_emails_8w', 'month']
categorical = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
binary = ['fraud_bool', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'foreign_request', 'keep_alive_session', 'has_other_cards']

# Splitting the dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fraud_bool
y_val = df_val.fraud_bool
y_test = df_test.fraud_bool
y_full_train = df_full_train.fraud_bool

del df_train['fraud_bool']
del df_val['fraud_bool']
del df_test['fraud_bool']
del df_full_train['fraud_bool']

# Creating dicts
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')
full_train_dicts = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)
X_full_train = dv.transform(full_train_dicts)

features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)

xgb_params = {
    'eta': 1, 
    'max_depth': 10,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'nthread': 16,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=5)

output_file = 'xgb_13aucpr.pkl'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')