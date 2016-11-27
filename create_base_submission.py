''' 
This function should be sourced in the following way

python create_base_submission.py
arg1 = whether scaling shoud be implemented

'''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from outputting_functions import *
from feature_engineering import *

log = sys.argv[1]

print('Reading in data..')

test = pd.read_csv('Data/test.csv')
train = pd.read_csv('Data/train.csv')

test['SalePrice'] = np.nan
y_train, y_test = train['SalePrice'].values, test['SalePrice'].values

if log:
    y_train = np.log1p(y_train)
    print('Taking the log of the outcome variable')

numeric_vars = ['LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
               'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
               'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'YearRemodAdd', 'YearBuilt']

print('Creating holdout set (30%) from the training set..')

train, valid, y_train, y_valid = train_test_split(train, y_train, test_size=0.3, random_state=0)             
X_train, X_test, X_valid = data_formatting(train, test, valid, numeric_vars, log)               
               
print('Training model..')

clf = RandomForestRegressor(n_estimators = 100, random_state=2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_holdout = clf.predict(X_valid)

score = cross_val_score(clf, X_valid, y_valid).mean()
print("Score on the holdout set = %.2f" % score)

if log:
    y_pred = np.expm1(y_pred)
    print('Unlogging')

output_sub(test["Id"], y_pred)

