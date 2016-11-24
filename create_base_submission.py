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

from outputting_functions import *
from feature_engineering import *

log = sys.argv[1]

print('Reading in data..')

test = pd.read_csv('Data/test.csv')
train = pd.read_csv('Data/train.csv')

test['SalePrice'] = ''
y_train, y_test = train['SalePrice'].values, test['SalePrice'].values

if log:
    y_train = np.log1p(y_train)
    print('Taking the log of the outcome variable')

dropping = ['SalePrice', 'Id']

X_train = train.drop(dropping, 1)
X_test = test.drop(dropping, 1)

numeric_vars = ['LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
               'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
               'ScreenPorch', 'PoolArea', 'MiscVal']

X_train, X_test = data_formatting(X_train, X_test, numeric_vars, log)               
               
print('Training model..')

clf = RandomForestRegressor(n_estimators = 100, random_state=2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

if log:
    y_pred = np.expm1(y_pred)
    print('Unlogging')

output_sub(test["Id"], y_pred)

