''' 
This function should be sourced in the following way

python create_base_submission.py True lassocv
arg1 = whether scaling shoud be implemented
arg2 = the type of model
    default is the base submission rf with 100 trees
    basecv = grid-search optimised rf
    lassocv = a grid search optimised lasso regression
    xgboost = a basic xgboost model
    svm = don't use currently as there is something odd happening with the scoring
arg3 = whether you'd like to output a feather (hahahah)
'''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from outputting_functions import *
from feature_engineering import *
from models import *

log = sys.argv[1]
model = sys.argv[2]
output = sys.argv[3]

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

print('Creating holdout set (25%) from the training set..')

train, valid, y_train, y_valid = train_test_split(train, y_train, test_size=0.25, random_state=0)             
X_train, X_test, X_valid = data_formatting(train, test, valid, numeric_vars, log, output)               

print('Training model..')

clf = train_model(X_train, y_train, model)

y_pred = clf.predict(X_test)
y_pred_holdout = clf.predict(X_valid)

score = cross_val_score(clf, X_valid, y_valid).mean()
print("Score on the holdout set = %.2f" % score)

if log:
    y_pred = np.expm1(y_pred)
    print('Unlogging')

output_sub(test["Id"], y_pred, model)

