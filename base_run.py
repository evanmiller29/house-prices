import pandas as pd
import lightgbm as lgbm
import functions as fn

import os
from os.path import join
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

data_dir = 'F:/Nerdy Stuff/Kaggle/House prices'

print('Reading in data..')

train = pd.read_csv(join(data_dir, 'train.csv'))
test = pd.read_csv(join(data_dir, 'test.csv'))
sub = pd.read_csv(join(data_dir, 'sample_submission.csv'))

y = train['SalePrice']

objs = train.columns[train.dtypes == 'object']

print('base run, only using numeric features..')

X = train.drop(objs, axis=1)
X = X.drop('SalePrice', axis=1)

X = X.fillna(-99)

model = lgbm.LGBMRegressor()
kf = KFold(n_splits=10)

rmse_valids = []

for i, (train_index, valid_index) in enumerate(kf.split(X)):

    print('Fold #%s' % (i + 1))

    X_train, X_valid = X.loc[train_index, :], X.loc[valid_index, :]
    y_train, y_valid = y[train_index], y[valid_index]

    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)

    rmse_valid = sqrt(mean_squared_error(y_valid, y_valid_pred))
    print('RMSE for validation set =%s' % rmse_valid)

    rmse_valids.append(rmse_valid)

