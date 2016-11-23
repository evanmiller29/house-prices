import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

test = pd.read_csv('Data/test.csv')
train = pd.read_csv('Data/train.csv')

test['SalePrice'] = ''
y_train, y_test = train['SalePrice'].values, test['SalePrice'].values

dropping = ['SalePrice', 'Id']

X_train = train.drop(dropping, 1)
X_test = test.drop(dropping, 1)

numeric_vars = ['LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
               'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
               'ScreenPorch', 'PoolArea', 'MiscVal']

X_train_numeric = X_train[numeric_vars]
X_test_numeric = X_test[numeric_vars]

X_train_cat = X_train.drop(numeric_vars, 1)
X_test_cat = X_test.drop(numeric_vars, 1)

X_train_cat = pd.get_dummies(X_train_cat)
X_test_cat = pd.get_dummies(X_test_cat)

X_train = pd.concat([X_train_numeric, X_train_cat], axis = 1)
X_test = pd.concat([X_test_numeric, X_test_cat], axis = 1)

train_cols = X_train.columns.values
test_cols = X_test.columns.values

diffs = list(set(train_cols) - set(test_cols))

for col in diffs:
    X_test[col] = 0

# Can unit test for different column numbers as a result of pd.get_dummies()    
    
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

clf = RandomForestRegressor(n_estimators = 100, random_state=2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

pred_df = pd.DataFrame(y_pred, index=test["Id"], columns=["SalePrice"])

print('Outputting predictions..')
pred_df.to_csv('output.csv', header=True, index_label='Id')