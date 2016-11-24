def unique_options(x, y):
    x = str(x)
    y = str(y)
    if x == y: return 0
    return 1
    
def second_recode(x, y):
        
    if y == 1: return x
    else: return np.nan

def data_formatting(train, test, numeric_vars, log):
    
    import pandas as pd
    import numpy as np
    
    print('Formatting data..')
    
    #### Want to first combine the two datasets and do the feature engineering there. 
    #### Then once you're done split them back into test and train sets
    
    ttl_df = pd.concat([train, test], 1)
    
    train_numeric = train[numeric_vars]   
    test_numeric = test[numeric_vars]
    
    if log:
        train_numeric == np.log1p(train_numeric)
        test_numeric == np.log1p(test_numeric)
        print('Taking the log of numeric variables')
        
    train_cat = train.drop(numeric_vars, 1)
    test_cat = test.drop(numeric_vars, 1)

    train_cat = pd.get_dummies(train_cat)
    test_cat = pd.get_dummies(test_cat)

    X_train = pd.concat([train_numeric, train_cat], axis = 1)
    X_test = pd.concat([test_numeric, test_cat], axis = 1)

    train_cols = X_train.columns.values
    test_cols = X_test.columns.values

    diffs = list(set(train_cols) - set(test_cols))

    for col in diffs:
        X_test[col] = 0
         
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()
    
    return(X_train, X_test)