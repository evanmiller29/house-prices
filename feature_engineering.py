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
    
    train['data'] = 'train'
    test['data'] = 'test'
    
    ttl_df = pd.concat([train, test], axis = 0)
    
    print('Formatting data..')
    
    ttl_numeric = ttl_df[numeric_vars]
    
    if log:
        ttl_numeric = np.log1p(ttl_numeric)
        print('Taking the log of numeric variables')
        
    train_cat = train.drop(numeric_vars, 1)
    test_cat = test.drop(numeric_vars, 1)
    ttl_cat = ttl_df.drop(numeric_vars, 1)
    
    data = ttl_cat['data']
    ttl_cat = ttl_cat.drop('data', 1)       
    ttl_cat = pd.get_dummies(ttl_cat)

    X_ttl = pd.concat([ttl_numeric, ttl_cat, data], axis = 1)
    X_train = X_ttl.loc[X_ttl.data == 'train', :]
    X_test = X_ttl.loc[X_ttl.data == 'test', :]
    
    X_train = X_train.drop('data', 1)
    X_test = X_test.drop('data', 1)
             
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
       
    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()
    
    return(X_train, X_test)