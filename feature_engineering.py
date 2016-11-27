def unique_options(x, y):
    x = str(x)
    y = str(y)
    if x == y: return 0
    return 1
    
def second_recode(x, y):
    
    import numpy as np
    
    if y == 1: return x
    else: return np.nan

def floor_error(x, y):
    x = str(x)
    if x.find('1Story') == 0 and y > 0: return 'Pos2ndfloor'
    else: return x    

def recent_remodel(x, y):
    if x >= y: return 1
    else: return 0

def year_diff(x, y):
    
    import math
    import numpy as np
    
    if math.isnan(x) or math.isnan(y): return np.nan
    else:
    
        x = int(float(x))
        y = int(float(y))
    
        return x - y
    
def data_formatting(train, test, valid, numeric_vars, log):
    
    import pandas as pd
    import numpy as np
    
    recent_threshold = 5
    
    train['data'] = 'train'
    test['data'] = 'test'
    valid['data'] = 'valid'
    
    ttl_df = pd.concat([train, test, valid], axis = 0)
    
    print('Recoding data errors..')
    
    ttl_df['HouseStyle_recode'] = ttl_df.apply(lambda x: floor_error(x['HouseStyle'], x['2ndFlrSF']), axis=1)
    
    print('Formatting data..')
    
    ttl_numeric = ttl_df[numeric_vars]
        
    print('Generating aggregate features..')
        
    grouped = ttl_df.loc[ttl_df['data'] == 'train', :].groupby('Neighborhood')
    med_neighbour_sale = grouped.SalePrice.median().reset_index()

    med_neighbour_sale.rename(columns = {'SalePrice':'AreaAverage'}, inplace = True)
    ttl_df = pd.merge(ttl_df, med_neighbour_sale, how = 'left', on='Neighborhood')
    
    ttl_numeric['AreaAverage'] = ttl_df['AreaAverage']
    
    print('Generating numeric features that will be scaled..')
    
    ttl_numeric['ttl_floor_SF'] = ttl_numeric.apply(lambda x: x['1stFlrSF'] + x['2ndFlrSF'] + x['BsmtFinSF1'] + x['BsmtFinSF2'], axis = 1)
    ttl_numeric['ttl_bath'] = ttl_numeric.apply(lambda x:  x['BsmtFullBath'] + 0.5 * x['BsmtHalfBath'] + x['FullBath'] + 0.5 * x['HalfBath'], axis = 1)
    
    if log:
        ttl_numeric = np.log1p(ttl_numeric)
        print('Taking the log of numeric variables')
        
    print('Generating numeric features that won\'t be scaled')
    
    ttl_numeric['Remodelled'] = ttl_numeric.apply(lambda x: unique_options(x['YearBuilt'], x['YearRemodAdd']), axis = 1)
    ttl_numeric['YearRemodAdd'] = ttl_numeric.apply(lambda x: second_recode(x['YearRemodAdd'], x['Remodelled']), axis = 1)
    ttl_numeric['YearsSinceRemodel'] = ttl_numeric.apply(lambda x: year_diff(x['YrSold'], x['YearRemodAdd']), axis=1)  
    ttl_numeric['recent_remodel'] = ttl_numeric.apply(lambda x: recent_remodel(x['YearsSinceRemodel'], recent_threshold), axis=1)
        
    ttl_cat = ttl_df.drop(numeric_vars, 1)
    ttl_cat = ttl_cat.drop('AreaAverage', 1)
    
    print('Generating new categorical features..')
    
    ttl_cat['Mult_Ext'] = ttl_cat.apply(lambda x: unique_options(x['Exterior1st'], x['Exterior2nd']), axis=1)
    ttl_cat['Mult_conds'] = ttl_cat.apply(lambda x: unique_options(x['Condition1'], x['Condition2']), axis = 1)   
    ttl_cat['Exterior2nd'] = ttl_cat.apply(lambda x: second_recode(x['Exterior2nd'], x['Mult_Ext']), axis = 1)
    ttl_cat['Condition2'] = ttl_cat.apply(lambda x: second_recode(x['Condition2'], x['Mult_conds']), axis = 1)
      
    dropping = ['SalePrice', 'Id']
    
    ttl_cat = ttl_cat.drop(dropping, axis = 1)
    
    data = ttl_cat['data']
    ttl_cat = ttl_cat.drop('data', 1)       
    ttl_cat = pd.get_dummies(ttl_cat)
    
    print('Re-separating into test and train sets..')
    
    ttl_numeric = ttl_numeric.reset_index()
    ttl_cat = ttl_cat.reset_index()
    
    X_ttl = pd.concat([ttl_numeric, ttl_cat, data], axis = 1)
    X_train = X_ttl.loc[X_ttl.data == 'train', :]
    X_test = X_ttl.loc[X_ttl.data == 'test', :]
    X_valid = X_ttl.loc[X_ttl.data == 'valid', :]
    
    X_train = X_train.drop('data', 1)
    X_test = X_test.drop('data', 1)
    X_valid = X_valid.drop('data', 1)
    
    # The recode of missing variables wasn't working, so had to find this hack :D
    
    X_train = X_train.replace(np.nan, {column: 0 for column in X_train.columns})
    X_test = X_test.replace(np.nan, {column: 0 for column in X_test.columns})
    X_valid = X_valid.replace(np.nan, {column: 0 for column in X_valid.columns})
    
    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()
    X_valid = X_valid.as_matrix()
    
    return(X_train, X_test, X_valid)