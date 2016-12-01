def train_model(X_train, y_train, model):

    from time import time
    
    if model == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        clf = RandomForestRegressor(n_estimators = 100, random_state=2)
        t0 = time()
        clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        
        return(clf)
    
    if model == 'rfcv':
        
        long_run = True
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        clf = RandomForestRegressor()
        param_grid = {"n_estimators": [50, 100, 200],
              "max_depth": [3, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False]}
        
        print('Running grid search for random forest')        
        
        t0 = time()
        
        if long_run:
            clf = GridSearchCV(clf, param_grid=param_grid)
            clf.fit(X_train, y_train)
            print('best params')
            print (clf.best_params_)
            print('best score')
            print (clf.best_score_)
        
        else:
            clf = RandomForestRegressor(n_estimators = 200, max_features = 10,
                            min_samples_leaf = 1, min_samples_split = 2, 
                            max_depth = None, bootstrap = False)
            
            clf.fit(X_train, y_train)
            
        print("done in %0.3fs" % (time() - t0))
        
        return(clf)
    
    if model == 'rfpca':
    
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA
        
        cols = X_train.shape[1]
        n_components = list(range(0, 200, 2))
        n_components[0] = 1
        
        pca = PCA()
        rf = RandomForestRegressor(n_estimators = 200, min_samples_split = 3,
                                    max_depth = None, min_samples_leaf = 1,
                                    bootstrap = True)
        
        pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])    
        t0 = time()
        
        clf = GridSearchCV(pipe,
                             dict(pca__n_components=n_components))

        clf.fit(X_train, y_train)    
        print("done in %0.3fs" % (time() - t0))
        
        return(clf)
    
    if model == 'lassocv':
        
        from sklearn.linear_model import LassoCV
        t0 = time()
        clf = LassoCV(alphas = [5, 1, 0.1, 0.001, 0.0005], max_iter = 5000)
        print('Fitting CV lasso model..')
        clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        return(clf)
    
    if model == 'svm_linear':
        
        import numpy as np
        from sklearn.svm import SVR
        
        clf = SVR(kernel='linear')
        clf.fit(X_train, y_train)
        
        return(clf)
        
    if model == 'xgboost':
        
        import xgboost as xgb
                
        clf = xgb.XGBRegressor()
        clf.fit(X_train, y_train)
        return(clf)

    if model == 'xgboostcv':
        
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV        
        
        target_param_grid = {
            'colsample_bytree':[0.4,0.6,0.8],
            'gamma':[0,0.03,0.1,0.3],
            'min_child_weight':[1.5,6,10],
            'learning_rate':[0.1,0.07],
            'max_depth':[3,5],
            'n_estimators':[10000],
            'reg_alpha':[1e-5, 1e-2,  0.75],
            'reg_lambda':[1e-5, 1e-2, 0.45],
            'subsample':[0.6,0.95]  
        }

                    
        xgb_model = xgb.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
             min_child_weight=1, gamma=0, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

        gsearch = GridSearchCV(estimator = xgb_model, param_grid = target_param_grid, n_jobs=6, scoring='neg_mean_squared_error')
        gsearch.fit(X_train, y_train)
                
        return(gsearch)