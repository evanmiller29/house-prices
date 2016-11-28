def train_model(X_train, y_train, model):

    from time import time
    
    if model == 'rfcv':
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        clf = RandomForestRegressor()
        param_grid = {"n_estimators": [50, 100, 200],
              "max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False]}
        
        print('Running grid search for random forest')        
        
        t0 = time()
        grid_search = GridSearchCV(clf, param_grid=param_grid)
        grid_search.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        return(grid_search)

    if model == 'lassocv':
        
        from sklearn.linear_model import LassoCV
        t0 = time()
        clf = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
        clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        return(clf)
        
    else:
        from sklearn.ensemble import RandomForestRegressor
        clf = RandomForestRegressor(n_estimators = 100, random_state=2)
        t0 = time()
        clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        
        return(clf)
