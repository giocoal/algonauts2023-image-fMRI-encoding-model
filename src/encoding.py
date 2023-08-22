from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import os 

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def compute_perason_numpy(pred, target):
    """
    Compute Pearson correlation coefficient between two arrays.
    Used to compute the correlation between the predicted and target fMRI data.
    during GridSearch.
    """
    corrcoef = list()
    for pred, target in zip(pred.T, target.T):

        s, _ = pearsonr(x=pred, y=target)
        corrcoef.append(s)

    return np.array(corrcoef)

def linear_regression(regression_type, 
                      features_train, 
                      features_val, 
                      features_test, 
                      lh_fmri_train, 
                      rh_fmri_train, 
                      save_predictions,
                      subject_submission_dir,
                      alpha_l = None,
                      alpha_r = None,
                      grid_search = False,
                      param_grid = {'alpha': [10, 100, 1e4, 2e4, 5e4, 1e5, 1e6]},
                      UseStandardScaler = False):
    # Fit linear regressions on the training data
    if UseStandardScaler:
        print('Standardizing features...')
        features_train = StandardScaler().fit_transform(features_train)
        features_val = StandardScaler().fit_transform(features_val)
        features_test = StandardScaler().fit_transform(features_test)
    if regression_type == 'ridge':
        if grid_search:
            print('Fitting ridge regressions on the training data...')
            #param_grid = {'alpha': [0.0001, 0.0002, 0.001, 0.01, 0.1, 1, 10, 100, 1e4, 2e4, 5e4, 1e5, 1e6]}
            grid_search_l = GridSearchCV(Ridge(), param_grid=param_grid, scoring=make_scorer(
                lambda x, y: np.median(compute_perason_numpy(x, y))), cv=5, n_jobs=5, verbose=1)
            grid_search_l.fit(X=features_train, y=lh_fmri_train)
            alpha_l = grid_search_l.best_params_['alpha']
            print("Best Param LH: {}".format(grid_search_l.best_params_))
            
            grid_search_r = GridSearchCV(Ridge(), param_grid=param_grid, scoring=make_scorer(
                lambda x, y: np.median(compute_perason_numpy(x, y))), cv=5, n_jobs=5, verbose=1)
            grid_search_r.fit(X=features_train, y=rh_fmri_train)
            print("Best Param RH: {}".format(grid_search_r.best_params_))
            alpha_r = grid_search_r.best_params_['alpha']
        print('Fitting ridge regressions on the training data...')
        reg_lh = Ridge(alpha=alpha_l).fit(features_train, lh_fmri_train)
        reg_rh = Ridge(alpha=alpha_r).fit(features_train, rh_fmri_train)
    elif regression_type == 'linear':
        print('Fitting linear regressions on the training data...')
        reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
        reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
    # Use fitted linear regressions to predict the validation and test fMRI data
    print('Predicting fMRI data on the validation and test data...')
    lh_fmri_val_pred = reg_lh.predict(features_val)
    lh_fmri_test_pred = reg_lh.predict(features_test)
    rh_fmri_val_pred = reg_rh.predict(features_val)
    rh_fmri_test_pred = reg_rh.predict(features_test)
    
    # Test submission files
    if save_predictions:
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
        np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)
        
    return lh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_val_pred, rh_fmri_test_pred #, alpha_l, alpha_r

def ridge_alpha_grid_search(features_train, 
                      lh_fmri_train, 
                      rh_fmri_train, 
                      param_grid = {'alpha': [10, 100, 1e3, 1e4, 2e4, 5e4, 1e5, 1e6]},
                      UseStandardScaler = False):
    # Fit linear regressions on the training data
    print('Fitting ridge regressions on the training data...')
    if UseStandardScaler:
        print('Standardizing features...')
        features_train = StandardScaler().fit_transform(features_train)
    #param_grid = {'alpha': [0.0001, 0.0002, 0.001, 0.01, 0.1, 1, 10, 100, 1e4, 2e4, 5e4, 1e5, 1e6]}
    grid_search_l = GridSearchCV(Ridge(), param_grid=param_grid, scoring=make_scorer(
        lambda x, y: np.median(compute_perason_numpy(x, y))), cv=5, n_jobs=5, verbose=1)
    grid_search_l.fit(X=features_train, y=lh_fmri_train)
    alpha_l = grid_search_l.best_params_['alpha']
    print("Best Param LH: {}".format(grid_search_l.best_params_))
    
    # grid_search_r = GridSearchCV(Ridge(), param_grid=param_grid, scoring=make_scorer(
    #     lambda x, y: np.median(compute_perason_numpy(x, y))), cv=5, n_jobs=5, verbose=1)
    # grid_search_r.fit(X=features_train, y=rh_fmri_train)
    # print("Best Param RH: {}".format(grid_search_r.best_params_))
    # alpha_r = grid_search_r.best_params_['alpha']
    alpha_r = alpha_l
        
    return alpha_l, alpha_r