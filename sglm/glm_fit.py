#fit GLM functions


import os
import csv
import yaml
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Optional 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV



def save_model(model_dict, config):
    """
    Save the model to the specified path in config.yaml
    """
    model_path = config['Project']['project_path'] + '/models'
    model_name = config['Project']['project_name'] + '_model.pkl'
    model_full_path = os.path.join(model_path, model_name)
    with open(model_full_path, 'wb') as f:
        pickle.dump(model_dict, f)



def split_data(X, y, config):
    """
    Split data into train and test sets
    Will use the config.yaml set values for train_size and test_size
    """

    train_size = config['train_test_split']['train_size']
    test_size = config['train_test_split']['test_size']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size)

    return X_train, X_test, y_train, y_test

def shift_series_range(series: pd.Series, shift_amt_range: Tuple[int], fill_value: Optional[float] = np.nan, shift_bounding_column: Optional[str] = None) -> pd.DataFrame:
    """
    Shift a series up/down by a range of shift amounts.

    Args:
        series (pd.Series): Series to be shifted up or down.
        shift_amt_range (Tuple[int]): Range of amounts to shift data (start, end).
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan).
        shift_bounding_column (Optional[str]): Column for grouping shifts (default: None).

    Returns:
        pd.DataFrame: DataFrame containing post-shift versions of the series.
    """
    shifted_series_list = []
    for shift_amt in range(shift_amt_range[0], shift_amt_range[1] + 1):
        shifted_series = shift_series(series, shift_amt, fill_value=fill_value, shift_bounding_column=shift_bounding_column)
        shifted_series_list.append(shifted_series.rename((f"{series.name}", f"{shift_amt}")))
    
    df_shifted_series = pd.concat(shifted_series_list, axis=1)
    return df_shifted_series

def shift_series(series: pd.Series, shift_amt: int, fill_value: Optional[float] = np.nan, shift_bounding_column: Optional[str] = None) -> pd.Series:
    """
    Shift a series up or down by a specified amount.

    Args:
        series (pd.Series): Series to be shifted up or down.
        shift_amt (int): Amount to shift data (positive for shift down, negative for shift up).
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan).
        shift_bounding_column (Optional[str]): Column for grouping shifts (default: None).

    Returns:
        pd.Series: Post-shift version of the series.
    """
    if shift_amt == 0:
        return series

    if shift_bounding_column:
        grouped_series = series.groupby(shift_bounding_column)
    else:
        grouped_series = series

    shifted_series = grouped_series.shift(periods=shift_amt, fill_value=fill_value)
    return shifted_series

def shift_array(setup_array: np.ndarray, shift_amt: int, fill_value: Optional[float] = np.nan) -> np.ndarray:
    """
    Shift a numpy array up or down by a specified amount.

    Args:
        setup_array (np.ndarray): Array to be shifted up or down.
        shift_amt (int): Amount to shift data (positive for shift down, negative for shift up).
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan).

    Returns:
        np.ndarray: Post-shift version of the array.
    """
    if shift_amt == 0:
        return setup_array

    blanks = np.ones((abs(shift_amt), setup_array.shape[1])) * fill_value

    if shift_amt > 0:
        shifted_array = np.concatenate((blanks, setup_array[:-shift_amt, :]), axis=0)
    else:
        shifted_array = np.concatenate((setup_array[-shift_amt:, :], blanks), axis=0)

    return shifted_array

def shift_predictors(config, df_source):
    """
    Shift predictors by the amounts specified in config.yaml
    """

    predictors = config['glm_params']['predictors']
    shift_bounds = config['glm_params']['predictors_shift_bounds'] if 'predictors_shift_bounds' in config['glm_params'] else {}
    shift_bounds_default = config['glm_params']['predictors_shift_bounds_default']
    list_predictors_and_shifts = [(predictor, shift_bounds.get(
        predictor, shift_bounds_default)) for predictor in predictors]

    
    list_predictors_shifted = []
    for predictor, predictor_shift_bounds in list_predictors_and_shifts:
        predictor_shifted = shift_series_range(
            df_source[predictor],
            predictor_shift_bounds,
            shift_bounding_column=['SessionName']
        )
        list_predictors_shifted.append(predictor_shifted)
    

    df_shifted = pd.concat(list_predictors_shifted, axis=1)
    srs_response = df_source[config['glm_params']['response']]
    non_nans = (df_shifted.isna().sum(axis=1) == 0)&~np.isnan(srs_response)
    df_predictors_fit = df_shifted[non_nans].copy()
    srs_response_fit = srs_response[non_nans].copy()

    return srs_response_fit, df_predictors_fit, list_predictors_and_shifts



def fit_EN(config, X_train, X_test, y_train, y_test):
        """
        Fit a GLM model using ElasticNet from scikit-learn
        Will pass in values from config file
        """
        
        alpha=config['glm_params']['glm_keyword_args']['alpha']
        fit_intercept=config['glm_params']['glm_keyword_args']['fit_intercept']
        max_iter=config['glm_params']['glm_keyword_args']['max_iter']
        warm_start=config['glm_params']['glm_keyword_args']['warm_start']
        l1_ratio=config['glm_params']['glm_keyword_args']['l1_ratio']      
        selection = config['glm_params']['glm_keyword_args']['selection'] 
        score_metric = config['glm_params']['glm_keyword_args']['score_metric']
        
    
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
                            max_iter=max_iter, copy_X=True, warm_start=warm_start,
                            selection=selection)
        
        model.fit(X_train, y_train)
        beta = model.coef_
        sparse_beta = model.sparse_coef_
        intercept = model.intercept_

        y_pred = model.predict(X_test)


        if score_metric == 'r2':
            score = calc_r2(y_pred, y_test)
        elif score_metric == 'mse':
            score = calc_mse(y_pred, y_test)
        elif score_metric == 'avg':
            score = model.score(y_pred, y_test)
    
        return model, y_pred, score, beta, intercept, sparse_beta

def fit_tuned_EN(config, X_train, X_test, y_train, y_test):
            """
            Fit a GLM model using ElasticNetCV from scikit-learn
            Will pass in values from config file. You will need to
            provide a list of alphas and l1_ratios to test.
            """
            
            alpha=config['glm_params']['glm_keyword_args']['alpha']
            n_alphas = config['glm_params']['glm_keyword_args']['n_alphas']
            cv = config['glm_params']['glm_keyword_args']['cv']
            fit_intercept=config['glm_params']['glm_keyword_args']['fit_intercept']
            max_iter=config['glm_params']['glm_keyword_args']['max_iter']
            l1_ratio=config['glm_params']['glm_keyword_args']['l1_ratio'] 
            n_jobs=config['glm_params']['glm_keyword_args']['n_jobs']      
            score_metric = config['glm_params']['glm_keyword_args']['score_metric']
            
            tuned_model = ElasticNetCV(alphas=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
                                max_iter=max_iter, copy_X=True, cv=cv, n_alphas=n_alphas, 
                                n_jobs=n_jobs)
            
            tuned_model.fit(X_train, y_train)

            best_alpha = tuned_model.alpha_
            best_l1r = tuned_model.l1_ratio_
            best_params = dict(alpha=best_alpha, l1_ratio=best_l1r)

            beta = tuned_model.coef_

            y_pred = tuned_model.predict(X_test)
    
            if score_metric == 'r2':
                score = calc_r2(y_pred, y_test)
            elif score_metric == 'mse':
                score = calc_mse(y_pred, y_test)
            elif score_metric == 'avg':
                score = tuned_model.score(y_pred, y_test)
        
            return tuned_model, y_pred, score, beta, best_params

def fit_ridge(config, X_train, X_test, y_train, y_test):
        """
        Fit a Ridge model using Ridge from scikit-learn
        Will pass in values from config file
        """
        
        alpha=config['glm_params']['ridge_keyword_args']['alpha']
        fit_intercept=config['glm_params']['ridge_keyword_args']['fit_intercept']
        max_iter=config['glm_params']['ridge_keyword_args']['max_iter']
        solver=config['glm_params']['ridge_keyword_args']['solver']      
        score_metric = config['glm_params']['ridge_keyword_args']['score_metric']
        
    
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, 
                            max_iter=max_iter, copy_X=True,
                            solver=solver)
        
        model.fit(X_train, y_train)
        beta = model.coef_
        intercept = model.intercept_

        y_pred = model.predict(X_test)


        if score_metric == 'r2':
            score = calc_r2(y_pred, y_test)
        elif score_metric == 'mse':
            score = calc_mse(y_pred, y_test)
        elif score_metric == 'avg':
            score = model.score(y_pred, y_test)
    
        return model, y_pred, score, beta, intercept

def fit_tuned_ridge(config, X_train, X_test, y_train, y_test):
            """
            Fit a Ridge model using RidgeCV from scikit-learn
            Will pass in values from config file. You will need to
            provide a list of alphas to test.
            """
            
            alpha=config['glm_params']['ridge_keyword_args']['alpha']
            cv = config['glm_params']['ridge_keyword_args']['cv']
            fit_intercept=config['glm_params']['ridge_keyword_args']['fit_intercept']
            gcv_mode=config['glm_params']['ridge_keyword_args']['gcv_mode']   
            score_metric = config['glm_params']['ridge_keyword_args']['score_metric']
            
            tuned_model = RidgeCV(alphas=alpha, fit_intercept=fit_intercept, 
                                cv=cv, scoring=score_metric, store_cv_values=False,
                                gcv_mode=gcv_mode, alpha_per_target=False)
            
            tuned_model.fit(X_train, y_train)

            best_alpha = tuned_model.alpha_
            best_score = tuned_model.best_score_
            best_params = dict(alpha=best_alpha,
                               best_score=best_score)
            beta = tuned_model.coef_

            y_pred = tuned_model.predict(X_test)
    
            if score_metric == 'r2':
                score = calc_r2(y_pred, y_test)
            elif score_metric == 'mse':
                score = calc_mse(y_pred, y_test)
            elif score_metric == 'avg':
                score = tuned_model.score(y_pred, y_test)
        
            return tuned_model, y_pred, score, beta, best_params




def calc_residuals(y_pred, y):
    """
    Calculate the residuals of the model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

    Returns
    -------
    residuals : array-like of shape (n_samples,)
        Residuals of the model
    """

    prediction = y_pred
    residuals = y - prediction
    avg_residuals = y - np.mean(y)

    return residuals, avg_residuals

def calc_r2(y_pred, y):
    """
    Calculate the r2 of the model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

    Returns
    -------
    r2 : float
        r2 of the model
    """

    residuals, avg_residuals = calc_residuals(y_pred, y)
    rss = np.sum(residuals**2)
    tss = np.sum(avg_residuals**2)

    if tss == 0:
        r2 = 0
    else:
        r2 = 1 - (rss/tss)

    return r2

def calc_mse(y_pred, y):
    """
    Calculate the mean squared error of the model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

    Returns
    -------
    mse : float
        mean squared error of the model
    """

    residuals, _ = self.calc_residuals(y_pred, y)
    mse = np.mean(residuals**2)

    return mse

def plot_and_save(config, y_pred, y_test, beta, df_predictors_shift):
    """
    Plot and save the predictions vs actual values and the model fit results
    Will be saved in the results folder of the project path
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('talk')
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette('deep')
    ax.scatter(y_pred, y_test,s=50, alpha=0.5, color=colors[0])
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.set_title('Predicted vs Actual Values')
    ax.grid(True)
    plt.savefig(config['Project']['project_path'] + '/results/predicted_vs_actual.png')
    plt.close()

    #plot histogram of residuals
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette('deep')
    ax.hist(y_test-y_pred, bins=50, color=colors[0])
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Count')
    ax.set_title('Residuals')
    ax.grid(True)
    plt.savefig(config['Project']['project_path'] + '/results/residuals.png')
    plt.close()

    #save model fit results
    model_fit_results = pd.Series(beta, index=df_predictors_shift.columns, name='coef').unstack(0, )
    model_fit_results.index = model_fit_results.index.astype(int)
    model_fit_results = model_fit_results.reindex(config['glm_params']['predictors'], axis=1)

    tup_y_lim = (np.inf, -np.inf)
    fig, axes = plt.subplots(1, len(model_fit_results.columns), figsize=(5*len(model_fit_results.columns), 5))
    axes = axes.flatten()
    for ipredictor, predictor in enumerate(model_fit_results.columns):
        axes[ipredictor].plot(model_fit_results.sort_index()[predictor])
        axes[ipredictor].set_title(predictor)
        axes[ipredictor].grid(True)
        
        tup_y_lim = (min(tup_y_lim[0], model_fit_results[predictor].min()-0.1),
                    max(tup_y_lim[1], model_fit_results[predictor].max()+0.1))

    for ax in axes:
        ax.set_ylim(tup_y_lim)
    fig.suptitle('GLM Coefficients Fit Results')
    plt.savefig(config['Project']['project_path'] + '/results/model_fit.png')
    plt.close()

    