from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from root import *


def xgboost_tune(df_train, df_val, XGBoost_eval_dir, max_depth,
                 min_child_weight, gamma, subsample, colsample_bytree, alpha,
                 params_df):
    """
    Train a model with the specified combination of hyperparameters and plot
    the model loss.
    :param df_train: pandas dataframe: the training set
    :param df_val: pandas dataframe: the validation set
    :param XGBoost_eval_dir: directory where the file containing
    hyperparameter values and the validation error is saved.
    :param max_depth: Maximum depth of a tree
    :param min_child_weight: Minimum sum of instance weight (hessian) needed
    in a child
    :param gamma: Minimum loss reduction required to make a further partition
    on a leaf node of the tree.
    :param subsample: Subsample ratio of the training instances.
    :param colsample_bytree: Subsample ratio of columns when constructing each
    tree.
    :param alpha: L1 regularization term on weights.
    :param params_df: dataframe to which the model parameters and validation
     error are written.
    :return:
    """

    x_train = df_train[['year_scaled', 'month_sin', 'month_cos',
                        'day_of_month_sin', 'day_of_month_cos', 'day_1',
                        'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
                        'hour_sin', 'hour_cos']]
    x_val = df_val[['year_scaled', 'month_sin', 'month_cos',
                    'day_of_month_sin', 'day_of_month_cos', 'day_1',
                    'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
                    'hour_sin', 'hour_cos']]
    y_train = df_train['log_Value']
    y_val = df_val['log_Value']

    model = XGBRegressor(
        n_estimators=100,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        alpha=alpha,
        seed=42)

    model = model.fit(
        x_train,
        y_train,
        eval_metric="mae",
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False,
        early_stopping_rounds=5,
    )

    '''Evaluate predictions'''

    eval_df = pd.DataFrame()
    eval_df['log_Value'] = y_val.copy()
    eval_df['pred_log_Value'] = model.predict(x_val)

    eval_df['Value'] = np.exp(eval_df['log_Value'])
    eval_df['pred_Value'] = np.exp(eval_df['pred_log_Value'])

    MAE = mean_absolute_error(eval_df['Value'], eval_df['pred_Value'])

    '''Append a list of hyperparameters and the model MAE to a nonlocal
    df'''

    iteration_params = [max_depth, min_child_weight, gamma, subsample,
                        colsample_bytree, alpha, MAE]

    params_df.loc[len(params_df)] = iteration_params

    '''Plot the model loss per epoch'''

    results = model.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(0, epochs)
    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    ax.plot(x_axis, results['validation_1']['mae'], label='Validation')
    ax.legend()
    plt.ylabel('Mean Absolute Error (log(y))')
    plt.title('XGBoost MAE loss')
    plt.savefig(XGBoost_eval_dir + f'loss'
                                   f'_max_depth_{max_depth}'
                                   f'_min_child_weight_{min_child_weight}'
                                   f'_gamma_{gamma}'
                                   f'_subsample_{subsample}'
                                   f'_colsample_bytree_{colsample_bytree}'
                                   f'_alpha_{alpha}.png', dpi=400)
    plt.clf()
    plt.close("all")
