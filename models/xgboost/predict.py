
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from root import *


def predict(df_original, df_future_month, outcome_var):

    """
    Use the model trained on all available data to forecast outcome values
    for each hour in the following month. Take the antilog of predictions to
    obtain the same unit as the original outcome values.
    Plot the predicted timeseries on its own and together with the input
    values.
    :param df_original: pandas dataframe: the original, untransformed input
    data
    :param df_future_month: pandas dataframe: the datetime features for the
    following month
    :param outcome_var: string: 'sales' or 'traffic'
    :return: None. Save plots of the predicted timeseries.
    """

    pred_output_dir = f"{root}/output/forecast/{outcome_var}/"
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)

    XGBoost_output_dir = f"{root}/output/models/XGBoost/{outcome_var}/"
    XGBoost_model = pickle.load(open(XGBoost_output_dir +
                                     'xgboost_trained.pickle.dat', 'rb'))

    '''Predict the values for the following month. Take the anti-log of the 
    predictions, since the outcome value was log-transformed before 
    training.'''

    df_future_month_pred = \
        df_future_month[['year_scaled', 'month_sin', 'month_cos',
                         'day_of_month_sin', 'day_of_month_cos', 'day_1',
                         'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
                         'hour_sin', 'hour_cos']]

    df_future_month['log_Value'] = XGBoost_model.predict(df_future_month_pred)
    df_future_month['Value'] = np.exp(df_future_month['log_Value'])

    '''Constrain the lower bound to that of the training data'''

    lower_bound = df_future_month['Value'].min()
    df_future_month['Value_bounded'] = df_future_month['Value']\
        .clip(lower=lower_bound)

    '''Create a date-hour column for plotting'''

    df_original.rename(columns={"day_of_month": 'day'}, inplace=True)
    df_original['date_hour'] = pd.to_datetime(df_original[['year', 'month',
                                                           'day', 'hour']])
    df_original.sort_values(by='date_hour', inplace=True)

    df_future_month.rename(columns={"day_of_month": 'day'}, inplace=True)
    df_future_month['date_hour'] = \
        pd.to_datetime(df_future_month[['year', 'month', 'day', 'hour']])
    df_future_month.sort_values(by='date_hour', inplace=True)

    '''Pot training data together with forecast timeseries'''

    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(df_original['date_hour'], df_original['Value'], label='data')
    ax.plot(df_future_month['date_hour'], df_future_month['Value'],
            label='forecast')
    ax.legend()
    plt.ylabel(outcome_var)
    plt.title(f'{outcome_var} timeseries')
    plt.xticks(rotation=45)
    plt.savefig(pred_output_dir + f'timeseries_train_pred_{outcome_var}.png',
                dpi=400)
    plt.clf()
    plt.close("all")

    '''Plot forecast timeseries'''

    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(df_future_month['date_hour'], df_future_month['Value'],
            label='forecast')
    ax.legend()
    plt.ylabel(outcome_var)
    plt.title(f'{outcome_var} timeseries')
    plt.xticks(rotation=45)
    plt.savefig(pred_output_dir + f'timeseries_pred_{outcome_var}.png',
                dpi=400)
    plt.clf()
    plt.close("all")

    '''Save forecast to csv'''

    df_future_month[['date_hour', 'Value']]\
        .to_csv(pred_output_dir + f'forecast_{outcome_var}', index=False)
