import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

from root import *


def split_date_time(df):
    """
    Create a date and hour column
    :param df: pandas dataframe with the column 'Date' in the format
    <YYYY-MM-DD HH:MM:SS>
    :return: df: pandas dataframe with a column 'date' in datetime format
    YYYY-MM-DD and integer column 'hour'
    """
    '''Split the date and time'''

    df['date'] = pd.to_datetime(df['Date'].str[:10])
    df['hour'] = df['Date'].str[11:13].astype(int)
    df = df.groupby(['date', 'hour']).sum().reset_index()

    return df


def eda_stats(df, outcome_var, stage):
    """
    Characterise the dataset and write descriptive values to a text file.
    :param df: pandas dataframe with the columns 'date' in datetime format
    YYYY-MM-DD, 'hour in integer format, and the column 'Value' of a numeric
    type.
    :param outcome_var: string; the outcome variable, 'sales' or 'traffic'
    :param stage: 'raw_data' or 'outliers_removed'
    :return: None. EDA stats saved to the EDA_output directory
    """

    eda_dir = f'{root}/output/EDA/{outcome_var}/{stage}/'
    if not os.path.exists(eda_dir):
        os.makedirs(eda_dir)
    eda_text_file = f"{eda_dir}/eda.txt"
    if os.path.exists(eda_text_file):
        os.remove(eda_text_file)

    file = open(eda_text_file, "w+")
    sys.stdout = file

    print(f'Min date: {df["date"].min()}')
    print(f'Max date: {df["date"].max()}')
    nr_days_represented = df['date'].nunique()
    print(f'Nr. days represented: {nr_days_represented}')
    nr_days_in_range = (df['date'].max() - df['date'].min()).days + 1
    nr_missing_days = nr_days_in_range - nr_days_represented
    print(f'Nr. days missing: {nr_missing_days}')
    perc_missing_days = round((nr_missing_days / nr_days_in_range) * 100, 2)
    print(f'Percentage days missing: {perc_missing_days}')
    print(f'Minimum Value: {df["Value"].min()}')
    print(f'Nr. of Values == zero: {(df["Value"] == 0).sum()}')
    print(f'Min Value before outlier removal: {df["Value"].min()}')
    print(f'Max Value before outlier removal: {df["Value"].max()}')

    file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def missing_day_placeholders(df):
    """
    Insert missing days in the date range.
    Assign a null value to the 'Value' column for the previously missing
    timepoints.
    :param df: pandas dataframe with the columns 'date' in datetime format
    YYYY-MM-DD, and the column 'Value' of a numeric
    type.
    :return: df: pandas dataframe with placeholder records for missing days
    """

    '''Insert missing days'''

    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    all_dates = pd.DataFrame(all_dates, columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = pd.merge(all_dates, df, on='date', how='left')

    '''Checksum'''

    nr_days_in_range = (df['date'].max() - df['date'].min()).days + 1
    nr_unique_days = df['date'].nunique()
    assert nr_unique_days == nr_days_in_range, \
        f'There are {nr_unique_days} days present, while there should be ' \
        f'{nr_days_in_range} consecutive days in this range'

    return df


def missing_hour_placeholders(df, fill_missing):
    """
    Insert missing hours in each day.
    Assign the fill_null parameter to the 'Value' column for the previously
    missing timepoints.
    :param df: pandas dataframe with the columns 'date' in datetime format
    YYYY-MM-DD, 'hour in integer format, and the column 'Value' of a numeric
    type.
    :param fill_null: Value to assign to missing hours.
    :return: df: pandas dataframe with placeholder records for missing hours
    """

    hours = list(df['hour'].unique())

    '''Pivot data so that each hour is represented for each date.
    Sum Value if an hour is represented > once.'''

    df.sort_values(['date', 'hour'])
    df_pivot = df.pivot_table(index=['date'],
                              columns=['hour'],
                              values=['Value'],
                              aggfunc=np.sum,
                              fill_value=fill_missing)

    '''Create a column for each hour which isn't represented on any date and 
    fill the 'Value' column with Null'''

    missing_hours = list(set(range(0, 24)) - set(hours))
    for hour in missing_hours:
        df_pivot['Value', hour] = np.NaN

    df = df_pivot.unstack().reset_index(name='Value')
    df = df.drop('level_0', axis=1)[['date', 'hour', 'Value']]
    df.sort_values(['date', 'hour'])

    return df


def additional_date_features(df, df_holidays):
    """
    From the 'date' column, create 'year', 'month', 'day_of_month' and
    'day_of_week' features.
    Merge the external file containing the dates of US holidays and create a
    binary indicator column, 'holiday'.
    Create a column, 'missing' to indicate whether the 'Value' column is Null.
    :param df: pandas dataframe containing a 'date' column in the format
    'yyyy-mm-dd HH:MM:SS'
    :param df_holidays: pandas dataframe containing a 'date' column of US
    holidays in string format 'yyyy-mm-dd'
    :param stage: 'raw_data' or 'outliers_removed'
    :return: df: pandas dataframe with 5 new date feature columns.
    """

    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day_of_month'] = pd.DatetimeIndex(df['date']).day
    df['day_of_week'] = df['date'].dt.day_name()

    df_holidays = df_holidays[['date', 'holiday']].copy()
    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    df_holidays['holiday'] = 1
    df = pd.merge(df, df_holidays, on='date', how='left')
    df['holiday'] = df['holiday'].fillna(0)

    df['missing'] = df['Value'].isnull().astype(int)
    df['missing_hours'] = df['missing'].groupby(df['date']).transform('sum')

    return df


def eda_missing(df, outcome_var, stage):
    """
    Create bar plots to characterise the percentage of a hours for
    which the value of the outcome variable is missing, per category of a
    datetime feature.
    :param df: pandas dataframe containing the columns 'day_of_week',
    'day_of_month', 'hour', 'holiday', and the outcome variable 'sales'
    or 'traffic'
    :param outcome_var: string; the outcome variable, 'sales' or 'traffic'
    :param stage: 'raw_data' or 'outliers_removed'
    :return: None. Plots are saved to the EDA directory in the project root
    """

    eda_plots_missing_datadir = f"{root}/output/EDA/{outcome_var}/{stage}/" \
                                f"perc_hours_missing/"
    if not os.path.exists(eda_plots_missing_datadir):
        os.makedirs(eda_plots_missing_datadir)

    for time_var in ['day_of_week', 'day_of_month', 'hour', 'holiday']:
        df_missing = df.groupby([time_var])['missing'].agg(
            nr_hours='count', nr_hours_missing_time_var='sum').reset_index()
        df_missing['perc_missing'] = \
            (df_missing['nr_hours_missing_time_var'] / df_missing['nr_hours']
             ) * 100

        plt.bar(df_missing[time_var], df_missing['perc_missing'])
        plt.xlabel(time_var)
        plt.ylabel(f'Percentage of hours for which each {time_var} is missing')
        plt.savefig(eda_plots_missing_datadir + f'{time_var}.png')
        plt.clf()
        plt.close("all")


def eda_distributions(df, outcome_var, stage):
    """
    Create box plots representing the distribution of the Value per
    hour by time feature.
    :param df: pandas dataframe containing the columns 'year', 'month',
    'day_of_week', 'day_of_month', 'hour', and the outcome variable 'sales'
    or 'traffic'
    :param outcome_var: string; the outcome variable, 'sales' or 'traffic'
    :param stage: 'raw_data' or 'outliers_removed'
    :return: None. Plots are saved to the EDA directory in the project root.
    """

    eda_plots_distribution_datadir = f"{root}/output/EDA/{outcome_var}/" \
                                     f"{stage}/hourly_value_distribution/"
    if not os.path.exists(eda_plots_distribution_datadir):
        os.makedirs(eda_plots_distribution_datadir)

    for time_var in ['year', 'month', 'day_of_week', 'day_of_month', 'hour',
                     'holiday']:
        sns.boxplot(x=time_var, y='Value', data=df,
                    orient='v') \
            .set(xlabel=time_var, ylabel=f'{outcome_var} per hour')
        plt.xticks(rotation=45)
        plt.savefig(eda_plots_distribution_datadir +
                    f'hourly_{outcome_var}_by_{time_var}.png')
        plt.clf()
        plt.close("all")


def remove_outliers(df):
    """
    Remove Value outliers, defined as hourly sales or traffic values above
    Q3+1.5*IQR. Values are grouped by day_of_week and hour before defining
    outlier thresholds. The lower bound is set to zero.
    :param df: pandas dataframe with columns 'day_of_week', 'hour', and 'Value'
    :return: df: pandas dataframe with outliers removed from the upper end of
    the Value distribution.
    """

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    df_outliers = (df.copy().groupby(['day_of_week', 'hour']).agg(
        Q1=('Value', percentile(25)),
        Q3=('Value', percentile(75)))).reset_index()

    df_outliers['IQR'] = df_outliers['Q3'] - df_outliers['Q1']
    df_outliers['lower_bound'] = df_outliers['Q1'] - 1.5 * df_outliers['IQR']
    df_outliers['upper_bound'] = df_outliers['Q3'] + 1.5 * df_outliers['IQR']

    df_outliers = df_outliers.reset_index()[['day_of_week', 'hour',
                                             'lower_bound', 'upper_bound']]

    df = df.merge(df_outliers, how='inner', on=['day_of_week', 'hour'])

    df = df[df['Value'] > df['lower_bound']]
    df = df[df['Value'] < df['upper_bound']]
    df.drop(['lower_bound', 'upper_bound'], axis=1, inplace=True)

    return df


def future_daterange(df):
    """
    Create a dataframe containing the dates of the following month, and each
    hour in each day.
    :param df: pandas dataframe containing a 'date' column
    :return: df_future_month: pandas dataframe containing a 'date' and 'hour'
    column for the following month.
    """

    '''Create date range'''

    min_date = df['date'].max() + pd.DateOffset(days=1)
    max_date = min_date + pd.DateOffset(months=1)
    df_future_month = pd.date_range(start=min_date, end=max_date)
    df_future_month = pd.DataFrame(df_future_month, columns=['date'])
    df_future_month['date'] = pd.to_datetime(df_future_month['date'])

    '''Create a column containing each hour in each day'''

    for hour in list(range(0, 24)):
        df_future_month[hour] = np.NaN

    df_future_month = pd.melt(df_future_month, id_vars=['date'],
                              value_vars=list(range(0, 24)),
                              var_name='hour', value_name='Value')
    df_future_month['hour'] = df_future_month['hour'].astype(int)

    return df_future_month


def transform_datetime_features(df, df_future_month, outcome_var):
    """
    Transform all features to the range (0, 1).
    Sine- and cosine- transform the month, day of month and hour features to
    represent their cyclic nature.
    Create dummy variables for the day of week.
    Log transform the outcome variable.
    :param df: pandas dataframe containing training data with the columns
    'date', 'month', 'day_of_month', 'hour', 'year', and 'Value'.
    :param df_future_month: pandas dataframe containing dates for the
    following month and the same features as the training data.
    :param outcome_var: string: 'sales' or 'traffic
    :return: df, df_future_month: the training data and input features for
    the dates to be predicted.
    """

    eda_datadir = f"{root}/output/EDA/{outcome_var}/feature_engineering/"
    if not os.path.exists(eda_datadir):
        os.makedirs(eda_datadir)

    df = df[df['Value'] >= 0]

    '''Join the future daterange before transforming the datetime features'''

    max_train_date = df['date'].max()
    df_all_dates = df.append(df_future_month)

    '''Sine- and cosine- transform the month, day of month and hour features to 
    represent their cyclic nature. The transformed features initially have a 
    range of -1 to 1. Shift the range to (0, 1).'''

    df_all_dates['month_sin'] = np.sin(
        (df_all_dates['month'] - 1) * (2. * np.pi / 12))
    df_all_dates['month_cos'] = np.cos(
        (df_all_dates['month'] - 1) * (2. * np.pi / 12))

    df_all_dates['day_of_month_sin'] = np.sin(
        (df_all_dates['day_of_month'] - 1)
        * (2. * np.pi / 31))
    df_all_dates['day_of_month_cos'] = np.cos(
        (df_all_dates['day_of_month'] - 1)
        * (2. * np.pi / 31))

    df_all_dates['hour_sin'] = np.sin(df_all_dates['hour'] * (2. * np.pi / 24))
    df_all_dates['hour_cos'] = np.cos(df_all_dates['hour'] * (2. * np.pi / 24))

    plt.plot(df_all_dates['day_of_month_sin'],
             df_all_dates['day_of_month_cos'])
    plt.savefig(eda_datadir + 'day_of_month_sin_cos.png')
    plt.clf()
    plt.close("all")

    cyclic_cols = df_all_dates[['month_sin', 'month_cos', 'day_of_month_sin',
                                'day_of_month_cos', 'hour_sin', 'hour_cos']]
    cyclic_cols = MinMaxScaler().fit_transform(cyclic_cols)
    df_all_dates[
        ['month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos',
         'hour_sin', 'hour_cos']] = cyclic_cols

    '''Create dummy variables for the day of week, since weekdays have 
    distinct characteristics'''

    df_all_dates['day_of_week_name'] = df_all_dates['day_of_week']
    le = LabelEncoder()
    le.fit(df_all_dates['day_of_week'])
    df_all_dates['day_of_week'] = le.transform(df_all_dates['day_of_week'])
    day_of_week_encoded = pd.get_dummies(df_all_dates['day_of_week'],
                                         prefix='day',
                                         drop_first=True)
    df_all_dates = pd.concat([df_all_dates, day_of_week_encoded], axis=1)

    '''Transform the year feature to have a range (0, 1).'''

    x, y = df_all_dates.year.min(), df_all_dates.year.max()
    df_all_dates['year_scaled'] = (df_all_dates.year - x) / (y - x)

    df = df_all_dates[df_all_dates['date'] <= max_train_date]
    df_future_month = df_all_dates[df_all_dates['date'] > max_train_date]

    '''Log transform the outcome variable'''

    pd.options.mode.chained_assignment = None
    df['Value'].replace(to_replace=0, value=0.0000001, inplace=True)

    df['log_Value'] = np.log(df['Value'].copy())

    return df, df_future_month


def data_split(df):

    """
    Split the data into a training set of 60% of time points (hours), a
    validation set of 20%, and a test set of 20%.
    :param df: pandas dataframe containing the preprocessed data.
    :return: df_train, df_val, df_test: the train, test, and validation
    dataframes
    """

    df_train, df_val, df_test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.6 * len(df)), int(.8 * len(df))])

    return df_train, df_val, df_test
