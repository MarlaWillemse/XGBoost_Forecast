from root import *
from preprocessing.data_preprocessing import *
from models.xgboost.gridsearch_cv import *
import multiprocessing as mp
from models.xgboost.train import *
from models.xgboost.predict import *

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


def main():

    tune_hyperparameters = 0

    df_sales = pd.read_csv(f'{root}/input_data/training_Sales.csv')
    df_traffic = pd.read_csv(f'{root}/input_data/training_Traffic.csv')
    df_holidays = pd.read_csv(f'{root}/input_data/US_national_holidays.csv',
                              names=['sequence', 'date', 'holiday'])

    for df, outcome_var in [(df_sales, 'sales'), (df_traffic, 'traffic')]:

        '''Perform exploratory data analysis (EDA) to determine how to handle
        missing timepoints. Save EDA results to:
        /EDA_output/<outcome_variable>/raw_data/'''

        df_eda = split_date_time(df)
        eda_stats(df_eda, outcome_var, stage='raw_data')
        df_eda = missing_day_placeholders(df_eda)
        df_eda = missing_hour_placeholders(df_eda, fill_missing=np.NaN)
        df_eda = additional_date_features(df_eda, df_holidays)
        eda_missing(df_eda, outcome_var, stage='raw_data')
        eda_distributions(df_eda, outcome_var, stage='raw_data')
        df_original = df_eda.copy()

        '''
        Data preprocessing for XGBoost:
        Remove hours with outlying values of sales or traffic, again create
        placeholders for missing timepoints, and repeat the EDA data
        characterization. Assign a value of zero to missing hours and remove
        missing days, since they don't affect the XGBoost model.
        Perform datetime feature engineering.
        Save the model input data.'''

        df = split_date_time(df)
        df = missing_hour_placeholders(df, fill_missing=0)
        df = additional_date_features(df, df_holidays)
        eda_distributions(df, outcome_var, 'outliers_removed')
        df_future_month = future_daterange(df)
        df_future_month = additional_date_features(df_future_month,
                                                   df_holidays)
        df, df_future_month = transform_datetime_features(df, df_future_month,
                                                          outcome_var)

        '''Train-validate-test split'''

        df_train, df_val, df_test = data_split(df)

        '''Hyperparameter tuning with gridsearch cross-validation:
        train XGBoost model with various hyperparameters and evaluate.'''

        if tune_hyperparameters:

            XGBoost_tuning_dir = f"{root}/output/hyperparameter_tuning/" \
                                 f"XGBoost/{outcome_var}/"
            if not os.path.exists(XGBoost_tuning_dir):
                os.makedirs(XGBoost_tuning_dir)

            hyperparams_CV_df = \
                pd.DataFrame(columns=['max_depth', 'min_child_weight', 'gamma',
                                      'subsample', 'colsample_bytree', 'alpha',
                                      'RMSE'])

            jobs = []

            for max_depth in [10, 50, 100, 300, 500]:
                for min_child_weight in [1]:
                    for gamma in [0, 0.1]:
                        for subsample in [0.7, 0.8, 0.9, 1]:
                            for colsample_bytree in [0.7, 0.8, 0.9, 1]:
                                for alpha in [10, 50, 100, 300, 500]:

                                    p = mp.Process(target=xgboost_tune,
                                                   args=(df_train, df_val,
                                                         XGBoost_tuning_dir,
                                                         max_depth,
                                                         min_child_weight,
                                                         gamma,
                                                         subsample,
                                                         colsample_bytree,
                                                         alpha,
                                                         hyperparams_CV_df))

                                    jobs.append(p)
                                    p.start()

            hyperparams_CV_df.to_csv(XGBoost_tuning_dir + "hyperparams_CV.csv",
                                     index=False)

        '''Evaluate the final model and then retrain it using the full
        dataset.'''

        xgboost_train_eval(df_train, df_val, df_test, outcome_var, eval=1)
        xgboost_train_eval(df_train, df_val, df_test, outcome_var, eval=0)

        '''Make predictions and plot them.'''

        predict(df_original, df_future_month, outcome_var)


if __name__ == '__main__':
    main()
