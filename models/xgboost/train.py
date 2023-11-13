from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from root import *
import seaborn as sns


def xgboost_train_eval(df_train, df_val, df_test, outcome_var, eval):
    """
    Either evaluate or train the final model.
    Evaluation is performed using the test set, which was not used for
    hyperparameter tuning or training. The MAE is calculated from the antilog
    of the outcome value and prediction, so that it is reported in the units of
    the original outcome variable.
    The final model used for prediction is trained using all available data,
    to produce the strongest possible predictor.
    :param df_train: pandas dataframe: the training set
    :param df_val: pandas dataframe: the validation set
    :param df_test: pandas dataframe: the test set
    :param outcome_var: string: 'sales' or 'traffic
    :param eval: True or False: indicates whether to perform the final model
    evaluation on the test set or whether to train a final model on all
    available data.
    :return: None. The trained model is saved to
    {root}/output/models/XGBoost/{outcome_var}/
    """

    XGBoost_output_dir = f"{root}/output/models/XGBoost/{outcome_var}/"
    if not os.path.exists(XGBoost_output_dir):
        os.makedirs(XGBoost_output_dir)

    XGBoost_eval_dir = f"{root}/output/evaluation/XGBoost/{outcome_var}/"
    if not os.path.exists(XGBoost_eval_dir):
        os.makedirs(XGBoost_eval_dir)

    '''Split X and y'''

    x = ['year_scaled', 'month_sin', 'month_cos', 'day_of_month_sin',
         'day_of_month_cos', 'day_1', 'day_2', 'day_3', 'day_4',
         'day_5', 'day_6', 'hour_sin', 'hour_cos']
    y = 'log_Value'

    x_train = df_train[x]
    y_train = df_train[y]
    x_val = df_val[x]
    y_val = df_val[y]
    x_test = df_test[x]
    y_test = df_test[y]

    '''Before forecasting the following month, train a model on the full 
    dataset'''

    if not eval:

        x_train = x_train.append(x_val).append(x_test)
        y_train = y_train.append(y_val).append(y_test)

    if outcome_var == 'sales':

        model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=1,
            alpha=500,
            seed=42)

    if outcome_var == 'traffic':

        model = XGBRegressor(
            n_estimators=100,
            max_depth=300,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=1,
            alpha=300,
            seed=42)

    if eval:
        eval_set = [(x_train, y_train), (x_val, y_val)]
    else:
        eval_set = [(x_train, y_train)]

    model = model.fit(
        x_train,
        y_train,
        eval_metric='mae',
        eval_set=eval_set,
        verbose=False,
        early_stopping_rounds=1
    )

    '''Save model'''

    if not eval:
        pickle.dump(model, open(XGBoost_output_dir +
                                'xgboost_trained.pickle.dat', 'wb'))

        '''Plot feature importance'''

        features = pd.DataFrame()
        features['feature'] = x_train.columns
        features['importance'] = model.feature_importances_
        features = features.sort_values('importance', ascending=False)
        features = features.head(50)

        plt.bar(features['feature'], features['importance'])
        plt.xticks(rotation=50)
        plt.xticks(fontsize=5)
        plt.title(f'{outcome_var} XGBoost feature importance')

        plt.savefig(XGBoost_eval_dir + f'xgboost_feature_importance.png',
                    dpi=400)

        plt.clf()
        plt.close("all")

    '''Evaluate the model using the test set, which has not been used for 
    either hyperparameter tuning or model training'''

    if eval:

        eval_df = pd.DataFrame()
        eval_df['log_Value'] = y_test.copy()
        eval_df['pred_log_Value'] = model.predict(x_test)

        eval_df['Value'] = np.exp(eval_df['log_Value'])
        eval_df['pred_Value'] = np.exp(eval_df['pred_log_Value'])

        MAE = mean_absolute_error(eval_df['Value'], eval_df['pred_Value'])
        MAE = round(MAE, 2)

        print(MAE)

        '''Plot the model loss per epoch'''

        results = model.evals_result()
        epochs = len(results['validation_0']['mae'])
        x_axis = range(0, epochs)
        sns.set_theme()
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['mae'], label='Train')
        ax.plot(x_axis, results['validation_1']['mae'], label='Test')
        ax.legend()
        plt.ylabel('Mean Absolute Error (log(y))')
        plt.title(f'{outcome_var} test MAE (original units): {MAE}')
        plt.savefig(XGBoost_eval_dir + 'final_model_eval.png', dpi=400)
        plt.clf()
        plt.close("all")
