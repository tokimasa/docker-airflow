import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

def evaluate(**kwargs):
    if os.path.abspath(os.getcwd()) == "/usr/local/airflow":
        work_dir = './dags'
        ## Only for Airflow to exchange data ##
        model_version = kwargs['ti'].xcom_pull(key='model_version')
        if model_version is None:
            model_version = 1
        #######################################
    else:
        work_dir = '..'
        model_version = 1

    # Data access
    df_test = pd.read_csv(work_dir+'/data/prepared_test_data.csv')
    y_test = df_test.pop('quality')
    X_test = df_test

    # Model candidates
    model_list = ['Lasso', 'Ridge', 'regr']
    score_dict = {}
    for model in model_list:
        with open(work_dir+'/models/model_'+model+'_v'+str(model_version)+'.pickle', 'rb') as f:
            model_ = pickle.load(f)
            y_pred = model_.predict(X_test)
        score_dict[model_] = mean_squared_error(y_test, y_pred)

    print(score_dict)
    if os.path.abspath(os.getcwd()) == "/usr/local/airflow":
        ## Only for Airflow to exchange data ##
        kwargs['ti'].xcom_push(key='best_model_score', value=max(score_dict.values()))
        #######################################

    # Find best model
    best_model = max(score_dict, key=score_dict.get)
    with open(work_dir+'/models/model_v'+str(model_version)+'.pickle', 'wb') as f:
        pickle.dump(best_model, f)

if __name__=="__main__":
	evaluate()