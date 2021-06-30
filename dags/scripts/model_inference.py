import pandas as pd
import pickle
import os

def inference(**kwargs):
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

    df_test = pd.read_csv(work_dir+'/data/prepared_test_data.csv')
    y_test = df_test.pop('quality')
    X_test = df_test

    with open(work_dir+'/models/model_v'+str(model_version)+'.pickle', 'rb') as f:
        model = pickle.load(f)
        print(model.predict(X_test))

if __name__=="__main__":
	inference()