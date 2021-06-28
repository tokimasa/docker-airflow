import pandas as pd
import pickle
import os

def inference():
    if os.path.abspath(os.getcwd()) == "/usr/local/airflow":
        work_dir = './dags'
    else:
        work_dir = '..'

    df_test = pd.read_csv(work_dir+'/data/prepared_test_data.csv')
    y_test = df_test.pop('quality')
    X_test = df_test

    with open(work_dir+'/models/model_v1.pickle', 'rb') as f:
        model = pickle.load(f)
        print(model.predict(X_test))


if __name__=="__main__":
	inference()