# from yellowbrick.regressor import ResidualsPlot, PredictionError
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import os

def concept_drift():
    if os.path.abspath(os.getcwd()) == "/usr/local/airflow":
        work_dir = './dags'
    else:
        work_dir = '..'

    df_train = pd.read_csv(work_dir+'/data/prepared_train_data.csv')
    df_test = pd.read_csv(work_dir+'/data/prepared_test_data.csv')
    y_train = df_train.pop('quality')
    X_train = df_train
    y_test = df_test.pop('quality')
    X_test = df_test

    with open(work_dir+'/models/model_v1.pickle', 'rb') as f:
        model = pickle.load(f)

    # visualizer = PredictionError(model)
    # visualizer.score(X_test, y_test)
    score = model.score(X_test, y_test)

    # prediction error score is R2
    # flag = visualizer.score_ > 0.5
    flag = score > 0.5
    print(flag)
    return flag

def branch_fork(flag):
    train_token = flag
    if train_token:
        return 'model_training'
    return 'result_inference'

def drift_detection():
    try:
        flag = concept_drift()
    except:
        flag = True
    branch_name = branch_fork(flag)
    print(branch_name)
    return branch_name

if __name__=="__main__":
	drift_detection()