from yellowbrick.regressor import ResidualsPlot, PredictionError
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import os
from scipy import stats

def load_data():
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
    return X_test, y_test, X_train, y_train

def concept_drift(**kwargs):
    if os.path.abspath(os.getcwd()) == "/usr/local/airflow":
        work_dir = './dags'
    else:
        work_dir = '..'

    with open(work_dir+'/models/model_v1.pickle', 'rb') as f:
        model = pickle.load(f)

    # Ploting
    visualizer = ResidualsPlot(model)
    visualizer.fit(kwargs['X_train'], kwargs['y_train'])
    visualizer.score(kwargs['X_test'], kwargs['y_test'])
    if not os.path.exists(work_dir+'/chart/'):
        os.mkdir(work_dir+'/chart/')
    visualizer.show(work_dir + '/chart/residual_plot.jpg')

    # Get testing data score on pre-trained model
    score = model.score(kwargs['X_test'], kwargs['y_test'])

    # Prediction error score is R2
    # Flag for model performance being worse
    flag = score < 0.5
    print('Concept Drift: ', flag, 'Score: ', score)
    return flag

def data_drift(**kwargs):
    feature_column = kwargs['X_test'].columns
    """
    Try one of following tests:
    Population Stability Index/ Kullback-Leibler(KL) divergence/ Jensen-Shannon(JS) divergence/ Kolmogorov-Smirnov(KS) test
    """
    # Check distribution of each feature between test & train dataset
    drift_flag = {}
    for col in feature_column:
        _, pvalue = stats.ks_2samp(kwargs['X_test'][col], kwargs['X_train'][col])
        print(col, pvalue)
        ## Reject the null hypothesis, two distributions are identical, if p-value < 0.05
        drift_flag[col] = pvalue < 0.05

    # Flag for testing data being different from training data
    flag = False
    for item in drift_flag.values():
        flag = flag | item
    print('Data Drift: ', flag, drift_flag.values())
    return flag

def branch_fork(flag):
    train_token = flag
    if train_token:
        return 'model_training'
    return 'inference'

def batch_showname(path):
    last_char = []
    for fname in os.listdir(path):
        # print(os.path.join(path, fname))
        file_name = fname.split(".")[0]
        last_char.append(int(list(file_name)[-1]))
    return max(last_char)

def drift_detection(**kwargs):
    X_test, y_test, X_train, y_train = load_data()
    try:
        concept_flag = concept_drift(X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
        data_flag = data_drift(X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
        flag = concept_flag | data_flag
    except:
        flag = True
    branch_name = branch_fork(flag)
    print(branch_name)

    if os.path.abspath(os.getcwd()) == "/usr/local/airflow":
        work_dir = './dags'
        ## Only for Airflow to exchange data ##
        # Update the last version of experiment model if drift flag is True and push it
        last_version = batch_showname(work_dir + '/models')
        if flag:
            if last_version is None:
                last_version = 1
            else:
                last_version += 1
        kwargs['ti'].xcom_push(key='model_version', value=last_version)
        print(last_version)
        #######################################
    else:
        work_dir = '..'

    return branch_name

if __name__=="__main__":
    X_test, y_test, X_train, y_train = load_data()
    drift_detection()