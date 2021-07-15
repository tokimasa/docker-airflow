from yellowbrick.regressor import ResidualsPlot, PredictionError
from sklearn.metrics import r2_score
import pickle
import os
from scipy import stats
from scripts.util import LoadData

def concept_drift(**kwargs):
    ld = LoadData()
    work_dir = ld.check_dir()

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
    return flag, score

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
    ld = LoadData()
    work_dir = ld.check_dir()
    X_train, X_test, y_train, y_test = ld.get_data(datatype='both')
    concept_flag, score = concept_drift(X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
    data_flag = data_drift(X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
    flag = concept_flag | data_flag
    kwargs['ti'].xcom_push(key='old_model_score', value=score)
    # try:
    #     concept_flag, score = concept_drift(X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
    #     data_flag = data_drift(X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
    #     flag = concept_flag | data_flag
    #     kwargs['ti'].xcom_push(key='old_model_score', value=score)
    # except:
    #     flag = True
    branch_name = branch_fork(flag)
    print(branch_name)

    ## Only for Airflow to exchange data ##
    # Update the last version of experiment model if drift flag is True and push it
    last_version = batch_showname(work_dir + '/models')
    if flag:
        if last_version is None:
            last_version = 1
        else:
            last_version += 1
    kwargs['ti'].xcom_push(key='model_version', value=last_version)
    #######################################
    return branch_name

if __name__=="__main__":
    ld = LoadData()
    work_dir = ld.check_dir()
    ld.get_data(datatype='both')
    drift_detection()