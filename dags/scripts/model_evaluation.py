import pickle
from sklearn.metrics import mean_squared_error
from scripts.util import LoadData

def evaluate(**kwargs):
    ld = LoadData()
    work_dir = ld.check_dir()
    X_test, y_test = ld.get_data(datatype='test')
    
    ## Only for Airflow to exchange data ##
    model_version = kwargs['ti'].xcom_pull(key='model_version')
    if model_version is None:
        model_version = 1

    # Model candidates
    model_list = ['Lasso', 'Ridge', 'regr']
    score_dict = {}
    for model in model_list:
        with open(work_dir+'/models/model_'+model+'_v'+str(model_version)+'.pickle', 'rb') as f:
            model_ = pickle.load(f)
            y_pred = model_.predict(X_test)
        score_dict[model_] = mean_squared_error(y_test, y_pred)

    # Find best model
    best_model = min(score_dict, key=score_dict.get)
    ## Only for Airflow to exchange data ##
    kwargs['ti'].xcom_push(key='best_model_score', value=min(score_dict.values()))

    old_model_score = kwargs['ti'].xcom_pull(key='old_model_score')
    new_model_score = min(score_dict.values())
    if new_model_score > old_model_score:
        with open(work_dir+'/models/model_v'+str(model_version)+'.pickle', 'wb') as f:
            pickle.dump(best_model, f)
    else:
        print('No better result!')

if __name__=="__main__":
    evaluate()