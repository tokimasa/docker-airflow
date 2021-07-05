import pickle
from scripts.util import LoadData

def inference(**kwargs):
    ld = LoadData()
    work_dir = ld.check_dir()
    X_test, y_test = ld.get_data(datatype='test')
    
    ## Only for Airflow to exchange data ##
    model_version = kwargs['ti'].xcom_pull(key='model_version')
    if model_version is None:
        model_version = 1

    with open(work_dir+'/models/model_v'+str(model_version)+'.pickle', 'rb') as f:
        model = pickle.load(f)
        print(model.predict(X_test))


if __name__=="__main__":
    inference()