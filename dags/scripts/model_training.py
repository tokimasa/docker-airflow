from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import os
from scripts.util import LoadData

def train(**kwargs):
    seed = 0
    ld = LoadData()
    work_dir = ld.check_dir()
    X, y = ld.get_data(datatype='train')
    
    ## Only for Airflow to exchange data ##
    model_version = kwargs['ti'].xcom_pull(key='model_version')
    if model_version is None:
        model_version = 1

    pipe1 = Pipeline([('scaler', StandardScaler()),
                      ('poly', PolynomialFeatures()),
                      ('lasso', linear_model.Lasso())])
    pipe2 = Pipeline([('scaler', StandardScaler()),
                      ('poly', PolynomialFeatures()),
                      ('ridge', linear_model.Ridge())])

    pipe3 = Pipeline([('scaler', StandardScaler()),
                     ('poly', PolynomialFeatures()),
                     ('regr', RandomForestRegressor(random_state=seed))])

    regr_param = {'regr__n_estimators': [1, 10, 100],
                  'regr__max_depth': [1, 5, 10],
                  }
    lasso_param = {'lasso__alpha': [0.005, 0.03, 0.06],
                   }
    ridge_param = {'ridge__alpha': [550, 600, 650],
                   }

    pipe_dict = {'Lasso': (pipe1, lasso_param),
                 'Ridge': (pipe2, ridge_param),
                 'regr': (pipe3, regr_param)
                 }

    for model, (pipes, param) in pipe_dict.items():
        print(model, pipes)
        if not os.path.isfile(work_dir+'/models/model_'+model+'_v'+str(model_version)+'.pickle'):
            search = GridSearchCV(pipes, param, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
            search.fit(X, y)
            # Save model in pickle
            with open(work_dir+'/models/model_'+model+'_v'+str(model_version)+'.pickle', 'wb') as f:
                pickle.dump(search, f)
                print('Model saved!')

            with open(work_dir+'/models/param_'+model+'_v'+str(model_version)+'.pickle', 'wb') as f:
                pickle.dump(search.best_estimator_, f)
                print('Param. saved!')

if __name__=="__main__":
    train()