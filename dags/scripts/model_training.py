from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def train():
    seed = 0
    df = pd.read_csv(r'..\data\prepared_train_data.csv')
    y = df.pop('quality')
    X = df

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
        search = GridSearchCV(pipes, param, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
        search.fit(X, y)
        # Save model in pickle
        with open(r'..\models\model_'+model+'_v1.pickle', 'wb') as f:
            pickle.dump(search, f)
            print('Model saved!')

        with open(r'..\models\param_'+model+'_v1.pickle', 'wb') as f:
            pickle.dump(search.best_estimator_, f)
            print('Param. saved!')

if __name__=="__main__":
	train()