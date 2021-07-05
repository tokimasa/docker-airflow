import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
from scripts.util import LoadData

def feature_selection():
    seed = 0
    ld = LoadData()
    work_dir = ld.check_dir()
    X_train, X_test, y_train, y_test = ld.split_data()

    # Feature importance by random forest & feature selection for recursive feature elimination
    regr = RandomForestRegressor(max_depth=6, random_state=seed)
    rfe = RFE(estimator=regr, n_features_to_select=2, step=1)
    rfe.fit(X_train, y_train)

    ranking = rfe.ranking_
    support = rfe.support_

    # Selected features/columns
    select_column = X_train.columns[support]
    print(select_column)
    X_train_rfe = X_train[select_column]
    X_test_rfe = X_test[select_column]

    train_rfe = np.vstack((X_train_rfe.T, y_train)).T
    test_rfe = np.vstack((X_test_rfe.T, y_test)).T

    rfe_column = list(select_column)
    rfe_column.append('quality')

    df_train_rfe = pd.DataFrame(train_rfe, columns=rfe_column)
    df_test_rfe = pd.DataFrame(test_rfe, columns=rfe_column)

    # Save selected data
    if not os.path.isfile(work_dir + '/data/prepared_train_data.csv'):
        df_train_rfe.to_csv(work_dir + '/data/prepared_train_data.csv', index=False)
        df_test_rfe.to_csv(work_dir + '/data/prepared_test_data.csv', index=False)
    return rfe_column

def prepare_data():
	rfe_column = feature_selection()

if __name__=="__main__":
	prepare_data()