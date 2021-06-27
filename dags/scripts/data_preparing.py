import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

def feature_selection():
    seed = 0
    # Data access
    # df = pd.read_csv(r'..\data\download_data.csv')
    df = pd.read_csv(r'./dags/data/download_data.csv')
    y = df.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=seed)

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
    # df_train_rfe.to_csv(r'..\data\prepared_train_data.csv', index=False)
    # df_test_rfe.to_csv(r'..\data\prepared_test_data.csv', index=False)
    df_train_rfe.to_csv(r'./dags/data/prepared_train_data.csv', index=False)
    df_test_rfe.to_csv(r'./dags/data/prepared_test_data.csv', index=False)
    return rfe_column

def prepare_data():
	rfe_column = feature_selection()

if __name__=="__main__":
	prepare_data()