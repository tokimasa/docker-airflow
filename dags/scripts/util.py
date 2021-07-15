import os
import pandas as pd
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self):
        self.airflow_entry = "/usr/local/airflow"
        self.airflow_wd = './dags'
        self.local_wd = './dags'
        self.ingested_data = '/data/download_data.csv'
        self.prepared_train = '/data/prepared_train_data.csv'
        self.prepared_test = '/data/prepared_test_data.csv'
        self.target_column = 'quality'
        self.seed = 0
        self.test_size = 0.3
        
    def check_dir(self):
        if os.path.abspath(os.getcwd()) == self.airflow_entry:
            self.wd = self.airflow_wd
        else:
            self.wd = self.local_wd
        return self.wd

    def split_data(self):
        df = pd.read_csv(self.wd+self.ingested_data)
        y = df.pop(self.target_column)
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=self.test_size, random_state=self.seed)
        return X_train, X_test, y_train, y_test
    
    def get_data(self, datatype='both'):
        if datatype == 'train':
            df = pd.read_csv(self.wd+self.prepared_train)
            y = df.pop(self.target_column)
            x = df
            return x, y
        elif datatype == 'test':
            df = pd.read_csv(self.wd+self.prepared_test)
            y = df.pop(self.target_column)
            x = df
            return x, y
        elif datatype == 'both':
            df = pd.read_csv(self.wd+self.prepared_train)
            df_ = pd.read_csv(self.wd+self.prepared_test)
            y_train = df.pop(self.target_column)
            X_train = df
            y_test = df_.pop(self.target_column)
            X_test = df_
            return X_train, X_test, y_train, y_test
