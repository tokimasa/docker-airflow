import pandas as pd
import pickle

def inference():
    # df_test = pd.read_csv(r'..\data\prepared_test_data.csv')
    df_test = pd.read_csv(r'./dags/data/prepared_test_data.csv')
    y_test = df_test.pop('quality')
    X_test = df_test

    # with open('..\models\model_v1.pickle', 'rb') as f:
    with open('./dags/models/model_v1.pickle', 'rb') as f:
        model = pickle.load(f)
        print(model.predict(X_test))


if __name__=="__main__":
	inference()