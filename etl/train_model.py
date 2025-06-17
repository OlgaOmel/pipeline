import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model_task():
    with open('/opt/airflow/tmp/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    y_train = pd.read_csv('/opt/airflow/tmp/y_train.csv').squeeze()

    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open('/opt/airflow/results/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('/opt/airflow/tmp/model.pkl', 'wb') as f:
        pickle.dump(model, f)