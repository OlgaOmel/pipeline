import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data_task():
    os.makedirs('/opt/airflow/tmp', exist_ok=True)
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    X.to_csv('/opt/airflow/tmp/X.csv', index=False)
    y.to_csv('/opt/airflow/tmp/y.csv', index=False)