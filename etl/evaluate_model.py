import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model_task():
    with open('/opt/airflow/tmp/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    y_test = pd.read_csv('/opt/airflow/tmp/y_test.csv').squeeze()

    with open('/opt/airflow/tmp/model.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }

 
    with open('/opt/airflow/results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)  # indent=4 для красивого форматирования

    return metrics