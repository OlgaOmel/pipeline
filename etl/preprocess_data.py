import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data_task():
    X = pd.read_csv('/opt/airflow/tmp/X.csv')
    y = pd.read_csv('/opt/airflow/tmp/y.csv').squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open('/opt/airflow/tmp/X_train.pkl', 'wb') as f:
        pickle.dump(X_train_scaled, f)
    with open('/opt/airflow/tmp/X_test.pkl', 'wb') as f:
        pickle.dump(X_test_scaled, f)
    y_train.to_csv('/opt/airflow/tmp/y_train.csv', index=False)
    y_test.to_csv('/opt/airflow/tmp/y_test.csv', index=False)