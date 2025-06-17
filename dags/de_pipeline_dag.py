from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from etl.load_data import load_data_task
from etl.preprocess_data import preprocess_data_task
from etl.train_model import train_model_task
from etl.evaluate_model import evaluate_model_task

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='ml_breast_cancer_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='ML pipeline with Airflow',
) as dag:

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data_task,
    )

    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_task,
    )

    load_data >> preprocess_data >> train_model >> evaluate_model