from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.utils.dates import days_ago
from scripts.data_ingestion import ingest_data
from scripts.data_preparing import prepare_data
from scripts.data_drift import drift_detection
from scripts.model_training import train
from scripts.model_inference import inference
from scripts.model_evaluation import evaluate

# Define some arguments for our DAG
default_args = {
    'owner': 'arocketman',
    'depends_on_past': False,
    'start_date': datetime(2021, 6, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Instantiate our DAG
dag = DAG(
    'orchestrated_experiment',
    default_args=default_args,
    description='A ML orchestrated experiment',
    schedule_interval=timedelta(minutes=30),
)


def validate_data():
	print('show data quality...')


with dag:
	drift_detection_task = BranchPythonOperator(
        task_id='drift_detection',
        python_callable=drift_detection
	)

	data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=ingest_data
	)

	data_validation_task = PythonOperator(
        task_id='data_validation',
        python_callable=validate_data
	)

	data_preparation_task = PythonOperator(
        task_id='data_preparation',
        python_callable=prepare_data
	)

	model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=train
	)

	model_evaluation_task = PythonOperator(
        task_id='model_evaluation',
        python_callable=evaluate
	)


	inference_task = PythonOperator(
        task_id='result_inference',
        python_callable=inference
	)

	data_ingestion_task >> data_validation_task >> data_preparation_task >> drift_detection_task >> [model_training_task, inference_task]
	model_training_task >> model_evaluation_task
