from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.hooks.base_hook import BaseHook
from airflow.contrib.operators.slack_webhook_operator import SlackWebhookOperator
from airflow.utils.dates import days_ago
from scripts.data_ingestion import ingest_data
from scripts.data_preparing import prepare_data
from scripts.data_drift import drift_detection
from scripts.model_training import train
from scripts.model_inference import inference
from scripts.model_evaluation import evaluate

def task_fail_slack_alert(context):
    slack_webhook_token = BaseHook.get_connection(SLACK_CONN_ID).password
    slack_msg = """
            :red_circle: Task Failed. 
            *Task*: {task}  
            *Dag*: {dag} 
            *Execution Time*: {exec_date}  
            *Log Url*: {log_url} 
            """.format(
            task=context.get('task_instance').task_id,
            dag=context.get('task_instance').dag_id,
            ti=context.get('task_instance'),
            exec_date=context.get('execution_date'),
            log_url=context.get('task_instance').log_url,
        )
    failed_alert = SlackWebhookOperator(
        task_id='slack_test',
        http_conn_id=SLACK_CONN_ID,
        webhook_token=slack_webhook_token,
        message=slack_msg,
        username='airflow')
    return failed_alert.execute(context=context)

# Define some arguments for our DAG
default_args = {
    'owner': 'arocketman',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'catchup': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': task_fail_slack_alert
}

# Instantiate our DAG
dag = DAG(
    'orchestrated_experiment',
    default_args=default_args,
    description='A ML orchestrated experiment',
    schedule_interval=timedelta(days=1),
)

SLACK_CONN_ID = 'slack_connection'
slack_webhook_token = BaseHook.get_connection(SLACK_CONN_ID).password

def validate_data():
	print('show data quality...')

def get_chart():
    return "https://gitlab-k8s.wzs.wistron.com.cn/10809115/docker-airflow-main/-/blob/master/dags/chart/residual_plot.jpg"

with dag:
        drift_detection_task = BranchPythonOperator(
        task_id='drift_detection',
        provide_context=True,
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
        provide_context=True,
        python_callable=train
        )

        model_evaluation_task = PythonOperator(
        task_id='model_evaluation',
        provide_context=True,
        python_callable=evaluate
        )

        inference_task_1 = PythonOperator(
        task_id='inference',
        provide_context=True,
        python_callable=inference
        )

        inference_task_2 = PythonOperator(
        task_id='retrained_inference',
        provide_context=True,
        python_callable=inference
        )

        git_push_task = BashOperator(
        task_id='git_push_task',
        # "scripts" folder is under "/usr/local/airflow/dags"
        bash_command="scripts/git_push.sh",
        trigger_rule='one_success',
        )

        post_performance_task = SlackWebhookOperator(
        task_id='post_performance',
        http_conn_id=SLACK_CONN_ID,
        webhook_token=slack_webhook_token,
        message=get_chart(),
        channel='#feed'
        )

        data_ingestion_task >> data_validation_task >> data_preparation_task >> drift_detection_task >> [model_training_task, inference_task_1]
        model_training_task >> model_evaluation_task >> inference_task_2
        [inference_task_1, inference_task_2] >> git_push_task >> post_performance_task
