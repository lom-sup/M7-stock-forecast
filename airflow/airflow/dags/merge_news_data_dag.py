import boto3
from airflow.operators.python import PythonOperator
import datetime
from airflow.decorators import dag, task
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor


@dag(
    dag_id="merge_news_data_dag",
    schedule_interval = "0 4 * * *",    # 매일 UTC 04:00 실행
    start_date=datetime.datetime(2025, 3, 2),
    catchup=False,
    tags=['glue', 'merge_news_data']
)
def merge_news_data_dag():
    
    def push_is_incremental(**kwargs):
        is_incremental = kwargs.get('dag_run').conf.get("is_incremental", Variable.get("is_incremental", "false"))
        is_incremental = str(is_incremental).lower()
        kwargs['ti'].xcom_push(key="is_incremental", value=is_incremental)

    push_is_incremental_task = PythonOperator(
        task_id="push_is_incremental",
        python_callable=push_is_incremental,
        provide_context=True
    )

    merge_news_data_task = GlueJobOperator(
        task_id='merge_news_data',
        job_name='ce5-glue-job-test',
        region_name='ap-northeast-2',
        num_of_dpus=2,
        script_args={
            "--JOB_NAME": "ce5-glue-job-test",
            "--is_incremental": "{{ ti.xcom_pull(task_ids='push_is_incremental', key='is_incremental') }}",    # XCom에서 값 가져오기
            "--data_interval_start": "{{ data_interval_start.isoformat() }}",
            "--data_interval_end": "{{ data_interval_end.isoformat() }}",
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-metrics": "true",
            "--enable-continuous-log-filter": "true",
            '--task_type': 'merge_news_data',
        },
        aws_conn_id='my_aws_conn',
        do_xcom_push=True,
    )

    @task
    def final_message():
        return "Merge News Data Task가 성공적으로 완료되었습니다."

    push_is_incremental_task >> merge_news_data_task >> final_message()

dag_instance = merge_news_data_dag()
