import boto3
import time
from airflow.operators.python import PythonOperator
import datetime
from airflow.decorators import dag, task
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Airflow Variable에서 is_incremental 값 가져오기
is_incremental = Variable.get("is_incremental", default_var=False)
if isinstance(is_incremental, str):
    is_incremental = is_incremental.strip().lower() == "true"

# 스케줄 인터벌 설정
schedule_interval = "0 3 * * *" if is_incremental else "@once"    # 매일 UTC 03:00 실행

# 기본 인자 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
}


@dag(
    dag_id="glue_job_dag",
    schedule_interval=schedule_interval,
    start_date=datetime.datetime(2025, 3, 2),
    default_args=default_args,
    catchup=True,
    tags=['glue', 'cleansing', 'data union']
)
def glue_job_dag():
    
    # is_incremental 값에 따라 실행할 Task를 결정하는 함수
    def choose_glue_task():
        return 'glue_incremental_main' if is_incremental else 'glue_full_refresh_main'

    branch_glue_task = BranchPythonOperator(
        task_id="choose_glue_task",
        python_callable=choose_glue_task,
    )
    
    """
    Glue Job 호출
    """
    glue_full_refresh = GlueJobOperator(
        task_id='glue_full_refresh_main',
        job_name='ce5-glue-job-test',  
        region_name='ap-northeast-2',
        num_of_dpus=2,
        script_args={
            "--JOB_NAME": "ce5-glue-job-test",
            '--is_incremental': 'false',
            "--data_interval_start": "{{ data_interval_start.isoformat() }}",
            "--data_interval_end": "{{ data_interval_end.isoformat() }}",
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-metrics": "true",
            "--enable-continuous-log-filter": "true",
            '--task_type': 'finnhub_cleansing,cnn_cleansing,nytd_cleansing',
        },
        aws_conn_id='my_aws_conn',
        do_xcom_push=True,  # Glue Job 실행 결과를 XCom에 저장
    )

    glue_incremental = GlueJobOperator(
        task_id='glue_incremental_main',
        job_name='ce5-glue-job-test',
        region_name='ap-northeast-2',
        num_of_dpus=2,
        script_args={
            "--JOB_NAME": "ce5-glue-job-test",
            '--is_incremental': 'true',
            "--data_interval_start": "{{ data_interval_start.isoformat() }}",
            "--data_interval_end": "{{ data_interval_end.isoformat() }}",
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-metrics": "true",
            "--enable-continuous-log-filter": "true",
            '--task_type': 'finnhub_cleansing,cnn_cleansing,nytd_cleansing',
        },
        aws_conn_id='my_aws_conn',
        do_xcom_push=True,
    )

    # S3 파일 저장 결과 확인 Task 추가
    @task
    def verify_s3_output():
        return check_s3_output()

    @task
    def final_message() -> str:
        return "Glue Full Refresh Job이 성공적으로 완료되었습니다."

    branch_glue_task >> [glue_full_refresh, glue_incremental]
    

dag_instance = glue_job_dag()

