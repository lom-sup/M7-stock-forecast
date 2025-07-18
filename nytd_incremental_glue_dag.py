from airflow.decorators import dag, task
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from datetime import datetime, timedelta

# DAG 기본 설정
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 3, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    default_args=default_args,
    schedule_interval="0 1 * * *",  # 매일 오전 1시 실행
    catchup=True,
    description="Daily Incremental(from 202503) NYTD data clensing Using Spark(Glue)"
)
def glue_s3_incremental_load():
    
    @task
    # data_interval_start=None
    def generate_script_args(data_interval_start=None) -> dict:
        """script_args를 생성하는 함수"""
        target_date = data_interval_start.strftime("%Y_%m_%d")# YYYY_MM_DD 포맷
        return {"--LOAD_TYPE": "incremental", "--TARGET_DATE": target_date}
    
    script_args = generate_script_args()
    
    glue_job = GlueJobOperator(
        task_id="nytd_incremental_glue_dag",
        job_name="glue_s3_processing",
        script_location="s3://your-bucket-name/scripts/nytd_glue_script.py", # 추후에 script 파일 위치에 따라 수정 필요
        script_args=script_args,
        iam_role_name="GlueExecutionRole",
        wait_for_completion=True,
    )
    
    # script_args >> glue_job

glue_s3_incremental_load()