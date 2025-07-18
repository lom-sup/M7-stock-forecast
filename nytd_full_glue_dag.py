from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.decorators import dag, task
from datetime import datetime, timedelta


# DAG 기본 설정
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025,3,1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    default_args = default_args,
    schedule_interval = "@once", # 딱 한 번만 실행
    catchup = False,
    description = "One-time FULL(2020~202502) NYTD data clensing Using Spark(Glue)"
)

def glue_s3_full_load():
    
    @task
    def generate_script_args() -> dict:
        "Full 인자 script_args로 넘겨주기"
        return {"--LOAD_TYPE": "full"}
    
    script_args = generate_script_args()  # 실행 결과를 변수에 저장

    glue_job = GlueJobOperator(
        task_id="nytd_full_glue_dag",
        job_name="glue_s3_processing",
        script_location="s3://your-bucket-name/scripts/nytd_glue_script.py", # 나중에 진짜 script 경로로 수정
        script_args=script_args,
        iam_role_name="GlueExecutionRole",
        wait_for_completion=True,
    )

    # script_args >> glue_job 자동으로 의존성 생성?
    # https://www.astronomer.io/docs/learn/airflow-decorators/#taskflow-to-traditional-operator 참고!
    
glue_s3_full_load()
