from datetime import datetime, timedelta
from config.nytd_api_get import *

from airflow.decorators import dag, task

@dag(
    schedule_interval="30 0 * * *",  # 매일 00:30 실행
    start_date=datetime(2025,3,1),
    catchup=False, # True
    default_args={
        "owner": "airflow",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["nyt", "crawler"]
)
def everyday_nytd_s3():
    
    @task
    def task_get_nytd_data(data_interval_start = None):
        """
        NYTD 데이터를 수집하고 parquet 파일로 저장 후 S3 업로드 파일 경로 반환
        """
        # context = get_current_context()  # 현재 DAG 실행 컨텍스트 가져오기
        # data_interval_start = context["data_interval_start"]  # pendulum.DateTime 객체로 가져옴. Taskflow에선 jinja 사용 불가?
        target_date = data_interval_start
        return fetch_data_and_save(target_date)

    @task
    def task_s3_data_write(file_info: dict):
        """
        S3 업로드
        """
        upload_to_s3(file_info)
    
    # Airflow에서 data_interval_start을 직접 전달
    file_info = task_get_nytd_data()
    task_s3_data_write(file_info)

dag_instance = everyday_nytd_s3()
