from airflow.decorators import dag, task
from datetime import datetime, timedelta
from config import press_index_crawl
from config import press_contents_crawl
from config import speeches_index_crawl
from config import speeches_contents_crawl

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(default_args=default_args,
     schedule_interval=None,         # 한 번만 실행
     start_date=datetime(2025, 3, 1),
     catchup=False,
     description="Full crawling: 2020-01-01 ~ 2025-02-28",
     tags=["full_crawling"])
def fomc_full_crawling_dag():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 2, 28)
    s3_prefix = "raw_data/FOMC/archived"
    
    @task
    def run_press_index():
        return press_index_crawl.update_press_releases(start_date, end_date, s3_prefix)

    @task
    def run_speeches_index():
        return speeches_index_crawl.update_index(start_date, end_date, s3_prefix)

    @task
    def run_press_contents():
        return press_contents_crawl.update_contents(s3_prefix)

    @task
    def run_speeches_contents():
        return speeches_contents_crawl.update_contents(s3_prefix)

    press_index_result = run_press_index()
    speeches_index_result = run_speeches_index()

    # 의존성 명시: 인덱스 태스크 완료 후 콘텐츠 크롤링 진행
    press_index_result >> run_press_contents()
    speeches_index_result >> run_speeches_contents()

full_crawling_job = fomc_full_crawling_dag()
