import datetime
from airflow.decorators import dag, task
from config.cnn_crawler import main
from airflow.models import Variable

is_incremental_str = Variable.get("is_incremental", default_var = False)
is_incremental = (is_incremental_str.lower() == "true")

schedule_interval="30 0 * * *" if is_incremental else "@once"

@dag(
    schedule_interval=schedule_interval,
    start_date=datetime.datetime(2025, 3, 1),
    catchup=True,
    default_args={"owner": "airflow", "retries": 3},
    tags=["cnn", "crawler"]
)

def cnn_crawler_dag():

    @task
    def cnn_crawler_task(**context):

        run_date = context["data_interval_start"].date()
        # 만약 “첫 DAG Run(3/1)은 full refresh, 그 외(3/2~)는 incremental” 로 하려면:
        if run_date == datetime.date(2025, 3, 1):
            target_date = run_date
            is_incremental = False
        else:
            target_date = run_date - datetime.timedelta(days=1)
            is_incremental = True
        
        main(target_date, is_incremental)


    cnn_crawler_task()

dag_instance = cnn_crawler_dag()
