from airflow.decorators import dag
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 18),
}

@dag(default_args=default_args, schedule_interval='0 5 * * *', dag_id='glue_finbert_sentiment_dag', catchup=False)
def glue_finbert_sentiment_dag():
    run_glue_job = GlueJobOperator(
        task_id='run_finbert_sentiment',
        job_name='ce5-glue-job-test',
        script_location='s3://de5-finalproj-team5/spark_script/BERT_script.zip',
        aws_conn_id='my_aws_conn',
        region_name='ap-northeast-2', 
        script_args={
            '--additional-python-modules': 'transformers,torch'
        }
    )
    
    return run_glue_job

dag = glue_finbert_sentiment_dag()
