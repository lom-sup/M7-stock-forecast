from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import botocore.exceptions
import logging

# 기본 설정
default_args = {
    "owner": "eb",
    "start_date": datetime(2025, 3, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# DAG 정의
with DAG(
    dag_id="finnhub_news_ingestion_dag",
    default_args=default_args,
    schedule_interval="30 0 * * *",  # 매일 UTC 기준 00:30 실행
    catchup=True,
) as dag:

    @task
    def get_date_strings(data_interval_start, data_interval_end):
        """날짜 범위를 문자열로 변환하여 반환"""
        
        # ISO 형식의 문자열을 datetime 객체로 변환
        start_date = datetime.fromisoformat(data_interval_start)
        end_date = datetime.fromisoformat(data_interval_end)
        
        return {
            "from_date": start_date.strftime("%Y-%m-%d"),
            "to_date": end_date.strftime("%Y-%m-%d"),
            "data_interval_start": start_date.strftime("%Y%m%d")  # S3 파일명 생성을 위해 사용
        }

    @task
    def fetch_finnhub_news_for_symbol(dates: dict, symbol: str) -> list:
        """ 
        Finnhub API 호출하여 뉴스 데이터를 가져오고,
        필요한 필드만 필터링하여 리스트로 반환
        """
        token = Variable.get("FINNHUB_API_KEY")
        if isinstance(dates, tuple):
            dates = dates[0]
        from_date = dates["from_date"]
        to_date = dates["to_date"]
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={token}"
        
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()  # HTTP 4xx, 5xx 에러 발생 시 예외 처리
            data = resp.json()
            
            logging.info(f"API Response for {symbol}: {data}")
            
            if not isinstance(data, list) or len(data) == 0:
                logging.warning(f"No news data found for {symbol} from {from_date} to {to_date}")
                return []

            filtered_data = []
            for item in data:
                row = {
                    "symbol": symbol,
                    "source": item.get("source", ""),
                    "datetime": str(item.get("datetime", "")),  # 유닉스 타임을 문자열로 저장
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                }
                filtered_data.append(row)
            return filtered_data
        
        except requests.exceptions.RequestException as e:
            logging.error(f"API 요청 실패: {symbol}, 에러: {e}")
            raise RuntimeError(f"Finnhub API 요청 실패: {e}")

    @task
    def write_news_to_s3(news_results, dates: dict):
        """ 데이터를 Parquet 파일로 변환 후 S3에 저장 """
        s3_hook = S3Hook(aws_conn_id="my_aws_conn")
        s3_client = s3_hook.get_conn()

        bucket_name = "de5-finalproj-team5"
        base_key = "raw_data/FINNHUB/2025/incremental/"
        timestamp = dates["data_interval_start"]
        filename = f"finnhub_m7_news_{timestamp}.parquet"
        s3_key = f"{base_key}{filename}"

        # 여러 심볼에 대한 결과 리스트를 하나로 합침
        all_data = [item for sublist in news_results for item in sublist]

        if not all_data:
            logging.warning(f"No news data found. Skipping S3 upload: {s3_key}")
            raise ValueError(f"No news data found for upload: {s3_key}")

        try:
            df = pd.DataFrame(all_data)
            table = pa.Table.from_pandas(df)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf)
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=buf.getvalue().to_pybytes()
            )
            logging.info(f"Successfully uploaded to s3://{bucket_name}/{s3_key}")
        except (botocore.exceptions.ClientError, botocore.exceptions.Boto3Error) as e:
            logging.error(f"S3 업로드 실패: {s3_key}, 에러: {e}")
            raise RuntimeError(f"S3 업로드 실패: {e}")

    # 날짜 문자열 생성 (XCom으로 전달됨)
    dates = get_date_strings(
        data_interval_start="{{ data_interval_start }}", 
        data_interval_end="{{ data_interval_end }}"
    )    
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

    # 태스크 매핑을 통해 각 심볼에 대해 fetch 태스크 실행
    news_results = (
        fetch_finnhub_news_for_symbol
        .partial(dates=dates)
        .expand(symbol=symbols)
    )
    write_news_to_s3(news_results, dates)
