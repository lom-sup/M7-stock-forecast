import requests
import logging
import os
from io import BytesIO
import time
import pandas as pd
from datetime import datetime
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# Global
API_KEY = Variable.get("NYTD_API_KEY")
BASE_URL = Variable.get("NYTD_BASE_URL")
companies = ["Apple", "Amazon", "Google", "Microsoft", "Facebook", "Tesla", "Netflix"]
news_desks = ["Business", "Financial", "Market Place", "Business Day", "DealBook", "Personal Investing"]
fl_fields = "headline,pub_date,lead_paragraph,web_url,news_desk"
news_desk_fq = 'news_desk:("' + '", "'.join(news_desks) + '")'

# AWS
# S3_bucket = "de5-finalproj-team5-test" # 추후에 test 제거!
S3_bucket = "de5-finalproj-team5"
AWS_CONN_ID = "my_aws_conn" # conn_id 수정 완료
# AWS_CONN_ID = "my_aws" # test


def fetch_page_data(url, params):
    """NYT API 호출, 429 이나 기타 에러 발생 시 Exception raise"""
    response = requests.get(url, params=params)
    
    time.sleep(10)
    
    # 정상 동작
    if response.status_code == 200:
        return response.json()
    
    # Alert
    elif response.status_code == 429:
        raise Exception("Rate limit hit (429).")
    else:
        raise Exception(f"HTTP error {response.status_code} for URL: {response.url}")


def fetch_data_and_save(target_date: datetime) -> dict:
    """
    NYT 데이터를 수집하고 config에 임시로 parquet 파일을 생성한 후,
    S3 업로드에 필요한 파일 정보(s3_key, binary data)를 반환
    """
    year = target_date.year
    month = str(target_date.month).zfill(2)
    day = str(target_date.day).zfill(2)
    begin_date = f"{year}{month}{day}"
    end_date = begin_date

    all_articles = []
    
    # 더 이상 기사 없을 때까지 기사 가져옴
    for company in companies:
        logging.info(f"Fetching articles for {company}...")
        page = 0
        while True:
            params = {
                "q": company,
                "fq": news_desk_fq,
                "begin_date": begin_date,
                "end_date": end_date,
                "sort": "oldest",
                "page": page,
                "fl": fl_fields,
                "facet": "true",
                "facet_fields": "news_desk",
                "facet_filter": "true",
                "api-key": API_KEY
            }
            try:
                data = fetch_page_data(BASE_URL, params)
            except Exception as e:
                logging.error(f"Error fetching data for {company} on page {page}: {e}")
                break

            docs = data.get("response", {}).get("docs", [])
            if not docs:
                break

            logging.info(f"{company} - page {page}: {len(docs)} articles found")
            for doc in docs:
                article = {
                    "headline": doc.get("headline", {}).get("main", ""),
                    "pub_date": doc.get("pub_date", ""),
                    "content": doc.get("lead_paragraph") or doc.get("snippet") or "",
                    "web_url": doc.get("web_url", ""),
                    "news_desk": doc.get("news_desk", "N/A"),
                    "stock": company
                }
                all_articles.append(article)
            page += 1
            time.sleep(15)
        logging.info(f"{company} data collection complete")
        time.sleep(10)
    
    if all_articles:
        df = pd.DataFrame(all_articles)
        file_dir = "/opt/airflow/config"
        os.makedirs(file_dir, exist_ok=True)
        file_path = "/opt/airflow/config/nyt_articles_incremental.parquet"
        
        df.to_parquet(file_path, engine="pyarrow", index=False)
        
        s3_key = f"raw_data/NYTD/{year}/incremental/nyt_articles_{year}_{month}_{day}.parquet"
        
        logging.info(f"Data saved to {file_path}. Ready to upload.")
        return {"s3_key": s3_key, "file_path": file_path}
    
    else:
        logging.info("No articles found for this date.")
        return {}
        


def upload_to_s3(file_info: dict):
    """
    파일 정보를 받아 S3에 config 폴더에서 파일을 업로드
    """
    if not file_info or "s3_key" not in file_info or "file_path" not in file_info:
        logging.info("No file info provided, skipping S3 upload.")
        return
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    s3_hook.load_file(
        filename=file_info["file_path"],
        key=file_info["s3_key"],
        bucket_name=S3_bucket,
        replace=True
    )

    logging.info(f"File uploaded to S3: {file_info['s3_key']}")