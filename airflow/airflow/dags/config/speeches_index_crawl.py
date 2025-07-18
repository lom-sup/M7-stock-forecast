import os
import pandas as pd
import json
import time
import logging
from datetime import datetime
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import boto3
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

S3_BUCKET = "de5-finalproj-team5"
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)
INDEX_CSV = os.path.join(DATA_PATH, "fed_speeches.csv")
INDEX_JSON = os.path.join(DATA_PATH, "fed_speeches.json")
INDEX_PARQUET = os.path.join(DATA_PATH, "fed_speeches.parquet")
AWS_CONN_ID = "my_aws_conn"

def upload_file_to_s3(local_path, bucket, s3_prefix):
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    creds = s3_hook.get_credentials()
    logging.info(f"Retrieved AWS credentials: {creds}")
    filename = os.path.basename(local_path)
    s3_key = f"{s3_prefix}/{filename}"
    try:
        s3_hook.load_file(
            filename=local_path,
            key=s3_key,
            bucket_name=bucket,
            replace=True
        )
        logging.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}")

def get_chrome_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def parse_date(date_text):
    formats = ["%B %d, %Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    return None

def update_index(start_date, end_date, s3_prefix):
    if os.path.exists(INDEX_CSV):
        df_existing = pd.read_csv(INDEX_CSV)
        existing_keys = set(zip(df_existing["date"].tolist(),
                                df_existing["title"].tolist(),
                                df_existing["speaker"].tolist(),
                                df_existing["location"].tolist()))
    else:
        df_existing = pd.DataFrame()
        existing_keys = set()

    base_url = "https://www.federalreserve.gov/newsevents/speeches.htm"
    driver = get_chrome_driver()
    driver.get(base_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@class, "row ng-scope")]'))
    )
    logging.info("🚀 인덱스 데이터 업데이트 시작")

    new_speeches = []
    page_number = 1
    stop_flag = False

    try:
        while True:
            logging.info(f"📄 {page_number}번째 페이지 크롤링 중...")
            speech_elements = driver.find_elements(By.XPATH, '//div[contains(@class, "row ng-scope")]')
            if not speech_elements:
                logging.warning("⚠️ 페이지 내 연설 요소를 찾을 수 없습니다.")
                break

            for speech in speech_elements:
                try:
                    date_text = speech.find_element(By.CLASS_NAME, "itemDate").text.strip()
                    date_obj = parse_date(date_text)
                    if date_obj is None:
                        continue
                    if end_date is not None and date_obj > end_date:
                        continue
                    if date_obj < start_date:
                        logging.info("🚨 기준 날짜 이전의 데이터가 발견되어 크롤링을 종료합니다.")
                        stop_flag = True
                        break

                    title = speech.find_element(By.CLASS_NAME, "itemTitle").text.strip()
                    speaker = speech.find_element(By.CLASS_NAME, "news__speaker").text.strip()
                    location = speech.find_element(By.CLASS_NAME, "result__location").text.strip()
                    composite_key = (date_text, title, speaker, location)

                    if composite_key in existing_keys:
                        logging.info(f"🚨 중복 데이터 건너뜀 ({date_text}, {title}, {speaker}, {location}).")
                        continue

                    link = speech.find_element(By.XPATH, './/a[@ng-bind-html="item.title"]').get_attribute("href")
                    new_speeches.append({
                        "date": date_text,
                        "title": title,
                        "speaker": speaker,
                        "location": location,
                        "link": link
                    })
                    existing_keys.add(composite_key)
                except Exception as e:
                    logging.warning(f"⚠️ 데이터 추출 오류: {e}")
            if stop_flag:
                break

            try:
                next_page = driver.find_element(By.XPATH, '//a[contains(@ng-click, "selectPage") and text()="Next"]')
                driver.execute_script("arguments[0].click();", next_page)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@class, "row ng-scope")]'))
                )
                page_number += 1
            except Exception:
                logging.info("🚨 다음 페이지가 없습니다. 크롤링 종료합니다.")
                break

    finally:
        driver.quit()

    if not df_existing.empty:
        df_new = pd.DataFrame(new_speeches)
        df_updated = pd.concat([df_new, df_existing], ignore_index=True)
        df_updated.drop_duplicates(subset=["date", "title", "speaker", "location"], inplace=True)
    else:
        df_updated = pd.DataFrame(new_speeches)

    df_updated.to_csv(INDEX_CSV, index=False, encoding="utf-8")
    df_updated.to_json(INDEX_JSON, orient="records", indent=4, force_ascii=False)
    df_updated.to_parquet(INDEX_PARQUET, index=False)
    logging.info(f"✅ 인덱스 데이터 업데이트 완료: 신규 {len(new_speeches)}개, 전체 {len(df_updated)}개 기록")

    upload_file_to_s3(INDEX_CSV, S3_BUCKET, s3_prefix)
    upload_file_to_s3(INDEX_JSON, S3_BUCKET, s3_prefix)
    upload_file_to_s3(INDEX_PARQUET, S3_BUCKET, s3_prefix)

    return new_speeches

if __name__ == "__main__":
    update_index(datetime(2020, 1, 1), datetime(2025, 2, 28), "raw_data/FOMC/archived")
