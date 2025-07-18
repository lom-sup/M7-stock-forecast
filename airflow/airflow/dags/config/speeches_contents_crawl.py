import os
import pandas as pd
import json
import logging
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

S3_BUCKET = "de5-finalproj-team5"
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)
INDEX_CSV = os.path.join(DATA_PATH, "fed_speeches.csv")
# processed_links 파일명을 분리하여 사용
PROCESSED_LINKS_FILE = os.path.join(DATA_PATH, "processed_speeches_contents_links.json")

def get_output_filenames():
    today_str = datetime.today().strftime("%Y%m%d")
    output_csv = os.path.join(DATA_PATH, f"fed_speeches_contents_{today_str}.csv")
    output_json = os.path.join(DATA_PATH, f"fed_speeches_contents_{today_str}.json")
    output_parquet = os.path.join(DATA_PATH, f"fed_speeches_contents_{today_str}.parquet")
    return output_csv, output_json, output_parquet

def upload_file_to_s3(local_path, bucket, s3_prefix):
    s3_client = boto3.client('s3')
    filename = os.path.basename(local_path)
    s3_key = f"{s3_prefix}/{filename}"
    try:
        s3_client.upload_file(local_path, bucket, s3_key)
        logging.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}")

def scrape_press_release(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        heading_div = soup.select_one("div.heading.col-xs-12.col-sm-8.col-md-8")
        if heading_div:
            date_elem = heading_div.select_one("p.article__time")
            title_elem = heading_div.select_one("h3.title")
            release_time_elem = heading_div.select_one("p.releaseTime")
            date_text = date_elem.get_text(strip=True) if date_elem else None
            title_text = title_elem.get_text(strip=True) if title_elem else None
            release_time_text = release_time_elem.get_text(strip=True) if release_time_elem else None
        else:
            date_text, title_text, release_time_text = None, None, None

        content_div = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        text_content = "\n".join([p.get_text(strip=True) for p in content_div.find_all("p")]) if content_div else ""

        return {
            "url": url,
            "date": date_text,
            "title": title_text,
            "release_time": release_time_text,
            "content": text_content,
        }
    except Exception as e:
        logging.warning(f"⚠️ 크롤링 오류 ({url}): {e}")
        return None

def update_contents(s3_prefix):
    try:
        df_index = pd.read_csv(INDEX_CSV)
    except Exception as e:
        logging.error(f"인덱스 파일을 불러올 수 없습니다: {e}")
        return

    if os.path.exists(PROCESSED_LINKS_FILE):
        with open(PROCESSED_LINKS_FILE, "r", encoding="utf-8") as f:
            processed_links = set(json.load(f))
    else:
        processed_links = set()

    new_records = df_index[~df_index["link"].isin(processed_links)]
    if new_records.empty:
        logging.info("새로운 컨텐츠 크롤링 대상이 없습니다.")
        return

    new_links = new_records["link"].tolist()
    logging.info(f"새로운 컨텐츠 크롤링 대상: {len(new_links)}개")

    results = []
    for idx, url in enumerate(new_links, start=1):
        logging.info(f"🔍 [{idx}/{len(new_links)}] {url} 크롤링 중...")
        result = scrape_press_release(url)
        if result:
            results.append(result)
            processed_links.add(url)

    if results:
        output_csv, output_json, output_parquet = get_output_filenames()
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False, encoding="utf-8")
        df_results.to_json(output_json, orient="records", indent=4, force_ascii=False)
        df_results.to_parquet(output_parquet, index=False)
        logging.info(f"✅ 신규 컨텐츠 크롤링 완료! {len(results)}개 데이터 저장됨: {output_csv}")

        upload_file_to_s3(output_csv, S3_BUCKET, s3_prefix)
        upload_file_to_s3(output_json, S3_BUCKET, s3_prefix)
        upload_file_to_s3(output_parquet, S3_BUCKET, s3_prefix)
    else:
        logging.info("새로운 컨텐츠가 없습니다.")

    with open(PROCESSED_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_links), f, ensure_ascii=False, indent=4)
    upload_file_to_s3(PROCESSED_LINKS_FILE, S3_BUCKET, s3_prefix)

if __name__ == "__main__":
    update_contents("raw_data/FOMC/archived")
