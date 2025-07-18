import os
import datetime
from datetime import timedelta
from dateutil import parser, tz
import pandas as pd
import re
import shutil

import requests
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import boto3
from airflow.models import Variable

# ------------------------------
# 전역 설정 (상수)
# ------------------------------
# 로컬 파일 저장 경로
LOCAL_DATA_DIR = "cnn_data"
FULL_REFRESH_OUTPUT_DIR = f"{LOCAL_DATA_DIR}/full"
INCREMENTAL_OUTPUT_DIR = f"{LOCAL_DATA_DIR}/incremental"
BACKFILL = f"{LOCAL_DATA_DIR}/fail"
os.makedirs(LOCAL_DATA_DIR, exist_ok = True)
os.makedirs(BACKFILL, exist_ok = True)
os.makedirs(FULL_REFRESH_OUTPUT_DIR, exist_ok = True)
os.makedirs(INCREMENTAL_OUTPUT_DIR, exist_ok = True)

# S3 정보
BUCKET_NAME = "de5-finalproj-team5"
BASE_FOLDER = "raw_data/CNN"

# 검색 키워드
KEYWORDS = [
    "apple", "google", "alphabet", "facebook", "meta",
    "microsoft", "amazon", "tesla", "Elon Musk", "nvidia"
]

# 뉴스 기간
NOW_DATE = datetime.date.today()
LOOKBACK_YEARS = 5
FULL_REFRESH_YEAR = NOW_DATE.year - LOOKBACK_YEARS 
INCREMENTAL_DATE = NOW_DATE - timedelta(days=1)

# ------------------------------
# 로컬 데이터 삭제
# ------------------------------
def remove_data_file():
    shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
    print("로컬 데이터 폴더 삭제 완료")

# ------------------------------
# S3에 저장
# ------------------------------
def upload_file_to_s3(is_incremental: bool):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=Variable.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=Variable.get("AWS_SECRET_ACCESS_KEY")
        )
    
    # 배치 모드에 따른 데이터 폴더 결정
    if not is_incremental:
        source_dir = FULL_REFRESH_OUTPUT_DIR
    else:
        source_dir = INCREMENTAL_OUTPUT_DIR

    # data 폴더 내 파일 목록 확인
    for entry in os.listdir(source_dir):
        source_path = os.path.join(source_dir, entry)

        # 파일명에서 연도 추출 
        match = re.search(r"(\d{4})", entry)
        if not match:
            # 파일명에 연도가 없으면 업로드 스킵
            continue

        year = match.group(1)  # 예: '2020'
        
        # S3에 업로드할 경로(키) 결정
        s3_key = f"{BASE_FOLDER}/{year}/{entry}" # s3://de5-finalproj-team5/raw_data/CNN/2025/파일명.py

        # 업로드 실행
        s3_client.upload_file(source_path, BUCKET_NAME, s3_key)
        print(f"Uploaded {entry} to s3://{BUCKET_NAME}/{s3_key}")

# ------------------------------
# 로컬에 데이터 저장
# ------------------------------
def create_data_file(article_data, is_incremental):
    article_df = pd.DataFrame(article_data)
    file_date = INCREMENTAL_DATE.strftime("%Y%m%d")

    if not article_df.empty:
        article_df = article_df.sort_values("keyword")
        valid_article_df = article_df[article_df["date"].str[:4] != "9999"]

        if is_incremental:
            valid_article_df.to_parquet(os.path.join(INCREMENTAL_OUTPUT_DIR, f"cnn_article_{file_date}.parquet"), index=False)            

        else:
            for i in range(0, LOOKBACK_YEARS + 1):
                article_year = NOW_DATE.year - i
                year_df = valid_article_df[valid_article_df["date"].str[:4].astype(int) == article_year]
                year_df.to_parquet(os.path.join(FULL_REFRESH_OUTPUT_DIR, f"cnn_article_{article_year}.parquet"), index=False)
        print(f"모든 데이터 저장 완료")
    else:
        article_df.to_parquet(os.path.join(FULL_REFRESH_OUTPUT_DIR, f"cnn_article_{file_date}_No_data.parquet"), index=False)
        print("기사 없음")


# ------------------------------
# 뉴스 상세페이지에서 데이터 수집
# ------------------------------
def get_article_data(all_links):
    """
        CNN 상세페이지에서 기사 제목, 날짜, 상세 내용 수집
    """
    articles = []

    for link in all_links:
        url = link["link"]
        res = requests.get(url)
        if res.status_code!= 200:
            print(f"HTTP request failed. Status Code: {res.status_code} URL: {url}")
            continue
        soup = BeautifulSoup(res.text, 'html.parser')

        # 기사 제목
        try:
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else "No title"
        except Exception as e:
            print(f"기사 제목 로딩 실패: {e}")
            title = "[FAIL] couldn't find title"
        
        # 기사 내용
        try:
            content_elem = soup.find('div', class_='article__content')
            content = content_elem.get_text() if content_elem else "No content"
        except Exception as e:
            print(f"기사 내용 로딩 실패: {e}")
            content = "[FAIL] couldn't find content"

        # 기사 날짜
        try:
            detail_pub_date_elem = soup.select_one('meta[property="article:published_time"]')
            if detail_pub_date_elem and "content" in detail_pub_date_elem.attrs:
                dt = parser.parse(detail_pub_date_elem["content"])
                detail_pub_date = dt.replace(tzinfo=None).isoformat(timespec='milliseconds')
            else:
                detail_pub_date = "9999-12-31"
        except Exception as e:
            print(f"기사 날짜 로딩 실패: {e}")
            detail_pub_date = "9999-12-31"

        articles.append({
            "keyword": link["keyword"],
            "date": detail_pub_date,
            "title": title,
            "content": content,
            "link": url
            })
        print(link["keyword"], detail_pub_date, title)

    return articles

# ------------------------------
# 뉴스 사이트에서 키워드 검색 및 상세 뉴스 링크 수집
# ------------------------------
def get_links(keyword, driver, target_date, is_incremental):
    """
    CNN front page에서 article links를 수집
    날짜 기준으로 수집(2020.01 ~ 2025.02)
    """
    links = []
    page_from = 0
    page = 1
    size = 100

    while True:
        url = f"https://edition.cnn.com/search?q={keyword}&from={page_from}&size={size}&page={page}&sort=newest&types=all&section="
        driver.get(url)
        has_new_article = False

        wait = WebDriverWait(driver, 30)

        try:   
            # 기사목록 요소 리스트
            card_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.card')))
            
            for card in card_elements:
                try:
                    pub_date_org = card.find_element(By.CLASS_NAME, 'container__date').text
                    pub_date = parser.parse(pub_date_org, fuzzy=True).date()
                except Exception as e:
                    print(f"기사 날짜 로딩 실패: {e}")
                    continue

                # full refresh 수집 일자 및 수집 종료 날짜
                start_date = datetime.date(target_date.year - 5, 1, 1)
                last_date = datetime.date(NOW_DATE.year, 3, 1)

                # full refresh or incremental
                if (not is_incremental and start_date <= pub_date < last_date) or (is_incremental and pub_date == target_date):
                    has_new_article = True
                    link = card.get_attribute('data-open-link')
                    # video 와 cnn-underscored 링크는 제외
                    if "/videos/" in link or "/cnn-underscored/" in link:
                        continue
                    links.append({
                        "keyword": keyword,
                        "link": link
                        })
                        
            if not has_new_article:
                print(f"이 페이지에는 최신 기사가 없으므로 페이지 순회 중단")
                break
            
        except Exception as e:
            if isinstance(e, TimeoutException) or isinstance(e, NoSuchElementException):
                print("마지막 페이지까지 검색 완료")
                return links

            print(f"[{keyword}] 페이지 {page}] 검색 결과 로딩 실패: {e}")
            return links
        
        # 다음 페이지로 이동
        page += 1
        page_from += size

    return links

# ------------------------------
# Main
# ------------------------------
def main(target_date: datetime.date,is_incremental: bool):
    all_links = []
    options = Options()
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
                        "AppleWebKit/537.36 (KHTML, like Gecko) " +
                        "Chrome/114.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    
    # 기사 링크 수집
    for keyword in KEYWORDS:
        print(f"=========Crawling {keyword}==========")
        linkd_results = get_links(keyword, driver, target_date, is_incremental)
        print(f"{keyword} news total: {len(linkd_results)}")
        all_links.extend(linkd_results)
    
    driver.quit()

    # 기사 데이터 저장
    article_data = get_article_data(all_links)
    
    # 파일 생성
    create_data_file(article_data, is_incremental)

    # S3에 저장
    upload_file_to_s3(is_incremental)

    # 호스트에서 생성된 파일 삭제
    remove_data_file()

if __name__ == '__main__':
    is_incremental = False
    target_date = datetime.date(2025, 3, 15)
    main(target_date, is_incremental)
