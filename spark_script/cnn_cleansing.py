from pyspark.sql.functions import col, lit, to_date, input_file_name, regexp_extract, udf
from pyspark.sql.types import StringType
from functools import reduce
import re
import datetime
from datetime import timedelta
from pyspark.sql import SparkSession, DataFrame
import sys
from awsglue.utils import getResolvedOptions

# --------------------------------------------------------------------------------
# 전역 변수 설정
# --------------------------------------------------------------------------------
NOW_DATE = datetime.date.today()
LOOKBACK_YEARS = 5
FULL_REFRESH_START_YEAR = NOW_DATE.year - LOOKBACK_YEARS
INCREMENTAL_DATE = NOW_DATE - timedelta(days=1)
CURRENT_YEAR = str(NOW_DATE.year)
INCREMENTAL_DATE_STR = INCREMENTAL_DATE.strftime("%Y%m%d")

base_file_path = "s3a://de5-finalproj-team5/raw_data/CNN"
output_file_path = "s3a://de5-finalproj-team5/staging_data/news"

# --------------------------------------------------------------------------------
# 전처리 UDF 생성: content 클렌징
# --------------------------------------------------------------------------------
def preprocess_content(content):
    """
    콘텐츠 전처리:
        1. 시작과 끝의 큰따옴표 제거
        2. 줄바꿈(\n) 기준 분할 후, 각 문장이 탭 3번 혹은 스페이스 12번으로 시작하는 경우만 남기고,
            만약 "Related article"이면 그 문장과 바로 뒤 문장은 건너뜀.
        3. 남은 문장을 공백으로 연결.
        4. 콘텐츠 시작 부분의 "CNN", "CNN Business" 등 접두사를 제거.
    """
    if not content:
        return content

    content = content.strip('“”"')
    lines = content.splitlines()
    processed_lines = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if line.strip() == "Related article":
            skip_next = True
            continue
        if not re.match(r'^(?:\t{3}| {12})', line):
            continue
        processed_lines.append(line.strip())
    new_content = " ".join(processed_lines)
    new_content = re.sub(r'^(?:[A-Za-z\s]+)?\bCNN(?: Business)?\s*-\s*', '', new_content)
    return new_content

preprocess_content_udf = udf(preprocess_content, StringType())

# --------------------------------------------------------------------------------
# 전처리 UDF 생성: 키워드 재정의
# --------------------------------------------------------------------------------
def map_keyword(keyword):
    if not keyword:
        return keyword
    keyword_mapping = {
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "elon musk": "TSLA",
    "nvidia": "NVDA"
    }
    return keyword_mapping.get(keyword.lower(), keyword.upper())

map_keyword_udf = udf(map_keyword, StringType())

# --------------------------------------------------------------------------------
# DataFrame 전처리 함수. 컬럼명 재정의 및 클렌징 UDF 적용.
# --------------------------------------------------------------------------------
def process_df(df: DataFrame) -> DataFrame:
    return df.withColumn("symbol", map_keyword_udf(col("keyword"))) \
            .withColumn("source", lit("CNN")) \
            .withColumn("datetime", col("date")) \
            .withColumn("headline", col("title")) \
            .withColumn("summary", preprocess_content_udf(col("content")))
            
# --------------------------------------------------------------------------------
# S3 데이터 읽기
# --------------------------------------------------------------------------------
def get_file_from_S3(spark: SparkSession, s3_file_path: str) -> DataFrame:
    return spark.read.parquet(s3_file_path)

# --------------------------------------------------------------------------------
# S3 staging에 데이터 저장
# --------------------------------------------------------------------------------
def save_file_to_S3_staging(df: DataFrame, target_file_path: str):
    try:
        df.write.mode("overwrite").parquet(target_file_path)
        return target_file_path
    except Exception as e:
        error_msg = f"{target_file_path} 경로에 파일 저장 실패: {e}"
        print(error_msg)
        return error_msg
    
# --------------------------------------------------------------------------------
# 배치 처리 함수: Full Refresh
# --------------------------------------------------------------------------------
def process_full_refresh(spark: SparkSession):
    """
    FULL REFRESH 모드: FULL_REFRESH_START_YEAR부터 현재 연도까지 각 연도별 파일을 읽어 union 처리
    """
    union_dfs = []

    for year in range(FULL_REFRESH_START_YEAR, NOW_DATE.year + 1):
        year_file_path = f"{base_file_path}/{year}/cnn_article_{year}.parquet"
        try:
            df_temp = get_file_from_S3(spark, year_file_path)
            union_dfs.append(df_temp)
        except Exception as e:
            print(f"[WARNING] {year}년 파일 읽는 중 에러 발생: {e}", file=sys.stderr)

    if union_dfs:
        df_union = reduce(lambda a, b: a.union(b), union_dfs)
        df_union_processed = process_df(df_union)
        df_union_final = df_union_processed.select("symbol", "source", "datetime", "headline", "summary")
        output_year_path = f"{output_file_path}/full"
        save_file_to_S3_staging(df_union_final, output_year_path)
        print("년별 파일(2020~2024 및 2025의 2월 데이터) 전처리 후 저장 완료:", output_year_path)
    else:
        print("[ERROR] 연도별 파일 union에 실패했습니다.", file=sys.stderr)

# --------------------------------------------------------------------------------
# 배치 처리 함수: Incremental
# --------------------------------------------------------------------------------
def process_incremental(spark: SparkSession):
    """
        - 현재 연도와 INCREMENTAL_DATE에 해당하는 파일만 읽어 전처리 후 저장
    """
    file_path = f"{base_file_path}/{CURRENT_YEAR}/cnn_article_{INCREMENTAL_DATE_STR}.parquet"
    try:
        df_daily = get_file_from_S3(spark, file_path)
        df_daily_processed = process_df(df_daily)
        df_daily_final = df_daily_processed.select("symbol", "source", "datetime", "headline", "summary")
        output_daily_path = f"{output_file_path}/incremental/CNN"
        save_file_to_S3_staging(df_daily_final, output_daily_path)
        print("Incremental 파일 전처리 후 저장 완료:", output_daily_path)
    except Exception as e:
        print(f"Incremental 파일 읽기 에러(날짜 {INCREMENTAL_DATE_STR}): {e}")

# --------------------------------------------------------------------------------
# Main 함수
# --------------------------------------------------------------------------------
def run(spark, is_incremental):
    if not is_incremental:
        process_full_refresh(spark)
    else:
        process_incremental(spark)

if __name__ == '__main__':
    spark = SparkSession.builder.appName("CNN_article_preprocessing") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    is_incremental = False
    run(spark, is_incremental)
