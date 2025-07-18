import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import regexp_replace, col
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from datetime import datetime, timedelta

# --------------------------------------------------------------------------------
# 전역 변수 설정
# --------------------------------------------------------------------------------
S3_BUCKET = "de5-finalproj-team5"

            
# --------------------------------------------------------------------------------
# S3 데이터 읽기 및 저장
# --------------------------------------------------------------------------------
def get_file_from_S3(spark: SparkSession, raw_data_2024: str, raw_data_2025: str) -> DataFrame:
    return spark.read.parquet(raw_data_2024, raw_data_2025)

def save_file_to_S3_staging(df_cleaned: DataFrame, output_file_path: str):
    try:
        # DynamicFrame 변환 및 파티션 1개로 조정
        dynamic_frame = DynamicFrame.fromDF(df_cleaned, glueContext, "dynamic_frame")
        df_single = dynamic_frame.toDF().coalesce(1)
        dynamic_frame_single = DynamicFrame.fromDF(df_single, glueContext, "dynamic_frame_single")

        # S3에 저장 (Parquet 형식)
        glueContext.write_dynamic_frame.from_options(
            frame=dynamic_frame_single,
            connection_type="s3",
            connection_options={"path": output_file_path},
            format="parquet"
        )

        df.write.mode("overwrite").parquet(output_file_path)
        return output_file_path
    
    except Exception as e:
        error_msg = f"{output_file_path} 경로에 파일 저장 실패: {e}"
        print(error_msg)
        return error_msg


# --------------------------------------------------------------------------------
# 배치 처리 함수: Full Refresh와 Incremental
# --------------------------------------------------------------------------------
# Full refresh 모드
def process_full_refresh(spark: SparkSession):
    """
    FULL REFRESH 모드:
        - 2024~2025년 2월까지의 데이터 처리
    """
    raw_data_2024 = f"s3://{S3_BUCKET}/raw_data/FINNHUB/2024/*.parquet"
    raw_data_2025 = f"s3://{S3_BUCKET}/raw_data/FINNHUB/2025/archived_until_202502/*.parquet"
    OUTPUT_PATH = f"s3://{S3_BUCKET}/staging_data/full/"
    
    union_dfs = get_file_from_S3(spark, raw_data_2024, raw_data_2025)
    
    
    # 중복 뉴스 제거
    df = union_dfs.dropDuplicates(["symbol", "datetime", "headline"])
    
    # URL, HTML, 추가 텍스트, 공백 정규식 패턴 정의
    url_pattern = r"http[s]?://\S+"
    html_pattern = r"<.*?>"
    extra_text_pattern = r"\[.*?\]|\(.*?\)"
    whitespace_pattern = r"\s+"

    # 데이터 전처리
    df_cleaned = df.withColumn("headline", regexp_replace(col("headline"), url_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), html_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), extra_text_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), whitespace_pattern, " ")) \
                    .withColumn("summary", regexp_replace(col("summary"), url_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), html_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), extra_text_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), whitespace_pattern, " "))

    # S3에 저장 (Parquet 형식)
    save_file_to_S3_staging(df_cleaned, OUTPUT_PATH)
    print("년별 파일(2024 및 2025의 2월 데이터) 전처리 후 저장 완료:", OUTPUT_PATH)
    
    
def process_incremental(spark: SparkSession, data_interval_start, data_interval_end):
    """
    INCREMENTAL 모드:
        - 현재 연도와 INCREMENTAL_DATE에 해당하는 파일만 읽어 전처리 후 저장
    """
    # URL, HTML, 추가 텍스트, 공백 정규식 패턴 정의
    url_pattern = r"http[s]?://\S+"
    html_pattern = r"<.*?>"
    extra_text_pattern = r"\[.*?\]|\(.*?\)"
    whitespace_pattern = r"\s+"
    
    # data_interval_start ~ data_interval_end 범위 내의 날짜들을 처리
    current_day = data_interval_start
    while current_day < data_interval_end:
        day_str = (data_interval_start - timedelta(days=1)).strftime("%Y%m%d")
        RAW_DATA_2025 = f"s3://{S3_BUCKET}/raw_data/FINNHUB/2025/incremental/finnhub_m7_news_{day_str}.parquet"
        OUTPUT_PATH = f"s3://{S3_BUCKET}/staging_data/incremental/FINNHUB/{day_str}"
        
        # S3에서 데이터 로드
        try:
            df = spark.read.parquet(RAW_DATA_2025)
        except Exception as e:
            print(f"{day_str} 파일을 읽는 중 오류 발생: {e}")
            current_day += timedelta(days=1)
            continue  # 해당 날짜는 건너뛰기
        
        # 중복 뉴스 제거
        df = df.dropDuplicates(["symbol", "datetime", "headline"])
        
        # 데이터 전처리
        df_cleaned = df.withColumn("headline", regexp_replace(col("headline"), url_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), html_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), extra_text_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), whitespace_pattern, " ")) \
                    .withColumn("summary", regexp_replace(col("summary"), url_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), html_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), extra_text_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), whitespace_pattern, " "))

        # S3에 저장 (Parquet 형식)
        save_file_to_S3_staging(df_cleaned, OUTPUT_PATH)
        print(f"{day_str} 처리 완료: {OUTPUT_PATH}")
        current_day += timedelta(days=1)

# --------------------------------------------------------------------------------
# Main 함수. main script에서 호출하기 위함으로 def main -> def run 으로 변경
# --------------------------------------------------------------------------------
def run(spark, is_incremental, data_interval_start, data_interval_end):

    if not is_incremental:
        process_full_refresh(spark)
    else:
        process_incremental(spark, data_interval_start, data_interval_end)


if __name__ == '__main__':
    spark = SparkSession.builder.appName("FINNHUB_article_preprocessing") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    is_incremental = False
    run(spark, is_incremental, data_interval_start, data_interval_end)
