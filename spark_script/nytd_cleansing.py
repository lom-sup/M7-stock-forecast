import sys
import boto3
import datetime
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit, to_date, when

def get_all_parquet_files(s3_client, bucket, prefix):
    """S3에서 특정 prefix 아래 모든 .parquet 파일 가져오기"""
    files = []
    continuation_token = None

    while True:
        response = (
            s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token)
            if continuation_token
            else s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        )
        if "Contents" in response:
            files.extend(
                [f"s3a://{bucket}/{obj['Key']}" 
                for obj in response.get("Contents", []) 
                if obj["Key"].endswith(".parquet")]
            )

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    return files

def list_parquet_files(s3_client, bucket, is_incremental, target_date=None):
    """
    - Full load: 전체 경로 중, 2020~2024 폴더는 그대로 포함하고,
    2025 폴더에서는 파일명에 '2025_01' 또는 '2025_02'가 포함된 파일만 선택.
    - Incremental load: "raw_data/NYTD/2025/incremental/" 경로에서 target_date가 키에 포함된 파일만 선택. 파일 목록 전부를 가져와 filter하니 오류 발생.
    """
    if is_incremental:
        prefix = "raw_data/NYTD/2025/incremental/"
        all_files = get_all_parquet_files(s3_client, bucket, prefix)
        if target_date:
            filtered_files = [f for f in all_files if target_date in f]
            return filtered_files
        else:
            return all_files
    else: # full인 경우, 딱 한번만 실행
        prefix = "raw_data/NYTD/"
        all_files = get_all_parquet_files(s3_client, bucket, prefix)
        filtered_files = []
        base_path = f"s3a://{bucket}/raw_data/NYTD/"
        for f in all_files:
            subpath = f.replace(base_path, "")
            # 2020~2024 폴더의 경우
            if subpath.startswith(("2020/", "2021/", "2022/", "2023/", "2024/")):
                filtered_files.append(f)
            # 2025 폴더의 경우, 파일명에 '2025_01' 또는 '2025_02'가 포함되어야 함.
            elif subpath.startswith("2025/"):
                filename = subpath.split("/")[-1]
                if "2025_01" in filename or "2025_02" in filename:
                    filtered_files.append(f)
        return filtered_files

def load_and_process_data(spark, s3_client, bucket, is_incremental, target_date=None):
    """데이터를 로드하고 전처리"""
    parquet_files = list_parquet_files(s3_client, bucket, is_incremental, target_date)

    if not parquet_files:
        print("[ERROR] No parquet files to read.")
        sys.exit(1)

    df = spark.read.parquet(*parquet_files)
    df_cleaned = df.withColumn("source", lit("NYTD")).withColumn("pub_date", to_date("pub_date", "yyyy-MM-dd'T'HH:mm:ssZ"))

    df_transformed = df_cleaned.select(
        when(col("stock") == "Apple", lit("AAPL"))
        .when(col("stock") == "Amazon", lit("AMZN"))
        .when(col("stock") == "Google", lit("GOOGL"))
        .when(col("stock") == "Microsoft", lit("MSFT"))
        .when(col("stock") == "Facebook", lit("META"))
        .when(col("stock") == "Tesla", lit("TSLA"))
        .when(col("stock") == "Netflix", lit("NVDA"))
        .otherwise(col("stock")).alias("symbol"),
        col("source"),
        col("pub_date").alias("datetime"),
        col("headline"),
        col("content").alias("summary")
    )
    df_transformed = df_transformed.dropDuplicates(["symbol", "datetime", "headline"])
    return df_transformed

def delete_existing_parquet_files(s3_client, bucket, prefix):
    """S3 폴더 내 기존 Parquet 파일 삭제, overwritten로 기존 데이터 삭제 불가한 경우 대비"""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" in response:
        for obj in response["Contents"]:
            if obj["Key"].endswith(".parquet"):
                s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                print(f"[INFO] Deleted existing file: {obj['Key']}")

def save_data(df_transformed, s3_client, bucket, is_incremental, target_date):
    """데이터를 저장하는 함수"""
    full_output_path = f"s3://{bucket}/staging_data/news/full/"
    # target_date는 "YYYY_MM_DD" 형식으로 전달된다고 가정
    dt = datetime.datetime.strptime(target_date, "%Y_%m_%d")
    ymd = dt.strftime("%Y%m%d")
    incremental_output_path = f"s3://{bucket}/staging_data/news/incremental/NYTD/{ymd}/"

    if is_incremental:
        # 저장 전에 해당 날짜 폴더 내 기존 파일 삭제
        prefix = incremental_output_path.replace(f"s3://{bucket}/", "")
        delete_existing_parquet_files(s3_client, bucket, prefix)
        df_transformed.repartition(1).write.mode("overwrite").parquet(incremental_output_path)
        print("[INFO] Incremental Load 완료")
    else:
        df_transformed.repartition(1).write.mode("overwrite").parquet(full_output_path)
        print("[INFO] Full Load 완료: nytd.parquet")

def run(spark, is_incremental, target_date):
    """SparkSession을 활용하여 데이터 처리 실행"""
    s3_client = boto3.client("s3")
    s3_bucket = "de5-finalproj-team5"

    df_transformed = load_and_process_data(spark, s3_client, s3_bucket, is_incremental)
    save_data(df_transformed, s3_client, s3_bucket, is_incremental, target_date)

if __name__ == '__main__':
    spark = SparkSession.builder.appName("NYTD_article_preprocessing") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .getOrCreate()

    # Glue DAG에서 is_incremental 및 target_date 파싱
    args = getResolvedOptions(sys.argv, ['is_incremental', 'target_date'])
    #is_incremental = args['is_incremental'].lower() == 'true'
    is_incremental = False
    target_date = args['target_date']  # "YYYY_MM_DD" 형식으로 전달됨
    run(spark, is_incremental, target_date)