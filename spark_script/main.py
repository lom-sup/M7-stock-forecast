import sys
import boto3
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import regexp_replace, regexp_extract, col, coalesce, lit, udf, to_date, when, from_unixtime, date_format, length, year, month, collect_list, struct
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from datetime import datetime, timedelta
# import datetime
import re
from functools import reduce
from pyspark.sql.types import StringType



# Glue Job 실행을 위한 인자 처리
args = getResolvedOptions(sys.argv, ["task_type", "is_incremental", "data_interval_start", "data_interval_end", "JOB_NAME"])

task_types = args["task_type"].split(",")
is_incremental = args['is_incremental'].lower() == 'true'

print(f"🔍 [INFO] 처리할 task_type 목록: {task_types}")
print(f"🔍 [INFO] is_incremental: {is_incremental}")

data_interval_start = datetime.fromisoformat(args["data_interval_start"])
data_interval_end = datetime.fromisoformat(args["data_interval_end"])

TARGET_DATE = data_interval_start.strftime("%Y_%m_%d")  # YYYY_MM_DD 형식


spark = SparkSession.builder.appName(args["JOB_NAME"]).getOrCreate()
glueContext = GlueContext(spark.sparkContext)

# Job 객체 초기화
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

S3_BUCKET = "de5-finalproj-team5"

# --------------------------------------------------------------------------------
# S3 데이터 읽기 및 저장
# --------------------------------------------------------------------------------
def get_file_from_S3(s3_file_path: str) -> DataFrame:
    try:
        df = spark.read.parquet(s3_file_path)
        if df is None or df.rdd.isEmpty():
            raise FileNotFoundError(f"{s3_file_path} 경로의 파일이 비어 있거나 존재하지 않습니다.")
        return df
    except AnalysisException as e:
        raise FileNotFoundError(f"{s3_file_path} 경로의 파일 분석 중 오류 발생: {e}")
    except Exception as e:
        raise RuntimeError(f"{s3_file_path} 경로의 파일 읽기 실패: {e}")


def save_file_to_S3_staging(df: DataFrame, target_file_path: str):
    try:
        df.write.mode("overwrite").parquet(target_file_path)
        return target_file_path
    except Exception as e:
        error_msg = f"{target_file_path} 경로에 파일 저장 실패: {e}"
        print(error_msg)
        return error_msg

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


# 키워드 클렌징
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
    
    print(f"🔍 [INFO] list_parquet_files() 실행됨 - is_incremental: {is_incremental}, target_date: {TARGET_DATE}")
    
    if is_incremental:
        prefix = "raw_data/NYTD/2025/incremental/"
        all_files = get_all_parquet_files(s3_client, bucket, prefix)
        
        print(f"📂 [DEBUG] {prefix} 경로에서 가져온 전체 파일 목록: {all_files}")
        
        if not all_files:
            print(f"❌ [ERROR] {prefix} 경로에서 아무 파일도 찾을 수 없음!")
            return []
        
        # if not target_date or target_date.lower() == "none":
        #     target_date = (data_interval_start - timedelta(days=1)).strftime("%Y_%m_%d")  # YYYY_MM_DD 형식
            
        filtered_files = [f for f in all_files if f.split('/')[-1].startswith(f"nyt_articles_{TARGET_DATE}")]
            
        print(f"📂 [DEBUG] 필터링된 파일 목록: {filtered_files}")
            
        if not filtered_files:
            print(f"There is no {TARGET_DATE} NYTD Data.")
            return []
            
        return filtered_files

    else: # full인 경우, 딱 한번만 실행
        prefix = "raw_data/NYTD/"
        all_files = get_all_parquet_files(s3_client, bucket, prefix)
        
        print(f"📂 [DEBUG] {prefix} 경로에서 가져온 전체 파일 목록: {all_files}")
        
        if not all_files:
            print(f"❌ [ERROR] {prefix} 경로에서 아무 파일도 찾을 수 없음!")
            return []
        
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
                    
        print(f"📂 [DEBUG] 필터링된 파일 목록: {filtered_files}")    
        
        if not filtered_files:
            print(f"❌ [ERROR] Full Load에서도 사용할 파일이 없습니다!")
            return []     
                    
        return filtered_files

def load_and_process_data(s3_client, is_incremental, target_date=None):
    """데이터를 로드하고 전처리"""
    print(f"🚀 [INFO] load_and_process_data 실행됨 - is_incremental: {is_incremental}, target_date: {TARGET_DATE}")
    
    parquet_files = list_parquet_files(s3_client, S3_BUCKET, is_incremental, TARGET_DATE)

    if not parquet_files:
        raise ValueError("[ERROR] No NYTD parquet files to read.")
    
    # 어차피 읽을 parquet는 많아봤자 8개 아래, 그냥 이름 다 보여주기
    print(f"🔍 Found parquet files: {parquet_files}")

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

# def delete_existing_parquet_files(s3_client, bucket, prefix):
#     """S3 폴더 내 기존 Parquet 파일 삭제, overwritten로 기존 데이터 삭제 불가한 경우 대비"""
#     response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
#     if "Contents" in response:
#         for obj in response["Contents"]:
#             if obj["Key"].endswith(".parquet"):
#                 s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
#                 print(f"[INFO] Deleted existing file: {obj['Key']}")

def save_data(df_transformed, s3_client, bucket, is_incremental, target_date, data_interval_start, data_interval_end):
    """데이터를 저장하는 함수"""
    # full_output_path = f"s3://{bucket}/staging_data/news/full/"
    # # target_date는 "YYYY_MM_DD" 형식으로 전달된다고 가정
    # dt = datetime.datetime.strptime(target_date, "%Y_%m_%d")
    # ymd = dt.strftime("%Y%m%d")
    # incremental_output_path = f"s3://{bucket}/staging_data/news/incremental/NYTD/{ymd}/"
    
    """데이터를 저장하는 함수"""
    full_output_path = f"s3://{bucket}/staging_data/news/full/"
    # DAG 에서 target_date가 None값으로 넘어와서 data_interval_end값으로 지정
    ymd = target_date.replace("_", "")
    incremental_output_path = f"s3://{bucket}/staging_data/news/incremental/NYTD/{ymd}/"

    if is_incremental:
        # `_SUCCESS`, `_$folder$` 파일 생성을 방지
        spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        
        # 기존 S3 폴더의 모든 파일 삭제
        print(f"[DELETE] 기존 S3 데이터 삭제 중... 경로: {incremental_output_path}")
        prefix = incremental_output_path.replace(f"s3://{bucket}/", "")
        
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" in response:
            for obj in response["Contents"]:
                s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                
        # delete_existing_parquet_files(s3_client, bucket, prefix)
        df_transformed.repartition(1).write.mode("overwrite").parquet(incremental_output_path)
        print("[INFO] Incremental Load 완료")
    else:
        df_transformed.repartition(1).write.mode("overwrite").parquet(full_output_path)
        print("[INFO] Full Load 완료: nytd.parquet")

# --------------------------------------------------------------------------------
# DataFrame 전처리 함수. 컬럼명 재정의 및 클렌징 UDF 적용.
# --------------------------------------------------------------------------------
def process_df(df: DataFrame) -> DataFrame:
    return df.withColumn("symbol", map_keyword_udf(col("keyword"))) \
            .withColumn("source", lit("CNN")) \
            .withColumn("datetime", date_format(col("date"), "yyyy-MM-dd")) \
            .withColumn("headline", col("title")) \
            .withColumn("summary", preprocess_content_udf(col("content")))
            

    
# --------------------------------------------------------------------------------
# staging_data -> analytics_data
# --------------------------------------------------------------------------------
def list_s3_parquet_files(bucket, prefix):
    """
    S3 버킷 내 특정 경로의 모든 Parquet 파일 목록을 반환
    """
    s3 = boto3.client("s3")
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    parquet_files = [
        f"s3a://{bucket}/{obj['Key']}" 
        for obj in objects.get("Contents", []) 
        if obj["Key"].endswith(".parquet")
    ]
    return parquet_files

def get_file_from_staging(spark: SparkSession, s3_bucket: str, base_s3_prefix: str, is_incremental: bool):
    """
    S3에 저장된 full 또는 incremental 데이터를 읽어와 하나의 DataFrame으로 반환
    """
    try:
        if not is_incremental:
            df = spark.read.parquet(f"s3a://{s3_bucket}/{base_s3_prefix}/full")
        else:
            dfs = []
            for source in sources:
                prefix = f"{base_s3_prefix}/incremental/{source}/"
                parquet_files = list_s3_parquet_files(s3_bucket, prefix)
                
                if parquet_files:
                    print(f"[INFO] {source}에서 {len(parquet_files)}개의 Parquet 파일을 찾았습니다.")
                    dfs.append(spark.read.parquet(*parquet_files))
                else:
                    print(f"[WARN] {source} 데이터 없음: {prefix}")
            
            if dfs:
                df = reduce(lambda a, b: a.union(b), dfs)
            else:
                print("[WARN] 읽어올 데이터가 없습니다.")
                sys.exit(1)
                
        return df.coalesce(1) if df else None
    
    except Exception as e:
        print(f"[ERROR] get_file_from_staging 실행 중 오류 발생: {e}")
        sys.exit(1)

def save_to_analytics(df: DataFrame, s3_bucket: str, output_s3_prefix: str):
    try:
        # output_s3_prefix = "analytic_data/news"
        output_path = f"s3a://{s3_bucket}/{output_s3_prefix}"
        file_count = 0
        saved_paths = []

        print(f"[DEBUG] 저장 경로: {output_path}")
        print(f"[DEBUG] 데이터프레임 총 행 수: {df.count()}")
        print(f"[DEBUG] 데이터프레임 스키마:")
        df.printSchema()
        
        # datetime을 강제로 string으로 변환 (타입 불일치 해결)
        df = df.withColumn("datetime", col("datetime").cast("string"))
        
        # NULL 값 확인
        df_null_check = df.filter(col("datetime").isNull())
        print(f"[DEBUG] datetime NULL인 데이터 개수: {df_null_check.count()}")
        df_null_check.show(5)
        
        # datetime 컬럼이 'YYYY-MM-DD' 또는 'YYYY-MM-DDTHH:mm:ss.SSS' 형태일 경우, 'YYYY-MM-DD'로 변환
        df = df.withColumn(
            "record_date",
            when(length(col("datetime")) > 10, to_date(col("datetime"), "yyyy-MM-dd'T'HH:mm:ss.SSS"))
            .when(regexp_extract(col("datetime"), "^[0-9]+$", 0) != "", to_date(from_unixtime(col("datetime").cast("bigint"), "yyyy-MM-dd")))
            .otherwise(to_date(col("datetime"), "yyyy-MM-dd"))
        )

        # 연도와 월 컬럼 추가
        df = df.withColumn("year", year(col("record_date"))).withColumn("month", month(col("record_date")))

        # 고유한 연-월 조합을 그룹화하여 저장
        for row in df.select("year", "month").distinct().collect():
            yyyy, mm = row["year"], row["month"]
            partition_df = df.filter((col("year") == yyyy) & (col("month") == mm))
            partition_output = f"{output_path}/year={yyyy}/month={mm}"
            
            partition_count = partition_df.count()
            print(f"[DEBUG] 저장 중: {partition_output}, 데이터 개수: {partition_count}")
            
            if partition_count == 0:
                print(f"[WARN] 월 {mm}에 대한 데이터가 없습니다.")
                continue
            
            partition_df.select("symbol", "source", "datetime", "headline", "summary") \
                .write.mode("overwrite").parquet(partition_output)
            
            saved_paths.append(partition_output)
            file_count += 1

        print(f"[INFO] 데이터 저장 완료: 총 {file_count}개의 파일이 파티션에 저장되었습니다.")
        return df.select("year", "month").distinct().collect(), saved_paths, file_count
    except Exception as e:
        print(f"[ERROR] save_to_analytics 실행 중 오류 발생: {e}")
        sys.exit(1)


if "cnn_cleansing" in task_types:
    
    data_interval_start = datetime.fromisoformat(args["data_interval_start"])
    data_interval_end = datetime.fromisoformat(args["data_interval_end"])
    is_incremental = args['is_incremental'].lower() == 'true'
    
    
    print("cnn_cleansing 개발 중...")
    """
        - 현재 연도와 INCREMENTAL_DATE에 해당하는 파일만 읽어 전처리 후 저장
    """
    # 날짜
    current_year = data_interval_start.strftime("%Y")
    incremental_date_str = data_interval_start.strftime("%Y%m%d")
    # 파일 경로
    base_file_path = "s3a://de5-finalproj-team5/raw_data/CNN"
    output_file_path = "s3a://de5-finalproj-team5/staging_data/news"
    file_path = f"{base_file_path}/{current_year}/cnn_article_{incremental_date_str}.parquet"
    
    try:
        df_daily = get_file_from_S3(file_path)
        df_daily_processed = process_df(df_daily)
        df_daily_final = df_daily_processed.select("symbol", "source", "datetime", "headline", "summary")
        output_daily_path = f"{output_file_path}/incremental/CNN/{incremental_date_str}"
        save_file_to_S3_staging(df_daily_final, output_daily_path)
        print("Incremental 파일 전처리 후 저장 완료:", output_daily_path)
    except Exception as e:
        print(f"Incremental 파일 읽기 에러(날짜 {incremental_date_str}): {e}")
        
if "finnhub_cleansing" in task_types:

    print(f"Incremental 실행 시작... finnhub_cleansing")
    # pass

    # 과거 데이터도 처리하기 위하여 datetime.today()가 아닌 다른 값 사용,
    # execution_date, data_interval_start, data_interval_end
    # data_interval_start은 실행날짜가 아니라 스케줄링에 따른 실행될 날짜!
    # 예) 스케줄링: 매일 새벽 3시
    # 3/15 수동 실행시 data_interval_start는 3/14 03:00 data_interval_end는 3/15 03:00
    
    
    # --------------------------------
    day_exp = data_interval_start.strftime("%Y%m%d") 

    # 🔹 S3 경로 설정
    RAW_DATA_2025 = f"s3://{S3_BUCKET}/raw_data/FINNHUB/2025/incremental/finnhub_m7_news_{day_exp}.parquet"
    OUTPUT_PATH = f"s3://{S3_BUCKET}/staging_data/news/incremental/FINNHUB/{day_exp}"
    
    try:
        # `_SUCCESS`, `_$folder$` 파일 생성을 방지
        spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        
        # 기존 S3 폴더의 모든 파일 삭제
        print(f"[DELETE] 기존 S3 데이터 삭제 중... 경로: {OUTPUT_PATH}")
        s3 = boto3.client("s3")
        bucket = S3_BUCKET
        prefix = OUTPUT_PATH.replace(f"s3://{bucket}/", "")

        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" in response:
            for obj in response["Contents"]:
                s3.delete_object(Bucket=bucket, Key=obj["Key"])

        print(f"[DELETE] 기존 데이터 삭제 완료!")
        
        # 🔹 S3에서 데이터 로드
        print(f"[LOAD] S3에서 데이터 로드 중... 경로: {RAW_DATA_2025}")
        df = spark.read.parquet(RAW_DATA_2025)

        # 🔹 중복 뉴스 제거
        print(f"[CLEAN] 중복 제거 중...")
        df = df.dropDuplicates(["symbol", "datetime", "headline"])
        
        # 🔹 datetime 컬럼을 UTC 시간을 제외한 날짜 값으로 변환
        df = df.withColumn("datetime", from_unixtime(col("datetime"), "yyyy-MM-dd"))

        # 🔹 데이터 전처리 (URL, HTML, 추가 텍스트 제거)
        print(f"[CLEAN] 데이터 전처리 중...")
        url_pattern = r"http[s]?://\S+"
        html_pattern = r"<.*?>"
        extra_text_pattern = r"\[.*?\]|\(.*?\)"
        whitespace_pattern = r"\s+"

        df_cleaned = df.withColumn("headline", regexp_replace(col("headline"), url_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), html_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), extra_text_pattern, "")) \
                    .withColumn("headline", regexp_replace(col("headline"), whitespace_pattern, " ")) \
                    .withColumn("summary", regexp_replace(col("summary"), url_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), html_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), extra_text_pattern, "")) \
                    .withColumn("summary", regexp_replace(col("summary"), whitespace_pattern, " "))

        # Parquet 파일을 1개만 유지하도록 설정
        df_single = df_cleaned.repartition(1)

        # S3에 저장 (Parquet 형식, 덮어쓰기 적용)
        print(f"[SAVE] 데이터 S3 저장 중... 경로: {OUTPUT_PATH}")
        df_single.write.mode("overwrite").parquet(OUTPUT_PATH)

        print(f"[SUCCESS] {day_exp} 처리 완료! 저장 경로: {OUTPUT_PATH}")

    except Exception as e:
        print(f"[ERROR] 처리 중 오류 발생! Task Type: finnhub_cleansing, 오류 내용: {e}")
        sys.exit(1)  # 치명적인 오류 발생 시 Glue Job 종료
    # --------------------------------
    
if "nytd_cleansing" in task_types:
    
    # NYT 데이터 클렌징 로직 추가
    # pass

    # --------------------------------
    """SparkSession을 활용하여 데이터 처리 실행"""
    s3_client = boto3.client("s3")
    
    df_transformed = load_and_process_data(s3_client, is_incremental)
    
    print(f"🔍 [INFO] TARGET_DATE: {TARGET_DATE}")
    
    save_data(df_transformed, s3_client, S3_BUCKET, is_incremental, TARGET_DATE, data_interval_start, data_interval_end)  

    # --------------------------------
    
    
if "merge_news_data" in task_types:
    try:
        s3_bucket = "de5-finalproj-team5"
        base_s3_prefix = "staging_data/news"
        output_s3_prefix = "analytic_data/news"

        sources = ["FINNHUB", "CNN", "NYTD"]
        df = get_file_from_staging(spark, s3_bucket, base_s3_prefix, is_incremental)
        if df is None:
            print("[WARN] S3에서 데이터를 읽어올 수 없습니다.")
            sys.exit(1)

        distinct_dates, saved_paths, file_count = save_to_analytics(df, s3_bucket, output_s3_prefix)

        for saved_date in distinct_dates:
            print(f"[INFO] 저장된 기사 날짜: {saved_date}")
        for saved_path in saved_paths:
            print(f"[INFO] 저장된 S3 경로: {saved_path}")
        print(f"[INFO] 총 저장된 파일(파티션) 갯수: {file_count}건")
        
        # staging 폴더에 데이터 삭제
        # remove_staging_data(spark, s3_bucket, base_s3_prefix, is_incremental)

    except Exception as e:
        print(f"[ERROR] save_to_analytic 처리 중 오류 발생: {e}")
        sys.exit(1)
    
job.commit()
