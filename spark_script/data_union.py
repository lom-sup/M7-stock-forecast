from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, to_date

# --------------------------------------------------------------------------------
# 전역 변수 설정
# --------------------------------------------------------------------------------
s3_bucket = "de5-finalproj-team5"
base_s3_prefix = "staging_data/news"
output_s3_prefix = "analytic_data/news"

sources = ["FINHUB", "CNN", "NYTD", "FOMC"]

# --------------------------------------------------------------------------------
# S3 staging_data 폴더에서 데이터 read
# --------------------------------------------------------------------------------
def get_file_from_S3(spark: SparkSession, s3_bucket: str, base_s3_prefix: str, is_incremental: bool):
    """
    S3에 저장된 full 또는 incremental 데이터를 읽어와 하나의 DataFrame으로 반환
    """
    if not is_incremental:
        df = spark.read.parquet(f"s3a://{s3_bucket}/{base_s3_prefix}/full")
    else:
        # 각 소스별 incremental 폴더의 데이터를 union
        dfs = []
        for source in sources:
            path = f"s3a://{s3_bucket}/{base_s3_prefix}/incremental/{source}"
            dfs.append(spark.read.parquet(path))
        if dfs:
            df = reduce(lambda a, b: a.union(b), dfs)
        else:
            df = None
    if df:
        return df.coalesce(1)
    else:
        return None
    
# --------------------------------------------------------------------------------
# S3 analytics_data 폴더에 데이터 저장
# --------------------------------------------------------------------------------
def save_to_S3(df: DataFrame, s3_bucket: str, output_s3_prefix: str):
    output_path = f"s3a://{s3_bucket}/{output_s3_prefix}"
    file_count = 0
    saved_paths = []

    # datetime를 date 타입(yyyy-mm-dd)으로 변환해 record_date 컬럼 추가
    df_with_rec_date = df.withColumn("record_date", to_date(col("datetime"), "yyyy-MM-dd'T'HH:mm:ss.SSS"))

    # 고유한 날짜 리스트 수집 
    distinct_dates = [row['record_date'] for row in df_with_rec_date.select("record_date").distinct().collect() if row['record_date'] is not None]

    for rec_date in distinct_dates:
        yyyy = rec_date.strftime('%Y')
        mm = rec_date.strftime('%m')

        partition_df = df_with_rec_date.filter(col("record_date") == lit(rec_date))
        partition_output = f"{output_path}/year={yyyy}/month={mm}"

        partition_df.select("symbol", "source", "datetime", "headline", "summary") \
            .write.mode("overwrite").parquet(partition_output)

        saved_paths.append(partition_output)
        file_count += 1

    print(f"[INFO] 데이터 저장 완료: 총 {file_count}개의 파일이 파티션에 저장되었습니다.")
    return distinct_dates, saved_paths, file_count

# --------------------------------------------------------------------------------
# staging 폴더의 데이터 삭제
# --------------------------------------------------------------------------------
def remove_staging_data(spark, s3_bucket: str, base_s3_prefix: str, is_incremental: bool):
    # 삭제 대상 prefix
    if not is_incremental:
        prefix = f"{base_s3_prefix}/full"
    else:
        prefix = f"{base_s3_prefix}/incremental"

    delete_path = f"s3a://{s3_bucket}/{prefix}"

    print(f"[INFO] Staging 폴더 삭제 시도 경로: {delete_path}")

    # SparkContext 및 Hadoop Configuration 가져오기
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()

    # 실제 삭제 로직 (Py4J를 통해 Hadoop FileSystem API 호출)
    try:
        # PySpark 내부 JVM 객체를 얻어온 뒤, Hadoop FileSystem API 사용
        jvm = sc._jvm
        hadoop_fs = jvm.org.apache.hadoop.fs
        path_obj = hadoop_fs.Path(delete_path)

        fs = path_obj.getFileSystem(hadoop_conf)
        # recursive=True 로 설정해야 하위 파일/폴더를 모두 삭제
        fs.delete(path_obj, True)

        print(f"[INFO] Staging 데이터 삭제 완료: {delete_path}")
    except Exception as e:
        print(f"[WARN] Staging 데이터 삭제 중 에러 발생: {e}")

# --------------------------------------------------------------------------------
# Main 함수. main script에서 호출하기 위함으로 def main -> def run 으로 변경
# --------------------------------------------------------------------------------
def run(spark, is_incremental):
    df = get_file_from_S3(spark, s3_bucket, base_s3_prefix, is_incremental)
    if df is None:
        print("[WARN] S3에서 데이터를 읽어올 수 없습니다.")
        return

    distinct_dates, saved_paths, file_count = save_to_S3(df, s3_bucket, output_s3_prefix)

    for saved_date in distinct_dates:
        print(f"[INFO] 저장된 기사 날짜: {saved_date}")
    for saved_path in saved_paths:
        print(f"[INFO] 저장된 S3 경로: {saved_path}")
    print(f"[INFO] 총 저장된 파일(파티션) 갯수: {file_count}건")

    # staging 폴더에 데이터 삭제
    remove_staging_data(spark, s3_bucket, base_s3_prefix, is_incremental)

if __name__ == '__main__':
    spark = SparkSession.builder.appName("CNN_article_preprocessing") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    is_incremental = False
    run(spark, is_incremental)