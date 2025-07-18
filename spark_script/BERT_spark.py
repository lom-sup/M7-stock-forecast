import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, lit, pandas_udf
from pyspark.sql.types import MapType, StringType, FloatType
from pyspark.sql.functions import PandasUDFType
import os


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 접속 정보 (보안정보는 실제 운영환경에서는 안전한 방식으로 관리할 것)
S3_BUCKET = "de5-finalproj-team5"
ACCESS_KEY = ""
SECRET_KEY = ""

# 입력 및 출력 경로 설정
speeches_input_pattern = f"s3a://{S3_BUCKET}/raw_data/FOMC/incremental/fed_speeches_contents_*.json"
output_path = f"s3a://{S3_BUCKET}/staging_data/news/full/fed_speeches_with_sentiment"

# finBERT 모델 이름 설정
model_name = "yiyanghkust/finbert-tone"

# SparkSession 생성 (필요한 의존성을 추가하여 클래스 누락 문제를 해결)


spark = SparkSession.builder \
    .appName("finbert-sentiment") \
    .config("spark.pyspark.python", "C:/Python313/python.exe") \
    .config("spark.pyspark.driver.python", "C:/Python313/python.exe") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2,org.apache.hadoop:hadoop-common:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.12.524") \
    .config("spark.jars.excludes", "com.amazonaws:aws-java-sdk,com.amazonaws:aws-java-sdk-core") \
    .getOrCreate()



# S3 접속 설정
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY)
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY)
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")
# S3의 JSON 파일 읽기 (multiLine 옵션 사용)
df = spark.read.option("multiLine", "true").json(speeches_input_pattern)
df.printSchema()

# pdf_texts 컬럼이 존재하면 결합, 없으면 빈 문자열로 처리
if "pdf_texts" in df.columns:
    df = df.withColumn("pdf_text_combined", concat_ws(" ", col("pdf_texts")))
else:
    df = df.withColumn("pdf_text_combined", lit(""))

# content와 pdf_text_combined를 결합하여 combined_text 컬럼 생성
df = df.withColumn("combined_text", concat_ws(" ", col("content"), col("pdf_text_combined")))

# 전역 변수 캐싱을 위한 모델/토크나이저 변수 초기화
global_tokenizer = None
global_model = None

def load_model():
    """
    각 워커(또는 파티션)에서 처음 호출 시에 모델과 토크나이저를 로드합니다.
    이후에는 캐시된 객체를 재사용합니다.
    """
    global global_tokenizer, global_model
    if global_tokenizer is None or global_model is None:
        logger.info("Loading model and tokenizer...")
        global_tokenizer = AutoTokenizer.from_pretrained(model_name)
        global_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return global_tokenizer, global_model

@pandas_udf(MapType(StringType(), FloatType()), PandasUDFType.SCALAR)
def sentiment_udf(texts: pd.Series) -> pd.Series:
    """
    Pandas UDF를 이용하여 각 배치 단위로 감성 분석을 수행합니다.
    각 텍스트에 대해 모델을 적용하고, 모델의 id2label을 사용하여 동적으로 라벨을 매핑합니다.
    """
    results = []
    tokenizer, model = load_model()
    for text in texts:
        try:
            if text is None or str(text).strip() == "":
                results.append(None)
            else:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                scores = outputs.logits.detach().numpy()[0]
                # 모델 config에서 동적으로 라벨을 가져옴
                labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
                sentiment = {label: float(score) for label, score in zip(labels, scores)}
                results.append(sentiment)
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            results.append(None)
    return pd.Series(results)

# UDF를 적용하여 감성 분석 결과 컬럼 추가
df = df.withColumn("finbert_sentiment", sentiment_udf(col("combined_text")))

# 결과를 S3에 JSON 형식으로 저장
df.write.mode("overwrite").json(output_path)

spark.stop()
