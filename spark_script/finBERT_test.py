from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, avg, count, lag, to_date, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from transformers import pipeline, AutoTokenizer

# 전역 변수: 각 executor에서 모델과 토크나이저가 한 번만 로드되도록 함
sentiment_pipeline_global = None
tokenizer_global = None

def load_sentiment_pipeline():
    """
    각 executor에서 FinBERT 모델과 토크나이저를 최초로 로드하는 함수.
    """
    global sentiment_pipeline_global, tokenizer_global
    if sentiment_pipeline_global is None:
        sentiment_pipeline_global = pipeline(
            "sentiment-analysis",
            model="yiyanghkust/finbert-tone",
            tokenizer="yiyanghkust/finbert-tone",
            return_all_scores=True
        )
    if tokenizer_global is None:
        tokenizer_global = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return sentiment_pipeline_global, tokenizer_global

def get_sentiment_info(text):
    """
    슬라이딩 윈도우 기법을 적용해 긴 뉴스 텍스트를 여러 조각으로 분할하여 FinBERT 감성 분석을 수행합니다.
    
    [주요 과정]
    1. 텍스트를 토크나이즈하여 최대 길이(512 토큰)를 초과하면 stride(256 토큰)로 분할합니다.
    2. 각 조각별로 감성 분석을 수행하여 긍정, 중립, 부정 확률을 누적 후 평균을 냅니다.
    3. 부정 확률에 NEGATIVE_WEIGHT를 곱해 예측 감성 산출 시 반영합니다.
    4. 최종적으로 예측 감성(pred_label)과 raw 확률(negative_prob, neutral_prob, positive_prob)을 반환합니다.
    """
    if text is None:
        return (None, None, None, None)
    
    try:
        sentiment_pipeline, tokenizer = load_sentiment_pipeline()
        max_length = 512
        stride = 256  # 슬라이딩 윈도우 이동 간격 (중복 포함)
        
        # 텍스트 토큰화
        tokens = tokenizer.tokenize(text)
        chunks = []
        
        # 텍스트 길이가 512 토큰을 초과하면 슬라이딩 윈도우 적용
        if len(tokens) > max_length:
            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i:i+max_length]
                chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
        else:
            chunks.append(text)
        
        # 각 조각별 감성 점수 누적
        aggregated_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        chunk_count = 0
        
        for chunk in chunks:
            results = sentiment_pipeline(chunk)
            if results and isinstance(results, list) and len(results) > 0:
                # results[0]는 [{'label': 'negative', 'score': ...}, ...] 형태입니다.
                chunk_scores = {d['label']: d['score'] for d in results[0]}
                aggregated_scores["negative"] += chunk_scores.get("negative", 0)
                aggregated_scores["neutral"] += chunk_scores.get("neutral", 0)
                aggregated_scores["positive"] += chunk_scores.get("positive", 0)
                chunk_count += 1
        
        if chunk_count == 0:
            return (None, None, None, None)
        
        # 평균 확률 계산
        avg_negative = aggregated_scores["negative"] / chunk_count
        avg_neutral = aggregated_scores["neutral"] / chunk_count
        avg_positive = aggregated_scores["positive"] / chunk_count
        
        # 경제 뉴스의 경우 부정에 더 가중을 주기 위해 negative 확률에 weight 적용
        NEGATIVE_WEIGHT = 1.5  # 가중치 값은 실험을 통해 조정 가능
        weighted_negative = avg_negative * NEGATIVE_WEIGHT
        
        # 가중치를 적용한 점수로 예측 감성 산출
        weighted_scores = {"negative": weighted_negative, "neutral": avg_neutral, "positive": avg_positive}
        pred_label = max(weighted_scores, key=weighted_scores.get)
        
        return (pred_label, float(avg_negative), float(avg_neutral), float(avg_positive))
    except Exception as e:
        return (None, None, None, None)

# UDF의 반환 스키마 정의: 예측 감성 및 raw 확률을 포함
sentiment_schema = StructType([
    StructField("sentiment_label", StringType(), True),
    StructField("negative_prob", DoubleType(), True),
    StructField("neutral_prob", DoubleType(), True),
    StructField("positive_prob", DoubleType(), True)
])
sentiment_udf = udf(get_sentiment_info, sentiment_schema)

def aggregate_news_features(df):
    """
    뉴스 데이터를 종목(symbol) 및 날짜별로 집계하고 파생 변수를 생성합니다.
    
    파생 변수:
    - avg_positive, avg_neutral, avg_negative: 각 날짜별 감성 확률의 평균
    - news_count: 해당 날짜의 뉴스 건수
    - lag 변수: 전일의 감성 점수와 뉴스 건수를 추가 (시계열 분석에 활용)
    """
    # datetime 컬럼을 date 형식으로 변환
    df = df.withColumn("date", to_date(col("datetime")))
    
    # 종목 및 날짜별로 집계
    agg_df = df.groupBy("symbol", "date").agg(
        avg("positive_prob").alias("avg_positive"),
        avg("neutral_prob").alias("avg_neutral"),
        avg("negative_prob").alias("avg_negative"),
        count("*").alias("news_count")
    )
    
    # 각 종목별 날짜 순으로 윈도우 함수 적용하여 lag 변수 추가
    windowSpec = Window.partitionBy("symbol").orderBy("date")
    agg_df = agg_df.withColumn("lag_avg_positive", lag("avg_positive", 1).over(windowSpec))
    agg_df = agg_df.withColumn("lag_avg_neutral", lag("avg_neutral", 1).over(windowSpec))
    agg_df = agg_df.withColumn("lag_avg_negative", lag("avg_negative", 1).over(windowSpec))
    agg_df = agg_df.withColumn("lag_news_count", lag("news_count", 1).over(windowSpec))
    
    return agg_df

def main():
    spark = SparkSession.builder.getOrCreate()
    
    # S3에 저장된 전처리된 뉴스 데이터를 읽어옵니다.
    input_path = "s3://your-bucket/path/to/news/parquet/"
    df = spark.read.parquet(input_path)
    
    # FinBERT 감성 분석 UDF를 적용해 summary 컬럼으로 감성 정보를 도출합니다.
    df = df.withColumn("sentiment_info", sentiment_udf(df.summary))
    df = df.select("*", "sentiment_info.*").drop("sentiment_info")
    
    # 감성 점수가 포함된 데이터 S3 저장
    sentiment_output_path = "s3://your-bucket/path/to/news_with_sentiment/"
    df.write.mode("overwrite").parquet(sentiment_output_path)
    
    # 뉴스 데이터를 종목 및 날짜별로 집계하여 파생 변수를 생성합니다.
    features_df = aggregate_news_features(df)
    
    # 파생 변수 데이터 S3 저장 (이후 주가 데이터와 병합하여 분석에 활용)
    features_output_path = "s3://your-bucket/path/to/news_features/"
    features_df.write.mode("overwrite").parquet(features_output_path)

if __name__ == "__main__":
    main()
