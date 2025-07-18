import json
import logging
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

def process_partition(iterator):
    # 각 파티션에서 필요한 모듈들을 로컬 임포트 (분산환경에서 매 파티션마다 모델을 로드)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    import json

    # 모델 및 토크나이저 로드 (GPU 사용 시 자동 감지)
    model_name = "yiyanghkust/finbert-tone"
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to("cuda" if device != -1 else "cpu")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )

    def get_sentiment_pipeline(text, max_length=512, stride=256):
        """
        긴 텍스트에 대해 슬라이딩 윈도우 기법을 적용하여 배치 처리 기반 감성 분석 수행.
        긍정, 중립, 부정 확률을 모두 반환합니다.
        """
        try:
            if text is None or not isinstance(text, str) or len(text.strip()) == 0:
                logging.error("입력 텍스트가 유효하지 않습니다.")
                return None

            special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
            adjusted_max = max_length - special_tokens  # 실제 사용할 토큰 수
            tokens = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = tokens['input_ids'][0]
            total_length = input_ids.shape[0]
            windows = []
            for i in range(0, total_length, stride):
                window_ids = input_ids[i: i + adjusted_max]
                if window_ids.shape[0] == 0:
                    break
                window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
                windows.append(window_text)
            if windows:
                results = sentiment_pipeline(windows, batch_size=8, truncation=True, max_length=max_length)
                # 긍정, 중립, 부정 확률을 저장할 리스트 초기화
                sentiments = {"positive": [], "neutral": [], "negative": []}
                for res in results:
                    label = res.get('label', '').lower()
                    if label in sentiments:
                        sentiments[label].append(res.get('score', 0))
                    else:
                        logging.warning(f"예상치 못한 라벨 반환됨: {label}")
                avg_sentiment = {label: float(np.mean(scores)) if scores else 0.0 
                                 for label, scores in sentiments.items()}
                return json.dumps(avg_sentiment)
            else:
                logging.error("분할된 텍스트 윈도우가 없습니다.")
                return None
        except Exception as e:
            logging.error(f"감성 분석 중 예외 발생: {e}")
            return None

    # 각 파티션 내의 각 행(row)에 대해 처리 수행
    for row in iterator:
        try:
            summary_text = row["summary"] if "summary" in row and row["summary"] else ""
            headline_text = row["headline"] if "headline" in row and row["headline"] else ""
            # summary가 존재하면 summary 사용, 없으면 headline 사용
            text_to_analyze = summary_text if summary_text.strip() != "" else headline_text
            sentiment = get_sentiment_pipeline(text_to_analyze)
            new_row = row.asDict()
            new_row['finbert_sentiment'] = sentiment
            yield new_row
        except Exception as e:
            logging.error(f"행 처리 중 오류 발생: {e}")
            continue

def main():
    # Glue 환경에서도 SparkSession을 사용합니다.
    spark = SparkSession.builder.appName("FinBERT Sentiment Analysis").getOrCreate()

    # S3에서 CSV 파일 읽기 (경로를 실제 S3 버킷 경로로 수정)
    input_path = "s3://de5-finalproj-team5/analytic_data/test/current_output.csv"
    try:
        df = spark.read.csv(input_path, header=True, inferSchema=True)
    except Exception as e:
        logging.error(f"S3에서 CSV 파일 읽기 실패: {e}")
        spark.stop()
        return

    # Spark RDD로 변환 후 각 파티션별 감성 분석 처리
    try:
        result_rdd = df.rdd.mapPartitions(process_partition)
    except Exception as e:
        logging.error(f"RDD 변환 또는 파티션 처리 중 오류 발생: {e}")
        spark.stop()
        return

    try:
        new_schema = df.schema.add("finbert_sentiment", StringType())
        result_df = spark.createDataFrame(result_rdd, schema=new_schema)
    except Exception as e:
        logging.error(f"새 DataFrame 생성 중 오류 발생: {e}")
        spark.stop()
        return

    # 결과를 S3에 CSV 파일로 저장 (출력 경로는 실제 S3 버킷 경로로 수정)
    output_path = "s3://de5-finalproj-team5/analytic_data/test/Bert_combined_data.csv"
    try:
        result_df.write.csv(output_path, header=True, mode="overwrite")
        logging.info(f"결과가 성공적으로 {output_path} 에 저장되었습니다.")
    except Exception as e:
        logging.error(f"S3에 결과 저장 실패: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Glue 기반 FinBERT 감성 분석 작업을 시작합니다...")
    main()
    logger.info("감성 분석 작업이 완료되었습니다.")
