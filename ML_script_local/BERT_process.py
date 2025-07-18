import logging
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# GPU 설정: GPU가 있으면 사용, 없으면 CPU 사용
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "yiyanghkust/finbert-tone"
logger.info("finBERT 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
logger.info("모델 로드 완료.")

def get_sentiment_distribution_sliding(text, max_length=512, stride=256):
    """
    긴 텍스트에 대해 슬라이딩 윈도우 기법을 적용하여, 각 window마다 모델의 raw logits를 계산하고
    소프트맥스를 적용하여 얻은 확률 분포의 평균을 반환합니다.
    반환 결과는 {'positive': x, 'neutral': y, 'negative': z} 형태입니다.
    """
    if text is None or not isinstance(text, str) or len(text.strip()) == 0:
        return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    
    # 전체 텍스트 토큰화 (truncation 없이)
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens['input_ids'][0]
    total_length = input_ids.shape[0]
    
    # BERT 계열 모델의 특별 토큰 수 (일반적으로 [CLS], [SEP] 2개)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    adjusted_max = max_length - special_tokens  # 실제 window에 사용할 토큰 수
    
    window_distributions = []
    for i in range(0, total_length, stride):
        window_ids = input_ids[i : i + adjusted_max]
        if window_ids.shape[0] == 0:
            break
        # window_ids를 디코딩하여 텍스트 조각 생성
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        inputs = tokenizer(window_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        label_probs = {model.config.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        for label in ['positive', 'neutral', 'negative']:
            if label not in label_probs:
                label_probs[label] = 0.0
        window_distributions.append(label_probs)
    
    if not window_distributions:
        return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    
    avg_distribution = {}
    for label in ['positive', 'neutral', 'negative']:
        avg_distribution[label] = float(np.mean([wd[label] for wd in window_distributions]))
    return avg_distribution

# CSV 파일 읽기 (symbol, source, datetime, headline, summary 컬럼 포함)
csv_file_path = "current_output.csv"  # CSV 파일 경로를 지정하세요.
logger.info(f"CSV 파일 {csv_file_path} 읽기 시작...")
df = pd.read_csv(csv_file_path)
logger.info(f"CSV 파일 읽기 완료. 총 {len(df)} 행.")

def analyze_row(row):
    idx = row.name
    if idx % 10 == 0:
        logger.info(f"감성 분석 처리 진행: 행 {idx+1} / {len(df)}")
    
    summary_text = row.get("summary", "")
    headline_text = row.get("headline", "")
    
    if (not isinstance(summary_text, str) or len(summary_text.strip()) == 0) and \
       (not isinstance(headline_text, str) or len(headline_text.strip()) == 0):
        logger.info(f"행 {idx+1}: summary와 headline 모두 비어있어 감성 분석을 건너뜁니다.")
        return {'finbert_positive': 0.0, 'finbert_neutral': 0.0, 'finbert_negative': 0.0}
    
    if isinstance(summary_text, str) and len(summary_text.strip()) > 0:
        text_to_analyze = summary_text
        logger.debug(f"행 {idx+1}: summary 컬럼 사용")
    else:
        text_to_analyze = headline_text
        logger.info(f"행 {idx+1}: summary가 비어있어 headline 컬럼 사용")
    
    sentiment = get_sentiment_distribution_sliding(text_to_analyze)
    return {
        'finbert_positive': sentiment.get('positive', 0.0),
        'finbert_neutral': sentiment.get('neutral', 0.0),
        'finbert_negative': sentiment.get('negative', 0.0)
    }

logger.info("CSV 데이터에 대한 감성 분석 시작 (summary가 없으면 headline으로 대체)...")
sentiment_results = df.apply(analyze_row, axis=1)
sentiments_df = pd.DataFrame(sentiment_results.tolist())
df = pd.concat([df, sentiments_df], axis=1)
logger.info("감성 분석 완료.")

output_csv = "output_with_sentiment_combined2.csv"
df.to_csv(output_csv, index=False)
logger.info(f"결과를 {output_csv} 파일로 저장 완료.")
