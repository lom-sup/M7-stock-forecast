# 뉴스 및 FOMC 감성 분석 기반 M7 주가 예측

## Notice!

> 본 브랜치는 팀 프로젝트의 결과물 기반으로 개인적으로 정리한 버전입니다.
> 민감한 정보(API 키 등)는 모두 대체되었으며, 예시 DAG 및 모델 코드 일부는 원본과 다를 수 있습니다.


<br/>


## 프로젝트 개요

CNN, NYT 뉴스 및 FOMC 발표 자료를 수집하고, FinBERT 기반 감성 분석과 LSTM 시계열 예측 모델을 결합하여  
**M7 기술주(GOOGL, AAPL, AMZN, MSFT, NVDA, META, TSLA)** 의 주가를 예측하는 것을 목표로 한다. 

- 뉴스 및 정책 발표의 **비정형 데이터 정량화**
- 비정형 데이터 기반 **시장 예측 가능성 탐색**
- 자동화된 파이프라인을 통한 **ELT, ETL 자동화**
- 피쳐 엔지니어링, 

---

## 기술 스택

| 분야               | 도구 및 기술                                                                 |
|--------------------|------------------------------------------------------------------------------|
| 데이터 수집 및 저장 | Python, Selenium, BeautifulSoup, AWS S3, AWS Glue, AWS Athena               |
| 데이터 처리 및 분석 | Apache Spark, Pandas, FinBERT (Hugging Face Transformers), Pytorch, Colab   |
| 시각화 및 대시보드  | Superset (Preset)                                                            |
| 자동화 및 인프라     | Airflow, Docker, AWS EC2                                                    |
| 코드 관리           | GitHub (CI/CD 예정)                                                         |

<br/>


---

## 수행 프로세스 요약

1. **일 1회 M7 관련 뉴스 자동 수집**
    - 뉴스 출처: NYT, FINNHUB, CNN, FOMC
    - 수집 방식: 언론사 API 및 웹 크롤링
    - 자동화: Airflow DAG로 스케줄링

2. **데이터 전처리 및 감성 분석**
    - Spark와 Glue를 통해 정제
    - FinBERT 모델로 긍/부정/중립 감성 점수 추출
    - 결과는 S3에 저장 후 Glue-Athena로 테이블 구성

3. **LSTM 기반 주가 예측 모델**
    - 입력: 주가 데이터 + 뉴스 감성 점수
    - 비교: LSTM, ARIMAX, stockonly(주가 데이터만 사용), stocksenti(감성 분석 데이터를 포함한 모델)
    - Colab에서 PyTorch 기반 LSTM 학습

4. **Superset을 통한 대시보드 시각화**
    - 모델  채택 모델 예측 결과 시각화
    - 뉴스 트렌드 분석

<br/>

---

## 주요 코드 및 구조

| 파일명                  | 설명                                               |
|-------------------------|----------------------------------------------------|
| `docker-compose.yaml`   | Airflow & Superset 환경 구성                       |
| `requirements.txt`      | Python 의존성 패키지 목록                          |
| `glue_job_dag.py`       | Glue 기반 전처리 DAG 정의                          |
| `main.py`               | Glue 전처리 전체 파이프라인                         |
| `BERT_process2.py`      | FinBERT 감성 분석 (RDD 기반)                       |
| `BERT_spark.py`         | FinBERT 감성 분석 (Pandas UDF 기반)                |
| `LSTM_stock_BERT_v1.py` | PyTorch 기반 LSTM 예측 모델                         |
| `LSTM_stock_BERT_v2.py` | 하이퍼파라미터 조정 버전                           |

---

## 데이터 구조 (S3)

<details>
<summary><strong>📂 de5-finalproj-team5</strong></summary>

```bash
de5-finalproj-team5/
├── raw_data/
│   ├── CNN/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── ...
│   │   └── 2025/
│   ├── FINNHUB/
│   ├── FOMC/
│   └── NYTD/
│
├── staging_data/news/
│   ├── full/
│   └── incremental/
│       ├── CNN/
│       │   └── 20250317/
│       ├── FINNHUB/
│       │   └── 20250317/
│       └── ...
│
├── analytic_data/news/
│   ├── year=2025/
│   │   ├── month=3/
│   │   └── ...
│   └── year=2020/
│
└── spark_script/
    └── main.py
```
</details>

- **파티션**: Raw → Staging → Analytic  
- **연-월별 파티션 저장**: `analytic_data/news/year=YYYY/month=MM/`
- **계층 구조 및 파티셔닝**으로 효율적 관리
- Glue + Athena에서 테이블 생성 후 분석 및 시각화에 활용
- Bert로 정량화된 데이터(analytic_data)는 LSTM Model Input Data로 사용

<br/>

---

## 시각화 예시

- 뉴스 감성 점수 시계열 변화
- M7 종목별 감성 변화와 예측 결과 비교
- 모델별 MSE, 예측값, 실제값 비교 (Radar/Line Chart)
- 종목별 특이점 탐색 및 분석

---

## 기대 효과

- AWS + Airflow 기반 자동화된 ETL 파이프라인 구현 경험
- FinBERT 및 LSTM 결합 모델을 통한 감성 기반 주가 예측 시도
- Superset 대시보드를 통한 실무형 보고서 제작 경험

