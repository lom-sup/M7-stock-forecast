# 뉴스 및 FOMC 감성 분석 기반 M7 주가 예측

## Notice!

> 본 Repo는 팀 프로젝트의 Repo 기반으로 개인적으로 정리한 버전입니다.
> 
> 민감한 정보(API 키 등)는 모두 대체되었으며, 예시 DAG 및 모델 코드 일부는 원본과 다를 수 있습니다.


<br/>


## 프로젝트 개요

> CNN, NYT 뉴스 및 FOMC 발표 자료를 수집하고, FinBERT 기반 **감성 분석**과 **LSTM 시계열 예측 모델**을 결합하여  
> **M7 기술주(GOOGL, AAPL, AMZN, MSFT, NVDA, META, TSLA)** 의 주가를 예측하는 것을 목표로 한다.

<br/>

- 뉴스 및 정책 발표의 **비정형 데이터 정량화**
- 비정형 데이터 기반 **시장 예측 가능성 탐색**
- 자동화된 파이프라인을 통한 **ELT, ETL 자동화**
- 주가-감성분석 포함모델(sentistock), 주가포함모델(stockonly), ARIMAX 등 다양한 모델 성능 비교 시각화 


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

## 주요 파일 명세 및 설명

<details>
<summary><strong>📂 폴더명/파일 </strong></summary>

### 📂 airflow/airflow
| 파일명                  | 설명                                                                  |
|-------------------------|-----------------------------------------------------------------------|
| `docker-compose.yaml`   | Airflow & Superset 환경을 Docker 기반으로 실행하기 위한 설정 파일        |
| `requirements.txt`      | Python 의존성 패키지 목록 (예: airflow, pandas 등)                      |
| `Dockerfile`            | Airflow 컨테이너 빌드 시 사용되는 실행 환경 정의                        |
| `dags/`                 | Airflow 스케줄러가 실행하는 DAG 파일 저장 폴더                          |

<br/>

### 📂 airflow/airflow/dags/config

| 파일명                         | 설명 |
|-------------------------------|------|
| `*crawl*.py`                  | 원본 데이터 출처에서 크롤링하여 S3에 Parquet로 저장 |
| `nytd_api_get.py`             | NYT API에서 M7 종목 관련 기사를 수집하여 S3에 Parquet로 저장 |

<br/>

### 📂 ML_script_local
| 파일명                  | 설명                                               |
|-------------------------|----------------------------------------------------|
| `BERT_process.py`      | FinBERT 감성 분석 (RDD 기반)                       |
| `ARIMA_stock_BERT_demo.py | ARIMAX 모델 테스트 코드                          |
| `LSTM_stock_BERT_v1.py` | PyTorch 기반 LSTM 예측 모델                         |
| `LSTM_stock_BERT_v2.py` | 하이퍼파라미터 조정 버전                           |

<br/>

### 📂 spark_script
| 파일명                      | 설명                                                                                  |
|-----------------------------|---------------------------------------------------------------------------------------|
| `main.py`                  |Glue Job 실행, NYT, CNN, FINNHUB 데이터 전처리 및 통합 로직을 포함한 메인 스크립트          |
| `ARIMA_stock_BERT_demo.py` |주가 예측을 위해 ARIMA 모델과 BERT 기반 감성 분석 결과를 결합한 데모 코드                  |
| `BERT_spark.py`            | 분산 처리 환경에서 Spark를 사용해 FinBERT 모델을 병렬로 실행하고 감성 점수 계산            |
| `LSTM_stock_BERT*.py`      | FinBERT 감성 점수 및 주가 데이터를 바탕으로 시계열 LSTM 모델을 훈련                       |
| `*cleansing*.py`           | 데이터 전처리(결측치 처리, 날짜 파싱 등)를 수행하여 S3에 저장하는 Spark 기반 Glue 스크립트  |
| `data_union.py`            | 여러 출처(CNN, NYT, FINNHUB)로부터 수집되어 전처리된 데이터를 하나로 병합                 |

<br/>

### 📂 team_project_readme
| 파일명                  | 설명                                    |
|-------------------------|-----------------------------------------|
| `README.md`             | 팀 프로젝트 당시 작성했던 README          |

</details>


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

