# 뉴스 및 FOMC 감성 분석 기반 M7 주가 예측

## 개요

CNN, NYT 뉴스 및 FOMC 발표 자료를 수집하고, FinBERT 기반 감성 분석과 LSTM 시계열 예측 모델을 결합하여  
**M7 기술주(GOOGL, AAPL, AMZN, MSFT, NVDA, META, TSLA)** 의 주가를 예측하는 프로젝트입니다.

- 뉴스 및 정책 발표의 **시장 영향력 정량화**  
- 비정형 데이터 기반 **시장 예측 가능성 탐색**

---

## 기술 스택

| 분야               | 도구 및 기술                                                                 |
|--------------------|------------------------------------------------------------------------------|
| 데이터 수집 및 저장 | Python, Selenium, BeautifulSoup, AWS S3, AWS Glue, AWS Athena               |
| 데이터 처리 및 분석 | Apache Spark, Pandas, FinBERT (Hugging Face Transformers), Pytorch, Colab   |
| 시각화 및 대시보드  | Superset (Preset)                                                            |
| 자동화 및 인프라     | Airflow, Docker, AWS EC2                                                    |
| 협업 및 배포        | GitHub (CI/CD 구현 예정)                                                    |

---

## 수행 과정
1. **일 1회 M7 관련 뉴스 자동 수집**

- 뉴스 출처: NYT, FINNHUB, CNN, FOMC
- 수집 방식: 언론사 공식 API 및 웹 크롤링 (Selenium, BeautifulSoup 활용)
- 자동화: Airflow DAG를 통해 매일 새벽 자동 실행

2. **데이터 전처리 및 감성 분석 수행**

- 수집된 뉴스 데이터를 Spark 및 AWS Glue로 전처리
- 경제 특화 감성 모델 FinBERT를 적용하여
- 각 뉴스 기사에 대해 긍정/중립/부정 감성 점수 추출
- 분석 결과는 S3 분석 영역에 저장 및 Athena 테이블 생성

3. **시계열 예측 모델(LSTM) 적용**

- M7 기업의 주가 데이터 및 감성 점수를 활용하여
- 다음날 종가를 예측하는 LSTM 모델 학습 및 추론
- 실험군: stock+sentiment / 비교군: stockOnly
- 모델 성능 비교 및 시각화: Superset 대시보드 제공

## 프로젝트 구조
![Image](https://github.com/user-attachments/assets/a25aca54-e0d1-4262-b008-c143373b5069)

## 수행 방법

### 1. EC2 내 Docker 환경 실행 (Airflow & Superset)

```bash
# 1. docker-compose.yaml이 있는 디렉토리에서 실행
docker-compose up -d

# 2. Airflow UI 접속: http://server:8080  (ID/PW: airflow / airflow)
# 3. Superset UI 접속: http://server:8088 (ID/PW: admin / admin123)
```

- Airflow DAGs: `./dags/` 디렉토리에 dag 배치하고 서버 폴더와 마운트 (`docker-compose.yaml`)
- Airflow Crawlering: `./dags/confing/` 디렉토리와 dags 디렉토리와 연동하여 실행 
- Superset 대시보드: Docker 환경에서 Superset 공식 이미지를 기반으로 컨테이너 빌드 및 실행 → UI 상에서 Athena 연동 및 대시보드 직접 구성

### 2. 의존 패키지 설치

```bash
pip install -r requirements.txt
```
- 각 Crawling 등 필요 패키지 공유 후 `requirements.txt`로 관리
### 3. 감성 분석 / 모델 학습

- **데이터 적재 및 전처리**: `airflow/airflow/dags` 폴더, NYPD, CNN, FINNHUB, FOMC 발표를 크롤링 하여 전처리(`main.py`)
- - **감성 분석 (Spark Glue 기반)**: `main.py`, `BERT_process2.py`, `BERT_spark.py` 참고  
- **주가 예측 (LSTM)**: LSTM_stock_BERT_v1.py, LSTM_stock_BERT_v2.py를 Google Colab에서 별도 실행(GPU 사용)

---

## 주요 코드 및 구조

| 파일명                  | 설명                                               |
|-------------------------|----------------------------------------------------|
| `docker-compose.yaml`   | Airflow & Superset 환경 구성                       |
| `requirements.txt`      | 필수 Python 패키지 목록                            |
| `glue_job_dag.py`       | 데이터 처리 DAG 정의                               |
| `main.py`               | Glue 기반 전처리 및 S3 저장                        |
| `BERT_process2.py`      | FinBERT 감성 분석 (Spark RDD 기반)                 |
| `BERT_spark.py`         | FinBERT 감성 분석 (Pandas UDF 기반)                |
| `LSTM_stock_BERT_v1.py` | LSTM 모델 실행                                     |
| `LSTM_stock_BERT_v2.py` | LSTM 모델 실행                                     |

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
</details> ```

- **분리된 영역**: Raw → Staging → Analytic  
- **연-월별 파티션 저장**: `analytic_data/news/year=YYYY/month=MM/`

---

## 대시보드 시각화 가능

- Superset 대시보드에서 **뉴스 트렌드, 거래량 상관관계, 모델 평가, 예측 결과** 시각화  
- Radar Chart 기반 모델 비교, 종목별 예측값 확인 가능

---

## 기대 효과

- AWS + Airflow 기반 **자동화 및 확장성 확보**  
- 뉴스/정책 발표 기반 **비정형 데이터 분석**  
- **코로나 이후 시장 변동성 대응** 모델  
- **비전문가 대상 대시보드 제공**으로 전달력 강화

---
