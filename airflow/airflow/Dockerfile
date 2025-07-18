FROM apache/airflow:2.9.1

###################################################################
# 1) root 사용자로 전환 후, 시스템 패키지 설치
###################################################################
USER root

# apt-get update & 필수 유틸 설치
RUN apt-get update && apt-get install -y \
    wget gnupg2 \
    build-essential \
    gcc \
    python3-dev

# Google Chrome 설치
RUN wget -q -O /tmp/google-chrome-stable_current_amd64.deb \
    https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
 && apt-get install -y /tmp/google-chrome-stable_current_amd64.deb \
 && rm /tmp/google-chrome-stable_current_amd64.deb

###################################################################
# 2) airflow 사용자로 전환 후, pip 패키지 설치
###################################################################
USER airflow

# Amazon Provider 설치 (pip)
RUN pip install --no-cache-dir apache-airflow-providers-amazon

# 작업 디렉터리 설정 (원하는 디렉터리로 변경 가능)
WORKDIR /app

# requirements.txt 복사
COPY requirements.txt .

# requirements.txt 설치
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

