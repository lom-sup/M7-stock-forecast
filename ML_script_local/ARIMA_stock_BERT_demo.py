import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# ---------------------------
# 1. 데이터 로딩 및 전처리 함수
# ---------------------------
def load_stock_data(file_path, symbol):
    """
    주가 JSON 파일을 로딩하여 컬럼명을 정리.
    입력 JSON의 컬럼 이름이 ("Close", "AAPL") 등 형태라면 Close, Open, High, Low, Volume으로 변경.
    """
    df = pd.read_json(file_path)
    new_cols = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith("('"):
            # 예: "('Close', 'AAPL')" -> "Close"
            new_cols[col] = col.split(",")[0].strip("('")
        else:
            new_cols[col] = col
    df = df.rename(columns=new_cols)
    df["Date"] = pd.to_datetime(df["('Date', '')"] if "('Date', '')" in df.columns else df["Date"])
    # 만약 Date 컬럼이 남아있으면 사용
    if "('Date', '')" in df.columns:
        df.drop(columns=["('Date', '')"], inplace=True)
    df["symbol"] = symbol
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_sentiment_data(file_path):
    """
    감성 CSV 파일 로딩 및 일별 감성 피처 생성.
    """
    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        df["Date_only"] = pd.to_datetime(df["Date"]).dt.date
    elif "datetime" in df.columns:
        df["Date_only"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        raise ValueError("Sentiment CSV must contain a date or datetime column.")
    if "symbol" not in df.columns:
        raise ValueError("Sentiment CSV must contain a 'symbol' column.")
    df["Raw_Sentiment"] = df["finbert_positive"] - df["finbert_negative"]
    df["Sentiment"] = df.groupby("symbol")["Raw_Sentiment"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["Sentiment_Diff"] = df.groupby("symbol")["Sentiment"].transform(lambda x: x.diff().fillna(0))
    df["Sentiment_Volatility"] = df.groupby("symbol")["Sentiment"].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    return df[["symbol", "Date_only", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]]

def merge_stock_and_sentiment(stock_df, sentiment_df, symbol):
    """
    주가 데이터와 감성 데이터를 symbol 기준으로 병합.
    주가 데이터는 이미 Date 컬럼을 포함.
    """
    stock_df["Date_only"] = stock_df["Date"].dt.date
    symbol_sent = sentiment_df[sentiment_df["symbol"] == symbol]
    daily_sent = symbol_sent.groupby("Date_only", as_index=False).agg({
        "Sentiment": "mean",
        "Sentiment_Diff": "mean",
        "Sentiment_Volatility": "mean"
    })
    merged = pd.merge(stock_df, daily_sent, on="Date_only", how="left")
    merged[["Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]] = merged[["Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]].fillna(0)
    # 반환 시 Date, Close, 그리고 감성 피처만 사용 (OHLCV는 주가 파일에서 사용)
    return merged[["Date", "Close", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]]

def _compute_metrics(y_true, y_pred, prev_close):
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    actual_dir = np.sign(y_true - prev_close)
    pred_dir   = np.sign(y_pred - prev_close)
    dir_acc = np.mean(actual_dir == pred_dir)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DirAcc": dir_acc}

# ---------------------------
# 2. 평가 함수: ARIMA (종가만 사용)
# ---------------------------
def evaluate_arima(stock_df, order=(5,1,0)):
    df = stock_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.asfreq("B")
    df["Close"] = df["Close"].ffill()
    df.dropna(inplace=True)
    
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    
    model = ARIMA(train["Close"], order=order).fit()
    forecast = model.forecast(steps=len(test))
    
    return _compute_metrics(test["Close"].values, forecast.values, df["Close"].shift(1).iloc[split:].values)

# ---------------------------
# 3. 평가 함수: ARIMAX-Stock (OHLCV 사용)
# ---------------------------
def evaluate_arimax_stock(stock_df, order=(5,1,0)):
    df = stock_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.asfreq("B")
    df["Close"] = df["Close"].ffill()
    # OHLCV: forward fill
    df[["Open","High","Low","Volume"]] = df[["Open","High","Low","Volume"]].ffill()
    df.dropna(inplace=True)
    
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    
    model = ARIMA(train["Close"], exog=train[["Open","High","Low","Volume"]], order=order).fit()
    forecast = model.forecast(steps=len(test), exog=test[["Open","High","Low","Volume"]])
    
    return _compute_metrics(test["Close"].values, forecast.values, df["Close"].shift(1).iloc[split:].values)

# ---------------------------
# 4. 평가 함수: ARIMAX-Combined (OHLCV + 감성 피처 사용)
# ---------------------------
def evaluate_arimax_combined(stock_df, sentiment_df, symbol, order=(5,1,0)):
    # 감성 데이터 병합
    merged = merge_stock_and_sentiment(stock_df, sentiment_df, symbol)
    merged = merged.set_index("Date").asfreq("B")
    # 주가 파일의 OHLCV 컬럼을 stock_df에서 가져와 병합 (날짜 기준 정렬)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df.set_index("Date", inplace=True)
    stock_df = stock_df.asfreq("B")
    ohlcv = stock_df[["Open","High","Low","Volume"]].ffill()
    merged = merged.join(ohlcv, how="left")
    
    merged["Close"] = merged["Close"].ffill()
    merged.fillna(0, inplace=True)
    
    split = int(len(merged) * 0.8)
    train, test = merged.iloc[:split], merged.iloc[split:]
    
    exogs = ["Open","High","Low","Volume","Sentiment","Sentiment_Diff","Sentiment_Volatility"]
    model = ARIMA(train["Close"], exog=train[exogs], order=order).fit()
    forecast = model.forecast(steps=len(test), exog=test[exogs])
    
    return _compute_metrics(test["Close"].values, forecast.values, merged["Close"].shift(1).iloc[split:].values)

# ---------------------------
# 5. 메인 실행: 종목별 평가 지표 비교 및 결과 저장
# ---------------------------
nasdaq_M7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
sentiment_file = "output_with_sentiment_combined2.csv"
sentiment_df = load_sentiment_data(sentiment_file) if os.path.exists(sentiment_file) else None

results_list = []
order = (5,1,0)  # 필요 시 튜닝

for sym in nasdaq_M7:
    stock_path = os.path.join("stock", f"{sym}_stock_data.json")
    if not os.path.exists(stock_path):
        print(f"Stock data file not found for {sym}!")
        continue
    stock_df = load_stock_data(stock_path, sym)
    
    try:
        metrics_arima = evaluate_arima(stock_df, order=order)
    except Exception as e:
        print(f"Error processing ARIMA for {sym}: {e}")
        metrics_arima = {}
    
    try:
        metrics_arimax_stock = evaluate_arimax_stock(stock_df, order=order)
    except Exception as e:
        print(f"Error processing ARIMAX-Stock for {sym}: {e}")
        metrics_arimax_stock = {}
    
    try:
        metrics_arimax_combined = evaluate_arimax_combined(stock_df, sentiment_df, sym, order=order)
    except Exception as e:
        print(f"Error processing ARIMAX-Combined for {sym}: {e}")
        metrics_arimax_combined = {}
    
    row = {"Symbol": sym}
    for k, v in metrics_arima.items():
        row[f"ARIMA_{k}"] = round(v, 6)
    for k, v in metrics_arimax_stock.items():
        row[f"StockExog_{k}"] = round(v, 6)
    for k, v in metrics_arimax_combined.items():
        row[f"Combined_{k}"] = round(v, 6)
    
    results_list.append(row)

results_df = pd.DataFrame(results_list)
print("===== 모델 평가 지표 비교 =====")
print(results_df)
results_df.to_csv("comparison_ohlcv_sentiment.csv", index=False)
