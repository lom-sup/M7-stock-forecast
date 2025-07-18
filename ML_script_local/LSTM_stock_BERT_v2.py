import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Loading and Preprocessing Functions
def load_stock_data(file_path, symbol):
    """Load stock JSON and preprocess columns."""
    df = pd.read_json(file_path)
    new_cols = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith("('"):
            new_cols[col] = col.split(",")[0].strip("('")
        else:
            new_cols[col] = col
    df = df.rename(columns=new_cols)
    df["Date"] = pd.to_datetime(df["Date"])
    df["symbol"] = symbol
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_sentiment_data(file_path):
    """Load sentiment CSV and create daily sentiment features, including additional ones."""
    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        df["Date_only"] = pd.to_datetime(df["Date"]).dt.date
    elif "datetime" in df.columns:
        df["Date_only"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        raise ValueError("Sentiment CSV must contain a date or datetime column.")
    if "symbol" not in df.columns:
        raise ValueError("Sentiment CSV must contain a 'symbol' column.")
    
    # 기본 감성 피처 생성
    df["Raw_Sentiment"] = df["finbert_positive"] - df["finbert_negative"]
    df["Sentiment"] = df.groupby("symbol")["Raw_Sentiment"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["Sentiment_Diff"] = df.groupby("symbol")["Sentiment"].transform(lambda x: x.diff().fillna(0))
    df["Sentiment_Volatility"] = df.groupby("symbol")["Sentiment"].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    df["Sentiment_MA5"] = df.groupby("symbol")["Sentiment"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    def pct_change(x): 
        return x.pct_change().fillna(0)
    df["Sentiment_Change"] = df.groupby("symbol")["Sentiment"].transform(pct_change)
    df["Sentiment_Volatility_MA5"] = df.groupby("symbol")["Sentiment"].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )
    
    # 중립 피처 생성
    if "finbert_neutral" in df.columns:
        df["Neutral"] = df["finbert_neutral"]
    else:
        df["Neutral"] = 1 - (df["finbert_positive"] + df["finbert_negative"])
    df["Neutral_MA5"] = df.groupby("symbol")["Neutral"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df["Neutral_Diff"] = df.groupby("symbol")["Neutral"].transform(lambda x: x.diff().fillna(0))
    df["Neutral_Volatility"] = df.groupby("symbol")["Neutral"].transform(lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
    
    return df[["symbol", "Date_only", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility",
               "Sentiment_MA5", "Sentiment_Change", "Sentiment_Volatility_MA5",
               "Neutral", "Neutral_Diff", "Neutral_Volatility", "Neutral_MA5"]]

def merge_stock_and_sentiment(stock_df, sentiment_df, symbol):
    """Merge daily stock prices with aggregated sentiment features for the same symbol."""
    stock_df["Date_only"] = stock_df["Date"].dt.date
    symbol_sent = sentiment_df[sentiment_df["symbol"] == symbol]
    daily_sent = symbol_sent.groupby("Date_only", as_index=False).agg({
        "Sentiment": "mean",
        "Sentiment_Diff": "mean",
        "Sentiment_Volatility": "mean",
        "Sentiment_MA5": "mean",
        "Sentiment_Change": "mean",
        "Sentiment_Volatility_MA5": "mean",
        "Neutral": "mean",
        "Neutral_Diff": "mean",
        "Neutral_Volatility": "mean",
        "Neutral_MA5": "mean"
    })
    merged = pd.merge(stock_df, daily_sent, on="Date_only", how="left")
    sentiment_cols = ["Sentiment", "Sentiment_Diff", "Sentiment_Volatility",
                      "Sentiment_MA5", "Sentiment_Change", "Sentiment_Volatility_MA5",
                      "Neutral", "Neutral_Diff", "Neutral_Volatility", "Neutral_MA5"]
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0)
    return merged[["Date", "Close"] + sentiment_cols]

# 2. Feature Generation and Scaling
def create_multistep_features(df, window=5, mode="stock_only", feature_list=None):
    """
    Create sequences of length 'window' from DataFrame.
    For combined mode, use features provided in feature_list if given.
    """
    X, y = [], []
    if mode == "stock_only":
        features = ["Close"]
    else:
        features = feature_list if feature_list is not None else [
            "Close", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility",
            "Sentiment_MA5", "Sentiment_Change", "Sentiment_Volatility_MA5",
            "Neutral", "Neutral_Diff", "Neutral_Volatility", "Neutral_MA5"
        ]
        for col in features:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in data for combined mode.")
    for i in range(len(df) - window):
        X.append(df[features].iloc[i:i+window].values)
        y.append(df["Close"].iloc[i+window])
    return np.array(X), np.array(y)

def scale_data(X_train, X_test, y_train, y_test):
    """
    Scale features using MinMaxScaler.
    Fit scaler only on training data to prevent data leakage.
    """
    num_samples, time_steps, num_features = X_train.shape
    X_train_flat = X_train.reshape(-1, num_features)
    X_test_flat  = X_test.reshape(-1, num_features)
    feature_scaler = MinMaxScaler().fit(X_train_flat)
    X_train_scaled = feature_scaler.transform(X_train_flat).reshape(num_samples, time_steps, num_features)
    X_test_scaled  = feature_scaler.transform(X_test_flat).reshape(X_test.shape[0], time_steps, num_features)
    y_train = y_train.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)
    target_scaler = MinMaxScaler().fit(y_train)
    y_train_scaled = target_scaler.transform(y_train)
    y_test_scaled  = target_scaler.transform(y_test)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, target_scaler

# 3. Model Definitions
def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(30, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_improved_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(50, activation='relu', return_sequences=True)),
        Dropout(0.2),
        LSTM(30, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='nadam', loss='mse')
    return model

def augment_data(X, noise_std=0.01):
    noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
    X_aug = X + noise
    return np.concatenate([X, X_aug], axis=0)

# 4. Evaluation Metric Calculation
def evaluate_predictions(y_true_scaled, y_pred_scaled, target_scaler, prev_closes=None):
    y_true = target_scaler.inverse_transform(y_true_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if prev_closes is not None:
        prev = np.array(prev_closes).reshape(-1, 1)
        actual_dir = np.sign(y_true - prev)
        pred_dir   = np.sign(y_pred - prev)
    else:
        actual_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir   = np.sign(y_pred[1:] - y_pred[:-1])
    dir_acc = np.mean(pred_dir == actual_dir)
    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DirAcc": dir_acc}
    return metrics, y_true, y_pred

# 5. Feature Selection using Correlation-based method
def select_features_by_corr(data_df, candidate_features, target="Close", top_n=6, threshold=0.1):
    """
    각 후보 피처와 target("Close") 간의 절대 Pearson 상관계수를 계산해,
    threshold 이상인 피처가 있다면 상위 top_n 피처를 선택합니다.
    만약 threshold 이상의 피처가 없으면, 후보 피처 전체를 반환합니다.
    항상 'Close'는 포함됩니다.
    """
    corr_matrix = data_df[candidate_features + [target]].corr().abs()
    corr_with_target = corr_matrix[target].drop(target)
    selected = corr_with_target[corr_with_target >= threshold].sort_values(ascending=False).head(top_n).index.tolist()
    if len(selected) == 0:
        selected = candidate_features
    return ["Close"] + selected

# 6. Training, Evaluation, and Saving Results for Each Company
def model_evaluation_and_save(stock_df, symbol, sentiment_df, result_folder, window=5, epochs=50, batch_size=32):
    # (a) 데이터 준비: 2020년 1월 1일부터 2023년 12월 31일까지 학습, 이후 테스트
    stock_df = stock_df[stock_df["Date"] >= pd.to_datetime("2020-01-01")].copy()
    stock_df.reset_index(drop=True, inplace=True)
    if sentiment_df is not None:
        data_df = merge_stock_and_sentiment(stock_df, sentiment_df, symbol)
    else:
        data_df = stock_df[["Date", "Close"]].copy()
    data_df.sort_values("Date", inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    
    # (b) 학습/테스트 분할 (시계열 기준)
    train_end_date = pd.to_datetime("2023-12-31")
    train_df = data_df[data_df["Date"] <= train_end_date].copy()
    test_df  = data_df[data_df["Date"] > train_end_date].copy()
    
    # (c) 시퀀스 생성
    # stock-only 시퀀스 (학습용)
    X_stock, y_stock = create_multistep_features(train_df, window=window, mode="stock_only")
    
    # Combined 모델: 피처 선택 및 시퀀스 생성 (학습용)
    X_comb, y_comb = None, None
    features_used = None
    candidate_features = [
        "Sentiment", "Sentiment_Diff", "Sentiment_Volatility",
        "Sentiment_MA5", "Sentiment_Change", "Sentiment_Volatility_MA5",
        "Neutral", "Neutral_Diff", "Neutral_Volatility", "Neutral_MA5"
    ]
    if sentiment_df is not None:
        features_used = select_features_by_corr(train_df, candidate_features, target="Close", top_n=6, threshold=0.1)
        X_comb, y_comb = create_multistep_features(train_df, window=window, mode="combined", feature_list=features_used)
    
    # (d) 테스트 시퀀스 생성 (테스트용)
    X_stock_test, y_stock_test = create_multistep_features(test_df, window=window, mode="stock_only")
    if sentiment_df is not None:
        X_comb_test, y_comb_test = create_multistep_features(test_df, window=window, mode="combined", feature_list=features_used)
    
    # (e) 스케일링 (Train만 fit)
    X_train_s_scaled, X_test_s_scaled, y_train_s_scaled, y_test_s_scaled, target_scaler_s = scale_data(
        X_stock, X_stock_test, y_stock, y_stock_test
    )
    if sentiment_df is not None:
        X_train_c_scaled, X_test_c_scaled, y_train_c_scaled, y_test_c_scaled, target_scaler_c = scale_data(
            X_comb, X_comb_test, y_comb, y_comb_test
        )
    
    # (f) 모델 학습
    model_stock = build_model(input_shape=X_train_s_scaled.shape[1:])
    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
    ]
    model_stock.fit(X_train_s_scaled, y_train_s_scaled, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1, callbacks=callbacks, verbose=0)
    
    model_comb = None
    if sentiment_df is not None:
        X_train_c_aug = augment_data(X_train_c_scaled, noise_std=0.01)
        y_train_c_aug = np.concatenate([y_train_c_scaled, y_train_c_scaled], axis=0)
        model_comb = build_improved_model(input_shape=X_train_c_scaled.shape[1:])
        model_comb.fit(X_train_c_aug, y_train_c_aug, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, callbacks=callbacks, verbose=0)
    
    # (g) 예측
    y_pred_s_scaled = model_stock.predict(X_test_s_scaled)
    if sentiment_df is not None:
        y_pred_c_scaled = model_comb.predict(X_test_c_scaled)
    
    # 테스트셋 이전 날 종가 계산: test_df 기준으로, 시퀀스 생성으로 인해 test_df의 앞부분은 사용 불가하므로 window 만큼 제거
    if len(X_test_s_scaled) > 0:
        # test_df의 시퀀스 생성 시 window 만큼이 loss됨을 감안
        prev_closes = test_df["Close"].iloc[window-1:-1].values
    else:
        prev_closes = None
    metrics_stock, y_test_s_inv, y_pred_s_inv = evaluate_predictions(
        y_test_s_scaled, y_pred_s_scaled, target_scaler_s, prev_closes=prev_closes
    )
    metrics_comb = {}
    y_pred_c_inv = None
    if sentiment_df is not None:
        metrics_comb, y_test_c_inv, y_pred_c_inv = evaluate_predictions(
            y_test_c_scaled, y_pred_c_scaled, target_scaler_c, prev_closes=prev_closes
        )
    
    # (h) 결과 저장
    results_stock_df = pd.DataFrame({
        "StockOnly": {k: round(v, 6) for k, v in metrics_stock.items()}
    }).T
    if sentiment_df is not None:
        results_comb_df = pd.DataFrame({
            "Stock+Sentiment": {k: round(v, 6) for k, v in metrics_comb.items()}
        }).T
        results_df = pd.concat([results_stock_df, results_comb_df])
    else:
        results_df = results_stock_df

    features_str = ", ".join(features_used) if features_used is not None else "N/A"
    print(f"========== {symbol} ==========")
    print("Selected Features:", features_str)
    print(results_df, "\n")
    os.makedirs(result_folder, exist_ok=True)
    results_df.to_csv(os.path.join(result_folder, f"{symbol}_evaluation.csv"), index=False)
    with open(os.path.join(result_folder, f"{symbol}_features_used.txt"), "w") as f:
        f.write(features_str)
    
    # (i) 테스트 예측 결과 저장
    if len(X_test_s_scaled) > 0:
        # test_df의 날짜에서 window 만큼 제거하여 실제 시퀀스 시작일에 맞춤
        test_dates = test_df["Date"].iloc[window:].values
        pred_df = pd.DataFrame({
            "Date": test_dates,
            "Actual": y_test_s_inv.flatten(),
            "Pred_StockOnly": y_pred_s_inv.flatten()
        })
        if sentiment_df is not None:
            pred_df["Pred_StockSentiment"] = y_pred_c_inv.flatten()
        pred_df.sort_values("Date", inplace=True)
        pred_df.to_csv(os.path.join(result_folder, f"{symbol}_predictions.csv"), index=False)

# Main execution: process each company
nasdaq_M7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
sentiment_file = "output_with_sentiment_combined2.csv"
sentiment_df = load_sentiment_data(sentiment_file) if os.path.exists(sentiment_file) else None

result_folder = "result_final3"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

for symbol in nasdaq_M7:
    stock_path = os.path.join("stock", f"{symbol}_stock_data.json")
    if not os.path.exists(stock_path):
        print(f"Stock data file not found for {symbol}!")
        continue
    stock_df = load_stock_data(stock_path, symbol)
    model_evaluation_and_save(stock_df, symbol, sentiment_df, result_folder, window=5, epochs=50, batch_size=32)
