import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 1. Data Loading and Preprocessing Functions
def load_stock_data(file_path, symbol):
    """Load stock JSON and preprocess columns."""
    df = pd.read_json(file_path)
    # Rename multi-index columns (e.g. "('Close', 'AAPL')" -> "Close")
    new_cols = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith("('"):
            new_cols[col] = col.split(",")[0].strip("('")  # take the first part like 'Close'
        else:
            new_cols[col] = col
    df = df.rename(columns=new_cols)
    df["Date"] = pd.to_datetime(df["Date"])  # ensure Date is datetime
    df["symbol"] = symbol
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_sentiment_data(file_path):
    """Load sentiment CSV and create daily sentiment features."""
    df = pd.read_csv(file_path)
    # Ensure there's a date column to work with
    if "Date" in df.columns:
        df["Date_only"] = pd.to_datetime(df["Date"]).dt.date
    elif "datetime" in df.columns:
        df["Date_only"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        raise ValueError("Sentiment CSV must contain a date or datetime column.")
    if "symbol" not in df.columns:
        raise ValueError("Sentiment CSV must contain a 'symbol' column.")
    # Compute raw sentiment and rolling features
    df["Raw_Sentiment"] = df["finbert_positive"] - df["finbert_negative"]
    # 3-day rolling average of sentiment (by symbol) to smooth out noise
    df["Sentiment"] = df.groupby("symbol")["Raw_Sentiment"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    # Daily sentiment change and volatility (3-day rolling std)
    df["Sentiment_Diff"] = df.groupby("symbol")["Sentiment"].transform(lambda x: x.diff().fillna(0))
    df["Sentiment_Volatility"] = df.groupby("symbol")["Sentiment"].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    # Return relevant columns
    return df[["symbol", "Date_only", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]]

def merge_stock_and_sentiment(stock_df, sentiment_df, symbol):
    """Merge daily stock prices with sentiment features for the same symbol."""
    stock_df["Date_only"] = stock_df["Date"].dt.date
    symbol_sent = sentiment_df[sentiment_df["symbol"] == symbol]
    # Aggregate sentiment by Date_only (mean of sentiment features for that day)
    daily_sent = symbol_sent.groupby("Date_only", as_index=False).agg({
        "Sentiment": "mean",
        "Sentiment_Diff": "mean",
        "Sentiment_Volatility": "mean"
    })
    merged = pd.merge(stock_df, daily_sent, on="Date_only", how="left")
    # Fill missing sentiment values with 0 (no news on that day)
    merged[["Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]] = merged[["Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]].fillna(0)
    # Keep only needed columns
    return merged[["Date", "Close", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]]

# 2. Feature Generation and Scaling for LSTM
def create_multistep_features(df, window=5, mode="stock_only"):
    """
    Create sequences of length 'window' from DataFrame.
    mode='stock_only': use only Close price as feature.
    mode='combined': use Close + sentiment features.
    """
    X, y = [], []
    if mode == "stock_only":
        features = ["Close"]
    else:
        features = ["Close", "Sentiment", "Sentiment_Diff", "Sentiment_Volatility"]
        # Ensure required columns exist
        for col in features:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in data for combined mode.")
    for i in range(len(df) - window):
        X.append(df[features].iloc[i:i+window].values)
        y.append(df["Close"].iloc[i+window])
    return np.array(X), np.array(y)

def scale_data(X_train, X_test, y_train, y_test):
    """Min-Max scale features and target for LSTM."""
    from sklearn.preprocessing import MinMaxScaler
    # Scale X (features) across all time steps for each feature column
    num_samples, time_steps, num_features = X_train.shape
    X_train_flat = X_train.reshape(-1, num_features)
    X_test_flat  = X_test.reshape(-1, num_features)
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(num_samples, time_steps, num_features)
    X_test_scaled  = feature_scaler.transform(X_test_flat).reshape(X_test.shape[0], time_steps, num_features)
    # Scale y (target)
    y_train = y_train.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled  = target_scaler.transform(y_test)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, target_scaler

# 3. LSTM Model Definition
def build_model(input_shape):
    """Build an LSTM model with dropout regularization."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(30, activation='relu'),
        Dropout(0.2),
        Dense(1)  # output layer for predicted Close price
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Evaluation Metric Calculation (including DirAcc)
def evaluate_predictions(y_true_scaled, y_pred_scaled, target_scaler, prev_closes=None):
    """
    Compute metrics (MSE, RMSE, MAE, MAPE, R2, DirAcc) 
    and return them along with un-scaled actual/predicted values.
    """
    # Inverse transform to original scale
    y_true = target_scaler.inverse_transform(y_true_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Directional Accuracy: compare sign of changes
    if prev_closes is not None:
        # Use provided baseline (previous day's close or open) for direction comparison
        prev = np.array(prev_closes).reshape(-1, 1)
        actual_dir = np.sign(y_true - prev)
        pred_dir   = np.sign(y_pred - prev)
    else:
        # Default: compare consecutive differences (for sequential data)
        actual_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir   = np.sign(y_pred[1:] - y_pred[:-1])
    dir_acc = np.mean(pred_dir == actual_dir)
    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DirAcc": dir_acc}
    return metrics, y_true, y_pred

# 5. Training, Evaluation, and Saving Results for Each Company
def model_evaluation_and_save(stock_df, symbol, sentiment_df, result_folder, window=5, epochs=50, batch_size=32):
    """
    Train LSTM models (stock-only and combined) for a given company, evaluate, and save results.
    """
    # (a) Prepare data
    # Filter stock data from 2020 onward (to align with sentiment data timeframe)
    stock_df = stock_df[stock_df["Date"] >= pd.to_datetime("2020-01-01")].copy()
    stock_df.reset_index(drop=True, inplace=True)
    if sentiment_df is not None:
        data_df = merge_stock_and_sentiment(stock_df, sentiment_df, symbol)
    else:
        data_df = stock_df[["Date", "Close"]].copy()
        # Add placeholder zero sentiment columns if needed (not strictly necessary for stock_only mode)
    data_df.sort_values("Date", inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    # Create feature sequences for stock-only and combined modes
    X_stock, y_stock = create_multistep_features(data_df, window=window, mode="stock_only")
    if sentiment_df is not None:
        X_comb, y_comb = create_multistep_features(data_df, window=window, mode="combined")
    else:
        X_comb, y_comb = None, None
    # Ensure the sequence counts match (they should, given we fill missing sentiment with 0)
    if X_comb is not None and len(X_stock) != len(X_comb):
        # Align lengths if there's any minor mismatch
        n = min(len(X_stock), len(X_comb))
        X_stock, y_stock = X_stock[:n], y_stock[:n]
        X_comb, y_comb = X_comb[:n], y_comb[:n]
    # (b) Split into training and test sets (80/20 split by sequence order)
    total_seq = len(X_stock)
    split_idx = int(total_seq * 0.8)
    X_train_s, X_test_s = X_stock[:split_idx], X_stock[split_idx:]
    y_train_s, y_test_s = y_stock[:split_idx], y_stock[split_idx:]
    if X_comb is not None:
        X_train_c, X_test_c = X_comb[:split_idx], X_comb[split_idx:]
        y_train_c, y_test_c = y_comb[:split_idx], y_comb[split_idx:]
    # (c) Scale features and target
    X_train_s_scaled, X_test_s_scaled, y_train_s_scaled, y_test_s_scaled, target_scaler_s = scale_data(X_train_s, X_test_s, y_train_s, y_test_s)[0:5]
    if X_comb is not None:
        X_train_c_scaled, X_test_c_scaled, y_train_c_scaled, y_test_c_scaled, target_scaler_c = scale_data(X_train_c, X_test_c, y_train_c, y_test_c)[0:5]
    # (d) Build and train LSTM models
    model_stock = build_model(input_shape=X_train_s_scaled.shape[1:])
    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
    ]
    model_stock.fit(X_train_s_scaled, y_train_s_scaled, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1, callbacks=callbacks, verbose=0)
    # Train combined model only if sentiment data is available
    model_comb = None
    if X_comb is not None:
        model_comb = build_model(input_shape=X_train_c_scaled.shape[1:])
        model_comb.fit(X_train_c_scaled, y_train_c_scaled, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, callbacks=callbacks, verbose=0)
    # (e) Make predictions on test set
    y_pred_s_scaled = model_stock.predict(X_test_s_scaled)
    if model_comb is not None:
        y_pred_c_scaled = model_comb.predict(X_test_c_scaled)
    # (f) Evaluate performance (invert scaling and compute metrics)
    # Determine baseline for direction comparison: previous day's close for each test point
    if len(X_test_s) > 0:
        # The index of each test target in the original data (since sequences are sequential)
        test_target_indices = np.arange(split_idx, total_seq) + window  # offset by window size
        prev_closes = data_df["Close"].iloc[test_target_indices - 1].values  # previous day close for each test day
    else:
        prev_closes = None
    metrics_stock, y_test_s_inv, y_pred_s_inv = evaluate_predictions(y_test_s_scaled, y_pred_s_scaled, target_scaler_s, prev_closes=prev_closes)
    metrics_comb = {}
    y_pred_c_inv = None
    if model_comb is not None:
        metrics_comb, y_test_c_inv, y_pred_c_inv = evaluate_predictions(y_test_c_scaled, y_pred_c_scaled, target_scaler_c, prev_closes=prev_closes)

    # (g) 
    results_stock_df = pd.DataFrame({
        "StockOnly": {k: round(v, 6) for k, v in metrics_stock.items()}
    }).T

    if model_comb is not None:
        results_comb_df = pd.DataFrame({
            "Stock+Sentiment": {k: round(v, 6) for k, v in metrics_comb.items()}
        }).T
        results_df = pd.concat([results_stock_df, results_comb_df])
    else:
        results_df = results_stock_df

    print(f"========== {symbol} ==========")
    print(results_df, "\n")
    os.makedirs(result_folder, exist_ok=True)
    results_df.to_csv(os.path.join(result_folder, f"{symbol}_evaluation.csv"), index=False)
    # Save predictions of test set to CSV
    if len(X_test_s) > 0:
        test_dates = data_df["Date"].iloc[test_target_indices].values
        pred_df = pd.DataFrame({
            "Date": test_dates,
            "Actual": y_test_s_inv.flatten(),
            "Pred_StockOnly": y_pred_s_inv.flatten()
        })
        if y_pred_c_inv is not None:
            pred_df["Pred_StockSentiment"] = y_pred_c_inv.flatten()
        pred_df.sort_values("Date", inplace=True)
        pred_df.to_csv(os.path.join(result_folder, f"{symbol}_predictions.csv"), index=False)


# Main execution: process each company and evaluate
nasdaq_M7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
sentiment_file = "output_with_sentiment_combined2.csv"
sentiment_df = load_sentiment_data(sentiment_file) if os.path.exists(sentiment_file) else None

result_folder = "result_final"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

for symbol in nasdaq_M7:
    stock_path = os.path.join("stock", f"{symbol}_stock_data.json")
    if not os.path.exists(stock_path):
        print(f"Stock data file not found for {symbol}!")
        continue
    stock_df = load_stock_data(stock_path, symbol)
    model_evaluation_and_save(stock_df, symbol, sentiment_df, result_folder, window=5, epochs=50, batch_size=32)
