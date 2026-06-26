# train_all_stocks.py

import yfinance as yf
import pandas as pd
import numpy as np

from train_ml import train_ml_pipeline
from train_rl import train_rl_pipeline
from backtest import backtest, save_backtest_results
from models.nlp_model import get_batch_sentiment


tickers = ["NFLX"]

for ticker in tickers:

    print("\n==============================")
    print(f"🚀 Training for {ticker}")
    print("==============================")

    # -----------------------
    # Download data
    # -----------------------
    df = yf.download(ticker, period="2y")

    if df.empty:
        print(f"⚠️ No data for {ticker}")
        continue

    df = df.reset_index()

    # -----------------------
    # ML Training (LSTM)
    # -----------------------
    lstm_model, scaler, predicted_returns = train_ml_pipeline(df)

    predicted_returns = np.array(predicted_returns).reshape(-1)

    # -----------------------
    # NLP Sentiment
    # -----------------------
    headlines = [
        f"{ticker} reports earnings growth",
        f"Market volatility affects {ticker}"
    ] * (len(df) // 2 + 1)

    headlines = headlines[:len(df)]

    sentiment_scores = np.array(get_batch_sentiment(headlines), dtype=float).reshape(-1)

    # -----------------------
    # Feature Engineering (SAFE)
    # -----------------------
    volumes = df["Volume"].to_numpy().reshape(-1).astype(float)

    sma50 = (
        df["Close"]
        .rolling(50)
        .mean()
        .bfill()
        .to_numpy()
        .reshape(-1)
        .astype(float)
    )

    sma200 = (
        df["Close"]
        .rolling(200)
        .mean()
        .bfill()
        .to_numpy()
        .reshape(-1)
        .astype(float)
    )

    # -----------------------
    # Align all lengths safely
    # -----------------------
    n = min(
        len(predicted_returns),
        len(sentiment_scores),
        len(volumes),
        len(sma50),
        len(sma200)
    )

    features = np.column_stack([
        predicted_returns[:n],
        sentiment_scores[:n],
        volumes[:n],
        sma50[:n],
        sma200[:n]
    ]).astype(float)

    # -----------------------
    # RL Training
    # -----------------------
    agent = train_rl_pipeline(df.iloc[:n], predicted_returns[:n], sentiment_scores[:n])

    # -----------------------
    # Backtest
    # -----------------------
    equity = backtest(agent, df["Close"].values[:n], features)

    equity = np.array(equity, dtype=float)

    strategy_returns = np.diff(equity) / (equity[:-1] + 1e-9)
    strategy_returns = np.append(strategy_returns, 0.0)

    # -----------------------
    # Save Results
    # -----------------------
    save_backtest_results(
        dates=df.index[:n],
        prices=df["Close"].values[:n],
        strategy_returns=strategy_returns,
        ticker=ticker
    )

    print(f"✅ Saved {ticker}_backtest.csv")