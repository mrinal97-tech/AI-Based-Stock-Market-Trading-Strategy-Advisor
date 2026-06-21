# train_all_stocks.py

# train_all_stocks.py

import yfinance as yf
import pandas as pd
import numpy as np
from train_ml import train_ml_pipeline
from train_rl import train_rl_pipeline
from backtest import backtest, save_backtest_results
from models.nlp_model import get_batch_sentiment

tickers = [
    "NVDA"
]

for ticker in tickers:

    print("\n==============================")
    print(f"🚀 Training for {ticker}")
    print("==============================")

    df = yf.download(ticker, period="2y")

    if df.empty:
        print(f"⚠️ No data for {ticker}")
        continue

    df = df.reset_index()

    # -----------------------
    # ML Training
    # -----------------------
    lstm_model, scaler, predicted_returns = train_ml_pipeline(df)

    # Already clean 1D from train_ml.py

    # -----------------------
    # NLP Sentiment
    # -----------------------
    headlines = [
        f"{ticker} reports earnings growth",
        f"Market volatility affects {ticker}"
    ] * (len(df) // 2 + 1)

    headlines = headlines[:len(df)]

    sentiment_scores = get_batch_sentiment(headlines)
    sentiment_scores = np.array(sentiment_scores).astype(float)

    # -----------------------
    # RL Training
    # -----------------------
    agent = train_rl_pipeline(df, predicted_returns, sentiment_scores)

    # ---------------------------------------
    #  5️⃣ Feature Creation (FORCE CLEAN FLOATS)
    # ---------------------------------------

    volumes = df['Volume'].values.astype(float)
    sma50 = df['Close'].rolling(50).mean().bfill().values.astype(float)
    sma200 = df['Close'].rolling(200).mean().bfill().values.astype(float)

    features = []

    for i in range(len(predicted_returns)):
        features.append((
        float(predicted_returns[i]),
        float(sentiment_scores[i]),
        float(volumes[i]),
        float(sma50[i]),
        float(sma200[i])
    ))

    # -----------------------
    # Backtest
    # -----------------------
    equity = backtest(agent, df['Close'].values, features)

    strategy_returns = np.diff(equity) / equity[:-1]
    strategy_returns = np.append(strategy_returns, 0)

    # -----------------------
    # Save Results
    # -----------------------
    save_backtest_results(
    dates=df.index,
    prices=df["Close"].values,
    strategy_returns=strategy_returns
)

    # Save uniquely for each stock
    filename = f"/kaggle/working/backtest_{ticker.replace('.', '_')}.csv"

    results_df = pd.read_csv("/kaggle/working/AI-Stock-Project-Collab/backtest_results.csv")
    results_df.to_csv(filename, index=False)

    print(f"✅ Saved {filename}")

print("\n🎉 ALL STOCKS TRAINED SUCCESSFULLY")