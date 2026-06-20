# main.py

# main.py

import yfinance as yf
import numpy as np
import pandas as pd
from train_ml import train_ml_pipeline
from train_rl import train_rl_pipeline
from backtest import backtest, save_backtest_results
from metrics import sharpe_ratio, max_drawdown, total_return
from models.nlp_model import get_sentiment_score


def main():

    ticker = "AAPL"   # Change if needed

    print(f"\n🚀 Training for {ticker}")

    # 2 YEARS DATA
    df = yf.download(ticker, period="2y")
    df = df.reset_index()

    # -----------------------
    # ML Training
    # -----------------------
    lstm_model, scaler, predicted_returns = train_ml_pipeline(df)

    # -----------------------
    # NLP Sentiment
    # -----------------------
    headlines = [
        f"{ticker} reports strong quarterly earnings",
        f"Market conditions impact {ticker}"
    ]

    sentiment_score = get_sentiment_score(headlines)
    sentiment_scores = [sentiment_score] * len(df)

    # -----------------------
    # RL Training
    # -----------------------
    agent = train_rl_pipeline(df, predicted_returns, sentiment_scores)

    # -----------------------
    # Feature Creation
    # -----------------------
    features = list(zip(
        predicted_returns,
        sentiment_scores,
        df['Volume'],
        df['Close'].rolling(50).mean().bfill(),
        df['Close'].rolling(200).mean().bfill()
    ))

    # -----------------------
    # Backtest
    # -----------------------
    equity = backtest(agent, df['Close'].values, features)

    strategy_returns = np.diff(equity) / equity[:-1]
    strategy_returns = np.append(strategy_returns, 0)

    save_backtest_results(
        dates=df.index,
        prices=df["Close"].values,
        strategy_returns=strategy_returns
    )

    print("\n===== FINAL PERFORMANCE =====")
    print("Sharpe Ratio:", sharpe_ratio(equity))
    print("Max Drawdown:", max_drawdown(equity))
    print("Total Return:", total_return(equity))


if __name__ == "__main__":
    main()
