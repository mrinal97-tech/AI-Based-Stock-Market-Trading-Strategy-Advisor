import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from models.nlp_model import get_batch_sentiment

st.set_page_config(layout="wide")
st.title("📊 AI-Based Stock Market Strategy Advisor")

# -----------------------------------
# USER INPUT
# -----------------------------------
st.sidebar.header("🔍 Analyze Stock")

ticker = st.sidebar.text_input("Enter Stock Ticker (Example: AAPL / TCS.NS)")
analyze_button = st.sidebar.button("🚀 Analyze")

if not analyze_button:
    st.info("👈 Enter a stock ticker and click Analyze to get AI trading insights")

# -----------------------------------
# RUN ONLY AFTER CLICK
# -----------------------------------
if analyze_button and ticker:

    ticker = ticker.upper()

    file_map = {
        "AAPL": "backtest_AAPL.csv",
        "MSFT": "backtest_MSFT.csv",
        "TSLA": "backtest_TSLA.csv",
        "GOOG": "backtest_GOOG.csv",
        "TCS.NS": "backtest_TCS_NS.csv",
        "INFY.NS": "backtest_INFY_NS.csv",
        "WIPRO.NS": "backtest_WIPRO_NS.csv",
        "NVDA": "NVDA_backtest.csv"
    }

    if ticker not in file_map:
        st.error("⚠ Backtest data not available for this ticker")
        st.stop()

    df = pd.read_csv(file_map[ticker])

    # -----------------------------------
    # CURRENCY HANDLING
    # -----------------------------------
    if ticker.endswith(".NS"):
        currency_symbol = "₹"
    else:
        currency_symbol = "$"

    # -----------------------------------
    # CURRENT PRICE
    # -----------------------------------
    stock_data = yf.download(ticker, period="5d", progress=False)

    if not stock_data.empty:
        current_price = float(stock_data["Close"].dropna().iloc[-1])
    else:
        current_price = 0.0

    # -----------------------------------
    # PREDICTED PRICE
    # -----------------------------------
    predicted_price = current_price * (1 + df["Total_Return"].iloc[0] * 0.01)

    # -----------------------------------
    # REAL NEWS SENTIMENT
    # -----------------------------------
    ticker_obj = yf.Ticker(ticker)
    headlines = []

    try:
        news = ticker_obj.news

        if news:
            for item in news[:8]:
                if isinstance(item, dict):
                    if "title" in item:
                        headlines.append(item["title"])
                    elif "content" in item and isinstance(item["content"], dict):
                        if "title" in item["content"]:
                            headlines.append(item["content"]["title"])

    except Exception:
        pass

    # fallback
    if len(headlines) == 0:
        headlines = [
            f"Market outlook discussion about {ticker}",
            f"Investor sentiment analysis for {ticker}"
        ]

    sentiment_scores = get_batch_sentiment(headlines)

    avg_sentiment = float(np.mean(sentiment_scores))

    positive_count = sum(1 for s in sentiment_scores if s > 0.1)
    negative_count = sum(1 for s in sentiment_scores if s < -0.1)
    neutral_count = len(sentiment_scores) - positive_count - negative_count

    if avg_sentiment > 0.10:
        sentiment_label = "Positive"
    elif avg_sentiment < -0.10:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # -----------------------------------
    # HYBRID DECISION
    # -----------------------------------
    if predicted_price > current_price and avg_sentiment > 0:
        strategy_decision = "BUY"
    elif predicted_price < current_price and avg_sentiment < 0:
        strategy_decision = "SELL"
    else:
        strategy_decision = "HOLD"

    # -----------------------------------
    # TOP METRICS
    # -----------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
    col2.metric("Projected Price", f"{currency_symbol}{predicted_price:.2f}")
    col3.metric("Sentiment", f"{sentiment_label} ({avg_sentiment:.2f})")
    col4.metric("Strategy Decision", strategy_decision)

    st.divider()

    # -----------------------------------
    # EQUITY CURVE
    # -----------------------------------
    st.subheader("📈 Strategy Equity Curve")

    fig1 = plt.figure(figsize=(8,4))
    plt.plot(df["Strategy_Equity"])
    plt.title("AI Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    st.pyplot(fig1)

    # -----------------------------------
    # BUY & HOLD
    # -----------------------------------
    st.subheader("📊 Buy & Hold Comparison")

    prices = df["Close"]
    buy_hold = prices / prices.iloc[0]

    fig2 = plt.figure(figsize=(8,4))
    plt.plot(df["Strategy_Equity"], label="AI Strategy")
    plt.plot(buy_hold, label="Buy & Hold")
    plt.legend()
    plt.title("AI Strategy vs Buy & Hold")
    st.pyplot(fig2)

    # -----------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------
    st.subheader("📋 Performance Metrics")

    metric_df = pd.DataFrame({
        "Metric": ["Sharpe Ratio", "Max Drawdown", "Total Return"],
        "Value": [
            round(df["Sharpe"].iloc[0], 3),
            round(df["Max_Drawdown"].iloc[0], 3),
            round(df["Total_Return"].iloc[0], 3)
        ]
    })

    st.table(metric_df)

    # -----------------------------------
    # SENTIMENT BREAKDOWN
    # -----------------------------------
    st.subheader("📰 Sentiment Breakdown")

    sentiment_df = pd.DataFrame({
        "Category": ["Positive", "Neutral", "Negative"],
        "Count": [positive_count, neutral_count, negative_count]
    })

    st.table(sentiment_df)