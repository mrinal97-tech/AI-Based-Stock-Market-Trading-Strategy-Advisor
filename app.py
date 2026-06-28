import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from models.nlp_model import get_batch_sentiment

st.set_page_config(layout="wide")
st.title("📊 AI-Based Stock Market Strategy Advisor")

# -----------------------------------
# CACHED FUNCTIONS
# -----------------------------------

@st.cache_data
def load_backtest_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data(ttl=300)
def get_current_price(ticker):
    stock_data = yf.download(ticker, period="5d", progress=False)
    if not stock_data.empty:
        return float(stock_data["Close"].dropna().iloc[-1])
    return 0.0

@st.cache_data(ttl=3600)
def get_news_headlines(ticker):
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

    if len(headlines) == 0:
        headlines = [
            f"Market outlook discussion about {ticker}",
            f"Investor sentiment analysis for {ticker}"
        ]

    return headlines

@st.cache_data(ttl=3600)
def get_cached_sentiment(headlines):
    return get_batch_sentiment(headlines)

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
        # US Stocks
        "AAPL": "Stock Data/US Stocks/backtest_AAPL.csv",
        "MSFT": "Stock Data/US Stocks/backtest_MSFT.csv",
        "GOOG": "Stock Data/US Stocks/backtest_GOOG.csv",
        "TSLA": "Stock Data/US Stocks/backtest_TSLA.csv",
        "NVDA": "Stock Data/US Stocks/NVDA_backtest.csv",
        "AMZN": "Stock Data/US Stocks/AMZN_backtest.csv",
        "META": "Stock Data/US Stocks/META_backtest.csv",
        "NFLX": "Stock Data/US Stocks/NFLX_backtest.csv",
        "AMD": "Stock Data/US Stocks/AMD_backtest.csv",
        "INTC": "Stock Data/US Stocks/INTC_backtest.csv",
        "JPM": "Stock Data/US Stocks/JPM_backtest.csv",
        "KO": "Stock Data/US Stocks/KO_backtest.csv",
        "V": "Stock Data/US Stocks/V_backtest.csv",
        "WMT": "Stock Data/US Stocks/WMT_backtest.csv",
        "XOM": "Stock Data/US Stocks/XOM_backtest.csv",

        # Indian Stocks
        "TCS.NS": "Stock Data/Indian Stocks/backtest_TCS_NS.csv",
        "INFY.NS": "Stock Data/Indian Stocks/backtest_INFY_NS.csv",
        "WIPRO.NS": "Stock Data/Indian Stocks/backtest_WIPRO_NS.csv",
        "RELIANCE.NS": "Stock Data/Indian Stocks/RELIANCE.NS_backtest.csv",
        "HDFCBANK.NS": "Stock Data/Indian Stocks/HDFCBANK.NS_backtest.csv",
        "HINDUNILVR.NS": "Stock Data/Indian Stocks/HINDUNILVR.NS_backtest.csv",

        # Commodities
        "GC=F": "Stock Data/commodities/GC=F_backtest.csv",
        "SI=F": "Stock Data/commodities/SI=F_backtest.csv",
        "CL=F": "Stock Data/commodities/CL=F_backtest.csv",
        "NG=F": "Stock Data/commodities/NG=F_backtest.csv",

        # Indices
        "^GSPC": "Stock Data/indices/GSPC_backtest.csv",
        "^IXIC": "Stock Data/indices/IXIC_backtest.csv",
        "^NSEI": "Stock Data/indices/NSEI_backtest.csv",
        "^BSESN": "Stock Data/indices/BSESN_backtest.csv",
    }

    if ticker not in file_map:
        st.error("⚠ Backtest data not available for this ticker")
        st.stop()

    df = load_backtest_data(file_map[ticker])

    if ticker.endswith(".NS"):
        currency_symbol = "₹"
    else:
        currency_symbol = "$"

    current_price = get_current_price(ticker)

    predicted_price = current_price * (1 + df["Total_Return"].iloc[0] * 0.01)

    headlines = get_news_headlines(ticker)
    sentiment_scores = get_cached_sentiment(tuple(headlines))

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

    if predicted_price > current_price and avg_sentiment > 0:
        strategy_decision = "BUY"
    elif predicted_price < current_price and avg_sentiment < 0:
        strategy_decision = "SELL"
    else:
        strategy_decision = "HOLD"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
    col2.metric("Projected Price", f"{currency_symbol}{predicted_price:.2f}")
    col3.metric("Sentiment", f"{sentiment_label} ({avg_sentiment:.2f})")
    col4.metric("Strategy Decision", strategy_decision)

    st.divider()

    st.subheader("📈 Strategy Equity Curve")

    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(df["Strategy_Equity"])
    plt.title("AI Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    st.pyplot(fig1)

    st.subheader("📊 Buy & Hold Comparison")

    prices = df["Close"]
    buy_hold = prices / prices.iloc[0]

    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(df["Strategy_Equity"], label="AI Strategy")
    plt.plot(buy_hold, label="Buy & Hold")
    plt.legend()
    plt.title("AI Strategy vs Buy & Hold")
    st.pyplot(fig2)

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

    st.subheader("📰 Sentiment Breakdown")

    sentiment_df = pd.DataFrame({
        "Category": ["Positive", "Neutral", "Negative"],
        "Count": [positive_count, neutral_count, negative_count]
    })

    st.table(sentiment_df)