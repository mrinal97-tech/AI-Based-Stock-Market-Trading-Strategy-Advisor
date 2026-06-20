import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def preprocess_data(df):

    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    # Momentum
    df['Returns'] = df['Close'].pct_change()

    # Volatility (rolling std)
    df['Volatility'] = df['Returns'].rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df


def create_sentiment_data(df):
    # Placeholder sentiment for demo
   np.random.seed(42)
   df['Sentiment'] = np.random.uniform(-1, 1, size=len(df))
 
