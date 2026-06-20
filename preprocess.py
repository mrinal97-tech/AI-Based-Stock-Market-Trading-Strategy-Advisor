import pandas as pd
import numpy as np
import datetime

def preprocess_data(df):
    if df.empty:
        return df
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

def create_sentiment_data(df):
    df['Sentiment'] = np.random.uniform(-1, 1, size=len(df))
    return df

try:
    file_path = "AAPL_historical_data.csv"
    
    historical_df = pd.read_csv(
        file_path, 
        skiprows=4, 
        index_col=0, 
        parse_dates=True,
        header=None,
        names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    )

    print("--- Original Data ---")
    print(historical_df.head())

    preprocessed_df = preprocess_data(historical_df)
    final_df = create_sentiment_data(preprocessed_df)

    print("\n--- Preprocessed Data with New Features ---")
    print(final_df.head())
    
    final_df.to_csv("AAPL_preprocessed_data.csv")
    print(f"\nData successfully saved to AAPL_preprocessed_data.csv")
    
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")