import yfinance as yf
import pandas as pd
import datetime

# --- Configuration ---
# You can change these values to fetch data for a different stock and time period.
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = datetime.date.today().strftime('%Y-%m-%d')

# --- Fetching the data ---
try:
    print(f"Fetching historical data for {TICKER} from {START_DATE} to {END_DATE}...")
    
    # The yf.download() function downloads the data directly into a pandas DataFrame.
    historical_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    
    # Display the first few rows of the DataFrame to confirm the data was fetched.
    print("\n--- Fetched Data Preview ---")
    print(historical_data.head())

    # --- Saving the data ---
    # It's a good practice to save the data for later use, especially for machine learning.
    file_path = f"{TICKER}_historical_data.csv"
    historical_data.to_csv(file_path)
    print(f"\nData successfully saved to {file_path}")

except Exception as e:
    print(f"An error occurred: {e}")