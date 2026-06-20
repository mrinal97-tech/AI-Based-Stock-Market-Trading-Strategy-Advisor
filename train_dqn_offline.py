import yfinance as yf
import numpy as np
import os
import joblib
from ml_models import train_dqn_agent

# ---------------------------
# Configuration
# ---------------------------
TICKER = "AAPL"          # change for other stocks
PERIOD = "2y"            # 2 years data
EPISODES = 5            # for quick training; increase later for better accuracy

MODEL_DIR = "models"
SCALER_DIR = "scalers"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

RL_MODEL_PATH = os.path.join(MODEL_DIR, f"{TICKER}_dqn_agent.keras")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.keras")  # placeholder if you use LSTM

# ---------------------------
# Fetch stock data
# ---------------------------
print(f"Fetching {PERIOD} historical data for {TICKER}...")
data = yf.download(TICKER, period=PERIOD)
print("Dataset length:", len(data))   # ✅ Put it here

if data.empty:
    raise ValueError("No data fetched. Check ticker or internet connection.")
data.reset_index(inplace=True)

prices = data['Close'].values
volumes = data['Volume'].values
sma50 = data['Close'].rolling(50).mean().bfill().values
sma200 = data['Close'].rolling(200).mean().bfill().values
# Sentiment placeholder
sentiments = np.random.uniform(-1, 1, size=len(data))
# Predicted prices placeholder (replace with LSTM predictions later)
predicted_prices = prices.copy()

# ---------------------------
# Train RL agent
# ---------------------------
print("Training RL agent...")
agent, scaler_price, scaler_pred, scaler_sent, scaler_vol, scaler_sma50, scaler_sma200 = train_dqn_agent(
    prices,
    predicted_prices,
    sentiments,
    volumes,
    sma50,
    sma200,
    episodes=EPISODES,
    save_path=RL_MODEL_PATH
)

# ---------------------------
# Save scalers
# ---------------------------
joblib.dump(scaler_price, os.path.join(SCALER_DIR, f"{TICKER}_scaler_price.pkl"))
joblib.dump(scaler_pred,  os.path.join(SCALER_DIR, f"{TICKER}_scaler_pred.pkl"))
joblib.dump(scaler_sent,  os.path.join(SCALER_DIR, f"{TICKER}_scaler_sent.pkl"))
joblib.dump(scaler_vol,   os.path.join(SCALER_DIR, f"{TICKER}_scaler_vol.pkl"))
joblib.dump(scaler_sma50, os.path.join(SCALER_DIR, f"{TICKER}_scaler_sma50.pkl"))
joblib.dump(scaler_sma200,os.path.join(SCALER_DIR, f"{TICKER}_scaler_sma200.pkl"))

print(f"✅ RL agent and scalers saved for {TICKER}!")
print(f"Model path: {RL_MODEL_PATH}")
