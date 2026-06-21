# train_ml.py

import numpy as np
from models.lstm_model import train_lstm, predict_return


def train_ml_pipeline(df):

    print("📈 Training LSTM model...")

    model, scaler = train_lstm(df)

    look_back = 60
    predicted_returns = []

    prices = df['Close'].values.astype(float)

    for i in range(look_back, len(prices)):

        recent_prices = prices[i-look_back:i]

        pred_return = predict_return(model, scaler, recent_prices)

        print("TYPE =", type(pred_return))
        print("VALUE =", pred_return)

        pred_return = float(np.squeeze(pred_return))

        predicted_returns.append(pred_return)

    # Pad beginning with floats (NOT ints)
    predicted_returns = [0.0 for _ in range(look_back)] + predicted_returns

    # 🔥 CRITICAL: force strict float64 numpy array
    predicted_returns = np.array(predicted_returns, dtype=np.float64)

    return model, scaler, predicted_returns