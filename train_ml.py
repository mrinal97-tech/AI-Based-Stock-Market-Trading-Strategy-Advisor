# train_ml.py

import numpy as np
from models.lstm_model import train_lstm, predict_return


def train_ml_pipeline(df):

    print("📈 Training LSTM model...")

    model, scaler = train_lstm(df)

    look_back = 60
    predicted_returns = []

    prices = df['Close'].values.astype(float).reshape(-1, 1)

    scaled_prices = scaler.transform(prices)

    for i in range(look_back, len(scaled_prices)):

        recent_window = scaled_prices[i - look_back:i]

        pred_return = predict_return(model, scaler, recent_window)

        # SAFE FIX
        pred_return = np.asarray(pred_return).squeeze().item()

        predicted_returns.append(float(pred_return))

    predicted_returns = [0.0] * look_back + predicted_returns

    return model, scaler, np.array(predicted_returns, dtype=np.float64)