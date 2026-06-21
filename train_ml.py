# train_ml.py

import numpy as np
from models.lstm_model import train_lstm, predict_return


def train_ml_pipeline(df):

    print("📈 Training LSTM model...")

    model, scaler = train_lstm(df)

    look_back = 60
    predicted_returns = []

    prices = df['Close'].values.astype(float).reshape(-1, 1)

    # IMPORTANT: scale entire series once
    scaled_prices = scaler.transform(prices)

    for i in range(look_back, len(scaled_prices)):

        recent_window = scaled_prices[i - look_back:i]

        # shape: (60, 1)
        pred_return = predict_return(model, scaler, recent_window)

        # SAFE conversion
        pred_return = float(np.array(pred_return).reshape(-1)[0])

        predicted_returns.append(pred_return)

    # pad start
    predicted_returns = [0.0] * look_back + predicted_returns

    predicted_returns = np.array(predicted_returns, dtype=np.float64)

    return model, scaler, predicted_returns