# models/lstm_model.py

# models/lstm_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
import joblib


def prepare_data_for_lstm(df, look_back=60):
    data = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(df, look_back=60):
    X, y, scaler = prepare_data_for_lstm(df, look_back)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((X_train.shape[1], 1))

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Save model
    model.save("models/lstm_model.h5")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    return model, scaler


def predict_return(model, scaler, recent_prices):
    """
    recent_prices: last 60 real prices
    Returns predicted return (NOT scaled price)
    """
    scaled = scaler.transform(np.array(recent_prices).reshape(-1, 1))
    X = scaled.reshape(1, len(recent_prices), 1)

    scaled_pred = model.predict(X, verbose=0)
    pred_price = scaler.inverse_transform(scaled_pred)[0][0]

    last_price = recent_prices[-1]
    if last_price is None or last_price == 0 or np.isnan(last_price):
        predicted_return = 0.0
    else:
        predicted_return = (pred_price - last_price) / (last_price + 1e-9)

    return predicted_return
