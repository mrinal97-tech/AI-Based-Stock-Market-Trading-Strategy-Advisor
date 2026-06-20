# train_rl.py

# train_rl.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from models.dqn_agent import train_dqn_agent


def train_rl_pipeline(df, predicted_returns, sentiment_scores):

    print("🤖 Training RL agent...")

    # ---------------------------------------
    # 1️⃣ Align lengths
    # ---------------------------------------
    min_len = min(
        len(df),
        len(predicted_returns),
        len(sentiment_scores)
    )

    df = df.tail(min_len)

    predicted_returns = predicted_returns[-min_len:]
    sentiment_scores = sentiment_scores[-min_len:]

    # ---------------------------------------
    # 2️⃣ Extract features
    # ---------------------------------------
    prices = df['Close'].values
    volumes = df['Volume'].values
    sma50 = df['Close'].rolling(50).mean().bfill().values
    sma200 = df['Close'].rolling(200).mean().bfill().values

    # ---------------------------------------
    # 3️⃣ FORCE CLEAN 1D FLOAT ARRAYS
    # ---------------------------------------
    predicted_returns = np.asarray(predicted_returns).reshape(-1).astype(float)
    sentiment_scores = np.asarray(sentiment_scores).reshape(-1).astype(float)
    volumes = np.asarray(volumes).reshape(-1).astype(float)
    sma50 = np.asarray(sma50).reshape(-1).astype(float)
    sma200 = np.asarray(sma200).reshape(-1).astype(float)

    # ---------------------------------------
    # 4️⃣ STACK FEATURES SAFELY
    # ---------------------------------------
    features_matrix = np.column_stack((
        predicted_returns,
        sentiment_scores,
        volumes,
        sma50,
        sma200
    ))

    # ---------------------------------------
    # 5️⃣ Normalize
    # ---------------------------------------
    scaler = StandardScaler()
    features_matrix = scaler.fit_transform(features_matrix)

    predicted_returns = features_matrix[:, 0]
    sentiment_scores = features_matrix[:, 1]
    volumes = features_matrix[:, 2]
    sma50 = features_matrix[:, 3]
    sma200 = features_matrix[:, 4]

    # ---------------------------------------
    # 6️⃣ Train DQN
    # ---------------------------------------
    agent = train_dqn_agent(
        prices=prices,
        predicted_returns=predicted_returns,
        sentiments=sentiment_scores,
        volumes=volumes,
        sma50=sma50,
        sma200=sma200,
        episodes=15,   # keep moderate for Kaggle
        save_path="models/rl_model.keras"
    )

    return agent