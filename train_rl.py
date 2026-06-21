# train_rl.py

# train_rl.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from models.dqn_agent import train_dqn_agent


def clean(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1e6, 1e6)


def train_rl_pipeline(df, predicted_returns, sentiment_scores):

    print("🤖 Training RL agent...")

    # -----------------------------
    # 1. ALIGN LENGTHS
    # -----------------------------
    min_len = min(len(df), len(predicted_returns), len(sentiment_scores))

    df = df.tail(min_len).reset_index(drop=True)

    predicted_returns = clean(predicted_returns)[-min_len:]
    sentiment_scores = clean(sentiment_scores)[-min_len:]

    # -----------------------------
    # 2. FEATURES
    # -----------------------------
    prices = clean(df["Close"].values)
    volumes = clean(df["Volume"].values)

    sma50 = clean(df["Close"].rolling(50).mean().bfill().values)
    sma200 = clean(df["Close"].rolling(200).mean().bfill().values)

    # -----------------------------
    # 3. STACK FEATURES
    # -----------------------------
    features_matrix = np.column_stack((
        predicted_returns,
        sentiment_scores,
        volumes,
        sma50,
        sma200
    ))

    # -----------------------------
    # 4. FINAL SAFETY CLEAN
    # -----------------------------
    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    features_matrix = np.clip(features_matrix, -1e6, 1e6)

    # -----------------------------
    # 5. SCALE
    # -----------------------------
    scaler = StandardScaler()
    features_matrix = scaler.fit_transform(features_matrix)

    predicted_returns = features_matrix[:, 0]
    sentiment_scores = features_matrix[:, 1]
    volumes = features_matrix[:, 2]
    sma50 = features_matrix[:, 3]
    sma200 = features_matrix[:, 4]

    # -----------------------------
    # 6. TRAIN DQN
    # -----------------------------
    agent = train_dqn_agent(
        prices=prices,
        predicted_returns=predicted_returns,
        sentiments=sentiment_scores,
        volumes=volumes,
        sma50=sma50,
        sma200=sma200,
        episodes=15,
        save_path="models/rl_model.keras"
    )

    return agent