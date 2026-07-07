# 📈 AI-Based Stock Market Trading Strategy Advisor

An end-to-end AI-powered algorithmic trading system that combines **Deep Learning (LSTM)**, **Natural Language Processing (FinBERT)**, and **Reinforcement Learning (Deep Q-Network)** to generate intelligent Buy, Hold, and Sell recommendations.

The system integrates historical market data, financial news sentiment, technical indicators, and reinforcement learning into a unified trading pipeline, accompanied by a Streamlit dashboard for visualization and performance evaluation.

---

# Overview

Traditional trading strategies rely solely on historical prices or technical indicators.

This project improves decision making by combining three AI paradigms:

- Time-series forecasting (LSTM)
- Financial sentiment analysis (FinBERT)
- Reinforcement learning (DQN)

The complete pipeline predicts market behavior, understands financial news sentiment, learns trading policies, and evaluates performance through backtesting.

---

# Problem Statement

Most retail trading systems fail because they only consider one source of information.

This project integrates:

✔ Historical prices

✔ Financial news

✔ Technical indicators

✔ Sequential learning

✔ Reinforcement learning

to build a more intelligent trading advisor.

---

# Key Features

## Historical Market Analysis

Downloads stock data using Yahoo Finance.

Includes

- Open
- High
- Low
- Close
- Volume

Supports both US and Indian stocks.

---

## LSTM Price Prediction

Uses historical prices to forecast short-term market movement.

Pipeline

Historical Prices

↓

Data Cleaning

↓

Normalization

↓

60-Day Sliding Window

↓

LSTM Training

↓

Predicted Return

---

## Financial Sentiment Analysis

Financial news headlines are processed using FinBERT.

Outputs

Positive

Neutral

Negative

along with confidence scores.

---

## Technical Indicators

The reinforcement learning agent uses

- SMA 50
- SMA 200
- RSI
- MACD
- Volatility
- Daily Returns
- Trading Volume

as state features.

---

## Deep Q-Network (DQN)

Learns an optimal trading strategy.

Actions

- Buy
- Hold
- Sell

The agent improves over multiple training episodes using reward optimization.

---

## Backtesting Engine

Evaluates strategy performance using historical data.

Metrics

- Sharpe Ratio
- Maximum Drawdown
- Total Return
- Equity Curve

---

## Interactive Dashboard

Built using Streamlit.

Displays

- Latest Stock Price
- Predicted Return
- Sentiment Score
- Buy/Hold/Sell Recommendation
- Equity Curve
- Performance Metrics

---

# AI Pipeline

Yahoo Finance

↓

Historical Prices

↓

Preprocessing

↓

LSTM Prediction

↓

Financial News

↓

FinBERT Sentiment

↓

Technical Indicators

↓

Feature Engineering

↓

Deep Q-Network

↓

Trading Action

↓

Backtesting

↓

Performance Dashboard

---

# Technology Stack

## Machine Learning

- TensorFlow
- Keras
- Scikit-Learn

---

## Deep Learning

- LSTM

---

## NLP

- FinBERT
- Transformers
- PyTorch

---

## Reinforcement Learning

- Deep Q-Network (DQN)

---

## Visualization

- Streamlit
- Matplotlib
- Pandas

---

## Data Source

- Yahoo Finance

---

# Machine Learning Pipeline

## Step 1

Download stock data.

↓

## Step 2

Normalize prices using MinMaxScaler.

↓

## Step 3

Generate 60-day sequences.

↓

## Step 4

Train LSTM.

↓

## Step 5

Predict returns.

↓

## Step 6

Collect financial news.

↓

## Step 7

Analyze sentiment using FinBERT.

↓

## Step 8

Compute technical indicators.

↓

## Step 9

Create RL state vector.

↓

## Step 10

Train Deep Q-Network.

↓

## Step 11

Generate Buy/Hold/Sell action.

↓

## Step 12

Backtest strategy.

---

# Project Structure

```
models/

    lstm_model.py

    nlp_model.py

    rl_model.py

training/

    train_ml.py

    train_rl.py

evaluation/

    metrics.py

    backtest.py

Stock Data/

Streamlit/

    app.py

README.md
```

---

# Performance Evaluation

The strategy is evaluated using

### Sharpe Ratio

Measures risk-adjusted return.

---

### Maximum Drawdown

Largest portfolio decline.

---

### Total Return

Overall portfolio growth.

---

### Equity Curve

Visualizes portfolio performance over time.

---

# Supported Stocks

Examples

- AAPL
- MSFT
- NVDA
- GOOG
- TSLA
- TCS.NS
- INFY.NS

Additional stocks can be added easily.

---

# Future Enhancements

- Real-time Trading APIs
- Broker Integration
- Portfolio Optimization
- Multi-Asset Support
- Transformer-based Price Prediction
- Risk Management Module
- Position Sizing
- Explainable AI
- Multi-Agent Reinforcement Learning
- Cloud Deployment
- Live Trading Dashboard

---

# Research Contribution

This project also formed the basis of a peer-reviewed research publication proposing an integrated AI trading framework combining

- LSTM
- FinBERT
- Deep Q-Network

for intelligent stock market decision making.

---

# Learning Outcomes

- Deep Learning
- Financial NLP
- Reinforcement Learning
- Time Series Forecasting
- Algorithmic Trading
- Feature Engineering
- AI Pipeline Design
- ML System Design
- Streamlit Deployment
- Research-driven AI Development

---

# Author

**Mrinal Kadam**

AI Engineer | Machine Learning Engineer | Software Engineer
