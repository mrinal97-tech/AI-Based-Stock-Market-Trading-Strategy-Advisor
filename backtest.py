# backtest.py

import pandas as pd
import numpy as np


# -----------------------------------
# BACKTEST FUNCTION
# -----------------------------------
def backtest(agent, prices, features):

    # 🔥 FORCE CLEAN 1D FLOAT ARRAY
    prices = np.asarray(prices).reshape(-1).astype(float)

    equity = [1.0]   # start with 1 unit capital
    position = 0

    # Safe return calculation
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, 0)

    for t in range(len(prices) - 1):

        state = np.array([features[t]])
        action = agent.act(state)

        new_position = action - 1   # 0=short,1=hold,2=long

        reward = position * returns[t]
        equity.append(equity[-1] * (1 + reward))

        position = new_position

    return np.array(equity)


# -----------------------------------
# SAVE RESULTS FUNCTION
# -----------------------------------
def save_backtest_results(dates, prices, strategy_returns):

    prices = np.asarray(prices).reshape(-1).astype(float)
    strategy_returns = np.asarray(strategy_returns).reshape(-1).astype(float)

    equity_curve = (1 + strategy_returns).cumprod()

    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
    max_drawdown = (equity_curve / np.maximum.accumulate(equity_curve) - 1).min()
    total_return = equity_curve[-1] - 1

    results_df = pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Strategy_Equity": equity_curve
    })

    results_df["Sharpe"] = sharpe
    results_df["Max_Drawdown"] = max_drawdown
    results_df["Total_Return"] = total_return

    results_df.to_csv("backtest_results.csv", index=False)

    print("✅ Backtest results saved as backtest_results.csv")