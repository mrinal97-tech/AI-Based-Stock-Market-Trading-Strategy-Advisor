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
def save_backtest_results(dates, prices, strategy_returns, ticker):

    import pandas as pd
    import numpy as np

    prices = np.asarray(prices, dtype=np.float64).reshape(-1)
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64).reshape(-1)

    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    strategy_returns = np.nan_to_num(strategy_returns, nan=0.0, posinf=0.0, neginf=0.0)

    equity_curve = (1 + strategy_returns).cumprod()

    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
    max_drawdown = (equity_curve / np.maximum.accumulate(equity_curve) - 1).min()
    total_return = equity_curve[-1] - 1

    min_len = min(len(dates), len(prices), len(equity_curve))

    results_df = pd.DataFrame({
        "Date": dates[:min_len],
        "Close": prices[:min_len],
        "Strategy_Equity": equity_curve[:min_len]
    })

    results_df["Sharpe"] = sharpe
    results_df["Max_Drawdown"] = max_drawdown
    results_df["Total_Return"] = total_return

    output_path = f"{ticker}_backtest.csv"
    results_df.to_csv(output_path, index=False)

    print(f"✅ Backtest saved: {output_path}")

    return output_path