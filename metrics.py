# metrics.py

import numpy as np


def calculate_returns(equity_curve):
    equity = np.array(equity_curve)
    return np.diff(equity) / equity[:-1]


def sharpe_ratio(equity_curve, risk_free_rate=0.0):
    returns = calculate_returns(equity_curve)
    if np.std(returns) == 0:
        return 0
    return (np.mean(returns) - risk_free_rate) / np.std(returns)


def max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return np.min(drawdown)


def total_return(equity_curve):
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
