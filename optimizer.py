import numpy as np
from scipy.optimize import minimize

trading_days = 252

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculates expected portfolio return and risk (volatility).
    """
    ret = np.dot(weights, mean_returns)* trading_days
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))* trading_days)
    return ret, risk

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Objective function: negative Sharpe ratio (because we minimize in scipy).
    """
    ret, risk = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / risk

def optimize_portfolio(price_data, risk_free_rate=0.01):
    """
    Runs portfolio optimization to maximize Sharpe ratio.

    Parameters:
        price_data (pd.DataFrame): Close prices from data_handler

    Returns:
        dict: optimal weights, return, risk, Sharpe
    """
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    
    init_guess = np.array([1.0 / num_assets] * num_assets)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    result = minimize(
        negative_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        weights = result.x
        ret, risk = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - risk_free_rate) / risk
        return {
            'weights': weights,
            'return': ret,
            'risk': risk,
            'sharpe': sharpe
        }
    else:
        raise ValueError("Optimization failed.")
