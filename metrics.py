import numpy as np
import pandas as pd

import config


def daily_returns(portfolio_value: pd.Series) -> pd.Series:
    return portfolio_value.pct_change().dropna()


def annualised_return(portfolio_value: pd.Series, periods_per_year: int = None) -> float:
    """CAGR based on the equity curve's start/end value and elapsed trading days."""
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    if len(portfolio_value) < 2:
        return np.nan
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1.0
    n_periods = len(portfolio_value) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return np.nan
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def sharpe_ratio(portfolio_value: pd.Series, risk_free_rate: float = None, periods_per_year: int = None) -> float:
    risk_free_rate = risk_free_rate if risk_free_rate is not None else config.RISK_FREE_RATE
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR

    rets = daily_returns(portfolio_value)
    if rets.std(ddof=1) == 0 or rets.empty:
        return np.nan

    period_rf = risk_free_rate / periods_per_year
    excess = rets - period_rf
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(periods_per_year)


def max_drawdown(portfolio_value: pd.Series) -> float:
    """Returns the maximum drawdown as a negative fraction (e.g. -0.15 = -15%)."""
    if portfolio_value.empty:
        return np.nan
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1.0
    return drawdown.min()


def win_rate(round_trips: pd.DataFrame) -> float:
    if round_trips is None or round_trips.empty:
        return np.nan
    return (round_trips["pnl"] > 0).mean()


def summarise(portfolio_value: pd.Series, round_trips: pd.DataFrame) -> dict:
    return {
        "annualised_return": annualised_return(portfolio_value),
        "sharpe_ratio": sharpe_ratio(portfolio_value),
        "win_rate": win_rate(round_trips),
        "max_drawdown": max_drawdown(portfolio_value),
        "num_round_trips": 0 if round_trips is None else len(round_trips),
        "total_pnl": 0.0 if round_trips is None or round_trips.empty else round_trips["pnl"].sum(),
        "final_portfolio_value": portfolio_value.iloc[-1] if len(portfolio_value) else np.nan,
    }
