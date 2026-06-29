"""Performance metrics computed from a monthly-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

MONTHS_PER_YEAR = 12


def equity_curve(monthly_returns: pd.Series, start_capital: float = C.START_CAPITAL) -> pd.Series:
    """Compound a monthly-return series into an equity curve."""
    return start_capital * (1.0 + monthly_returns.fillna(0.0)).cumprod()


def total_return(monthly_returns: pd.Series) -> float:
    return float((1.0 + monthly_returns.fillna(0.0)).prod() - 1.0)


def cagr(monthly_returns: pd.Series) -> float:
    n = monthly_returns.notna().sum()
    if n == 0:
        return float("nan")
    growth = (1.0 + monthly_returns.fillna(0.0)).prod()
    years = n / MONTHS_PER_YEAR
    return float(growth ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def max_drawdown(monthly_returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve (negative number)."""
    curve = (1.0 + monthly_returns.fillna(0.0)).cumprod()
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def sharpe(monthly_returns: pd.Series, rf_annual: float = C.RISK_FREE_ANNUAL) -> float:
    """Annualised Sharpe ratio from monthly returns."""
    r = monthly_returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    rf_monthly = rf_annual / MONTHS_PER_YEAR
    excess = r - rf_monthly
    return float(excess.mean() / excess.std() * np.sqrt(MONTHS_PER_YEAR))


def summarize(monthly_returns: pd.Series) -> dict[str, float]:
    """Bundle the headline metrics for one strategy."""
    return {
        "total_return": total_return(monthly_returns),
        "cagr": cagr(monthly_returns),
        "max_drawdown": max_drawdown(monthly_returns),
        "sharpe": sharpe(monthly_returns),
    }
