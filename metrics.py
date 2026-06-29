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


def information_coefficient(preds: pd.DataFrame, fwd: pd.DataFrame) -> dict[str, float]:
    """Cross-sectional rank IC of the model's predictions vs realised returns.

    The IC (information coefficient) is the per-month Spearman rank correlation
    between predicted edge and the realised forward return across the names
    scored that month. It measures the model's RAW signal quality — independent
    of selection, sizing, or costs — which is exactly what to optimise when
    improving the model (a strong backtest can come from one lucky bet; a
    positive, stable IC is the honest sign of edge).

    Parameters
    ----------
    preds : DataFrame indexed by (date, ticker) with a ``pred`` column.
    fwd   : [date x ticker] realised forward-return panel.

    Returns mean IC, IC information ratio (mean/std across months), hit rate
    (fraction of months with IC>0), month count, and a t-stat. Typical equity
    cross-sectional ICs are small: ~0.03-0.06 mean is already a real edge.
    """
    nan = {"mean_ic": float("nan"), "ic_ir": float("nan"),
           "hit_rate": float("nan"), "n_months": 0, "t_stat": float("nan")}
    if preds is None or preds.empty or "pred" not in preds.columns:
        return nan

    ics: list[float] = []
    for date, grp in preds.groupby(level=0):
        if date not in fwd.index:
            continue
        tickers = grp.index.get_level_values(1)
        realised = fwd.loc[date].reindex(tickers).to_numpy()
        pair = pd.DataFrame({"p": grp["pred"].to_numpy(), "r": realised}).dropna()
        if len(pair) < 5:  # too few names for a meaningful cross-section
            continue
        ic = pair["p"].rank().corr(pair["r"].rank())  # Spearman
        if pd.notna(ic):
            ics.append(float(ic))

    if not ics:
        return nan
    arr = np.asarray(ics)
    mean_ic = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    ic_ir = mean_ic / std if std and not np.isnan(std) and std != 0 else float("nan")
    t_stat = (mean_ic / (std / np.sqrt(len(arr)))
              if std and not np.isnan(std) and std != 0 else float("nan"))
    return {"mean_ic": mean_ic, "ic_ir": ic_ir,
            "hit_rate": float((arr > 0).mean()), "n_months": len(arr),
            "t_stat": t_stat}


def sortino(monthly_returns: pd.Series, rf_annual: float = C.RISK_FREE_ANNUAL) -> float:
    """Annualised Sortino ratio — like Sharpe but penalises only DOWNSIDE vol.

    Upside swings shouldn't count as 'risk'; Sortino divides excess return by the
    std of negative months only.
    """
    r = monthly_returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_monthly = rf_annual / MONTHS_PER_YEAR
    excess = r - rf_monthly
    # Downside deviation = RMS of below-target returns over the FULL sample
    # (standard Sortino). Finite for >=1 nonzero down month, unlike std of the
    # negatives alone (which is NaN for a single down month).
    dd = float(np.sqrt(np.square(np.minimum(excess.to_numpy(), 0.0)).mean()))
    if dd == 0 or pd.isna(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(MONTHS_PER_YEAR))


def calmar(monthly_returns: pd.Series) -> float:
    """CAGR divided by the magnitude of max drawdown — return per unit of pain."""
    mdd = abs(max_drawdown(monthly_returns))
    if mdd == 0 or pd.isna(mdd):
        return float("nan")
    return float(cagr(monthly_returns) / mdd)


def win_rate(monthly_returns: pd.Series) -> float:
    """Fraction of months with a positive return."""
    r = monthly_returns.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def summarize(monthly_returns: pd.Series) -> dict[str, float]:
    """Bundle the headline metrics for one strategy."""
    return {
        "total_return": total_return(monthly_returns),
        "cagr": cagr(monthly_returns),
        "max_drawdown": max_drawdown(monthly_returns),
        "sharpe": sharpe(monthly_returns),
        "sortino": sortino(monthly_returns),
        "calmar": calmar(monthly_returns),
        "win_rate": win_rate(monthly_returns),
    }
