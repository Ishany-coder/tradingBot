"""Feature engineering: momentum, volatility, point-in-time fundamentals.

All features for a stock at month-end ``t`` use ONLY information available on
or before ``t``. The forward-return target (t -> t+1) is realised at t+1 and is
never used as a feature — it is the thing the model predicts.
"""

from __future__ import annotations

import pandas as pd

import config as C
import data


# --- price-based features ---------------------------------------------------

def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily close panel to month-end (last observation)."""
    return daily.resample("ME").last()


def momentum_panels(mprices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the three momentum features, each a [date x ticker] panel.

    * ``mom_12_1``: return from t-12mo to t-1mo (skips the most recent month).
    * ``mom_6``: trailing 6-month return.
    * ``mom_3``: trailing 3-month return.
    """
    mom_12_1 = mprices.shift(C.MOM_SKIP) / mprices.shift(C.MOM_LONG) - 1.0
    mom_6 = mprices / mprices.shift(C.MOM_MED) - 1.0
    mom_3 = mprices / mprices.shift(C.MOM_SHORT) - 1.0
    return {"mom_12_1": mom_12_1, "mom_6": mom_6, "mom_3": mom_3}


def volatility_panel(daily: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Trailing 60-day daily-return std, sampled at each month-end."""
    daily_ret = daily.pct_change()
    vol = daily_ret.rolling(C.VOL_WINDOW_DAYS).std()
    # Align to month-ends (last available value on/before each month-end).
    return vol.reindex(daily.index).ffill().reindex(month_index, method="ffill")


def forward_return_panel(mprices: pd.DataFrame) -> pd.DataFrame:
    """Forward 1-month return: price[t+1]/price[t]-1. NaN in the final month."""
    return mprices.shift(-1) / mprices - 1.0


def sector_momentum(mprices: pd.DataFrame, etfs: list[str]) -> pd.DataFrame:
    """12-1 momentum for the sector ETFs themselves (Stage 1 ranking input)."""
    sub = mprices[[e for e in etfs if e in mprices.columns]]
    return sub.shift(C.MOM_SKIP) / sub.shift(C.MOM_LONG) - 1.0


# --- point-in-time fundamentals (Backtest B) --------------------------------

# yfinance line-item labels drift between tickers/versions; try several.
_INCOME_NET = ["Net Income", "Net Income Common Stockholders",
               "Net Income From Continuing Operation Net Minority Interest"]
_INCOME_REV = ["Total Revenue", "Operating Revenue"]
_BAL_EQUITY = ["Stockholders Equity", "Common Stock Equity",
               "Total Stockholder Equity"]
_BAL_DEBT = ["Total Debt"]
_BAL_DEBT_PARTS = ["Long Term Debt", "Current Debt", "Short Long Term Debt"]


def _row(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _extract(inc: pd.DataFrame, bal: pd.DataFrame):
    """Pull the four needed line-item series from one statement pair."""
    if inc.empty or bal.empty:
        return None
    net = _row(inc, _INCOME_NET)
    rev = _row(inc, _INCOME_REV)
    equity = _row(bal, _BAL_EQUITY)
    debt = _row(bal, _BAL_DEBT)
    if debt is None:  # reconstruct from parts if "Total Debt" absent
        parts = [_row(bal, [p]) for p in _BAL_DEBT_PARTS]
        parts = [p for p in parts if p is not None]
        debt = sum(parts) if parts else None
    if net is None or equity is None:
        return None
    return net, rev, equity, debt


def _record(period_end, ttm_ni, ttm_rev, eq, dbt) -> dict:
    """Assemble one point-in-time fundamentals record with its availability date.

    availability = fiscal period-end + FUND_LAG_QUARTERS quarters, approximating
    real filing delay so the backtest never reacts to a report before it could
    plausibly have been published.
    """
    available = pd.Timestamp(period_end) + pd.DateOffset(months=3 * C.FUND_LAG_QUARTERS)
    roe = ttm_ni / eq if eq and eq != 0 and pd.notna(ttm_ni) else float("nan")
    de = dbt / eq if eq and eq != 0 and pd.notna(dbt) else float("nan")
    margin = ttm_ni / ttm_rev if ttm_rev and ttm_rev != 0 and pd.notna(ttm_ni) else float("nan")
    return {"available": available, "ttm_net_income": ttm_ni, "ttm_revenue": ttm_rev,
            "equity": eq, "total_debt": dbt, "roe": roe, "de": de, "margin": margin}


def _quarterly_records(inc, bal) -> list[dict]:
    ex = _extract(inc, bal)
    if ex is None:
        return []
    net, rev, equity, debt = ex
    cols = sorted(set(net.dropna().index) & set(equity.dropna().index))
    if len(cols) < 4:  # need 4 quarters to form a TTM figure
        return []
    net_s, eq_s = net.reindex(cols), equity.reindex(cols)
    rev_s = rev.reindex(cols) if rev is not None else pd.Series(index=cols, dtype=float)
    debt_s = debt.reindex(cols) if debt is not None else pd.Series(index=cols, dtype=float)
    recs = []
    for i in range(3, len(cols)):  # 4 trailing quarters => TTM
        window = cols[i - 3:i + 1]
        recs.append(_record(cols[i], net_s.loc[window].sum(min_count=4),
                            rev_s.loc[window].sum(min_count=1),
                            eq_s.loc[cols[i]], debt_s.loc[cols[i]]))
    return recs


def _annual_records(inc, bal) -> list[dict]:
    ex = _extract(inc, bal)
    if ex is None:
        return []
    net, rev, equity, debt = ex
    cols = sorted(set(net.dropna().index) & set(equity.dropna().index))
    recs = []
    for c in cols:  # an annual net income already spans 12 months (TTM-equivalent)
        recs.append(_record(c, net.get(c, float("nan")),
                            rev.get(c, float("nan")) if rev is not None else float("nan"),
                            equity.get(c, float("nan")),
                            debt.get(c, float("nan")) if debt is not None else float("nan")))
    return recs


def build_fundamentals_timeline(ticker: str, force: bool = False) -> pd.DataFrame:
    """Point-in-time fundamentals timeline merging annual + quarterly reports.

    Returns a DataFrame indexed by ``available`` (the earliest date the data may
    be used) with columns: ttm_net_income, ttm_revenue, equity, total_debt, roe,
    de, margin. Empty if the ticker lacks usable data.

    Annual statements (~4yr of history) form the backbone; quarterly statements
    (~5 recent quarters) add recency. Where both land on the same availability
    date, the quarterly (more granular TTM) record wins.
    """
    raw = data.get_fundamentals(ticker, force=force)
    annual = _annual_records(raw["income_a"], raw["balance_a"])
    quarterly = _quarterly_records(raw["income_q"], raw["balance_q"])
    if not annual and not quarterly:
        return pd.DataFrame()

    # Merge: dict keyed by availability date; quarterly overrides annual.
    merged: dict = {}
    for r in annual:
        merged[r["available"]] = r
    for r in quarterly:
        merged[r["available"]] = r

    out = pd.DataFrame(sorted(merged.values(), key=lambda r: r["available"]))
    return out.set_index("available").sort_index()


def fundamentals_asof(timeline: pd.DataFrame, asof: pd.Timestamp) -> dict | None:
    """Most recent fundamentals row whose availability date <= ``asof``.

    Returns None if nothing is available yet (=> exclude the name this month).
    No forward-fill of stale numbers beyond selecting the latest *published*
    report — that is correct point-in-time behaviour, not lookahead.
    """
    if timeline.empty:
        return None
    eligible = timeline[timeline.index <= asof]
    if eligible.empty:
        return None
    return eligible.iloc[-1].to_dict()
