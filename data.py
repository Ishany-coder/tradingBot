"""Data layer: yfinance fetch + on-disk cache.

yfinance is slow and rate-limited, so every network pull is cached to disk
(parquet for price panels, pickle for the nested fundamentals structure).
Delete the files in ``data/cache`` or pass ``force=True`` to refresh.

Nothing here looks into the future; this module only *fetches*. Point-in-time
discipline (lagging fundamentals) lives in ``features.py``.
"""

from __future__ import annotations

import pickle
import warnings

import pandas as pd
import yfinance as yf

import config as C

warnings.filterwarnings("ignore", category=FutureWarning)

# yfinance keeps a sqlite cache for timezones / cookies. Its default location
# (a user cache dir) can be unwritable or corrupt, producing errors like
# "unable to open database file" / "database is locked". Point it at our own
# writable cache dir so downloads are reliable.
_YF_TZ_CACHE = C.CACHE_DIR / "yf_tz"
_YF_TZ_CACHE.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(_YF_TZ_CACHE))
except Exception:  # noqa: BLE001 - older yfinance lacks this; harmless
    pass


# --- price panel ------------------------------------------------------------

def get_prices(tickers: list[str], start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Daily adjusted-close panel (columns = tickers, index = dates).

    Cached by the sorted ticker set + date range so different universes don't
    collide. Auto-adjusted so splits/dividends are already baked in.
    """
    key = f"prices_{abs(hash((tuple(sorted(tickers)), start, end)))}.parquet"
    path = C.CACHE_DIR / key
    if path.exists() and not force:
        return pd.read_parquet(path)

    # threads=False serialises access to yfinance's sqlite tz cache, avoiding
    # "database is locked" when many tickers download at once.
    close = _download_close(tickers, start, end)

    # Retry any tickers that came back empty, one at a time. Transient cache /
    # rate-limit failures usually clear on a serial retry.
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if missing:
        print(f"[data] retrying {len(missing)} tickers individually: {missing[:8]}…")
        for t in missing:
            single = _download_close([t], start, end)
            if t in single.columns and not single[t].dropna().empty:
                close[t] = single[t]

    still_missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if still_missing:
        print(f"[data] no price data after retry (genuinely unavailable): {still_missing}")

    close = close.dropna(how="all").sort_index()
    close.to_parquet(path)
    return close


def _download_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Single yfinance call -> Close panel (columns = tickers). Robust to shapes."""
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, group_by="column", threads=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[data] download error for {tickers[:4]}…: {exc}")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].copy()
    close = raw[["Close"]].copy()
    close.columns = tickers[:1]
    return close


# --- ETF constituents -------------------------------------------------------

def get_etf_holdings(etf: str, force: bool = False) -> list[str]:
    """Current top-N holdings (tickers) of one sector ETF.

    yfinance only exposes the *current* top holdings, not historical
    membership, so the backtested universe carries survivorship bias. This is
    documented in the dashboard.
    """
    path = C.CACHE_DIR / f"holdings_{etf}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    holdings: list[str] = []
    try:
        fd = yf.Ticker(etf).funds_data
        top = fd.top_holdings  # DataFrame indexed by symbol
        if top is not None and len(top):
            holdings = [str(s).upper() for s in top.index[: C.TOP_HOLDINGS_N]]
    except Exception as exc:  # noqa: BLE001 - yfinance raises many shapes
        print(f"[data] holdings fetch failed for {etf}: {exc}")

    # Drop obviously non-equity / malformed symbols.
    holdings = [h for h in holdings if h.isalpha() and 1 <= len(h) <= 5]
    with open(path, "wb") as fh:
        pickle.dump(holdings, fh)
    return holdings


# --- quarterly fundamentals -------------------------------------------------

def get_fundamentals(ticker: str, force: bool = False) -> dict[str, pd.DataFrame]:
    """Raw income statement + balance sheet for one ticker, quarterly AND annual.

    Returns ``{"income_q","balance_q","income_a","balance_a"}`` with the
    yfinance shape (rows = line items, columns = fiscal period-end dates).

    Why both: yfinance now serves only ~5 quarters of quarterly statements, far
    short of a 5-year backtest. Annual statements still go back ~4 years, so we
    merge them — annual provides the multi-year backbone, quarterly the recent
    granularity. Empty frames on failure; callers treat missing data as
    "exclude this name", never forward-fill (that would be a mild lookahead).
    """
    path = C.CACHE_DIR / f"fund_{ticker}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    out = {k: pd.DataFrame() for k in ("income_q", "balance_q", "income_a", "balance_a")}
    try:
        tk = yf.Ticker(ticker)
        for key, attr in (("income_q", "quarterly_income_stmt"),
                          ("balance_q", "quarterly_balance_sheet"),
                          ("income_a", "income_stmt"),
                          ("balance_a", "balance_sheet")):
            df = getattr(tk, attr)
            if df is not None:
                out[key] = df
    except Exception as exc:  # noqa: BLE001
        print(f"[data] fundamentals fetch failed for {ticker}: {exc}")

    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def get_current_fundamentals(ticker: str, force: bool = False) -> dict[str, float]:
    """Current fundamentals snapshot for the LIVE holdings view.

    No lookahead risk in the present — we genuinely have today's numbers today.
    Used only by the dashboard's current-holdings panel, never by the backtest.
    """
    path = C.CACHE_DIR / f"snap_{ticker}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    snap = {"roe": float("nan"), "de": float("nan"), "margin": float("nan"),
            "net_income": float("nan")}
    try:
        info = yf.Ticker(ticker).info
        snap["roe"] = _f(info.get("returnOnEquity"))
        snap["de"] = _f(info.get("debtToEquity"))
        snap["margin"] = _f(info.get("profitMargins"))
        snap["net_income"] = _f(info.get("netIncomeToCommon"))
    except Exception as exc:  # noqa: BLE001
        print(f"[data] snapshot fetch failed for {ticker}: {exc}")

    with open(path, "wb") as fh:
        pickle.dump(snap, fh)
    return snap


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")
