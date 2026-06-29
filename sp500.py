"""Free point-in-time S&P 500 universe: historical index membership + sector map.

Membership comes from the MIT-licensed fja05680/sp500 dataset — a survivorship-
bias-free record of S&P 500 constituents on every change-date back to 1996. We
as-of join it so each backtest month only trades names that were ACTUALLY in the
index that month (dropped/delisted names included under their then-current
symbol). This replaces the survivorship-biased "current ETF top-10 holdings"
universe.

Sector tags (needed for the strategy's Stage-1 sector-momentum ranking) are
pulled from yfinance once per ticker and cached, then mapped to the 11 SPDR
sector ETFs. No paid data source anywhere in this module.

Caveats inherited from the source: community-maintained (not an official S&P DJI
feed, occasional symbol/date errors), and yfinance often lacks fully delisted
names so some historical members will have no price data (handled gracefully:
the data layer just drops names it can't fetch).
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import requests

import config as C

# fja05680/sp500 "S&P 500 Historical Components & Changes (Updated).csv".
# URL is percent-encoded (spaces -> %20, & -> %26). Confirmed HTTP 200 (2026-06).
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
_LOCAL_CSV = C.CACHE_DIR / "sp500_historical.csv"
_SECTOR_CACHE = C.CACHE_DIR / "sp500_sectors.json"
_UA = {"User-Agent": "tradingBot research (ghoshsanjoy@gmail.com)"}

# GICS (Wikipedia) AND yfinance sector strings -> SPDR sector ETF. Both naming
# styles are included so the map works regardless of which source named a stock.
SECTOR_TO_ETF: dict[str, str] = {
    # GICS names
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
    # yfinance names
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Basic Materials": "XLB",
}


def _to_yahoo(ticker: str) -> str:
    """Class-share symbols use dots in the dataset (BRK.B); yfinance wants dashes."""
    return ticker.strip().replace(".", "-")


# --- membership -------------------------------------------------------------

class Membership:
    """Point-in-time S&P 500 membership, queryable as-of any date."""

    def __init__(self, changes: pd.DataFrame):
        # changes: sorted, columns date (Timestamp) + members (frozenset).
        self._df = changes
        self._dates = changes["date"].to_numpy()

    def asof(self, date) -> set[str]:
        """Set of tickers in the index as of ``date`` (most recent snapshot <= date)."""
        ts = pd.Timestamp(date)
        # rightmost snapshot whose date <= ts
        idx = self._df["date"].searchsorted(ts, side="right") - 1
        if idx < 0:
            return set()
        return set(self._df["members"].iloc[idx])

    def union(self, start, end) -> set[str]:
        """Every ticker that was a member at any snapshot in [start, end], plus the
        snapshot active at ``start`` (so names present-then-dropped are included)."""
        ts0, ts1 = pd.Timestamp(start), pd.Timestamp(end)
        names = set(self.asof(ts0))
        mask = (self._df["date"] >= ts0) & (self._df["date"] <= ts1)
        for ms in self._df.loc[mask, "members"]:
            names |= set(ms)
        return names


def load_membership(force: bool = False) -> Membership:
    """Download (and cache) the historical-components CSV; return a Membership."""
    if _LOCAL_CSV.exists() and not force:
        df = pd.read_csv(_LOCAL_CSV, parse_dates=["date"])
    else:
        resp = requests.get(SP500_CSV_URL, headers=_UA, timeout=60)
        resp.raise_for_status()
        _LOCAL_CSV.write_bytes(resp.content)
        df = pd.read_csv(_LOCAL_CSV, parse_dates=["date"])

    df = df.sort_values("date").reset_index(drop=True)
    df["members"] = df["tickers"].apply(
        lambda s: frozenset(_to_yahoo(t) for t in str(s).split(",") if t.strip()))
    return Membership(df[["date", "members"]])


# --- sector tags ------------------------------------------------------------

def sector_map(tickers: list[str], force: bool = False,
               progress=None) -> dict[str, str]:
    """{ticker: SPDR sector ETF} via yfinance ``.info['sector']``, cached to JSON.

    Tickers whose sector is unknown / unmapped are omitted (they simply become
    non-selectable, which is safe — never guessed into a sector).
    """
    cache: dict[str, str] = {}
    if _SECTOR_CACHE.exists() and not force:
        try:
            cache = json.loads(_SECTOR_CACHE.read_text())
        except json.JSONDecodeError:
            cache = {}

    missing = [t for t in tickers if t not in cache]
    if missing:
        import yfinance as yf
        for i, t in enumerate(missing):
            sector = ""
            try:
                sector = yf.Ticker(t).info.get("sector") or ""
            except Exception:  # noqa: BLE001 - yfinance raises many shapes
                sector = ""
            cache[t] = sector
            if progress is not None:
                progress(i + 1, len(missing))
            elif (i + 1) % 25 == 0:
                print(f"[sp500] sectors {i + 1}/{len(missing)}…")
        _SECTOR_CACHE.write_text(json.dumps(cache, indent=0))

    out = {}
    for t in tickers:
        etf = SECTOR_TO_ETF.get(cache.get(t, ""))
        if etf:
            out[t] = etf
    return out


# --- assembled universe -----------------------------------------------------

def build_universe(years: int, force: bool = False,
                   sector_progress=None) -> tuple[dict[str, list[str]], Membership]:
    """Assemble the point-in-time S&P 500 universe over a ``years``-year window.

    Returns ({etf: [member tickers in that sector]}, Membership). The dict feeds
    backtest.build_samples as ``universe_override``; the Membership is threaded
    into the backtest so selection at month t is restricted to that month's
    actual index members (true point-in-time, no survivorship bias).

    The window matches backtest.build_samples: end=today, start reaches back
    ``years + 1`` years (the +1 is the model/momentum warmup).
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * (years + 1)))

    members = load_membership(force=force)
    union = sorted(members.union(start, end))
    sectors = sector_map(union, force=force, progress=sector_progress)

    uni: dict[str, list[str]] = {etf: [] for etf in C.SECTOR_ETFS}
    for ticker, etf in sectors.items():
        uni[etf].append(ticker)
    for etf in uni:
        uni[etf].sort()
    return uni, members
