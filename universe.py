"""Universe construction: sector ETFs -> their constituent stocks.

Builds the map {ETF: [stock tickers]} from current top holdings and the
reverse {stock: sector ETF} lookup the backtest uses to bucket names.
"""

from __future__ import annotations

import json

import config as C
import data


def build_universe(force: bool = False) -> dict[str, list[str]]:
    """Return {etf: [constituent tickers]} for all 11 sector ETFs.

    Cached to JSON so we don't re-hit yfinance for holdings every run.
    """
    path = C.CACHE_DIR / "universe.json"
    if path.exists() and not force:
        return json.loads(path.read_text())

    uni: dict[str, list[str]] = {}
    for etf in C.SECTOR_ETFS:
        uni[etf] = data.get_etf_holdings(etf, force=force)
        print(f"[universe] {etf}: {len(uni[etf])} holdings")
    path.write_text(json.dumps(uni, indent=2))
    return uni


def stock_to_sector(universe: dict[str, list[str]]) -> dict[str, str]:
    """Reverse map {stock ticker: owning ETF}.

    If a stock appears in more than one ETF's top holdings, the first ETF in
    config order wins (deterministic).
    """
    rev: dict[str, str] = {}
    for etf in C.SECTOR_ETFS:  # iterate in config order for determinism
        for stk in universe.get(etf, []):
            rev.setdefault(stk, etf)
    return rev


def all_tickers(universe: dict[str, list[str]]) -> list[str]:
    """Every ticker we need prices for: ETFs + constituents + benchmark."""
    tickers = set(C.SECTOR_ETFS) | {C.BENCHMARK}
    for stocks in universe.values():
        tickers.update(stocks)
    return sorted(tickers)
