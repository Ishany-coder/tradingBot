"""Free point-in-time fundamentals from SEC EDGAR's XBRL companyfacts API.

Why EDGAR over yfinance for fundamentals: every XBRL datapoint carries a
``filed`` date — the date the value actually became public — so we can build a
TRUE point-in-time timeline (a value is only usable once its filing date has
passed). yfinance only exposes fiscal period-end dates and ~5 quarters of
history; EDGAR is free, complete back to the 2009 XBRL phase-in, and exact.

This module produces a timeline with the SAME shape as
``features.build_fundamentals_timeline`` (indexed by an availability date with
roe/de/margin/ttm_net_income columns) so ``features.fundamentals_asof`` consumes
it unchanged. The availability date here IS the SEC filing date, so no synthetic
FUND_LAG_QUARTERS is applied — the lag is real.

Verified live against Apple (CIK 0000320193) on 2026-06-29.

Compliance: SEC requires a descriptive User-Agent with a contact email (blank UA
=> HTTP 403) and limits to 10 requests/second. We fetch one bulk companyfacts
file per ticker, sleep between calls, and cache the parsed timeline to disk.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import time

import pandas as pd
import requests

import config as C

_UA = {"User-Agent": "tradingBot research (ghoshsanjoy@gmail.com)",
       "Accept-Encoding": "gzip, deflate"}
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_CIK_CACHE = C.CACHE_DIR / "sec_cik_map.json"
_REQUEST_GAP_S = 0.12  # stay under the 10 req/s fair-access limit

# US-GAAP tags, in fallback order (verified against Apple). Duration (flow)
# concepts have a 'start'; instant (stock) concepts do not.
_NI_TAGS = ["NetIncomeLoss"]
_REV_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax",
             "Revenues", "SalesRevenueNet"]
_EQ_TAGS = ["StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
_LIAB_TAGS = ["Liabilities"]
_DEBT_TAGS = ["LongTermDebt", "LongTermDebtNoncurrent"]


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


# --- ticker -> CIK ----------------------------------------------------------

def _load_cik_map(force: bool = False) -> dict[str, str]:
    if _CIK_CACHE.exists() and not force:
        return json.loads(_CIK_CACHE.read_text())
    j = requests.get(_TICKER_MAP_URL, headers=_UA, timeout=30).json()
    cmap = {r["ticker"].upper(): str(r["cik_str"]).zfill(10) for r in j.values()}
    _CIK_CACHE.write_text(json.dumps(cmap))
    return cmap


def ticker_to_cik(ticker: str, force: bool = False) -> str | None:
    # EDGAR uses dashes for class shares (BRK-B) like yfinance; query upper-case.
    return _load_cik_map(force=force).get(ticker.upper())


# --- raw companyfacts -> event stream ---------------------------------------

def _rows(facts: dict, tags: list[str], want_duration: bool,
          merge: bool = False) -> list[dict]:
    """USD rows for the given tags (``prio`` = index of the tag they came from).

    want_duration=True keeps flow rows (have 'start'); False keeps instant rows.
    merge=False: first non-empty tag wins (correct for true alternates like the
    two equity tags). merge=True: union ALL tags — needed for revenue, where the
    ASC-606 tag only covers ~2018+ and legacy tags carry the earlier years; the
    ``prio`` tiebreak then prefers the earlier-listed (preferred) tag per period.
    """
    g = facts.get("facts", {}).get("us-gaap", {})
    collected: list[dict] = []
    for prio, t in enumerate(tags):
        if t not in g:
            continue
        out = []
        for unit, rows in g[t].get("units", {}).items():
            if unit != "USD":
                continue
            for r in rows:
                has_start = "start" in r
                if want_duration and not has_start:
                    continue
                if not want_duration and has_start:
                    continue
                rec = {"filed": r["filed"], "end": r["end"], "val": r["val"],
                       "fy": r.get("fy"), "fp": r.get("fp"), "prio": prio}
                if want_duration:
                    rec["dur"] = (_d(r["end"]) - _d(r["start"])).days
                out.append(rec)
        if out and not merge:
            return out
        collected.extend(out)
    return collected


def _latest_annual(rows: list[dict], asof: dt.date) -> float | None:
    """Latest FULL-YEAR (10-K, ~365d duration) value known by ``asof``.

    A 10-K's annual figure is an unambiguous trailing-twelve-month number — no
    quarter stitching needed. We deliberately avoid reconstructing quarters /
    Q4 (single-quarter filers like AAPL omit a standalone Q4; YTD-cumulative
    filers like JPM break a naive trailing-4 sum — both corrupt TTM). De-dupes
    restatements: latest fiscal period-end wins, latest ``filed`` breaks ties.
    """
    elig = [r for r in rows if _d(r["filed"]) <= asof and (r.get("dur") or 0) >= 350]
    if not elig:
        return None
    # latest period-end; latest restatement; preferred tag (lowest prio) breaks ties.
    elig.sort(key=lambda r: (r["end"], r["filed"], -r.get("prio", 0)))
    return float(elig[-1]["val"])


def _latest_instant(rows: list[dict], asof: dt.date) -> float | None:
    """Newest period-end instant (balance-sheet) value known by ``asof``.

    Balance-sheet items update every 10-Q, so equity/debt stay current even
    though net income only refreshes annually. Latest restatement wins on ties.
    """
    elig = [r for r in rows if _d(r["filed"]) <= asof]
    if not elig:
        return None
    elig.sort(key=lambda r: (r["end"], r["filed"]))
    return float(elig[-1]["val"])


def _snapshot(streams: dict, asof: dt.date) -> dict | None:
    """Compute roe/de/margin from the event streams as known at ``asof``.

    Net income & revenue: latest annual (10-K). Equity & debt: latest quarterly
    balance sheet. ROE = annual NI / current equity (numerator annual, denom
    current — the standard, robust convention).
    """
    ni = _latest_annual(streams["net_income"], asof)
    rev = _latest_annual(streams["revenue"], asof)
    equity = _latest_instant(streams["equity"], asof)
    liab = _latest_instant(streams["liabilities"], asof)
    ltd = _latest_instant(streams["long_term_debt"], asof)
    debt = ltd if ltd is not None else liab  # strict debt, else total liabilities

    # Non-positive equity (None / zero / NEGATIVE) makes roe & de sign-flip into
    # nonsense (common for heavy-buyback names). Treat as missing, not a value.
    if ni is None or equity is None or equity <= 0:
        return None
    return {
        "ttm_net_income": ni,           # full-year; name kept for schema parity
        "ttm_revenue": rev if rev is not None else float("nan"),
        "equity": equity,
        "total_debt": debt if debt is not None else float("nan"),
        "roe": ni / equity,
        "de": (debt / equity) if debt is not None else float("nan"),
        "margin": (ni / rev) if (rev not in (None, 0)) else float("nan"),
    }


def build_timeline(ticker: str, force: bool = False) -> pd.DataFrame:
    """Point-in-time fundamentals timeline for ``ticker`` from SEC EDGAR.

    Indexed by ``available`` (= the SEC filing date) with columns roe, de,
    margin, ttm_net_income, ttm_revenue, equity, total_debt — the same contract
    as ``features.build_fundamentals_timeline``, so ``features.fundamentals_asof``
    works on it unchanged. Empty DataFrame if the ticker has no usable XBRL data.
    """
    cache = C.CACHE_DIR / f"edgar_{ticker}.pkl"
    if cache.exists() and not force:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    out = pd.DataFrame()
    ok = False
    try:
        cik = ticker_to_cik(ticker, force=force)  # inside try: a CIK-map network
        if cik is not None:                        # blip must not abort the whole run
            time.sleep(_REQUEST_GAP_S)
            facts = requests.get(_FACTS_URL.format(cik=cik), headers=_UA,
                                 timeout=60).json()
            streams = {
                "net_income": _rows(facts, _NI_TAGS, True),
                "revenue": _rows(facts, _REV_TAGS, True, merge=True),
                "equity": _rows(facts, _EQ_TAGS, False),
                "liabilities": _rows(facts, _LIAB_TAGS, False),
                "long_term_debt": _rows(facts, _DEBT_TAGS, False),
            }
            # Evaluate a snapshot at each distinct filing date => PIT timeline.
            filed_dates = sorted({r["filed"] for rows in streams.values()
                                  for r in rows})
            recs = {}
            for fd in filed_dates:
                snap = _snapshot(streams, _d(fd))
                if snap is not None:
                    snap["available"] = pd.Timestamp(fd)
                    recs[pd.Timestamp(fd)] = snap
            if recs:
                out = (pd.DataFrame(sorted(recs.values(),
                                           key=lambda r: r["available"]))
                       .set_index("available").sort_index())
        ok = True  # reached here cleanly (cik=None is a legitimate empty)
    except Exception as exc:  # noqa: BLE001 - network/JSON/shape failures
        print(f"[edgar] fundamentals fetch failed for {ticker}: {exc}")

    # Cache only on success — never persist an empty result produced by a
    # transient failure (that would permanently drop the ticker until force=True).
    if ok:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out
