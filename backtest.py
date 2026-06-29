"""Backtest engine: builds the sample matrix, runs the walk-forward model,
then simulates the monthly two-stage strategy with transaction costs.

Runs two variants on the same data:
  * A (baseline)  : price-only model, NO quality screen.
  * B (full)      : quality screen + model on all seven features.

Simulation only. No broker, no real orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import datetime as dt
import numpy as np
import pandas as pd

import config as C
import data
import edgar
import features as F
import metrics
import model
import montecarlo
import sizing
import universe as U

# Base features, cross-sectionally z-scored per month (``_z`` columns built in
# build_samples). Z-scoring puts every name on the same per-month scale, which
# is what a cross-sectional ranker should compare — raw momentum/fundamentals
# live on wildly different scales and drift over time.
PRICE_FEATURES = ["mom_12_1_z", "mom_6_z", "mom_3_z", "vol_z"]
ALL_FEATURES = PRICE_FEATURES + ["roe_z", "de_z", "margin_z"]

# Raw columns that get a per-month cross-sectional z-score sibling (``_z``).
_Z_BASE = ["mom_12_1", "mom_6", "mom_3", "vol", "roe", "de", "margin"]


@dataclass
class BacktestResult:
    label: str
    monthly_returns: pd.Series          # indexed by decision month-end
    equity: pd.Series
    holdings_history: dict               # {date: {ticker: conviction weight}}
    pred_history: dict                   # {date: {ticker: predicted return}}
    conf_history: dict                   # {date: {ticker: confidence prob}}
    sector_history: dict                 # {date: [ranked etf list]}
    summary: dict
    mc: dict
    drop_log: dict = field(default_factory=dict)   # {date: n_excluded} (B only)
    ic: dict = field(default_factory=dict)         # signal quality (see metrics)


@dataclass
class Bundle:
    """Everything the dashboard needs, computed once."""
    universe: dict
    stock_sector: dict
    mprices: pd.DataFrame
    sector_mom: pd.DataFrame
    fwd: pd.DataFrame
    samples: pd.DataFrame
    result_a: BacktestResult
    result_b: BacktestResult
    spy_returns: pd.Series


# --- sample matrix ----------------------------------------------------------

def build_samples(force: bool = False, universe_override: dict | None = None,
                  years: int | None = None, fundamentals_source: str = "yfinance"):
    """Assemble the (date, ticker) sample matrix and supporting panels.

    universe_override : use this {etf: [tickers]} map instead of the live
        current-holdings universe (e.g. a point-in-time 2020 snapshot, or the
        S&P 500 point-in-time universe).
    years : history window in years (defaults to config.BACKTEST_YEARS).
    fundamentals_source : "yfinance" (default) or "edgar" — the latter uses SEC
        EDGAR for true point-in-time fundamentals (filing-dated, no synthetic
        lag); see edgar.py.

    Returns (samples, mprices, sector_mom, fwd, universe, stock_sector).
    """
    uni = universe_override if universe_override is not None else U.build_universe(force=force)
    stock_sector = U.stock_to_sector(uni)
    tickers = U.all_tickers(uni)
    stocks = sorted(stock_sector)

    yrs = years if years is not None else C.BACKTEST_YEARS
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * (yrs + 1)))  # +1yr warmup
    daily = data.get_prices(tickers, start.isoformat(), end.isoformat(), force=force)

    mprices = F.month_end_prices(daily)
    moms = F.momentum_panels(mprices)
    vol = F.volatility_panel(daily, mprices.index)
    fwd = F.forward_return_panel(mprices)
    sector_mom = F.sector_momentum(mprices, list(C.SECTOR_ETFS))

    # Point-in-time fundamentals timeline per stock (built once). EDGAR gives
    # filing-dated, multi-year fundamentals; yfinance is the thin fallback.
    if fundamentals_source == "edgar":
        timelines = {s: edgar.build_timeline(s, force=force) for s in stocks}
    else:
        timelines = {s: F.build_fundamentals_timeline(s, force=force) for s in stocks}

    # Long-format sample matrix over (month, stock).
    records = []
    for stk in stocks:
        if stk not in mprices.columns:
            continue
        tl = timelines[stk]
        for date in mprices.index:
            row = {
                "date": date,
                "ticker": stk,
                "mom_12_1": _at(moms["mom_12_1"], date, stk),
                "mom_6": _at(moms["mom_6"], date, stk),
                "mom_3": _at(moms["mom_3"], date, stk),
                "vol": _at(vol, date, stk),
                "target": _at(fwd, date, stk),
                "roe": float("nan"),
                "de": float("nan"),
                "margin": float("nan"),
                "net_income": float("nan"),
            }
            f = F.fundamentals_asof(tl, date)
            if f is not None:
                row["roe"] = f["roe"]
                row["de"] = f["de"]
                row["margin"] = f["margin"]
                row["net_income"] = f["ttm_net_income"]
            records.append(row)

    samples = pd.DataFrame(records).set_index(["date", "ticker"]).sort_index()

    # Cross-sectional z-score of each base feature within its month: puts all
    # names on a comparable per-month scale for the model. NaN base => NaN z
    # (those rows are dropped by the model's dropna, same as before).
    for col in _Z_BASE:
        g = samples.groupby(level=0)[col]
        z = (samples[col] - g.transform("mean")) / g.transform("std")
        # Clip to ±4 SD so a single outlier (e.g. a tiny-equity ROE blow-up or a
        # momentum spike) can't dominate the scale-sensitive MLP. Tree models are
        # monotone-invariant within the clip, so this only helps.
        samples[col + "_z"] = z.replace([np.inf, -np.inf], np.nan).clip(-4, 4)

    return samples, mprices, sector_mom, fwd, uni, stock_sector


def _at(panel: pd.DataFrame, date, col):
    try:
        return float(panel.at[date, col])
    except (KeyError, ValueError):
        return float("nan")


# --- selection + simulation -------------------------------------------------

def _top_sectors(sector_mom: pd.DataFrame, date) -> list[str]:
    """Stage 1: ranked sector ETFs by 12-1 momentum at ``date`` (best first)."""
    row = sector_mom.loc[date].dropna()
    return list(row.sort_values(ascending=False).index)


def _select(date, preds_at_t, samples, stock_sector, top_sectors, use_screen,
            members=None):
    """Stage 2: select the book (by edge) and size it (by conviction).

    ``preds_at_t`` is a DataFrame indexed by ticker with ``pred`` (edge) and
    ``confidence`` columns. Selection ranks by ``pred``; sizing uses
    confidence / volatility. Returns (weights dict, n_dropped_by_screen).

    ``members`` (set or None): if given, only names that were index members at
    ``date`` are eligible — point-in-time membership gating that removes
    survivorship bias (a name added to the index later can't be picked earlier).
    """
    chosen = set(top_sectors[: C.N_SECTORS])
    # Candidates = predicted names whose owning sector is in the top 3 (and, if
    # membership is enforced, that were actually in the index this month).
    cands = [t for t in preds_at_t.index
             if stock_sector.get(t) in chosen
             and (members is None or t in members)]
    dropped = 0

    if use_screen:
        kept = []
        # Sector-relative D/E median uses only point-in-time candidate data.
        de_by_sector: dict[str, list[float]] = {}
        for t in cands:
            de = samples.at[(date, t), "de"]
            if pd.notna(de):
                de_by_sector.setdefault(stock_sector[t], []).append(de)
        medians = {s: pd.Series(v).median() for s, v in de_by_sector.items()}

        for t in cands:
            row = samples.loc[(date, t)]
            ni, roe, de = row["net_income"], row["roe"], row["de"]
            sec = stock_sector[t]
            # Missing fundamentals => exclude (never guess / forward-fill).
            if pd.isna(ni) or pd.isna(roe) or pd.isna(de) or sec not in medians:
                dropped += 1
                continue
            if ni > 0 and roe > 0 and de < medians[sec]:
                kept.append(t)
            else:
                dropped += 1
        cands = kept

    if not cands:
        return {}, dropped

    # Selection: top N by predicted return (edge).
    ranked = preds_at_t.loc[cands, "pred"].sort_values(ascending=False)
    book = list(ranked.index[: C.N_STOCKS_MAX])

    # Sizing: conviction = confidence / volatility, capped + normalised.
    candidates = {
        t: {"confidence": float(preds_at_t.at[t, "confidence"]),
            "vol": _at_scalar(samples, date, t, "vol")}
        for t in book
    }
    return sizing.conviction_weights(candidates), dropped


def _at_scalar(samples, date, ticker, col):
    try:
        return float(samples.at[(date, ticker), col])
    except (KeyError, ValueError):
        return float("nan")


def run_variant(label, samples, feature_cols, sector_mom, fwd, stock_sector,
                use_screen, method="gbm", seed=None, with_mc=True,
                membership=None) -> BacktestResult:
    """Walk-forward predict, then simulate monthly rebalancing.

    method  : model method "gbm" | "lambdarank" | "mlp" (see model.py).
    seed    : override the model ``random_state`` (re-train robustness loop).
    with_mc : run the Monte-Carlo drawdown reshuffle (``MC_RUNS`` permutations).
        Set False when calling this in a tight loop to skip that cost.
    membership : optional object with ``.asof(date) -> set`` for point-in-time
        index-membership gating in selection (S&P 500 PIT universe).
    """
    preds = model.walk_forward_predict(samples, feature_cols, method=method, seed=seed)
    if preds.empty:
        # Data too thin to ever reach MIN_TRAIN_MONTHS (typically Backtest B
        # when yfinance fundamentals don't cover enough history). Return an
        # empty result so the other variant + dashboard still work.
        print(f"[backtest] {label}: no predictions (data too thin); empty result.")
        return _empty_result(label)

    decision_months = sorted(preds.index.get_level_values(0).unique())

    monthly_ret = {}
    holdings_hist, pred_hist, conf_hist, sector_hist, drop_log = {}, {}, {}, {}, {}
    prev_w: dict[str, float] = {}

    for t in decision_months:
        if t not in sector_mom.index:
            continue
        top_sectors = _top_sectors(sector_mom, t)
        sector_hist[t] = top_sectors[: C.N_SECTORS]

        preds_at_t = preds.xs(t, level=0)  # DataFrame: pred, confidence
        members = membership.asof(t) if membership is not None else None
        weights, dropped = _select(t, preds_at_t, samples, stock_sector,
                                   top_sectors, use_screen, members=members)
        if use_screen:
            drop_log[t] = dropped
        if not weights:
            prev_w = {}
            continue

        holdings_hist[t] = weights
        pred_hist[t] = {k: float(preds_at_t.at[k, "pred"]) for k in weights}
        conf_hist[t] = {k: float(preds_at_t.at[k, "confidence"]) for k in weights}

        # Transaction cost on turnover vs the prior book (buys + sells).
        turnover = sum(abs(weights.get(k, 0.0) - prev_w.get(k, 0.0))
                       for k in set(weights) | set(prev_w))
        cost = turnover * C.COST_PER_TRADE

        # Realised CONVICTION-WEIGHTED return over t -> t+1. Drop names whose
        # forward return is missing and renormalise over the rest. NaN in the
        # final month => skip (can't realise).
        avail = {k: w for k, w in weights.items()
                 if k in fwd.columns and pd.notna(fwd.at[t, k])}
        if not avail:
            prev_w = weights
            continue
        wsum = sum(avail.values())
        gross = sum((w / wsum) * fwd.at[t, k] for k, w in avail.items())
        monthly_ret[t] = gross - cost
        prev_w = weights

    # Raw signal quality (rank IC of predictions vs realised returns), computed
    # over every scored name — independent of selection/sizing/costs.
    ic = metrics.information_coefficient(preds, fwd)

    mr = pd.Series(monthly_ret).sort_index()
    if mr.empty:
        print(f"[backtest] {label}: no realised monthly returns; empty result.")
        return _empty_result(label, holdings_hist, pred_hist, conf_hist,
                             sector_hist, drop_log, ic)
    summary = metrics.summarize(mr)
    mc = (montecarlo.reshuffle_drawdowns(mr) if with_mc
          else {"p50_drawdown": float("nan"), "p95_drawdown": float("nan")})

    return BacktestResult(
        label=label,
        monthly_returns=mr,
        equity=metrics.equity_curve(mr),
        holdings_history=holdings_hist,
        pred_history=pred_hist,
        conf_history=conf_hist,
        sector_history=sector_hist,
        summary=summary,
        mc=mc,
        drop_log=drop_log,
        ic=ic,
    )


def current_book(result: BacktestResult):
    """Latest decision-date target book: (date, {ticker: {weight, pred, confidence}}).

    This is what the live trader trades toward and the dashboard displays.
    Returns (None, {}) if the variant produced no book.
    """
    if not result.holdings_history:
        return None, {}
    d = max(result.holdings_history)
    w = result.holdings_history[d]
    p = result.pred_history.get(d, {})
    c = result.conf_history.get(d, {})
    book = {t: {"weight": w[t], "pred": p.get(t, float("nan")),
                "confidence": c.get(t, float("nan"))} for t in w}
    return d, book


def _empty_result(label, holdings=None, preds=None, confs=None, sectors=None,
                  drops=None, ic=None) -> BacktestResult:
    """A result carrying no realised returns (variant was data-starved).

    Any holdings/predictions discovered before returns ran out are kept so the
    dashboard can still show the most recent book if one exists.
    """
    empty = pd.Series(dtype=float)
    nan_summary = {"total_return": float("nan"), "cagr": float("nan"),
                   "max_drawdown": float("nan"), "sharpe": float("nan")}
    return BacktestResult(
        label=label, monthly_returns=empty, equity=empty,
        holdings_history=holdings or {}, pred_history=preds or {},
        conf_history=confs or {}, sector_history=sectors or {},
        summary=nan_summary,
        mc={"p50_drawdown": float("nan"), "p95_drawdown": float("nan")},
        drop_log=drops or {},
        ic=ic or {},
    )


def run_all(force: bool = False, universe_override: dict | None = None,
            years: int | None = None, method: str = "gbm",
            membership=None, fundamentals_source: str = "yfinance",
            variant: str = "both") -> Bundle:
    """Run the strategy and bundle results for the dashboard.

    Pass universe_override + years to backtest a point-in-time universe (e.g.
    the 2020 holdings over a 9-year window, or the S&P 500 PIT universe).
    method selects the model (gbm|lambdarank|mlp); membership enforces
    point-in-time index membership; fundamentals_source picks yfinance|edgar.
    variant : "both" | "A" | "B" — build only the requested variant(s). Building
        one (the single live model) skips the other walk-forward, ~halving time.
    Defaults reproduce the live current-holdings GBM backtest.
    """
    samples, mprices, sector_mom, fwd, uni, stock_sector = build_samples(
        force=force, universe_override=universe_override, years=years,
        fundamentals_source=fundamentals_source)

    if variant in ("both", "A"):
        result_a = run_variant("A: momentum-only", samples, PRICE_FEATURES,
                               sector_mom, fwd, stock_sector, use_screen=False,
                               method=method, membership=membership)
    else:
        result_a = _empty_result("A: momentum-only")
    if variant in ("both", "B"):
        result_b = run_variant("B: momentum + quality", samples, ALL_FEATURES,
                               sector_mom, fwd, stock_sector, use_screen=True,
                               method=method, membership=membership)
    else:
        result_b = _empty_result("B: momentum + quality")

    # Full SPY buy-and-hold monthly (forward) return series over ALL months.
    # Consumers reindex to whichever variant's grid they display, so a short
    # variant (e.g. B) never truncates the SPY comparison for a longer one (A).
    spy = mprices[C.BENCHMARK] if C.BENCHMARK in mprices.columns else pd.Series(dtype=float)
    spy_returns = (spy.shift(-1) / spy - 1.0) if not spy.empty else pd.Series(dtype=float)

    _warn_overfit(result_a)
    _warn_overfit(result_b)
    _warn_thin_data(result_b)

    return Bundle(
        universe=uni, stock_sector=stock_sector, mprices=mprices,
        sector_mom=sector_mom, fwd=fwd, samples=samples,
        result_a=result_a, result_b=result_b, spy_returns=spy_returns,
    )


def _warn_overfit(r: BacktestResult):
    s = r.summary.get("sharpe", float("nan"))
    if pd.notna(s) and s > C.SHARPE_WARN:
        print(f"[WARNING] {r.label}: Sharpe {s:.2f} > {C.SHARPE_WARN}. "
              "Suspiciously high — likely overfitting or a lookahead bug, "
              "not a real edge. Audit the walk-forward loop.")


def _warn_thin_data(r: BacktestResult):
    if not r.drop_log:
        return
    avg_drop = sum(r.drop_log.values()) / len(r.drop_log)
    if avg_drop >= C.N_STOCKS_MAX:
        print(f"[WARNING] {r.label}: ~{avg_drop:.0f} names dropped/month by the "
              "quality screen. ~5yr yfinance fundamentals may be too thin to "
              "trust the screen; a real point-in-time source (FMP, Sharadar) "
              "would be needed for a credible result.")
