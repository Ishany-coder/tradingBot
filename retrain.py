"""Re-train robustness: re-run the 2020 walk-forward model N times, each with a
different GBM random seed, and measure how the outcome varies.

Why this is different from ``robustness.bootstrap_beat_spy``:

  * bootstrap_beat_spy resamples ONE realised return path — it asks "is the edge
    a fluke of one ordering of months?" and is instant (no re-training).
  * THIS module actually re-fits the model from scratch on every run. The
    GradientBoosting model subsamples 80% of rows per tree (``subsample=0.8``),
    so a different ``random_state`` yields a genuinely different model, different
    book, and a different return stream. It asks the harder question: "does the
    edge survive the model's OWN randomness, or did seed 42 just get lucky?"

Because every run is a full walk-forward backtest, this is SLOW. The expensive
sample matrix is built once by the caller and reused across all runs; only the
model fit + simulation (the part that actually depends on the seed) repeats.

The window is sliced to 2020-01-01 onward and each run is compared to a SPY
buy-and-hold over that run's exact months, matching ``backtest_2020.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import backtest
import metrics

START = pd.Timestamp("2020-01-01")


def retrain_beat_spy(bundle, n: int = 100, variant: str = "A",
                     base_seed: int = 1000, progress=None) -> dict:
    """Re-train the 2020 model ``n`` times and aggregate the results.

    Parameters
    ----------
    bundle : anything exposing ``samples``, ``sector_mom``, ``fwd``,
        ``stock_sector`` and ``spy_returns`` for the 2020 point-in-time run
        (a ``backtest.Bundle`` does).
    n : number of re-trains (each is a full walk-forward backtest).
    variant : "A" (momentum-only, the credible 2020 test) or "B" (+ quality
        screen — usually starved of fundamentals history back to 2020).
    base_seed : seeds used are ``base_seed + i`` for i in range(n).
    progress : optional callback ``progress(done, total)`` for a UI bar.

    Returns a dict of averaged metrics + the per-run excess-return samples, or
    ``{"error": ...}`` if no run reached the 2020 window.
    """
    if variant.upper().startswith("B"):
        feature_cols, use_screen, label = (
            backtest.ALL_FEATURES, True, "B: momentum + quality")
    else:
        feature_cols, use_screen, label = (
            backtest.PRICE_FEATURES, False, "A: momentum-only")

    samples = bundle.samples
    sector_mom = bundle.sector_mom
    fwd = bundle.fwd
    stock_sector = bundle.stock_sector

    runs = []
    window = None
    for i in range(n):
        seed = base_seed + i
        params = {**C.GBM_PARAMS, "random_state": seed}
        r = backtest.run_variant(label, samples, feature_cols, sector_mom, fwd,
                                 stock_sector, use_screen=use_screen,
                                 params=params, with_mc=False)
        mr = r.monthly_returns
        mr = mr[mr.index >= START]
        if progress is not None:
            progress(i + 1, n)
        if mr.empty:
            continue
        window = (mr.index.min(), mr.index.max())
        spy_w = bundle.spy_returns.reindex(mr.index)
        strat_tot = metrics.total_return(mr)
        spy_tot = float((1.0 + spy_w.fillna(0.0)).prod() - 1.0)
        runs.append({
            "seed": seed,
            "total_return": strat_tot,
            "cagr": metrics.cagr(mr),
            "sharpe": metrics.sharpe(mr),
            "max_drawdown": metrics.max_drawdown(mr),
            "final": C.START_CAPITAL * (1.0 + strat_tot),
            "spy_total": spy_tot,
            "excess": strat_tot - spy_tot,
            "beat": strat_tot > spy_tot,
            "months": len(mr),
        })

    if not runs:
        return {"error": "no re-train reached the 2020 window (data too thin)"}

    df = pd.DataFrame(runs)
    exc = df["excess"].to_numpy()
    spy_total = float(df["spy_total"].mean())
    return {
        "n": len(df),
        "variant": label,
        "first": window[0],
        "last": window[1],
        "months": int(df["months"].median()),
        "win_rate": float(df["beat"].mean()),
        "wins": int(df["beat"].sum()),
        "mean_total_return": float(df["total_return"].mean()),
        "median_total_return": float(df["total_return"].median()),
        "mean_final": float(df["final"].mean()),
        "mean_cagr": float(df["cagr"].mean()),
        "mean_sharpe": float(df["sharpe"].mean()),
        "mean_max_drawdown": float(df["max_drawdown"].mean()),
        "spy_total": spy_total,
        "spy_final": C.START_CAPITAL * (1.0 + spy_total),
        "mean_excess": float(exc.mean()),
        "p5_excess": float(np.percentile(exc, 5)),
        "p95_excess": float(np.percentile(exc, 95)),
        "excess_samples": exc,
    }
