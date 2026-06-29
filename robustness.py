"""Robustness test: how often does the strategy beat the S&P?

A single backtest gives one number. This asks whether that outperformance is a
robust property of the return distribution or a fluke of one particular path,
by PAIRED BLOCK-BOOTSTRAP resampling the realised monthly returns:

  * strategy and SPY returns are kept paired by month (same month sampled for
    both) so each draw is a fair head-to-head over the same simulated history,
  * months are drawn in contiguous blocks (default 6) to preserve momentum /
    autocorrelation that single-month resampling would destroy,
  * for each of N draws we compound both series and check whether the strategy's
    total return exceeds SPY's.

Reports the win rate (fraction of draws the strategy beats SPY) plus the
distribution of excess total return. This is fast (no re-training), so it backs
a one-click button.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_beat_spy(strat: pd.Series, spy: pd.Series, n: int = 100,
                       block: int = 6, seed: int = 12345) -> dict:
    """Paired block-bootstrap of strategy vs SPY monthly returns.

    Returns a dict with win_rate, mean/median/p5/p95 excess total return, the
    raw excess-return samples (for a histogram), and the single actual-path
    excess for reference.
    """
    df = pd.concat([strat.rename("s"), spy.rename("b")], axis=1).dropna()
    s = df["s"].to_numpy()
    b = df["b"].to_numpy()
    m = len(s)
    if m < block + 1:
        return {"error": f"need >{block} aligned months, have {m}"}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    excess = np.empty(n)
    wins = 0

    for i in range(n):
        # Draw contiguous blocks of months (with replacement) until we cover m.
        starts = rng.integers(0, m - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:m]
        strat_tot = np.prod(1.0 + s[idx]) - 1.0
        spy_tot = np.prod(1.0 + b[idx]) - 1.0
        excess[i] = strat_tot - spy_tot
        if strat_tot > spy_tot:
            wins += 1

    actual_excess = float(np.prod(1.0 + s) - 1.0) - float(np.prod(1.0 + b) - 1.0)
    return {
        "n": n,
        "block": block,
        "months": m,
        "win_rate": wins / n,
        "mean_excess": float(np.mean(excess)),
        "median_excess": float(np.median(excess)),
        "p5_excess": float(np.percentile(excess, 5)),
        "p95_excess": float(np.percentile(excess, 95)),
        "samples": excess,
        "actual_excess": actual_excess,
    }
