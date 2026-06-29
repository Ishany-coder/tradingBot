"""Monte Carlo reshuffle of monthly returns.

A single backtest reports one drawdown path. By reshuffling the order of the
realised monthly returns many times we get a distribution of *plausible*
drawdowns the same return stream could have produced, and report its tail
(P95) rather than trusting the one historical ordering.

Note: this resamples the order of returns, so total compounded return is
unchanged across runs; only path-dependent stats (drawdown) vary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import metrics


def reshuffle_drawdowns(monthly_returns: pd.Series, runs: int = C.MC_RUNS,
                        seed: int = C.MC_SEED) -> dict[str, float]:
    """Return P50 / P95 max-drawdown over ``runs`` reshuffles.

    P95 here is the 95th percentile of drawdown *magnitude* — i.e. a worse
    (deeper) drawdown than the median, the number to plan around.
    """
    r = monthly_returns.dropna().values
    if len(r) < 2:
        return {"p50_drawdown": float("nan"), "p95_drawdown": float("nan")}

    rng = np.random.default_rng(seed)
    dds = np.empty(runs)
    for i in range(runs):
        shuffled = pd.Series(rng.permutation(r))
        dds[i] = metrics.max_drawdown(shuffled)  # negative numbers

    # More-negative = worse. P95 worst-case => 5th percentile of the signed dd.
    return {
        "p50_drawdown": float(np.percentile(dds, 50)),
        "p95_drawdown": float(np.percentile(dds, 5)),
    }
