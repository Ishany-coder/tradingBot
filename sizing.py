"""Conviction position sizing: weight ∝ confidence / volatility.

Replaces equal weight. Edge (predicted return) decides *which* names are held
(selection, in backtest.py); this module decides *how much* capital each gets:
higher confidence and lower volatility => more dollars. Weights are long-only,
capped per name, and sum to 1.
"""

from __future__ import annotations

import config as C


def conviction_weights(candidates: dict[str, dict],
                       max_weight: float = C.MAX_WEIGHT) -> dict[str, float]:
    """Map {ticker: {"confidence": c, "vol": v}} -> {ticker: weight}.

    raw_i = confidence_i / vol_i ; normalise; cap at ``max_weight`` and
    redistribute the excess to uncapped names until everything fits.
    Falls back to equal weight if no usable confidence/vol is available.
    """
    raw = {}
    for t, d in candidates.items():
        conf = d.get("confidence")
        vol = d.get("vol")
        if conf is None or vol is None or vol <= 0 or conf <= 0:
            continue
        raw[t] = conf / vol

    names = list(candidates)
    if not raw:  # nothing usable -> equal weight the selected book
        if not names:
            return {}
        w = 1.0 / len(names)
        return {t: w for t in names}

    # Any selected name without a usable score gets a tiny floor so it isn't
    # silently dropped; give it the minimum positive raw score.
    floor = min(raw.values())
    for t in names:
        raw.setdefault(t, floor)

    total = sum(raw.values())
    weights = {t: r / total for t, r in raw.items()}
    return _apply_cap(weights, max_weight)


def _apply_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Cap each weight at ``cap``, redistributing excess to uncapped names.

    Iterates because redistributing can push another name over the cap. If the
    cap is infeasible (n·cap < 1), returns equal weights at the cap.
    """
    n = len(weights)
    if n == 0:
        return {}
    if n * cap <= 1.0:  # cannot fit under the cap -> everyone at the cap
        return {t: 1.0 / n for t in weights}

    w = dict(weights)
    for _ in range(100):
        over = {t: v for t, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for t in over:
            w[t] = cap
        uncapped = [t for t in w if w[t] < cap - 1e-12]
        pool = sum(w[t] for t in uncapped)
        if pool <= 0:
            break
        for t in uncapped:  # distribute excess proportionally to current weight
            w[t] += excess * (w[t] / pool)

    s = sum(w.values())
    return {t: v / s for t, v in w.items()}  # final renormalise
