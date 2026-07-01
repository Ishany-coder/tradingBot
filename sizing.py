"""Position sizing: inverse-volatility (default), conviction, or equal weight.

Edge (predicted rank) decides *which* names are held (selection, in
backtest.py); this module decides *how much* capital each gets. Weights are
long-only and capped per name; the sum may be < 1 (the remainder is cash — a
cap is a cap, never silently violated).

Why inverse-vol is the default: the conviction numerator (classifier
P(beat median)) is uncalibrated and, at the measured IC (~0.01), statistically
indistinguishable from 0.5 + noise — sizing on it concentrates capital on noise
and generates turnover. 1/vol keeps the risk-balancing part and drops the noise.
Set config.SIZING_MODE = "conviction" to restore the old behaviour.
"""

from __future__ import annotations

import config as C


def conviction_weights(candidates: dict[str, dict],
                       max_weight: float = C.MAX_WEIGHT,
                       mode: str | None = None) -> dict[str, float]:
    """Map {ticker: {"confidence": c, "vol": v}} -> {ticker: weight}.

    mode: "inverse_vol" | "conviction" | "equal" (default = config.SIZING_MODE).
    Weights are capped at ``max_weight`` per name. If the cap is infeasible
    (n·cap < 1) every name sits AT the cap and the remainder stays cash.
    """
    mode = mode or C.SIZING_MODE
    names = list(candidates)
    if not names:
        return {}

    raw: dict[str, float] = {}
    if mode != "equal":
        for t, d in candidates.items():
            vol = d.get("vol")
            if vol is None or vol <= 0 or vol != vol:  # missing/NaN/zero vol
                continue
            if mode == "conviction":
                conf = d.get("confidence")
                if conf is None or conf <= 0 or conf != conf:
                    continue
                raw[t] = conf / vol
            else:  # inverse_vol
                raw[t] = 1.0 / vol

    if not raw:  # equal mode, or nothing usable -> equal weight the book
        w = min(1.0 / len(names), max_weight)
        return {t: w for t in names}

    # Any selected name without a usable score gets the minimum positive raw
    # score so it isn't silently dropped.
    floor = min(raw.values())
    for t in names:
        raw.setdefault(t, floor)

    total = sum(raw.values())
    weights = {t: r / total for t, r in raw.items()}
    return _apply_cap(weights, max_weight)


def _apply_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Cap each weight at ``cap``, redistributing excess to uncapped names.

    Iterates because redistribution can push another name over the cap. If the
    cap is infeasible (n·cap < 1), every name sits AT the cap and the book holds
    the remainder as cash — the cap is never violated (the old behaviour of
    returning 1/n each silently blew through it on thin months).
    """
    n = len(weights)
    if n == 0:
        return {}
    if n * cap <= 1.0:  # cannot fully invest under the cap -> cap + cash
        return {t: cap for t in weights}

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
    if s > 1.0 + 1e-9:  # normalise only if somehow over-invested
        w = {t: v / s for t, v in w.items()}
    return w


def apply_sector_cap(weights: dict[str, float], stock_sector: dict[str, str],
                     cap: float = C.MAX_SECTOR_WEIGHT) -> dict[str, float]:
    """Scale down any sector whose aggregate weight exceeds ``cap``.

    Freed weight stays as cash (not force-fed into other sectors) — the point is
    limiting concentration, not staying fully invested at any cost.
    """
    by_sec: dict[str, float] = {}
    for t, w in weights.items():
        by_sec[stock_sector.get(t, "?")] = by_sec.get(stock_sector.get(t, "?"), 0.0) + w
    out = dict(weights)
    for sec, tot in by_sec.items():
        if tot > cap + 1e-12:
            scale = cap / tot
            for t in out:
                if stock_sector.get(t, "?") == sec:
                    out[t] *= scale
    return out
