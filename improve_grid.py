"""Pre-registered 32-cell improvement grid — see IMPROVEMENTS.md (binding rule).

Selection window 2012-2019 (fresh), validation 2020->now (secondary). Naive rank
model (no training) => the whole grid runs in minutes once panels are built.

Run:  python improve_grid.py            -> writes improve_grid_results.jsonl + verdict
"""
from __future__ import annotations

import itertools
import json
import time

import pandas as pd

import config as C
import backtest
import metrics
import robustness
import sp500

SEL_START, SEL_END = pd.Timestamp("2012-01-01"), pd.Timestamp("2019-12-31")
VAL_START = pd.Timestamp("2020-01-01")
YEARS = 15
BOOT_N = 2000
RESULTS = "improve_grid_results.jsonl"

OVERLAYS = {           # (REGIME_OFF_EXPOSURE, TARGET_VOL)
    "both": (0.5, 0.18),
    "gate_only": (0.5, 99.0),
    "vol_only": (1.0, 0.18),
    "none": (1.0, 99.0),
}


def _win(mr: pd.Series, spy_returns: pd.Series, lo, hi=None) -> dict:
    sub = mr[(mr.index >= lo) & ((mr.index <= hi) if hi is not None else True)]
    if len(sub) < 24:
        return {"win": None, "months": len(sub)}
    spy = spy_returns.reindex(sub.index)
    b = robustness.bootstrap_beat_spy(sub, spy, n=BOOT_N)
    if "error" in b:
        return {"win": None, "months": len(sub)}
    return {"win": round(b["win_rate"], 3), "months": len(sub),
            "total": round(metrics.total_return(sub), 3),
            "sharpe": round(metrics.sharpe(sub), 2),
            "maxdd": round(metrics.max_drawdown(sub), 3)}


def _preds(samples: pd.DataFrame, signal: str) -> pd.DataFrame:
    if signal == "mom_z":
        p = samples[["mom_12_1_z"]].dropna().rename(columns={"mom_12_1_z": "pred"})
    else:  # mom_over_vol: risk-adjusted momentum, z-scored within month
        raw = (samples["mom_12_1"] / samples["vol"]).replace(
            [float("inf"), -float("inf")], float("nan"))
        g = raw.groupby(level=0)
        z = ((raw - g.transform("mean")) / g.transform("std")).clip(-4, 4)
        p = z.dropna().to_frame("pred")
    p["confidence"] = 0.5
    return p


def main():
    t0 = time.time()
    print(f"[grid] building {YEARS}y S&P 500 PIT panels (price-only)…", flush=True)
    uni, members = sp500.build_universe(YEARS)
    samples, mprices, sector_mom, fwd, _u, ss = backtest.build_samples(
        universe_override=uni, years=YEARS, fundamentals_source="none")
    spy_m = mprices[C.BENCHMARK]
    spy_returns = (spy_m.shift(-1) / spy_m - 1.0)
    print(f"  built in {time.time()-t0:.0f}s — "
          f"{samples.index.get_level_values(1).nunique()} tickers, "
          f"{samples.index.get_level_values(0).nunique()} months", flush=True)

    preds_cache = {s: _preds(samples, s) for s in ("mom_z", "mom_over_vol")}

    base = (C.REGIME_OFF_EXPOSURE, C.TARGET_VOL, C.N_SECTORS, C.N_STOCKS_MAX)
    rows = []
    cells = list(itertools.product(OVERLAYS, (3, 4), (15, 25),
                                   ("mom_z", "mom_over_vol")))
    for i, (ov, nsec, nstk, sig) in enumerate(cells, 1):
        C.REGIME_OFF_EXPOSURE, C.TARGET_VOL = OVERLAYS[ov]
        C.N_SECTORS, C.N_STOCKS_MAX = nsec, nstk
        r = backtest.run_variant("cell", samples, backtest.PRICE_FEATURES,
                                 sector_mom, fwd, ss, use_screen=False,
                                 with_mc=False, membership=members,
                                 spy_monthly=spy_m,
                                 preds_override=preds_cache[sig])
        mr = r.monthly_returns
        row = {"overlay": ov, "n_sectors": nsec, "n_stocks": nstk, "signal": sig,
               "sel": _win(mr, spy_returns, SEL_START, SEL_END),
               "val": _win(mr, spy_returns, VAL_START)}
        rows.append(row)
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        s, v = row["sel"].get("win"), row["val"].get("win")
        print(f"  [{i:2d}/32] {ov:9s} sec={nsec} stk={nstk} {sig:12s} "
              f"sel={s if s is not None else '—'} val={v if v is not None else '—'}",
              flush=True)
    C.REGIME_OFF_EXPOSURE, C.TARGET_VOL, C.N_SECTORS, C.N_STOCKS_MAX = base

    # Binding decision rule (IMPROVEMENTS.md): sel>=0.55 AND val>=0.50,
    # maximize min(sel, val).
    qual = [r for r in rows
            if (r["sel"].get("win") or 0) >= 0.55 and (r["val"].get("win") or 0) >= 0.50]
    print(f"\n[grid] done in {time.time()-t0:.0f}s — {len(qual)}/32 cells qualify.")
    if qual:
        best = max(qual, key=lambda r: min(r["sel"]["win"], r["val"]["win"]))
        print("WINNER:", json.dumps(best))
    else:
        top = sorted(rows, key=lambda r: -(min(r["sel"].get("win") or 0,
                                               r["val"].get("win") or 0)))[:5]
        print("NO CELL QUALIFIES. Top 5 by min(sel,val):")
        for r in top:
            print(" ", json.dumps(r))


if __name__ == "__main__":
    main()
