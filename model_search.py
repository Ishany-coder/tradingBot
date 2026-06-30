"""Model search: evaluate candidate strategy configs against the S&P 500.

Goal: find a config that beats SPY in >=60% of paired block-bootstrap resamples
("reliably beat the S&P 60% of the time"). We report it over the FULL 2020->now
window AND over a RECENT (2023->now) slice — the recent slice is a rough
out-of-sample check (a config that only beats S&P in the early bull but not
recently is overfit/regime-dependent).

Honest caveat baked into the output: we also report the information coefficient
(IC). A high bootstrap win-rate with near-zero IC means the edge is magnitude /
concentration in a bull market, not stock-picking skill.

Efficiency: build the (expensive) sample matrix ONCE per universe, then evaluate
every (method, variant) on it — so a 4-config sp500 batch is one build + four
walk-forwards, not four full rebuilds.

Run:  python model_search.py <batch>     # batch = "2020" | "2026"
Each evaluated model is appended to model_search_results.jsonl.
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

import config as C
import backtest
import metrics
import robustness
import sp500
import universe_2020

START = pd.Timestamp("2020-01-01")
RECENT = pd.Timestamp("2023-01-01")
RESULTS = "model_search_results.jsonl"
PIT_YEARS = 9
BOOT_N = 2000


def _winrate(mr: pd.Series, spy_returns: pd.Series, since: pd.Timestamp):
    sub = mr[mr.index >= since]
    spy = spy_returns.reindex(sub.index)
    if len(sub) < 13:
        return None
    b = robustness.bootstrap_beat_spy(sub, spy, n=BOOT_N)
    return None if "error" in b else b


def _build_panels(universe: str):
    """Build the sample matrix + panels ONCE for a universe; return everything
    run_variant needs plus the SPY forward-return series and membership."""
    members = None
    if universe == "sp500":
        uni, members = sp500.build_universe(PIT_YEARS)
        panels = backtest.build_samples(universe_override=uni, years=PIT_YEARS,
                                        fundamentals_source="edgar")
    elif universe == "pit2020":
        panels = backtest.build_samples(
            universe_override=universe_2020.HOLDINGS_2020, years=PIT_YEARS)
    else:
        panels = backtest.build_samples()
    samples, mprices, sector_mom, fwd, _uni, stock_sector = panels
    spy = mprices[C.BENCHMARK] if C.BENCHMARK in mprices.columns else pd.Series(dtype=float)
    spy_returns = (spy.shift(-1) / spy - 1.0) if not spy.empty else pd.Series(dtype=float)
    return samples, sector_mom, fwd, stock_sector, spy_returns, members


def run_batch(batch: str):
    universe, configs = BATCHES[batch]
    print(f"=== model search batch '{batch}' ({universe}) — {len(configs)} configs, "
          f"bootstrap n={BOOT_N}, target win_rate>=0.60 ===", flush=True)
    t0 = time.time()
    print(f"building sample matrix for {universe} once…", flush=True)
    samples, sector_mom, fwd, stock_sector, spy_returns, members = _build_panels(universe)
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    for name, method, variant in configs:
        print(f"\n>>> {name} ({universe}/{method}/{variant}) …", flush=True)
        ts = time.time()
        feat = backtest.PRICE_FEATURES if variant == "A" else backtest.ALL_FEATURES
        try:
            res = backtest.run_variant(variant, samples, feat, sector_mom, fwd,
                                       stock_sector, use_screen=(variant == "B"),
                                       method=method, with_mc=False, membership=members)
            mr = res.monthly_returns
            mr = mr[mr.index >= START]
        except Exception as exc:  # noqa: BLE001
            row = {"name": name, "universe": universe, "method": method,
                   "variant": variant, "status": f"error: {exc}"}
            mr = pd.Series(dtype=float)
        if not mr.empty:
            spy = spy_returns.reindex(mr.index)
            spy_tot = float((1.0 + spy.fillna(0.0)).prod() - 1.0)
            full = _winrate(mr, spy_returns, START)
            rec = _winrate(mr, spy_returns, RECENT)
            ic = res.ic or {}
            row = {"name": name, "universe": universe, "method": method,
                   "variant": variant, "status": "ok", "months": len(mr),
                   "total_return": round(metrics.total_return(mr), 4),
                   "spy_total": round(spy_tot, 4),
                   "edge": round(metrics.total_return(mr) - spy_tot, 4),
                   "cagr": round(metrics.cagr(mr), 4),
                   "sharpe": round(metrics.sharpe(mr), 3),
                   "ic_mean": round(ic.get("mean_ic", float("nan")), 4),
                   "win_rate": round(full["win_rate"], 4) if full else None,
                   "win_rate_recent": round(rec["win_rate"], 4) if rec else None,
                   "secs": round(time.time() - ts)}
        else:
            row = {"name": name, "universe": universe, "method": method,
                   "variant": variant,
                   "status": locals().get("row", {}).get("status", "empty"),
                   "secs": round(time.time() - ts)}
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if row.get("status") == "ok":
            wr, wrr = row["win_rate"], row["win_rate_recent"]
            flag = "✅ BEATS 60%" if (wr or 0) >= 0.60 else "❌ <60%"
            rec_s = f"{wrr:.0%}" if wrr is not None else "n/a"
            print(f"    win_rate(2020→)={wr:.0%} {flag} | recent(2023→)={rec_s} | "
                  f"edge={row['edge']:+.1%} | CAGR={row['cagr']:+.2%} | "
                  f"Sharpe={row['sharpe']:.2f} | IC={row['ic_mean']:+.4f} | "
                  f"{row['months']}mo | {row['secs']}s", flush=True)
        else:
            print(f"    {row['status']}", flush=True)
    print(f"\n=== batch done in {time.time()-t0:.0f}s ===", flush=True)


BATCHES = {
    "2020": ("pit2020", [("2020-A-gbm", "gbm", "A"),
                         ("2020-A-lambdarank", "lambdarank", "A"),
                         ("2020-A-mlp", "mlp", "A"),
                         ("2020-B-gbm", "gbm", "B"),
                         ("2020-B-lambdarank", "lambdarank", "B")]),
    "2026": ("sp500", [("2026-B-lambdarank", "lambdarank", "B"),
                       ("2026-A-lambdarank", "lambdarank", "A"),
                       ("2026-B-gbm", "gbm", "B"),
                       ("2026-A-gbm", "gbm", "A")]),
    # Neural-net (seed-ensembled MLP) on the S&P 500 breadth universe.
    "nn": ("sp500", [("nn-A-mlp", "mlp", "A"),
                     ("nn-B-mlp", "mlp", "B")]),
    # Ensemble/stack: rank-blend of GBM + lambdarank + MLP on the S&P 500.
    "ensemble": ("sp500", [("ens-A", "ensemble", "A"),
                           ("ens-B", "ensemble", "B")]),
}


if __name__ == "__main__":
    run_batch(sys.argv[1] if len(sys.argv) > 1 else "2020")
