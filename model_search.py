"""Model search: evaluate candidate strategy configs against the S&P 500.

Goal: find a config that beats SPY in >=60% of paired block-bootstrap resamples
("reliably beat the S&P 60% of the time"), TUNED on the 2020 universe (in-sample)
and then VALIDATED on the S&P 500 PIT universe through 2026 (out-of-sample).

Honest caveat baked into the output: we also report the information coefficient
(IC). A high bootstrap win-rate with near-zero IC means the edge is magnitude /
concentration in a bull market, not stock-picking skill, and is unlikely to
survive out-of-sample. The 2020 -> 2026 split is the real test.

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
RESULTS = "model_search_results.jsonl"
PIT_YEARS = 9
BOOT_N = 2000


def _build(universe: str, method: str, variant: str):
    if universe == "sp500":
        uni, members = sp500.build_universe(PIT_YEARS)
        return backtest.run_all(universe_override=uni, years=PIT_YEARS,
                                method=method, variant=variant,
                                membership=members, fundamentals_source="edgar")
    if universe == "pit2020":
        return backtest.run_all(method=method, variant=variant,
                                universe_override=universe_2020.HOLDINGS_2020,
                                years=PIT_YEARS)
    return backtest.run_all(method=method, variant=variant)


def evaluate(name: str, universe: str, method: str, variant: str) -> dict:
    t0 = time.time()
    bundle = _build(universe, method, variant)
    res = bundle.result_a if variant == "A" else bundle.result_b
    mr = res.monthly_returns
    mr = mr[mr.index >= START]
    out = {"name": name, "universe": universe, "method": method, "variant": variant}
    if mr.empty:
        out.update({"status": "empty (data too thin / screen starved)",
                    "secs": round(time.time() - t0)})
        return out
    spy = bundle.spy_returns.reindex(mr.index)
    spy_tot = float((1.0 + spy.fillna(0.0)).prod() - 1.0)
    boot = robustness.bootstrap_beat_spy(mr, spy, n=BOOT_N)
    ic = res.ic or {}
    out.update({
        "status": "ok",
        "months": len(mr),
        "total_return": round(metrics.total_return(mr), 4),
        "spy_total": round(spy_tot, 4),
        "edge": round(metrics.total_return(mr) - spy_tot, 4),
        "cagr": round(metrics.cagr(mr), 4),
        "sharpe": round(metrics.sharpe(mr), 3),
        "ic_mean": round(ic.get("mean_ic", float("nan")), 4),
        "win_rate": round(boot.get("win_rate", float("nan")), 4),
        "mean_excess": round(boot.get("mean_excess", float("nan")), 4),
        "p5_excess": round(boot.get("p5_excess", float("nan")), 4),
        "secs": round(time.time() - t0),
    })
    return out


BATCHES = {
    # In-sample tuning on the 2020 point-in-time universe (yfinance fundamentals,
    # so variant B is usually screen-starved -> reported empty, which is honest).
    "2020": [
        ("2020-A-gbm", "pit2020", "gbm", "A"),
        ("2020-A-lambdarank", "pit2020", "lambdarank", "A"),
        ("2020-A-mlp", "pit2020", "mlp", "A"),
        ("2020-B-gbm", "pit2020", "gbm", "B"),
        ("2020-B-lambdarank", "pit2020", "lambdarank", "B"),
    ],
    # Out-of-sample validation on the S&P 500 PIT universe (EDGAR fundamentals).
    "2026": [
        ("2026-B-lambdarank", "sp500", "lambdarank", "B"),
        ("2026-A-lambdarank", "sp500", "lambdarank", "A"),
        ("2026-B-gbm", "sp500", "gbm", "B"),
        ("2026-A-gbm", "sp500", "gbm", "A"),
    ],
}


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "2020"
    configs = BATCHES[batch]
    print(f"=== model search batch '{batch}' — {len(configs)} configs, "
          f"bootstrap n={BOOT_N}, target win_rate>=0.60 ===", flush=True)
    for name, universe, method, variant in configs:
        print(f"\n>>> {name} ({universe}/{method}/{variant}) …", flush=True)
        try:
            row = evaluate(name, universe, method, variant)
        except Exception as exc:  # noqa: BLE001
            row = {"name": name, "universe": universe, "method": method,
                   "variant": variant, "status": f"error: {exc}"}
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if row.get("status") == "ok":
            flag = "✅ BEATS 60%" if row["win_rate"] >= 0.60 else "❌ <60%"
            print(f"    win_rate={row['win_rate']:.0%} {flag} | edge={row['edge']:+.1%} "
                  f"| CAGR={row['cagr']:+.2%} | Sharpe={row['sharpe']:.2f} "
                  f"| IC={row['ic_mean']:+.4f} | {row['months']}mo | {row['secs']}s",
                  flush=True)
        else:
            print(f"    {row['status']}", flush=True)
    print("\n=== batch done ===", flush=True)


if __name__ == "__main__":
    main()
