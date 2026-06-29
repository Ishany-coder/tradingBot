"""One-off: backtest from 2020 -> current and compare to S&P (SPY buy-and-hold).

Extends the data window so the walk-forward model can start predicting in 2020
(needs 12mo momentum warmup + 24mo training before the first 2020 prediction),
then slices results to 2020-01-01 onward and compares to a SPY buy-and-hold over
the identical months. Run: python backtest_2020.py
"""
from __future__ import annotations

import pandas as pd

import config as C
# Widen the window BEFORE building samples: ~9yr of history => first prediction
# lands ~2019, giving full 2020 coverage after slicing.
C.BACKTEST_YEARS = 9

import universe as U          # noqa: E402
import universe_2020          # noqa: E402

# Override the universe with POINT-IN-TIME 2020 holdings (kills survivorship
# bias). Only affects this script; the live trader still uses current holdings.
U.build_universe = lambda force=False: universe_2020.HOLDINGS_2020

import backtest  # noqa: E402  (import after config + universe override)
import metrics   # noqa: E402

START = pd.Timestamp("2020-01-01")
CAP = C.START_CAPITAL


def window(mr: pd.Series) -> pd.Series:
    return mr[mr.index >= START]


def report(name: str, mr: pd.Series):
    if mr.empty:
        print(f"\n{name}: no returns in window.")
        return None
    eq = CAP * (1 + mr.fillna(0)).cumprod()
    s = {
        "first": mr.index.min().date(),
        "last": mr.index.max().date(),
        "months": len(mr),
        "total_return": metrics.total_return(mr),
        "cagr": metrics.cagr(mr),
        "max_dd": metrics.max_drawdown(mr),
        "sharpe": metrics.sharpe(mr),
        "final_$": eq.iloc[-1],
    }
    print(f"\n=== {name} ({s['first']} -> {s['last']}, {s['months']} months) ===")
    print(f"  total return : {s['total_return']:+.1%}")
    print(f"  CAGR         : {s['cagr']:+.2%}")
    print(f"  max drawdown : {s['max_dd']:.2%}")
    print(f"  Sharpe       : {s['sharpe']:.2f}")
    print(f"  $100k -> ${s['final_$']:,.0f}")
    return s


def main():
    print("Running 2020->now backtest (downloading ~9yr history; be patient)…")
    b = backtest.run_all(force=False)  # new date range => prices re-download

    # Strategy A (price-based, full history).
    mr_a = window(b.result_a.monthly_returns)

    # SPY buy-and-hold over the SAME months (return t -> t+1, indexed at t).
    spy = b.mprices[C.BENCHMARK]
    spy_fwd = (spy.shift(-1) / spy - 1.0).reindex(mr_a.index)

    sa = report("STRATEGY A — 2020 point-in-time universe (no survivorship bias)", mr_a)
    sp = report("S&P 500 (SPY buy & hold)", spy_fwd)

    # Backtest B if it reaches the window at all.
    mr_b = window(b.result_b.monthly_returns)
    if not mr_b.empty:
        report("STRATEGY B (+ quality screen, limited fundamentals history)", mr_b)
    else:
        print("\nSTRATEGY B: no coverage back to 2020 (fundamentals history too "
              "short) — expected. A is the 2020 comparison.")

    if sa and sp:
        diff = sa["total_return"] - sp["total_return"]
        print("\n================= VERDICT =================")
        print(f"  Strategy A : {sa['total_return']:+.1%}  (${sa['final_$']:,.0f})")
        print(f"  S&P 500    : {sp['total_return']:+.1%}  (${sp['final_$']:,.0f})")
        print(f"  Edge vs S&P: {diff:+.1%} total  |  "
              f"CAGR {sa['cagr']-sp['cagr']:+.2%}")
        if sa["sharpe"] > C.SHARPE_WARN:
            print(f"  [WARNING] Sharpe {sa['sharpe']:.2f} > {C.SHARPE_WARN}: "
                  "suspiciously high — suspect overfit/lookahead, not real edge.")
        print("==========================================")


if __name__ == "__main__":
    main()
