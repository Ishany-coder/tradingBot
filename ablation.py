"""Ablation ladder + sensitivity tests: does the ML layer actually earn its keep?

The ladder (each rung adds one ingredient; the ML must beat rung 3 to justify
existing — beating SPY is NOT the bar):

  1. SPY buy & hold
  2. Equal-weight the top-3 momentum sector ETFs, monthly (Stage 1 alone)
  3. Stage 1 + NAIVE 12-1 momentum rank of stocks, top 15 — zero ML,
     run through the IDENTICAL pipeline (caps, overlays, costs)
  4. The full trained ensemble through the same pipeline

Also: MTUM benchmark (momentum-factor ETF), cost sensitivity (5/10/15/20 bp),
sizing-mode comparison (inverse-vol vs conviction vs equal), overlay on/off,
and the Deflated Sharpe of the deployed config given the search's trial count.

Run:  python ablation.py            (S&P 500 PIT universe; ~25 min first pass)
Writes ABLATION.md and prints the verdict.
"""
from __future__ import annotations

import json
import time

import pandas as pd

import config as C
import backtest
import data
import metrics
import model
import robustness
import sp500

START = pd.Timestamp("2020-01-01")
BOOT_N = 2000
PIT_YEARS = 9


def _series_report(mr: pd.Series, spy_returns: pd.Series) -> dict:
    mr = mr[mr.index >= START]
    if mr.empty:
        return {"error": "empty"}
    spy = spy_returns.reindex(mr.index)
    boot = robustness.bootstrap_beat_spy(mr, spy, n=BOOT_N)
    return {
        "months": len(mr),
        "total": round(metrics.total_return(mr), 4),
        "cagr": round(metrics.cagr(mr), 4),
        "sharpe": round(metrics.sharpe(mr), 3),
        "maxdd": round(metrics.max_drawdown(mr), 4),
        "win_vs_spy": round(boot["win_rate"], 3) if "error" not in boot else None,
    }


def _naive_preds(samples: pd.DataFrame) -> pd.DataFrame:
    """Rung-3 predictor: within-month 12-1 momentum z-score, no ML at all."""
    p = samples[["mom_12_1_z"]].dropna().rename(columns={"mom_12_1_z": "pred"})
    p["confidence"] = 0.5
    return p


def main():
    t0 = time.time()
    print("[ablation] building S&P 500 PIT panels…", flush=True)
    uni, members = sp500.build_universe(PIT_YEARS)
    samples, mprices, sector_mom, fwd, _u, stock_sector = backtest.build_samples(
        universe_override=uni, years=PIT_YEARS, fundamentals_source="edgar")
    spy_m = mprices[C.BENCHMARK]
    spy_returns = (spy_m.shift(-1) / spy_m - 1.0)
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    rows: dict[str, dict] = {}

    # --- rung 1: SPY -----------------------------------------------------------
    rows["1. SPY buy&hold"] = _series_report(
        spy_returns[spy_returns.index >= START], spy_returns)
    rows["1. SPY buy&hold"]["win_vs_spy"] = None  # vs itself is meaningless

    # --- MTUM benchmark ---------------------------------------------------------
    try:
        end = pd.Timestamp.today()
        mt = data.get_prices(["MTUM"], "2019-06-01", end.date().isoformat())["MTUM"]
        mt_m = mt.resample("ME").last()
        mt_r = (mt_m.shift(-1) / mt_m - 1.0)
        rows["1b. MTUM (momentum ETF)"] = _series_report(mt_r, spy_returns)
    except Exception as exc:  # noqa: BLE001
        rows["1b. MTUM (momentum ETF)"] = {"error": str(exc)}

    # --- rung 2: EW top-3 sector ETFs -------------------------------------------
    etf_ret = {}
    for t in sector_mom.index:
        row = sector_mom.loc[t].dropna()
        if len(row) < C.N_SECTORS:
            continue
        top = list(row.sort_values(ascending=False).index[: C.N_SECTORS])
        fr = [fwd.at[t, e] for e in top if e in fwd.columns and pd.notna(fwd.at[t, e])]
        if fr:
            etf_ret[t] = sum(fr) / len(fr)
    rows["2. EW top-3 sector ETFs"] = _series_report(
        pd.Series(etf_ret).sort_index(), spy_returns)

    # --- rung 3: naive momentum through the SAME pipeline ------------------------
    naive = _naive_preds(samples)
    r3 = backtest.run_variant("naive", samples, backtest.PRICE_FEATURES, sector_mom,
                              fwd, stock_sector, use_screen=False, with_mc=False,
                              membership=members, spy_monthly=spy_m,
                              preds_override=naive)
    rows["3. naive 12-1 rank (no ML)"] = _series_report(r3.monthly_returns, spy_returns)

    # --- rung 4: full ensemble (train once, reuse preds below) -------------------
    print("[ablation] training ensemble (once)…", flush=True)
    preds_ens = model.walk_forward_predict(samples, backtest.PRICE_FEATURES,
                                           method="ensemble")
    r4 = backtest.run_variant("ensemble", samples, backtest.PRICE_FEATURES,
                              sector_mom, fwd, stock_sector, use_screen=False,
                              with_mc=False, membership=members, spy_monthly=spy_m,
                              preds_override=preds_ens)
    rows["4. full ensemble"] = _series_report(r4.monthly_returns, spy_returns)

    # ML delta: rung 4 vs rung 3 excess months, bootstrapped head-to-head.
    m3 = r3.monthly_returns[r3.monthly_returns.index >= START]
    m4 = r4.monthly_returns[r4.monthly_returns.index >= START]
    both = pd.concat([m4.rename("ml"), m3.rename("naive")], axis=1).dropna()
    hh = robustness.bootstrap_beat_spy(both["ml"], both["naive"], n=BOOT_N)
    ml_delta = {"ml_beats_naive_pct": round(hh["win_rate"], 3) if "error" not in hh else None,
                "mean_excess": round(hh.get("mean_excess", float("nan")), 4)}

    # --- cost sensitivity (reuses trained preds; both rungs) --------------------
    cost_rows = {}
    base_cost = C.COST_PER_TRADE
    for bp in (5, 10, 15, 20):
        C.COST_PER_TRADE = bp / 10000.0
        rml = backtest.run_variant("e", samples, backtest.PRICE_FEATURES, sector_mom,
                                   fwd, stock_sector, use_screen=False, with_mc=False,
                                   membership=members, spy_monthly=spy_m,
                                   preds_override=preds_ens)
        rnv = backtest.run_variant("n", samples, backtest.PRICE_FEATURES, sector_mom,
                                   fwd, stock_sector, use_screen=False, with_mc=False,
                                   membership=members, spy_monthly=spy_m,
                                   preds_override=naive)
        cost_rows[f"{bp}bp"] = {
            "ensemble": _series_report(rml.monthly_returns, spy_returns),
            "naive": _series_report(rnv.monthly_returns, spy_returns),
        }
    C.COST_PER_TRADE = base_cost

    # --- sizing-mode comparison (same preds) ------------------------------------
    sizing_rows = {}
    base_mode = C.SIZING_MODE
    for mode in ("inverse_vol", "conviction", "equal"):
        C.SIZING_MODE = mode
        rm = backtest.run_variant("s", samples, backtest.PRICE_FEATURES, sector_mom,
                                  fwd, stock_sector, use_screen=False, with_mc=False,
                                  membership=members, spy_monthly=spy_m,
                                  preds_override=preds_ens)
        rep = _series_report(rm.monthly_returns, spy_returns)
        # turnover proxy: mean |Δw| per month
        books = [rm.holdings_history[d] for d in sorted(rm.holdings_history)]
        turns = [sum(abs(books[i].get(t, 0) - books[i-1].get(t, 0))
                     for t in set(books[i]) | set(books[i-1])) / 2
                 for i in range(1, len(books))]
        rep["avg_turnover"] = round(sum(turns) / len(turns), 3) if turns else None
        sizing_rows[mode] = rep
    C.SIZING_MODE = base_mode

    # --- overlays on/off ---------------------------------------------------------
    tv, ro = C.TARGET_VOL, C.REGIME_OFF_EXPOSURE
    C.TARGET_VOL, C.REGIME_OFF_EXPOSURE = 99.0, 1.0  # disable both overlays
    r_no = backtest.run_variant("no-overlay", samples, backtest.PRICE_FEATURES,
                                sector_mom, fwd, stock_sector, use_screen=False,
                                with_mc=False, membership=members, spy_monthly=spy_m,
                                preds_override=preds_ens)
    C.TARGET_VOL, C.REGIME_OFF_EXPOSURE = tv, ro
    overlay_rows = {"overlays ON": rows["4. full ensemble"],
                    "overlays OFF": _series_report(r_no.monthly_returns, spy_returns)}

    # --- deflated Sharpe of the deployed config ----------------------------------
    trials = []
    try:
        for ln in open("model_search_results.jsonl"):
            rec = json.loads(ln)
            if rec.get("sharpe") is not None:
                trials.append(rec["sharpe"] / (12 ** 0.5))  # to monthly scale
    except FileNotFoundError:
        pass
    dsr = metrics.deflated_sharpe(m4, n_trials=max(len(trials), 13),
                                  trial_sharpes=trials or None)

    # --- write report -------------------------------------------------------------
    def fmt(d):
        if "error" in d:
            return f"error: {d['error']}"
        win = f"{d['win_vs_spy']:.0%}" if d.get("win_vs_spy") is not None else "—"
        return (f"total {d['total']:+.1%} · CAGR {d['cagr']:+.1%} · Sharpe "
                f"{d['sharpe']:.2f} · maxDD {d['maxdd']:.0%} · beats SPY {win}")

    lines = ["# Ablation ladder — does the ML earn its keep?", "",
             f"S&P 500 PIT universe, 2020→now, costs {base_cost*1e4:.0f}bp, "
             f"overlays ON (regime gate + vol target), bootstrap n={BOOT_N}.", ""]
    for k in ["1. SPY buy&hold", "1b. MTUM (momentum ETF)",
              "2. EW top-3 sector ETFs", "3. naive 12-1 rank (no ML)",
              "4. full ensemble"]:
        lines.append(f"- **{k}** — {fmt(rows[k])}")
    lines += ["",
              f"**ML delta (rung 4 vs rung 3 head-to-head):** ensemble beats the "
              f"no-ML baseline in **{ml_delta['ml_beats_naive_pct']:.0%}** of "
              f"resamples (mean excess {ml_delta['mean_excess']:+.1%}).",
              "",
              "## Cost sensitivity", ""]
    for bp, rr in cost_rows.items():
        lines.append(f"- **{bp}** — ensemble: {fmt(rr['ensemble'])}")
        lines.append(f"  - naive: {fmt(rr['naive'])}")
    lines += ["", "## Sizing mode (same ensemble preds)", ""]
    for mode, rr in sizing_rows.items():
        lines.append(f"- **{mode}** — {fmt(rr)} · turnover {rr['avg_turnover']}")
    lines += ["", "## Risk overlays", ""]
    for k, rr in overlay_rows.items():
        lines.append(f"- **{k}** — {fmt(rr)}")
    lines += ["", "## Deflated Sharpe (multiple-testing haircut)", "",
              f"- observed monthly SR {dsr['sr_monthly']:.3f}, expected-max under "
              f"{dsr['n_trials']} trials {dsr['sr0_monthly']:.3f} → "
              f"**DSR = {dsr['dsr']:.2f}** (want ≥ 0.95; below that the edge is "
              f"not distinguishable from selection luck)."]
    text = "\n".join(lines)
    open("ABLATION.md", "w").write(text)
    print("\n" + text)
    print(f"\n[ablation] done in {time.time()-t0:.0f}s -> ABLATION.md", flush=True)


if __name__ == "__main__":
    main()
