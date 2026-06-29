"""Streamlit dashboard for the two-stage momentum strategy.

Run:  streamlit run app.py

Simulation only — no broker connection, no real orders.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
import data
import backtest as B
import robustness
import universe_2020

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")

PIT_YEARS = 9  # window for the 2020 point-in-time backtest (reaches back to 2020)


@st.cache_data(show_spinner="Running backtest (downloading data on first run)…")
def load_bundle(mode: str = "current", force: bool = False) -> B.Bundle:
    if mode == "pit2020":
        return B.run_all(force=force,
                         universe_override=universe_2020.HOLDINGS_2020,
                         years=PIT_YEARS)
    return B.run_all(force=force)


# --- sidebar ----------------------------------------------------------------
st.sidebar.title("Controls")

univ_mode = st.sidebar.radio(
    "Universe / backtest mode",
    ["Current holdings (survivorship-biased)", "2020 point-in-time (realistic)"],
    help="Current = today's top-10 ETF holdings tested back in time (inflated by "
         "survivorship bias). 2020 = the holdings as of early 2020 (honest test).",
)
mode = "pit2020" if univ_mode.startswith("2020") else "current"

variant = st.sidebar.radio(
    "Strategy variant",
    ["B: momentum + quality", "A: momentum-only"],
    help="A is the price-only control; B adds the fundamental quality screen.",
)
if st.sidebar.button("↻ Refresh data (re-download)"):
    load_bundle.clear()
    st.rerun()

if mode == "pit2020":
    st.sidebar.caption(
        f"2020 point-in-time universe · {PIT_YEARS}y window · ${C.START_CAPITAL:,.0f} · "
        f"cost {C.COST_PER_TRADE*100:.3f}%/trade · no survivorship bias"
    )
else:
    st.sidebar.caption(
        f"Current holdings: {len(C.SECTOR_ETFS)} ETFs × top {C.TOP_HOLDINGS_N} · "
        f"{C.BACKTEST_YEARS}y window · ${C.START_CAPITAL:,.0f} · "
        f"cost {C.COST_PER_TRADE*100:.3f}%/trade · ⚠️ survivorship-biased"
    )

st.title("📈 Two-Stage Momentum Strategy")
st.caption("Sector momentum → stock selection · walk-forward GBM · **simulation only, no real orders**")

# Gate the heavy download/backtest behind an explicit click. Nothing downloads
# on page load; the dashboard builds only after Run is pressed (and again after
# switching universe, since that needs different price data).
if st.sidebar.button("▶ Run backtest / load data", type="primary"):
    st.session_state.loaded_mode = mode
if st.session_state.get("loaded_mode") != mode:
    st.info(
        "Nothing downloaded yet. Press **▶ Run backtest / load data** in the "
        "sidebar to fetch prices and run the "
        f"**{'2020 point-in-time' if mode == 'pit2020' else 'current holdings'}** "
        "backtest. Switching universe needs a fresh click — the data differs."
    )
    st.stop()

bundle = load_bundle(mode=mode)
result = bundle.result_b if variant.startswith("B") else bundle.result_a
other = bundle.result_a if variant.startswith("B") else bundle.result_b

# --- current holdings -------------------------------------------------------
if not result.holdings_history:
    st.warning(
        f"**{result.label}** produced no book — yfinance fundamentals are too "
        "thin over this window to run the quality screen. The momentum-only "
        "variant (A) below still works. For a credible quality-screened "
        "backtest, a real point-in-time fundamentals source (Financial Modeling "
        "Prep, Sharadar) is needed."
    )
    if not other.holdings_history:
        st.error("Neither variant produced holdings. Try Refresh.")
        st.stop()
    result = other  # fall back to the variant that has data
    st.info(f"Showing **{result.label}** instead.")

last_date = max(result.holdings_history)
weights = result.holdings_history[last_date]
preds = result.pred_history[last_date]
confs = result.conf_history.get(last_date, {})

rows = []
for tkr, w in weights.items():
    snap = data.get_current_fundamentals(tkr)  # live view: current fundamentals OK
    rows.append({
        "Ticker": tkr,
        "Sector": C.SECTOR_ETFS.get(bundle.stock_sector.get(tkr, ""), "—"),
        "Weight": w,
        "Confidence": confs.get(tkr, float("nan")),
        "Predicted growth": preds.get(tkr, float("nan")),
        "ROE (now)": snap["roe"],
        "D/E (now)": snap["de"],
        "Margin (now)": snap["margin"],
    })
holdings_df = pd.DataFrame(rows)

st.subheader(f"Current Holdings — rebalanced {last_date:%b %Y}")
st.caption("Weights are **conviction-sized** (confidence ÷ volatility), not equal. "
           "‘Confidence’ = model P(beats median next month). ‘Predicted growth’ is an "
           "**estimate**, not a guarantee.")

# Panel 1: treemap, sized by conviction weight, colored by sector.
tm = holdings_df.copy()
tm["label"] = (tm["Ticker"] + "<br>" + (tm["Weight"] * 100).round(1).astype(str) + "%"
               + "<br>conf " + (tm["Confidence"] * 100).round(0).astype(str) + "%"
               + "<br>est " + (tm["Predicted growth"] * 100).round(1).astype(str) + "%")
fig_tm = px.treemap(
    tm, path=[px.Constant("Book"), "Sector", "label"], values="Weight",
    color="Sector", title=None,
)
fig_tm.update_traces(textinfo="label")
fig_tm.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=420)
st.plotly_chart(fig_tm, width="stretch")

# Panel 2: sortable holdings table with a sort toggle.
sort_by = st.radio(
    "Sort holdings by",
    ["Position size (largest first)", "Predicted growth (highest first)",
     "Confidence (highest first)"],
    horizontal=True,
)
col = ("Weight" if sort_by.startswith("Position")
       else "Confidence" if sort_by.startswith("Confidence")
       else "Predicted growth")
table = holdings_df.sort_values(col, ascending=False).reset_index(drop=True)
st.dataframe(
    table.style.format({
        "Weight": "{:.1%}", "Confidence": "{:.0%}", "Predicted growth": "{:+.2%}",
        "ROE (now)": "{:.2f}", "D/E (now)": "{:.2f}", "Margin (now)": "{:.2%}",
    }),
    width="stretch",
)

st.divider()

# --- equity curve + metrics -------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Equity Curve — strategy vs buy-and-hold SPY")
    eq = result.equity
    spy_eq = (C.START_CAPITAL * (1 + bundle.spy_returns.reindex(eq.index).fillna(0)).cumprod())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=result.label, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=spy_eq.index, y=spy_eq.values, name="Buy & hold SPY",
                             line=dict(width=2, dash="dash")))
    fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0),
                      yaxis_title="Portfolio value ($)", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Metrics")
    s = result.summary
    st.metric("CAGR", f"{s['cagr']:.2%}")
    st.metric("Max drawdown", f"{s['max_drawdown']:.2%}")
    st.metric("Sharpe", f"{s['sharpe']:.2f}")
    st.metric("Monte Carlo P95 drawdown", f"{result.mc['p95_drawdown']:.2%}",
              help=f"P50: {result.mc['p50_drawdown']:.2%} over {C.MC_RUNS} reshuffles")
    if pd.notna(s["sharpe"]) and s["sharpe"] > C.SHARPE_WARN:
        st.warning(f"Sharpe {s['sharpe']:.2f} > {C.SHARPE_WARN}: suspiciously high — "
                   "likely overfitting or a lookahead bug, not a real edge.")

st.divider()

# --- robustness: how often do we beat the S&P? ------------------------------
st.subheader("🎲 Robustness — do we beat the S&P on average?")
st.caption("Paired block-bootstrap of the monthly returns: resample the history "
           "many times and count how often the strategy's total return beats SPY. "
           "Tests whether the edge is robust or a fluke of one path (no re-training, "
           "so it's instant).")

rc1, rc2 = st.columns([1, 3])
with rc1:
    n_tests = st.number_input("Number of tests", min_value=10, max_value=5000,
                              value=100, step=10)
    run_robust = st.button("▶ Run robustness tests vs S&P", type="primary")

if run_robust:
    strat = result.monthly_returns
    spy = bundle.spy_returns.reindex(strat.index)
    res = robustness.bootstrap_beat_spy(strat, spy, n=int(n_tests))
    if "error" in res:
        st.warning(f"Not enough data: {res['error']}")
    else:
        win = res["win_rate"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Beat S&P", f"{win:.0%}", help=f"{int(win*res['n'])}/{res['n']} resamples")
        m2.metric("Avg excess return", f"{res['mean_excess']:+.1%}",
                  help="mean strategy-minus-SPY total return across resamples")
        m3.metric("Actual path excess", f"{res['actual_excess']:+.1%}")
        st.caption(f"5th–95th percentile excess: {res['p5_excess']:+.1%} … "
                   f"{res['p95_excess']:+.1%}  ·  block={res['block']}mo  ·  "
                   f"{res['months']} months  ·  variant: {result.label}  ·  "
                   f"universe: {'2020 point-in-time' if mode=='pit2020' else 'current (biased)'}")
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=res["samples"] * 100, nbinsx=40,
                                    name="excess return"))
        hist.add_vline(x=0, line=dict(color="red", dash="dash"),
                       annotation_text="break-even vs SPY")
        hist.update_layout(height=300, margin=dict(t=10, l=0, r=0, b=0),
                           xaxis_title="strategy − SPY total return (%)",
                           yaxis_title="resamples")
        st.plotly_chart(hist, width="stretch")
        verdict = ("✅ robustly beats S&P" if win >= 0.6 else
                   "⚠️ edge is marginal / luck-dependent" if win >= 0.45 else
                   "❌ does NOT reliably beat S&P")
        st.write(f"**Verdict: {verdict}** — beats S&P in {win:.0%} of resampled histories.")

st.divider()

# --- re-train robustness: run the 2020 model many times ---------------------
st.subheader("🔁 Re-train Test — run the 2020 model 100× (slow, honest)")
st.caption(
    "Actually **re-fits the walk-forward model from scratch** many times, each "
    "with a different random seed. The gradient-boosting model subsamples 80% of "
    "rows per tree, so a new seed yields a genuinely different model and book — "
    "this asks whether the edge survives the model's *own* randomness, not just "
    "one lucky path. Always uses the **2020 point-in-time** universe (no "
    "survivorship bias), variant A. Each run is a full backtest, so 100 runs take "
    "several minutes."
)

tc1, tc2 = st.columns([1, 3])
with tc1:
    n_retrain = st.number_input("Number of re-trains", min_value=5, max_value=500,
                                value=100, step=5, key="n_retrain")
    run_retrain = st.button("▶ Re-train the 2020 model & average results",
                            type="primary")

if run_retrain:
    import retrain
    bundle2020 = load_bundle(mode="pit2020")  # cached; samples built once
    prog = st.progress(0.0, text="Re-training…")
    with st.spinner("Running re-trains (each is a full walk-forward backtest)…"):
        res = retrain.retrain_beat_spy(
            bundle2020, n=int(n_retrain), variant="A",
            progress=lambda d, t: prog.progress(d / t, text=f"Re-training {d}/{t}…"))
    prog.empty()
    if "error" in res:
        st.warning(f"Could not run re-train test: {res['error']}")
    else:
        win = res["win_rate"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Beat S&P", f"{win:.0%}",
                  help=f"{res['wins']}/{res['n']} re-trains beat buy-and-hold SPY")
        m2.metric("Avg total return", f"{res['mean_total_return']:+.1%}",
                  help=f"S&P over the same {res['months']} months: {res['spy_total']:+.1%}")
        m3.metric("Avg CAGR", f"{res['mean_cagr']:+.2%}")
        m4.metric("Avg Sharpe", f"{res['mean_sharpe']:.2f}")
        n1, n2, n3 = st.columns(3)
        n1.metric(f"Avg final value (${C.START_CAPITAL/1000:.0f}k → )",
                  f"${res['mean_final']:,.0f}")
        n2.metric("S&P final value", f"${res['spy_final']:,.0f}")
        n3.metric("Avg excess vs S&P", f"{res['mean_excess']:+.1%}",
                  help="mean strategy-minus-SPY total return across re-trains")
        st.caption(
            f"5th–95th pct excess: {res['p5_excess']:+.1%} … {res['p95_excess']:+.1%}  ·  "
            f"{res['n']} re-trains  ·  {res['first']:%b %Y}–{res['last']:%b %Y}  ·  "
            f"variant {res['variant']}  ·  2020 point-in-time universe (no survivorship bias)")
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=res["excess_samples"] * 100, nbinsx=30,
                                    name="excess return"))
        hist.add_vline(x=0, line=dict(color="red", dash="dash"),
                       annotation_text="break-even vs S&P")
        hist.update_layout(height=300, margin=dict(t=10, l=0, r=0, b=0),
                           xaxis_title="strategy − S&P total return since 2020 (%)",
                           yaxis_title="re-trains")
        st.plotly_chart(hist, width="stretch")
        verdict = ("✅ robustly beats S&P across random seeds" if win >= 0.6 else
                   "⚠️ edge depends on the random seed (marginal)" if win >= 0.45 else
                   "❌ does NOT reliably beat S&P")
        st.write(f"**Verdict: {verdict}** — beats S&P in {win:.0%} of "
                 f"{res['n']} independent re-trains since 2020.")

st.divider()

# --- head-to-head A vs B ----------------------------------------------------
st.subheader("A vs B — does the quality screen earn its keep?")
def _hh(r: B.BacktestResult) -> dict:
    return {
        "Variant": r.label,
        "CAGR": r.summary["cagr"],
        "Max drawdown": r.summary["max_drawdown"],
        "Sharpe": r.summary["sharpe"],
        "MC P95 drawdown": r.mc["p95_drawdown"],
    }
hh = pd.DataFrame([_hh(bundle.result_a), _hh(bundle.result_b)])
st.dataframe(
    hh.style.format({"CAGR": "{:.2%}", "Max drawdown": "{:.2%}", "Sharpe": "{:.2f}",
                     "MC P95 drawdown": "{:.2%}"}),
    width="stretch", hide_index=True,
)

a, b = bundle.result_a.summary, bundle.result_b.summary
if pd.notna(a["sharpe"]) and pd.notna(b["sharpe"]):
    better_sharpe = b["sharpe"] - a["sharpe"]
    verdict = "improved" if better_sharpe > 0 else "did NOT improve"
    st.caption(
        f"Quality screen {verdict} Sharpe (Δ {better_sharpe:+.2f}). "
        "Caveat: variant A also lacks the fundamental *features*, so this is a "
        "bundle comparison, not a clean screen-only ablation — the best ~5yr "
        "yfinance data allows."
    )
else:
    st.caption(
        "One variant produced no returns (usually B, when yfinance fundamentals "
        "are too thin) — head-to-head comparison unavailable. A real "
        "point-in-time fundamentals source would be needed for a credible B."
    )
if result.drop_log:
    avg_drop = sum(result.drop_log.values()) / len(result.drop_log)
    st.caption(f"Quality screen drops ~{avg_drop:.0f} names/month on average "
               "(missing fundamentals are excluded, never guessed).")

st.divider()

# --- sector view ------------------------------------------------------------
st.subheader("Sector View — current top 3 selected sectors")
sm_row = bundle.sector_mom.loc[last_date].dropna().sort_values(ascending=False)
sec_rows = []
for rank, (etf, mom) in enumerate(sm_row.items(), start=1):
    sec_rows.append({
        "Rank": rank,
        "Sector": C.SECTOR_ETFS.get(etf, etf),
        "ETF": etf,
        "12-1 momentum": mom,
        "Selected": "✅" if etf in result.sector_history.get(last_date, []) else "",
    })
sec_df = pd.DataFrame(sec_rows)
st.dataframe(
    sec_df.style.format({"12-1 momentum": "{:+.2%}"}),
    width="stretch", hide_index=True,
)

st.divider()

# --- live execution (Alpaca paper) ------------------------------------------
st.subheader("⚙️ Live Execution — Alpaca paper account")
st.caption("Read-only view. The trader loop (`run_loop.py`) places the orders — "
           "**paper account only, no real money**.")


@st.cache_data(ttl=30, show_spinner=False)
def _account_snapshot():
    """Pull live equity + positions; tolerate any broker/network failure."""
    from broker import PaperBroker
    bk = PaperBroker()
    return {"equity": bk.get_equity(), "positions": bk.get_positions(),
            "open": bk.is_market_open()}


try:
    snap = _account_snapshot()
    c1, c2, c3 = st.columns(3)
    c1.metric("Account equity", f"${snap['equity']:,.2f}")
    c2.metric("Open positions", str(len(snap["positions"])))
    c3.metric("Market", "🟢 open" if snap["open"] else "🔴 closed")

    invest = snap["equity"] * C.INVEST_FRACTION
    cur_mv = {t: p["market_value"] for t, p in snap["positions"].items()}
    rows = []
    names = set(weights) | set(cur_mv)
    for t in names:
        tgt = weights.get(t, 0.0) * invest
        rows.append({
            "Ticker": t,
            "Target $": tgt,
            "Held $": cur_mv.get(t, 0.0),
            "Drift $": tgt - cur_mv.get(t, 0.0),
            "In book": "✅" if t in weights else "— (exit)",
        })
    exec_df = pd.DataFrame(rows).sort_values("Target $", ascending=False)
    st.dataframe(exec_df.style.format({"Target $": "${:,.0f}", "Held $": "${:,.0f}",
                                       "Drift $": "${:,.0f}"}),
                 width="stretch", hide_index=True)
    st.caption(f"Orders fire only when |Drift| > "
               f"max({C.REBALANCE_BAND:.0%} equity, ${C.MIN_ORDER_USD:.0f}).")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not reach Alpaca paper account: {exc}")

# Dry-run / live status + recent trade log.
mode = ("LIVE armed" if C.DRYRUN_FLAG.exists() else "DRY-RUN pending (first run sends nothing)")
stopped = " · ⛔ STOP file present" if C.STOP_FILE.exists() else ""
st.write(f"**Trader status:** {mode}{stopped}")

if C.TRADE_LOG.exists():
    import json
    lines = C.TRADE_LOG.read_text().strip().splitlines()[-5:]
    st.write("**Recent runs:**")
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
            n = len(rec.get("orders", []))
            st.text(f"{rec['time']}  {rec.get('mode','?'):<22}  "
                    f"equity=${rec.get('equity',0):,.0f}  orders={n}")
        except Exception:  # noqa: BLE001
            continue
else:
    st.caption("No trades yet. Run `python trader.py` for the first (dry-run) cycle.")
