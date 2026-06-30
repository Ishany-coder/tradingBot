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
import metrics
import robustness
import sp500
import universe_2020

st.set_page_config(page_title="Momentum Strategy Dashboard", layout="wide")

PIT_YEARS = 9  # window for the 2020 point-in-time backtest (reaches back to 2020)


@st.cache_data(show_spinner="Running backtest (downloading data on first run)…")
def load_bundle(mode: str = "current", method: str = "gbm",
                force: bool = False) -> B.Bundle:
    if mode == "sp500":
        # Free point-in-time S&P 500 universe (survivorship-bias-free membership
        # + SEC EDGAR fundamentals). First build is slow: it fetches sectors and
        # EDGAR fundamentals for ~600 names, then caches them.
        uni, members = sp500.build_universe(PIT_YEARS, force=force)
        return B.run_all(force=force, universe_override=uni, years=PIT_YEARS,
                         method=method, membership=members,
                         fundamentals_source="edgar", variant=C.STRATEGY_VARIANT)
    if mode == "pit2020":
        return B.run_all(force=force, method=method, variant=C.STRATEGY_VARIANT,
                         universe_override=universe_2020.HOLDINGS_2020,
                         years=PIT_YEARS)
    return B.run_all(force=force, method=method, variant=C.STRATEGY_VARIANT)


def _naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tz is not None else ts


@st.cache_data(ttl=120, show_spinner=False)
def live_inception_vs_spy() -> dict | None:
    """Live PAPER equity history (from the trade log + current account) vs a
    buy-and-hold S&P seeded with the same capital on the inception day.

    This is the FORWARD test — the only verdict that matters. Returns None until
    the trader has logged at least one run with equity.
    """
    import json
    if not C.TRADE_LOG.exists():
        return None
    points: dict[pd.Timestamp, float] = {}
    for ln in C.TRADE_LOG.read_text().strip().splitlines():
        try:
            r = json.loads(ln)
            if r.get("equity") and r.get("time"):
                points[_naive(r["time"])] = float(r["equity"])
        except Exception:  # noqa: BLE001
            continue
    if not points:
        return None

    # Append the freshest live equity so the curve ends at "now".
    try:
        from broker import PaperBroker
        points[_naive(pd.Timestamp.now(tz="UTC"))] = PaperBroker().get_equity()
    except Exception:  # noqa: BLE001
        pass

    dates = sorted(points)
    strat = [points[d] for d in dates]
    start_d, start_e = dates[0], strat[0]

    # SPY buy-and-hold from inception, same starting capital.
    spy_eq = None
    try:
        end = _naive(pd.Timestamp.now(tz="UTC")) + pd.Timedelta(days=2)
        spy = data.get_prices(["SPY"], start_d.date().isoformat(),
                              end.date().isoformat())["SPY"].dropna()
        if not spy.empty:
            spy.index = [_naive(i) for i in spy.index]
            spy0 = spy.iloc[0]
            spy_eq = []
            for d in dates:
                px = spy[[i <= d for i in spy.index]]
                spy_eq.append(start_e * (px.iloc[-1] / spy0) if len(px) else start_e)
    except Exception:  # noqa: BLE001
        spy_eq = None
    return {"dates": dates, "strategy": strat, "spy": spy_eq,
            "start_date": start_d, "start_eq": start_e}


# --- sidebar ----------------------------------------------------------------
st.sidebar.title("Controls")

# Single live model (one model to test/trade/improve). The A/B variant choice is
# fixed by config.STRATEGY_VARIANT — no more dual-model confusion.
VARIANT = C.STRATEGY_VARIANT
VARIANT_NAME = {"A": "A · momentum-only", "B": "B · momentum + quality screen"}[VARIANT]

_UNIV_OPTS = ["S&P 500 point-in-time — the REAL test (RECOMMENDED, ~500 names, honest)",
              "Current holdings (~30 names · survivorship-biased → INFLATED, fake-good)",
              "2020 ETF-holdings (~30 narrow names · loses to S&P · not a fair test)"]
_UNIV_DEFAULT = 0  # always default to the S&P 500 universe (the honest, real one)
univ_mode = st.sidebar.radio(
    "Test universe",
    _UNIV_OPTS, index=_UNIV_DEFAULT,
    help="ONLY the S&P 500 universe is an honest test: real historical index "
         "membership (~500 names, no survivorship bias) + EDGAR fundamentals — "
         "the live universe (model beats S&P ~78%). 'Current holdings' is "
         "survivorship-biased so it looks fake-GREAT (~97%); '2020 ETF-holdings' "
         "is a narrow ~30-name slice that loses (~9%). Judge the model ONLY on "
         "S&P 500.",
)
mode = ("current" if univ_mode.startswith("Current") else
        "pit2020" if univ_mode.startswith("2020") else "sp500")

# ONE model only — the neural network (seed-ensembled MLP). No algorithm picker.
method = C.LIVE_METHOD
st.sidebar.caption("🧠 **Model: Neural Network** (seed-ensembled MLP · momentum) — "
                   "the single live model. Beat the S&P in 78% of bootstrap "
                   "resamples (66% recent). Set in config.LIVE_METHOD.")
if st.sidebar.button("↻ Refresh data (re-download)"):
    st.session_state.force_reload = True  # consumed by the load below (force=True)
    load_bundle.clear()
    st.rerun()

if mode == "sp500":
    st.sidebar.caption(
        f"S&P 500 point-in-time · {PIT_YEARS}y window · EDGAR fundamentals · "
        f"${C.START_CAPITAL:,.0f} · cost {C.COST_PER_TRADE*100:.3f}%/trade · "
        f"no survivorship bias · model: {method}"
    )
    st.sidebar.caption(
        "⚠️ Sector tags are present-day (free yfinance) — a name reclassified "
        "across GICS sectors mid-history uses today's sector. Minor; membership, "
        "fundamentals, and prices are all point-in-time correct."
    )
elif mode == "pit2020":
    st.sidebar.caption(
        f"2020 point-in-time universe · {PIT_YEARS}y window · ${C.START_CAPITAL:,.0f} · "
        f"cost {C.COST_PER_TRADE*100:.3f}%/trade · no survivorship bias · model: {method}"
    )
else:
    st.sidebar.caption(
        f"Current holdings: {len(C.SECTOR_ETFS)} ETFs × top {C.TOP_HOLDINGS_N} · "
        f"{C.BACKTEST_YEARS}y window · ${C.START_CAPITAL:,.0f} · "
        f"cost {C.COST_PER_TRADE*100:.3f}%/trade · ⚠️ survivorship-biased · model: {method}"
    )

st.title("📈 Two-Stage Momentum Strategy")
st.caption(f"Sector momentum → walk-forward selection · single live model "
           f"**{VARIANT_NAME}** · the dashboard backtests; the paper trader places "
           f"the orders (paper only).")

_UNIV_NAME = {"sp500": "S&P 500 point-in-time", "pit2020": "2020 point-in-time",
              "current": "current holdings"}

# Gate the heavy download/backtest behind an explicit click. Nothing downloads
# on page load; the dashboard builds only after Run is pressed (and again after
# switching universe OR model, since each needs a fresh backtest).
if st.sidebar.button("▶ Run backtest / load data", type="primary"):
    st.session_state.loaded_key = (mode, method)
if st.session_state.get("loaded_key") != (mode, method):
    extra = (" The S&P 500 build fetches sectors + EDGAR fundamentals for ~600 "
             "names on first run (several minutes), then caches." if mode == "sp500"
             else "")
    st.info(
        f"Nothing loaded for **{_UNIV_NAME[mode]} · {method}** yet. Press "
        f"**▶ Run backtest / load data** in the sidebar to fetch data and run it. "
        f"Changing universe or model needs a fresh click.{extra}"
    )
    st.stop()

bundle = load_bundle(mode=mode, method=method,
                     force=st.session_state.pop("force_reload", False))
result = bundle.result_a if VARIANT == "A" else bundle.result_b

# --- current holdings -------------------------------------------------------
if not result.holdings_history:
    st.error(
        f"**{result.label}** produced no book on the **{_UNIV_NAME[mode]}** "
        "universe — the data is too thin here (e.g. variant B's quality screen "
        "needs fundamentals this universe lacks). Try the **S&P 500** universe "
        "(EDGAR fundamentals) or switch model in config. Nothing to show."
    )
    st.stop()

last_date = max(result.holdings_history)
weights = result.holdings_history[last_date]
preds = result.pred_history[last_date]
confs = result.conf_history.get(last_date, {})

# gbm/mlp predict a forward RETURN (show as %); lambdarank's "pred" is an
# ordinal within-month ranking SCORE, not a return — show it as a score.
pred_is_return = method in ("gbm", "mlp")
pred_col = "Predicted growth" if pred_is_return else "Edge score (rank)"

rows = []
for tkr, w in weights.items():
    snap = data.get_current_fundamentals(tkr)  # live view: current fundamentals OK
    rows.append({
        "Ticker": tkr,
        "Sector": C.SECTOR_ETFS.get(bundle.stock_sector.get(tkr, ""), "—"),
        "Weight": w,
        "Confidence": confs.get(tkr, float("nan")),
        pred_col: preds.get(tkr, float("nan")),
        "ROE (now)": snap["roe"],
        "D/E (now)": snap["de"],
        "Margin (now)": snap["margin"],
    })
holdings_df = pd.DataFrame(rows)

st.subheader(f"Current Holdings — rebalanced {last_date:%b %Y}")
st.caption("Weights are **conviction-sized** (confidence ÷ volatility), not equal. "
           "‘Confidence’ = model P(beats median next month). " +
           ("‘Predicted growth’ is an **estimate**, not a guarantee."
            if pred_is_return else
            "‘Edge score’ is the lambdarank model's **within-month ranking score** "
            "(higher = preferred), not a return."))

# Panel 1: treemap, sized by conviction weight, colored by sector.
tm = holdings_df.copy()
if pred_is_return:
    tail = "<br>est " + (tm[pred_col] * 100).round(1).astype(str) + "%"
else:
    tail = "<br>score " + tm[pred_col].round(2).astype(str)
tm["label"] = (tm["Ticker"] + "<br>" + (tm["Weight"] * 100).round(1).astype(str) + "%"
               + "<br>conf " + (tm["Confidence"] * 100).round(0).astype(str) + "%" + tail)
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
    ["Position size (largest first)", f"{pred_col} (highest first)",
     "Confidence (highest first)"],
    horizontal=True,
)
col = ("Weight" if sort_by.startswith("Position")
       else "Confidence" if sort_by.startswith("Confidence")
       else pred_col)
table = holdings_df.sort_values(col, ascending=False).reset_index(drop=True)
st.dataframe(
    table.style.format({
        "Weight": "{:.1%}", "Confidence": "{:.0%}",
        pred_col: "{:+.2%}" if pred_is_return else "{:+.3f}",
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
    mr = result.monthly_returns
    spy_mr = bundle.spy_returns.reindex(mr.index)
    spy_cagr = metrics.cagr(spy_mr)
    spy_total = metrics.total_return(spy_mr)

    r1, r2 = st.columns(2)
    r1.metric("Total return", f"{s['total_return']:.1%}",
              delta=f"{s['total_return']-spy_total:+.1%} vs S&P")
    r2.metric("CAGR", f"{s['cagr']:.2%}",
              delta=f"{s['cagr']-spy_cagr:+.2%} vs S&P")
    r3, r4 = st.columns(2)
    r3.metric("Sharpe", f"{s['sharpe']:.2f}")
    r4.metric("Sortino", f"{s.get('sortino', float('nan')):.2f}",
              help="Like Sharpe but penalises only downside volatility.")
    r5, r6 = st.columns(2)
    r5.metric("Max drawdown", f"{s['max_drawdown']:.2%}")
    r6.metric("Calmar", f"{s.get('calmar', float('nan')):.2f}",
              help="CAGR ÷ |max drawdown| — return per unit of pain.")
    r7, r8 = st.columns(2)
    r7.metric("Win rate (mo)", f"{s.get('win_rate', float('nan')):.0%}",
              help="Fraction of months with a positive return.")
    # Book stats: average names held + average monthly turnover.
    books = [result.holdings_history[d] for d in sorted(result.holdings_history)]
    avg_names = (sum(len(b) for b in books) / len(books)) if books else float("nan")
    turns = [sum(abs(books[i].get(t, 0.0) - books[i - 1].get(t, 0.0))
                 for t in set(books[i]) | set(books[i - 1])) / 2
             for i in range(1, len(books))]
    avg_turn = (sum(turns) / len(turns)) if turns else float("nan")
    r8.metric("Avg names", f"{avg_names:.0f}")

    ic = result.ic or {}
    if ic.get("n_months"):
        st.metric("Signal IC (mean)", f"{ic['mean_ic']:+.3f}",
                  help="Cross-sectional rank IC of predictions vs realised "
                       f"returns — the honest signal-quality measure. IR "
                       f"{ic['ic_ir']:.2f} · hit {ic['hit_rate']:.0%} · t "
                       f"{ic['t_stat']:.1f} over {ic['n_months']} months. "
                       "A stable mean ~0.03+ is a real edge; near 0 = no signal.")
    c1, c2 = st.columns(2)
    c1.metric("Avg turnover/mo", f"{avg_turn:.0%}",
              help="Average fraction of the book replaced each rebalance "
                   "(higher = more trading cost).")
    c2.metric("MC P95 drawdown", f"{result.mc['p95_drawdown']:.2%}",
              help=f"P50: {result.mc['p50_drawdown']:.2%} over {C.MC_RUNS} reshuffles")
    if pd.notna(s["sharpe"]) and s["sharpe"] > C.SHARPE_WARN:
        st.warning(f"Sharpe {s['sharpe']:.2f} > {C.SHARPE_WARN}: suspiciously high — "
                   "likely overfitting or a lookahead bug, not a real edge.")
    if ic.get("n_months") and abs(ic["mean_ic"]) < 0.02:
        st.warning(f"IC {ic['mean_ic']:+.3f} is near zero — the per-name signal is "
                   "weak; returns are mostly sector/beta/concentration, which is "
                   "fragile out-of-sample. Don't over-trust the backtest edge.")

st.divider()

# --- live forward test: paper account vs S&P since inception ----------------
st.subheader("📡 Live forward test — paper account vs S&P (since inception)")
st.caption("The real verdict. Your live Alpaca **paper** equity vs a buy-and-hold "
           "S&P seeded with the same capital on day one. The backtest above is the "
           "past; this is the model proving (or disproving) itself going forward. "
           "Populates as the trader logs runs.")
_live = live_inception_vs_spy()
if not _live:
    st.info("No live runs logged yet. Once the paper trader has run, this tracks "
            "your real equity vs S&P from the first logged run.")
else:
    days = max(1, (_live["dates"][-1] - _live["start_date"]).days)
    strat_now = _live["strategy"][-1]
    strat_ret = strat_now / _live["start_eq"] - 1.0
    lf = go.Figure()
    lf.add_trace(go.Scatter(x=_live["dates"], y=_live["strategy"],
                            name="Strategy (paper)", line=dict(width=2)))
    if _live["spy"]:
        lf.add_trace(go.Scatter(x=_live["dates"], y=_live["spy"], name="S&P buy & hold",
                                line=dict(width=2, dash="dash")))
    lf.update_layout(height=320, margin=dict(t=10, l=0, r=0, b=0),
                     yaxis_title="Value ($)", legend=dict(orientation="h"))
    st.plotly_chart(lf, width="stretch")
    g1, g2, g3 = st.columns(3)
    g1.metric("Days live", f"{days}")
    if _live["spy"]:
        spy_ret = _live["spy"][-1] / _live["start_eq"] - 1.0
        g2.metric("Strategy return", f"{strat_ret:+.2%}",
                  delta=f"{strat_ret - spy_ret:+.2%} vs S&P")
        g3.metric("S&P return", f"{spy_ret:+.2%}")
    else:
        g2.metric("Strategy return", f"{strat_ret:+.2%}")
    st.caption(f"Inception {_live['start_date']:%b %d %Y} at "
               f"${_live['start_eq']:,.0f}. Too short to mean anything yet — judge "
               "over months, not days. (Live ≠ backtest; expect divergence.)")

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
                   f"universe: {_UNIV_NAME[mode]}")
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
    bundle2020 = load_bundle(mode="pit2020", method=method)  # cached; samples reused
    prog = st.progress(0.0, text="Re-training…")
    with st.spinner("Running re-trains (each is a full walk-forward backtest)…"):
        res = retrain.retrain_beat_spy(
            bundle2020, n=int(n_retrain), variant="A", method=method,
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
