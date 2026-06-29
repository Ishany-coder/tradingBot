# Momentum Equity Strategy + Dashboard — Design

Date: 2026-06-28
Status: Approved

## Goal

Python app: two-stage momentum equity strategy (sector momentum → stock
selection) with a walk-forward ML prediction model, realistic backtest, and a
Streamlit dashboard. **Simulation only — no broker connection, no real orders.**

Stack: `yfinance` (prices + fundamentals), `pandas` (data layer),
`scikit-learn` (`GradientBoostingRegressor`), `streamlit` (UI), `plotly`
(treemap + charts).

## Strategy

### Stage 1 — Sector momentum
- Universe: 11 sector ETFs — XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC.
- Each month rank sectors by **12-1 momentum** (return t-12mo → t-1mo, skipping
  the most recent month).
- Select top 3 sectors.

### Stage 2 — Stock selection
- Constituents: yfinance `funds_data.top_holdings` (current top ~10 per ETF).
  Caveat: current holdings only → survivorship bias; no historical membership.
- **Quality screen (Backtest B only)** — keep stocks passing ALL:
  - positive TTM net income (profitable)
  - ROE > 0
  - debt-to-equity below the sector median
- **Momentum** — 12-1 for survivors.
- Rank by the model's predicted forward 1-month return.
- Buy top 10–15 names, **equal weight**. Rebalance **monthly**.

## Prediction model
- `GradientBoostingRegressor` predicts **forward 1-month return**.
- Features: 12-1 momentum, 6m momentum, 3m momentum, ROE, debt-to-equity,
  profit margin, 60-day volatility.
- **Walk-forward / expanding window** (no random splits). At decision month `t`,
  train only on samples whose forward return is already realized
  (`s <= t-1`), then predict month `t`. Never sees `t → t+1` actuals.
- Output "predicted growth" per held stock. Labeled in UI as an **estimate, not
  a guarantee**.

## No-lookahead — the auditable core
```
months sorted; first prediction after MIN_TRAIN_MONTHS (default 24)
for decision month t:
    # sample at month s: features known at s, target = return s->s+1
    # return s->s+1 is realized at s+1. At t we only know returns up to t.
    # train ONLY on s where s+1 <= t  (i.e. s <= t-1). zero future leak.
    train = all (X_s, y_s) for s <= t-1
    fit GBM on train
    predict forward return for month-t candidates
    Stage1: top 3 sectors by 12-1
    Stage2: [B] quality screen -> survivors; rank by predicted; top 10-15 eq-wt
    apply COST x turnover
```

## Fundamentals — point-in-time (Backtest B)
- Pull yfinance **quarterly** income statement + balance sheet (~4-5yr).
- yfinance gives fiscal **period-end** dates, not filing dates. Companies file
  weeks-to-months later. Lag each report by **FUND_LAG_QUARTERS = 2** (config,
  conservative) before it is usable at a decision date.
- At each monthly decision date, use only the most recent report whose
  lagged-availability date is `<= decision date`.
- **Missing-data rule:** if a stock lacks valid fundamentals at a decision date,
  **exclude** it that month. No forward-fill of stale numbers (mild lookahead),
  no guessing. Log drop count per month; warn if a large fraction routinely
  drops (signals ~5yr yfinance data too thin → need a real point-in-time source
  like FMP / Sharadar).
- **Forbidden:** current-snapshot proxy across history (lookahead +
  survivorship). Never used in the backtest.

## Backtest A vs B (the experiment)
- **A — baseline:** price-only, no quality screen. Ranks survivors with the GBM
  trained on **price-only features** (mom 12-1/6/3 + vol). Long history.
- **B — full:** quality screen + GBM on all 7 features, fundamentals lagged 2Q,
  missing excluded.
- Compare on the **overlapping window**: CAGR, max drawdown, Sharpe, MC-P95
  drawdown side by side. State explicitly whether the quality screen improved
  risk-adjusted return / drawdown; if not, flag it as possibly unnecessary.
- Caveat (shown in UI): A also lacks fundamental *features*, so this is a
  bundle comparison, not a perfectly clean screen-only ablation — best the data
  allows.

## Backtest realism
- Transaction cost **0.05% per trade** (commission + slippage) on every buy/sell,
  charged on turnover at each rebalance.
- No lookahead anywhere — only data available at the decision date.
- Report: total return, CAGR, max drawdown, Sharpe, equity curve.
- **Monte Carlo:** reshuffle monthly returns 1000x → report P50 and P95
  drawdown (not just single-path drawdown).
- **Sharpe > 2.5 → print warning** (likely overfitting or lookahead, not edge).

## Live current-holdings view
- The live dashboard view uses **current** fundamentals freely (no lookahead in
  the present). Point-in-time lag rules apply ONLY to the historical backtest.

## Dashboard (Streamlit)
1. **Current holdings** treemap — tile size = weight, color = sector, label =
   ticker + weight + predicted growth.
2. **Holdings table** — toggle sort: (a) position size desc, (b) predicted
   growth desc.
3. **Equity curve** — strategy vs buy-and-hold SPY, same chart.
4. **Metrics panel** — CAGR, max drawdown, Sharpe, MC-P95 drawdown.
5. **Sector view** — current top-3 sectors + their momentum ranks.

## Config block (config.py)
SECTOR_ETFS, N_SECTORS=3, N_STOCKS_MIN=10, N_STOCKS_MAX=15, COST=0.0005,
FUND_LAG_QUARTERS=2, MIN_TRAIN_MONTHS=24, LOOKBACKS (12-1/6/3), VOL_WINDOW=60,
BACKTEST_YEARS=5, START_CAPITAL=100_000, MC_RUNS=1000, SHARPE_WARN=2.5,
TOP_HOLDINGS_N=10, CACHE_DIR.

## Module layout
- `config.py` — tunables
- `data.py` — yfinance fetch + parquet disk cache
- `universe.py` — sector ETFs + top-10 holdings
- `features.py` — momentum, vol, point-in-time fundamentals
- `model.py` — GBM walk-forward
- `metrics.py` — CAGR / max drawdown / Sharpe
- `montecarlo.py` — reshuffle → P50/P95 drawdown
- `backtest.py` — monthly rebalance engine, costs, runs A & B
- `app.py` — Streamlit dashboard
- `requirements.txt`, `README.md`

## Out of scope
- Real broker orders (Alpaca etc.). Strictly simulation.
- Intraday data, options, shorting.
