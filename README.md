# Momentum Equity Strategy + Dashboard + Auto-Trader

Two-stage momentum equity strategy (sector momentum → stock selection) driven by
an **ensemble machine-learning model**, walk-forward trained with strict
no-lookahead discipline, on a **survivorship-bias-free S&P 500 universe** with
**free point-in-time SEC EDGAR fundamentals**. Includes conviction-based sizing,
a realistic backtest, a Streamlit dashboard, and an **automated Alpaca paper
trader**.

> **Paper account only.** The broker layer hard-refuses any endpoint that is not
> `paper-api.alpaca.markets`. No real money is ever at risk. This is a research /
> educational tool, **not investment advice**. Backtested outperformance is not a
> promise of future results — see the honesty notes throughout.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # includes lightgbm; macOS: brew install libomp
streamlit run app.py
```

In the dashboard: pick the **S&P 500 point-in-time** universe (the default and
the only honest test — see *Universes* below) and press **▶ Run backtest / load
data**. First run downloads + caches data (~15–20 min for the full S&P 500 build,
then cached); later runs reuse the cache.

---

## How the model works

The strategy makes one decision per month: **which ~15 stocks to hold, and how
much of each.** It does this in two stages.

### Stage 1 — sector momentum
Rank the 11 SPDR sector ETFs by **12-1 momentum** (return from t-12 months to
t-1 month, skipping the most recent month to avoid short-term reversal). Keep the
top `N_SECTORS` (default 3). Only stocks in those leading sectors are eligible.

### Stage 2 — model-ranked stock selection
Within the leading sectors, an ML model **ranks** every eligible stock by its
expected forward 1-month performance. The top `N_STOCKS_MAX` (default 15) are
bought, **conviction-weighted** (see *Position sizing*), and rebalanced monthly.

### The model = an ensemble of three models

`config.LIVE_METHOD = "ensemble"`. Each month the pipeline runs **three different
learners** and blends them (this is the "run multiple models and combine them"
design). Blending decorrelated models is more robust than trusting any one:

| Sub-model | What it is | Role |
|---|---|---|
| **GBM** | `sklearn.GradientBoostingRegressor` — predicts forward 1-mo return | pointwise edge |
| **lambdarank** | `lightgbm.LGBMRanker` (learning-to-rank) — optimises the monthly *ranking* directly | ordinal ranker |
| **NN** | seed-ensembled `sklearn.MLPRegressor` (5 nets averaged, `MLP_SEEDS`) | neural net, different inductive bias |

**How they combine** (`model._fit_ensemble`): each sub-model scores every stock
for the month; each score is converted to a **cross-sectional rank**; the three
ranks are **averaged**. Selection uses that blended rank. Rank-averaging (not
score-averaging) keeps any one model from dominating just because its raw scores
are on a bigger scale. Confidence (for sizing) is the mean of the three models'
confidence heads.

### Features (inputs)
Each `(month, stock)` row is described by, all **cross-sectionally z-scored within
the month** (and clipped to ±4 SD so one outlier can't dominate):

- **Momentum**: 12-1, 6-month, 3-month returns
- **Volatility**: 60-day daily-return std
- **ETF relative strength** (`etf_rs`): the stock's trailing-120d beta to a basket
  of the *currently leading* sector ETFs — i.e. "is this name riding the ETFs
  that are working?" (pure price data, no look-ahead)
- **Fundamentals** (variant B only): ROE, debt-to-equity, profit margin, from SEC
  EDGAR point-in-time filings

### Variants A and B
- **A — momentum-only** (deployed): price + ETF-RS features, no fundamental screen.
- **B — momentum + quality**: adds a fundamental screen (positive TTM net income,
  ROE > 0, debt/equity below sector median) + fundamental features.

On the S&P 500 universe, **A** wins (the quality screen drops too many momentum
leaders); `config.STRATEGY_VARIANT = "A"`.

---

## How the model is trained

**Walk-forward, expanding window, retrained from scratch every month.** There is
no single saved "trained model" — each monthly decision trains fresh on all
history available at that point. This is the auditable core (`model.py`,
heavily commented).

### The no-lookahead contract
A training sample is one `(month s, stock)` row: its **features** come from data
known at month-end `s`; its **target** is the return `s → s+1`, only realised at
`s+1`. To predict month `t`, the model may train **only on rows with date < t**
(so every training target is already realised before `t`). It never sees a return
at or after the decision date. The classifier's "winner" label likewise compares
a sample to the median of *its own past month*, so no future information leaks.

### Per-month training loop
For each decision month `t` (from `MIN_TRAIN_MONTHS`, default 24, onward):
1. **Slice** training rows = all `(s, stock)` with `s < t`.
2. **Fit** all three sub-models on that slice:
   - GBM regressor on realised forward returns;
   - lambdarank on per-month integer relevance grades (`RANK_GRADES` quantile
     buckets of the target), grouped by month;
   - the NN as **5 independently-seeded MLPs**, each behind a `StandardScaler`
     refit on the training slice only (no scaling leakage) — their predictions
     averaged (single-seed MLPs are too noisy to use alone);
   - plus a confidence head per model = P(stock beats the cross-sectional median
     next month).
3. **Predict** the rows at month `t`, rank-blend the three, and that is the book
   for `t → t+1`.

Because every month refits on a longer window, the model "continuously trains" as
new data arrives. The per-month fits are independent, so they run in parallel
across CPU cores (`joblib`).

### Universes (what data it trains on)
- **S&P 500 point-in-time** (deployed): real historical index membership from a
  free MIT-licensed dataset (`sp500.py`), so each month only trades names that
  were *actually* in the index then — **no survivorship bias**. ~9-year window
  (`years=9`) so the first prediction lands ~2019 and 2020→now is fully covered.
- **Current holdings** / **2020 point-in-time**: two narrow ~30-name ETF-holding
  sets. **Do not judge the model on these** — they're a lagging slice of the
  index (2020 loses to the S&P; "current holdings" is survivorship-*inflated*).
  Kept only for comparison.

### Point-in-time fundamentals (SEC EDGAR)
`edgar.py` pulls each company's fundamentals from the free SEC XBRL `companyfacts`
API keyed on the **actual filing date** (`filed`), so a value is only usable once
its filing became public — true point-in-time, no synthetic lag. Uses annual
(10-K) net income & revenue with the latest quarterly balance-sheet equity/debt;
de-dupes restatements by latest filing; requires equity > 0. yfinance is the
fallback source on the narrow universes.

---

## Position sizing — conviction, not equal weight

Selection picks *which* names (top-N by blended rank); sizing picks *how much*:

```
weight_i  ∝  confidence_i / volatility_i      # high conviction + low vol => more $
```

capped at `MAX_WEIGHT` (25%) per name, long-only, normalised to 100%.
`confidence` is the ensemble's blended P(beats cross-sectional median next month).

---

## How good is it? (measured, with caveats)

Judged by a **paired block-bootstrap win-rate** vs SPY: resample the monthly
return history many times, count the fraction of resampled paths where the
strategy's total return beats buy-and-hold SPY. Full model ranking is in
[`MODELS.md`](MODELS.md). Deployed ensemble, S&P 500 universe, 2020→now:

| | win-rate 2020→ | recent 2023→ (out-of-sample) | edge | CAGR | Sharpe |
|---|---|---|---|---|---|
| **Ensemble (deployed)** | **79%** | **66%** | +312% | +31% | 1.01 |
| lambdarank alone (best single) | 88% | 73% | +310% | +31% | 1.04 |

> **Read this honestly.** The **information coefficient (IC ≈ 0.012)** — the rank
> correlation of predictions with realised returns — is *tiny*. That means the
> high win-rate is driven by the large compounded magnitude of a concentrated
> momentum book during the 2020–2024 tech bull, **not** by reliable per-name
> stock-picking skill. The recent-slice number is the best out-of-sample signal,
> but the whole sample is one bull regime. The **live paper forward curve** (the
> 📡 dashboard panel) is the only test that can't be curve-fit. A backtest win-rate
> is not a promise of beating the S&P going forward.

The dashboard prints a warning whenever IC is near zero or Sharpe > 2.5
(suspiciously high usually = overfitting / a lookahead bug, not a real edge).

---

## Backtest realism

- Transaction cost **0.05% per trade** charged on turnover at every rebalance.
- **No lookahead** anywhere — only data available at the decision date.
- Reports total return, CAGR, max drawdown, Sharpe, Sortino, Calmar, monthly win
  rate, average names, turnover, IC, and the equity curve vs buy-and-hold SPY.
- **Monte Carlo**: `MC_RUNS` (1000) reshuffles of the monthly returns → P50/P95
  drawdown (one historical path understates tail risk).
- **Robustness / re-train buttons**: bootstrap the win-rate, or re-fit the model
  under many seeds, to separate a real edge from one lucky path.

---

## Automated paper trading

`trader.py` runs one cycle: build the ensemble target book → read live Alpaca
equity + positions → diff to target dollars → place fractional-$ market orders to
close the gap. `run_loop.py` repeats it every `RECOMPUTE_HOURS` (default 6),
**market hours only**. Universe / model / variant default to the `config.LIVE_*`
values so the loop trades exactly the deployed model.

```bash
python trader.py            # one cycle (first ever run is a dry-run)
python trader.py --dry-run  # plan only, never sends
python run_loop.py          # unattended loop (this is what runs continuously)
```

**Dry-run once, then full auto.** The first real cycle prints the plan and sends
nothing, then writes `data/state/dryrun_done.flag`; every cycle after is fully
automatic.

**Run unattended (macOS launchd) — keeps trading across reboots:**
```bash
cp com.tradingbot.trader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tradingbot.trader.plist   # start + restart on boot
launchctl unload ~/Library/LaunchAgents/com.tradingbot.trader.plist # stop
```
Keep the machine awake (`caffeinate -is &`) or the loop pauses on sleep.

### Safety guardrails
- **Paper-only**: broker asserts `paper-api` in the URL or refuses to run.
- **Cash only**, ≤ `INVEST_FRACTION` (98%) invested; never levered. **25% max** per name.
- **Rebalance band**: a name trades only when target vs current weight drifts past
  `max(3% equity, $10)` — stops the loop churning a monthly signal into costs.
- **Market-hours gate**: orders submit only when the market is open; the loop
  also skips the (expensive) rebuild while the market is closed.
- **Kill switch**: `touch data/state/STOP` halts all trading (delete to resume);
  or set `ENABLED = False` in config.
- Every cycle appends a full JSON record (equity, book, orders, IC, turnover) to
  `data/state/trade_log.jsonl`.

---

## Dashboard panels

1. **Current holdings** treemap + sortable table (weight, confidence, score).
2. **Equity curve** — strategy vs buy-and-hold SPY (backtest).
3. **Metrics** — total return & excess vs S&P, CAGR, Sharpe, Sortino, Calmar,
   monthly win rate, avg names, turnover, IC, Monte Carlo P95 drawdown.
4. **📡 Live forward test** — real paper equity vs S&P since inception (the honest test).
5. **🎲 Robustness** — bootstrap win-rate vs S&P.
6. **🔁 Re-train test** — re-fit under many seeds to check the edge isn't luck.
7. **Sector view** — the top-3 selected sectors and their momentum ranks.
8. **Live execution** — Alpaca paper equity, positions vs target, trader status.

---

## Configuration

All tunables live in `config.py`. Key deployment switches:

| Key | Meaning | Deployed |
|---|---|---|
| `LIVE_METHOD` | model: `gbm` \| `lambdarank` \| `mlp` \| `ensemble` | `ensemble` |
| `LIVE_UNIVERSE` | `current` \| `pit2020` \| `sp500` | `sp500` |
| `STRATEGY_VARIANT` | `A` (momentum-only) \| `B` (+ quality screen) | `A` |
| `MLP_SEEDS` | nets averaged in the NN | 5 |
| `RECOMPUTE_HOURS` | loop cadence (market hours only) | 6 |

Plus `N_SECTORS`, book size, momentum lookbacks, cost, window, Monte Carlo runs,
rebalance band, kill switch. **Restart the Streamlit server and the loop after
editing `config.py`** — Python caches the module at import.

## File layout

| File | Purpose |
|---|---|
| `config.py` | all tunables + live-model switches |
| `data.py` | yfinance price/fundamentals fetch + on-disk cache |
| `sp500.py` | free point-in-time S&P 500 membership + sector map |
| `edgar.py` | free SEC EDGAR point-in-time fundamentals |
| `universe.py` | sector ETFs → constituents (narrow universes) |
| `features.py` | momentum, volatility, ETF relative strength, PIT fundamentals |
| `model.py` | walk-forward training + GBM / lambdarank / NN / **ensemble** |
| `sizing.py` | conviction weights (confidence ÷ vol, capped) |
| `metrics.py` | CAGR / drawdown / Sharpe / Sortino / Calmar / IC / win rate |
| `montecarlo.py` | drawdown distribution |
| `robustness.py` | bootstrap win-rate vs S&P |
| `retrain.py` | re-fit under many seeds (overfit check) |
| `backtest.py` | sample matrix + rebalance engine |
| `model_search.py` | evaluate/compare model configs (→ `MODELS.md`) |
| `broker.py` | Alpaca paper client (paper-only guard) |
| `trader.py` | one automated trading cycle (CLI) |
| `run_loop.py` | unattended market-hours loop (runs continuously) |
| `com.tradingbot.trader.plist` | launchd template for unattended runs |
| `app.py` | Streamlit dashboard |
| `MODELS.md` | model-search results + honesty notes |
