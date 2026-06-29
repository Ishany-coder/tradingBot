# Momentum Equity Strategy + Dashboard + Auto-Trader

Two-stage momentum equity strategy (sector momentum → stock selection) with a
walk-forward machine-learning prediction model, conviction-based position
sizing, a realistic backtest, a Streamlit dashboard, and an **automated
Alpaca paper trader**.

> **Paper account only.** The broker layer hard-refuses any endpoint that is not
> `paper-api.alpaca.markets`. No real money is ever at risk. This is a research /
> educational tool, not investment advice.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First launch downloads price + fundamentals data from yfinance and caches it to
`data/cache/` (parquet/pickle). Subsequent launches are fast. Use the sidebar
**↻ Refresh** button to re-download.

## Strategy

**Stage 1 — sector momentum.** Rank the 11 SPDR sector ETFs by *12-1 momentum*
(return from t-12mo to t-1mo, skipping the most recent month). Hold the top 3.

**Stage 2 — stock selection.** Within the chosen sectors:
- (Backtest B) **quality screen**: positive TTM net income, ROE > 0, and
  debt-to-equity below the sector median.
- rank survivors by the model's predicted forward 1-month return,
- buy the top 10–15, **equal weight**, rebalanced **monthly**.

## Prediction model

`sklearn.GradientBoostingRegressor` predicts each stock's forward 1-month
return from: 12-1 / 6-month / 3-month momentum, ROE, debt-to-equity, profit
margin, and 60-day volatility.

**Walk-forward (expanding window) — no lookahead.** To decide the book held over
month `t → t+1`, the model trains only on samples whose target return is already
realised by `t` (i.e. months strictly before `t`). It never sees a return that
unfolds at or after the decision date. The loop is heavily commented in
`model.py`; this is the single most important correctness property.

The predicted score is an **estimate, not a guarantee**, and is labelled that
way in the UI.

## Backtest A vs B

| | A — baseline | B — full |
|---|---|---|
| Quality screen | no | yes |
| Model features | price-only (momentum + vol) | all seven |
| History | long (price only) | limited by fundamentals |

They are compared head-to-head (CAGR, max drawdown, Sharpe, Monte Carlo P95
drawdown) so you can see whether the quality screen actually earns its keep.

> **Caveat:** A also lacks the fundamental *features*, so this is a bundle
> comparison, not a perfectly clean screen-only ablation — the best the
> available data allows.

## Backtest realism

- Transaction cost **0.05% per trade** charged on turnover at every rebalance.
- **No lookahead** anywhere — only data available at the decision date.
- Reports total return, CAGR, max drawdown, Sharpe, and the equity curve vs
  buy-and-hold SPY.
- **Monte Carlo**: 1000 reshuffles of the monthly returns → P50 and P95
  drawdown (the single historical path understates tail risk).
- If Sharpe > 2.5, a **warning** is printed/shown — suspiciously high usually
  means overfitting or a lookahead bug, not a real edge.

## Known data limitations (read this)

- **ETF constituents:** yfinance only exposes the *current* top ~10 holdings per
  ETF, not historical membership → **survivorship bias** in the universe.
- **Fundamentals history is thin.** yfinance now serves only ~5 quarters of
  quarterly statements, so the backtest merges **annual** statements (~4 years)
  for the backbone with **quarterly** for recency. Each report is lagged by
  `FUND_LAG_QUARTERS` (default 2) quarters to approximate filing delay. Stocks
  with missing fundamentals at a decision date are **excluded** that month
  (never forward-filled or guessed). If a large fraction drops out, the app
  warns that the data is too thin to trust the quality screen — a real
  point-in-time source (Financial Modeling Prep, Sharadar) would be needed for a
  credible Backtest B.
- The **live current-holdings view** uses current fundamentals freely (no
  lookahead in the present); the point-in-time lag applies only to the backtest.

## Position sizing — conviction, not equal weight

Books are **not** equal weight. Selection picks *which* names (top-N by predicted
return = edge); sizing picks *how much* capital each gets:

```
weight_i  ∝  confidence_i / volatility_i      # high conviction + low vol => more $
```

capped at 25% per name, long-only, normalised to 100%. `confidence` is a second
model head — a `GradientBoostingClassifier` predicting P(the stock beats the
cross-sectional median next month), run through the same walk-forward loop. Both
the predicted return and the confidence are shown per holding in the dashboard.

## Automated paper trading

`trader.py` runs one full cycle: compute conviction targets → read live Alpaca
equity + positions → diff to target dollars → place fractional-$ market orders
to close the gap. `run_loop.py` repeats it every `RECOMPUTE_HOURS` (default 1).

```bash
python trader.py            # one cycle (first ever run is a dry-run)
python trader.py --dry-run  # plan only, never sends
python run_loop.py          # unattended hourly loop
```

**Dry-run once, then full auto.** The very first real cycle prints the order plan
and sends nothing, then writes `data/state/dryrun_done.flag`. Every cycle after
is fully automatic — zero human input on orders.

**Run unattended (macOS launchd):**
```bash
cp com.tradingbot.trader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tradingbot.trader.plist   # starts loop, restarts on boot
launchctl unload ~/Library/LaunchAgents/com.tradingbot.trader.plist # stop
```
Cron alternative (every hour, market days):
```
0 * * * 1-5 cd /Users/ishanghosh/projects/tradingBot && .venv/bin/python trader.py >> data/state/cron.log 2>&1
```

### Safety guardrails
- **Paper-only**: broker asserts `paper-api` in the URL or refuses to run.
- **Cash only**, ≤ `INVEST_FRACTION` (98%) of equity invested; never levered.
- **25% max** per position.
- **Rebalance band**: a name only trades when its target vs current weight drifts
  past `max(3% of equity, $10)` — stops an hourly cadence from churning a
  monthly-horizon signal into costs. (Side effect: target positions below ~3% of
  equity may not be entered; lower `REBALANCE_BAND` to include them.)
- **Market-hours gate**: orders submit only when the market is open (fractional
  orders are rejected otherwise); closed runs recompute + log only.
- **Kill switch**: `touch data/state/STOP` halts all trading; or set
  `ENABLED=False` in config.
- Every cycle appends a full JSON record to `data/state/trade_log.jsonl`.

> **Honesty note:** trading a 1-month-forward signal hourly adds little
> statistical edge; the rebalance band makes most hourly cycles no-ops by design,
> giving you a live-updating view without bleeding transaction costs.

## Dashboard panels

1. **Current holdings** treemap — tiles sized by weight, coloured by sector,
   labelled with ticker + weight + predicted growth.
2. **Holdings table** — sort by position size or by predicted growth.
3. **Equity curve** — strategy vs buy-and-hold SPY.
4. **Metrics** — CAGR, max drawdown, Sharpe, Monte Carlo P95 drawdown.
5. **Sector view** — the top-3 selected sectors and their momentum ranks.
6. **Live execution** — Alpaca paper equity, current positions vs target weights,
   trader status (dry-run/live), and the recent trade log (read-only).

## Configuration

All tunables live in `config.py`: universe, `N_SECTORS`, book size, momentum
lookbacks, volatility window, `FUND_LAG_QUARTERS`, `MIN_TRAIN_MONTHS`, GBM
params, cost, backtest window, starting capital, Monte Carlo runs, and the
Sharpe warning threshold.

## File layout

| File | Purpose |
|---|---|
| `config.py` | all tunables |
| `data.py` | yfinance fetch + on-disk cache |
| `universe.py` | sector ETFs → constituents |
| `features.py` | momentum, volatility, point-in-time fundamentals |
| `model.py` | walk-forward GBM regressor (edge) + classifier (confidence) |
| `sizing.py` | conviction weights (confidence ÷ vol, capped) |
| `metrics.py` | CAGR / drawdown / Sharpe |
| `montecarlo.py` | drawdown distribution |
| `backtest.py` | rebalance engine, runs A & B |
| `broker.py` | Alpaca paper client (paper-only guard) |
| `trader.py` | one automated trading cycle (CLI) |
| `run_loop.py` | unattended hourly loop |
| `com.tradingbot.trader.plist` | launchd template for unattended runs |
| `app.py` | Streamlit dashboard |
