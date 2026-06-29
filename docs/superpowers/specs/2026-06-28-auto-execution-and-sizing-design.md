# Auto-Execution + Conviction Sizing — Design

Date: 2026-06-28
Status: Approved

## Goal

Extend the momentum strategy from a sim-only dashboard into an **automated paper
trader** with **conviction-based position sizing** (no more equal weight) and
**zero human input** on orders.

Three additions:
1. A confidence/probability head on the model → per-stock confidence.
2. Conviction position sizing: `weight ∝ confidence / volatility` (not equal).
3. Automated order execution against the Alpaca **paper** account, recomputed
   hourly, dry-run once then fully automatic.

## Constraints

- **Paper account only.** `broker.py` asserts the endpoint contains
  `paper-api` and refuses to run otherwise. No live-money path exists.
- Cash only, no leverage. Target invested ≤ 98% of equity (cash buffer).
- No human input on orders after the first dry-run.

## Model — add a confidence head (`model.py`)

Keep the `GradientBoostingRegressor` (predicted forward 1-month return = *edge*).
Add a `GradientBoostingClassifier` predicting **P(stock's forward return beats
the cross-sectional median that month)** = *confidence*. Both run through the
same expanding-window walk-forward loop (identical no-lookahead contract: train
on months strictly before `t`, predict month `t`).

`walk_forward_predict` returns a DataFrame indexed by (date, ticker) with
columns `pred` (return) and `confidence` (probability in [0,1]). The classifier
label at month `s` is `1` if `target_s > median(target_s across that month)`.

## Sizing — `sizing.py` (new)

Given the selected book at a decision date with each name's `confidence` and
`volatility`:

```
raw_i   = confidence_i / volatility_i      # high confidence + low vol => more $
w_i     = raw_i / sum(raw)                 # normalise to 100%
cap at MAX_WEIGHT (0.25); redistribute excess to uncapped names; renormalise
```

No shorts (weights ≥ 0). Edge enters via **selection** (top-N by predicted
return); confidence/vol enters via **sizing**. This is the user-chosen scheme.

## Selection (unchanged)

Stage 1 sector momentum → top 3 sectors. Stage 2 quality screen (Backtest B) →
rank survivors by predicted return → top `N_STOCKS_MIN..MAX`. Only the *weights*
change from equal to conviction.

## Backtest integration (`backtest.py`)

Replace equal-weight books with `sizing.py` weights. Portfolio monthly return =
`Σ wᵢ · fwd_returnᵢ`; transaction cost on turnover of the conviction weights.
A and B both use conviction sizing (A still has no quality screen / price-only
features; its confidence comes from the price-only classifier).

## Broker — `broker.py` (new)

Thin Alpaca client (alpaca-py or REST) reading `.env`
(`ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`).

- `assert_paper()` — refuse if base URL lacks `paper-api`.
- `get_equity()`, `get_positions()` → {ticker: market_value}.
- `submit_notional(ticker, dollars, side)` — fractional $ market orders, DAY TIF.
- `is_market_open()` via Alpaca clock.
- `dry_run` mode: log intended orders, send nothing.

## Trader — `trader.py` (new, CLI)

One run = the unit of work (cron/launchd/loop call it):
1. `assert_paper()`; honor `STOP` kill-file and `ENABLED` flag.
2. Run pipeline (cached) → latest decision date → selected book + conviction
   target weights.
3. `equity = get_equity()`; target `$_i = w_i · equity · INVEST_FRACTION`.
4. Diff vs current positions. Place an order for name `i` only if
   `|target_$ - current_$| > max(REBALANCE_BAND · equity, MIN_ORDER_USD)`
   (anti-churn). Sell names no longer in the book entirely.
5. **Dry-run-once:** if `data/state/dryrun_done.flag` is absent, force dry-run,
   print the order plan, write the flag, exit. Subsequent runs trade live.
6. Append a JSON record per run to `data/state/trade_log.jsonl`:
   timestamp, equity, per-name {target_w, confidence, pred, action, dollars}.

Market-hours: if closed, recompute + log but skip order submission (unless
`--allow-closed`, which lets Alpaca queue for next open).

## Scheduling — `run_loop.py` (new) + launchd

`run_loop.py`: loop that calls one `trader.run()` per hour during market hours,
sleeps otherwise. A `com.tradingbot.trader.plist` launchd template runs it
unattended on macOS. Cron alternative documented in README.

## Dashboard (`app.py`)

- Holdings treemap/table: tiles now reflect **conviction weights**; add a
  **Confidence** column and show it on tiles.
- New **Execution panel**: live Alpaca equity, current positions vs target
  weights, last N trade-log entries, dry-run/live status, market open/closed,
  next-run note. Read-only (the loop does the trading).

## Config additions (`config.py`)

`MAX_WEIGHT=0.25`, `INVEST_FRACTION=0.98`, `REBALANCE_BAND=0.03`,
`MIN_ORDER_USD=10`, `RECOMPUTE_HOURS=1`, `ENABLED=True`, `CONF_LABEL="beats
cross-sectional median"`, state dir `data/state/`.

## Guardrails summary

Paper-only assert · cash only ≤98% · 25% max position · rebalance band anti-churn
· market-hours gate · `STOP` kill-file · dry-run once · full per-run logging.

## Honesty note

Hourly execution of a 1-month-horizon signal adds little statistical edge and
risks churn; the rebalance band makes most hourly runs no-ops by design, giving
fresh live updates without cost bleed. Documented in README + UI.

## Out of scope

Live-money trading, options/shorting, intraday alpha, multi-account.
