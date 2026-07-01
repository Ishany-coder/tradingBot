# Gap analysis — this bot vs top systematic strategies

From a 3-lens professional audit (risk, ML methodology, execution) + research on
what top funds (AQR/Man-style momentum implementations, academic best practice)
and best open-source bots (freqtrade, QuantConnect LEAN, Nautilus) do.
Severity = how much it matters for actually beating the S&P out-of-sample.

## CRITICAL

| # | Flaw | What top shops do | Fix (effort) |
|---|---|---|---|
| 1 | **Book = 15 names, ~all semis.** Only diversification is top-3-sector gating; no per-sector cap, no correlation input. Effective breadth ≈ 1–2 independent bets. | Optimizers with factor covariance; sector/correlation caps everywhere. | Cap ≤6 names & ≤40% weight per sector; later LedoitWolf-shrunk inverse-variance sizing. (medium) |
| 2 | **Always 98% long, every regime.** No vol targeting, no trend gate, no drawdown de-risking. Momentum's documented crash risk (2009, 2020) unprotected. | Vol-managed portfolios (Moreira-Muir), TSMOM regime gates (Moskowitz), Barroso-Santa-Clara momentum-vol scaling. | `invest_fraction = min(0.98, TARGET_VOL/realized_vol)`; halve exposure when SPY 12-1 < 0. (medium) |
| 3 | **Selection bias in the headline number.** Deployed config = best of ~13+ trials; 79%/88% win-rates never discounted for multiple testing (no Deflated Sharpe / PBO). | Bailey & López de Prado deflated Sharpe, PBO; report trials count. | Persist per-trial return series; add `deflated_sharpe()`; report it in MODELS.md. (medium) |
| 4 | **IC 0.012 never significance-tested in reporting; random-rank ablation never run.** The stock-picking layer may add ~nothing over sector momentum. | Signal t-stats + ablations standard. | Log IC t-stat/IR everywhere; run the random-pred ablation — if returns match, the ML layer is decoration. (small) |
| 5 | **No monitoring/alerting.** Log-only errors, laptop launchd, no dead-man switch. Bot can die or misbehave silently for days. | freqtrade ships Telegram alerts + healthchecks; LEAN has live monitors. | `alerts.py` → Telegram/Discord webhook on orders+errors; healthchecks.io ping per cycle. (small) |

## HIGH

| # | Flaw | Fix (effort) |
|---|---|---|
| 6 | **No automated kill switch** — no HWM-drawdown halt, no daily-loss limit, no halt on consecutive order errors, no IC-floor gate. Unattended bot needs automated stops. | HWM tracking in `trader.run()`: equity < HWM×0.8 → write STOP. ~30 lines. (small) |
| 7 | **N_STOCKS_MIN=10 unenforced; 25% cap silently violated** on thin months (2-name book → 50/50). | Empty-book fallback below min; fix `_apply_cap` infeasible branch → cash remainder. (small) |
| 8 | **Sizing uses uncalibrated predict_proba** (IC~0 signal → confidence ≈ 0.5+noise; doubles positions on noise). | Shrink toward 0.5 (λ≈0.25) or plain 1/vol until reliability curve validated; MAX_WEIGHT 25%→~12%. (small) |
| 9 | **No open-order awareness / idempotency** — pending orders invisible to the planner; no client_order_id; crash mid-cycle can double-send. | Cancel-all open orders at cycle start; deterministic client_order_id. (small) |
| 10 | **No turnover circuit breaker** — a thin-data month can silently flip the whole book mid-month at market. | If planned turnover >40% and not a new signal month → dry-run + alert. (small) |
| 11 | **Bootstrap win-rate ≠ "beats S&P"** — resamples ONE bull-regime path; leverage artifact inflates it. | Headline = p5/median/p95 excess CAGR, risk-matched. (small) |
| 12 | **Delisted names still missing prices** → residual survivorship bias (SIVB-class left tail absent). | Log monthly PIT-member price coverage; consider Stooq for delisted. (large) |

## MEDIUM (selected)

- **Buys-into-crashes**: 6h loop mechanically buys a name gapping down intramonth (band is symmetric). Fix: suppress buy-side rebalances on names below entry-month price. (small)
- **Sells and buys fired without fill-wait** → transient overspend risk. Fix: sells → poll fills → buys. (medium)
- **Naive equal ensemble deployed at 79% when lambdarank alone was 88%** — no per-head IC/correlation analysis backing the blend. (small analysis)
- **No turnover banding**: hold incumbents to rank ≤30, admit at ≤15 (Novy-Marx-Velikov) — halves turnover for free. (small)
- **No signal-speed diversification**: 4 weekly tranches instead of one monthly cliff. (medium)
- **Monte-Carlo drawdown uses i.i.d. permutation**, contradicting the repo's own block-bootstrap reasoning. Fix: reuse block resampler. (small)
- **`data.py` cache key uses salted `hash()`** — every process restart re-downloads prices. Fix: md5. (small)
- **Order timing**: market orders at arbitrary loop-wake times incl. open/close chop; no slippage measurement vs cost model. (medium)
- **No liquidity screens** (fine at paper size; matters at real size). (small)
- **Defensive factor blend** (quality/low-vol composite) for regime balance. (medium)

## The honest bottom line

The bot's edge as measured is **one bull-regime sector-momentum bet with an
untested stock-picking layer (IC≈0.012), no crash protection, sized by an
uncalibrated probability, evaluated by a metric that flatters it, selected with
uncorrected multiple testing, running unmonitored on a laptop.** Top systematic
shops differ less in signal cleverness and more in: risk overlays, statistical
discipline about their own results, and operational hardening. That is where
every high-value fix above lives.

## Suggested order of work

1. **Safety first (all small):** #5 alerts, #6 kill switch, #7 breadth/cap fix,
   #9 open orders, #10 turnover breaker, buys-into-crashes fix.
2. **Risk overlays (medium):** #1 sector caps, #2 vol targeting + trend gate.
3. **Statistical honesty (small→medium):** #4 ablation + IC t-stats, #3 deflated
   Sharpe, #11 honest headline metric.
4. **Quality (medium):** turnover banding, tranching, ensemble analysis,
   sell-then-buy sequencing.
5. **Data (large):** delisted-price coverage.
