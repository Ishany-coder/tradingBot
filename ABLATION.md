# Ablation ladder — does the ML earn its keep?

S&P 500 PIT universe, 2020→now, costs 5bp, overlays ON (regime gate + vol target), bootstrap n=2000.

- **1. SPY buy&hold** — total +154.6% · CAGR +15.7% · Sharpe 0.94 · maxDD -24% · beats SPY —
- **1b. MTUM (momentum ETF)** — total +182.0% · CAGR +17.5% · Sharpe 0.90 · maxDD -30% · beats SPY 38%
- **2. EW top-3 sector ETFs** — total +173.5% · CAGR +17.0% · Sharpe 1.03 · maxDD -12% · beats SPY 56%
- **3. naive 12-1 rank (no ML)** — total +134.9% · CAGR +15.5% · Sharpe 0.96 · maxDD -18% · beats SPY 28%
- **4. full ensemble** — total +132.1% · CAGR +16.8% · Sharpe 0.94 · maxDD -17% · beats SPY 27%

**ML delta (rung 4 vs rung 3 head-to-head):** ensemble beats the no-ML baseline in **48%** of resamples (mean excess +2.3%).

## Cost sensitivity

- **5bp** — ensemble: total +132.1% · CAGR +16.8% · Sharpe 0.94 · maxDD -17% · beats SPY 27%
  - naive: total +134.9% · CAGR +15.5% · Sharpe 0.96 · maxDD -18% · beats SPY 28%
- **10bp** — ensemble: total +127.2% · CAGR +16.4% · Sharpe 0.92 · maxDD -17% · beats SPY 24%
  - naive: total +130.5% · CAGR +15.2% · Sharpe 0.95 · maxDD -18% · beats SPY 25%
- **15bp** — ensemble: total +122.3% · CAGR +15.9% · Sharpe 0.90 · maxDD -17% · beats SPY 21%
  - naive: total +126.1% · CAGR +14.8% · Sharpe 0.93 · maxDD -18% · beats SPY 23%
- **20bp** — ensemble: total +117.5% · CAGR +15.4% · Sharpe 0.88 · maxDD -18% · beats SPY 19%
  - naive: total +121.8% · CAGR +14.4% · Sharpe 0.91 · maxDD -18% · beats SPY 21%

## Sizing mode (same ensemble preds)

- **inverse_vol** — total +132.1% · CAGR +16.8% · Sharpe 0.94 · maxDD -17% · beats SPY 27% · turnover 0.354
- **conviction** — total +133.0% · CAGR +16.9% · Sharpe 0.95 · maxDD -17% · beats SPY 26% · turnover 0.354
- **equal** — total +127.4% · CAGR +16.4% · Sharpe 0.91 · maxDD -18% · beats SPY 24% · turnover 0.329

## Risk overlays

- **overlays ON** — total +132.1% · CAGR +16.8% · Sharpe 0.94 · maxDD -17% · beats SPY 27%
- **overlays OFF** — total +227.0% · CAGR +24.4% · Sharpe 1.07 · maxDD -20% · beats SPY 68%

## Deflated Sharpe (multiple-testing haircut)

- observed monthly SR 0.271, expected-max under 13 trials 0.086 → **DSR = 0.92** (want ≥ 0.95; below that the edge is not distinguishable from selection luck).

## Follow-up cell: naive, overlays OFF (for a fair no-overlay ladder)

- **naive 12-1 rank, overlays OFF** — total +199.2% · CAGR +20.4% · Sharpe 1.00 · maxDD -21% · beats SPY 52%
- (ensemble overlays OFF, from above: +227.0% · CAGR +24.4% · Sharpe 1.07 · maxDD -20% · beats SPY 68%)

## Conclusions (the honest read)

1. **The ML stack is retired.** In the deployed pipeline (overlays ON) the
   trained ensemble beats the naive momentum rank in **48% of resamples — a coin
   flip**. Without overlays the ensemble's point estimate is higher (+227% vs
   +199%, 68% vs 52%), but the ensemble config was the best of 13+ searched
   trials and its **DSR is 0.92 < 0.95** — that gap is not distinguishable from
   selection luck. Per the pre-registered criterion, the live model is now the
   naive momentum rank (identical measured performance, no 20-minute retrains,
   no seed variance, far fewer overfit knobs). Trained methods remain for research.
2. **Stage 1 (sector momentum) is most of the strategy.** Plain EW top-3 sector
   ETFs: +173%, Sharpe 1.03, maxDD −12%, beats SPY 56% — with zero stock picking.
   The stock-selection layer adds return points in the bull sample but also risk;
   its skill contribution is unproven.
3. **The risk overlays are expensive insurance in this sample**: they cut
   2020→2026 total return from +199/227% to ~+132% and the beat-SPY rate from
   52–68% to ~27%, while trimming maxDD only ~3pts (this sample's crashes were
   V-shaped; the overlays' value case is prolonged bears, which this sample
   doesn't contain). They are kept **deliberately** as crash protection — that
   is a values choice, not a data-fitted one; softening them by tuning on this
   same sample would be exactly the snooping we just paid to eliminate.
4. **Costs matter but don't flip conclusions**: 5→20bp shaves ~4pts of win-rate
   from every stock rung; dividends are total-return consistent on both sides
   (auto-adjusted prices for strategy and SPY alike).
5. **Sizing**: conviction ≈ inverse-vol (indistinguishable — the confidence
   numerator was noise, as suspected); equal-weight slightly worse. Inverse-vol
   stays (same result, no dependence on an uncalibrated classifier).
6. **Honest bottom line**: with crash protection ON, this strategy's 2020→2026
   backtest is roughly SPY-class total return with a different risk profile —
   not a reliable S&P-beater. The live paper curve is the only evaluation that
   counts from 2026-07-01 onward.