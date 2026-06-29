# Model search — beat the S&P 60% of the time?

Metric: paired block-bootstrap (n=2000) win-rate of strategy total return vs SPY.
`win_rate (2020→)` = full window; `recent (2023→)` = rough out-of-sample check.
Target: **win-rate ≥ 60%**.

Honest caveat: a high win-rate with near-zero **IC** (information coefficient) is
magnitude/concentration in a bull market, not stock-picking skill. The recent
slice is the most useful generalization signal; even so, the 2020–2026 sample is
mostly a tech bull, so the bootstrap does not test a regime change.

## Results

| Model | Universe | win 2020→ | recent 2023→ | edge | CAGR | Sharpe | IC | verdict |
|---|---|---|---|---|---|---|---|---|
| A-lambdarank | S&P 500 PIT | **88%** | **73%** | +310% | +30.7% | 1.04 | +0.0158 | ✅ beats 60% (best) |
| A-gbm | S&P 500 PIT | 84% | 70% | +292% | +30.1% | 1.02 | +0.0040 | ✅ beats 60% |
| B-lambdarank | S&P 500 PIT | 45% | 54% | +40% | +18.0% | 0.79 | +0.0100 | ❌ (currently LIVE) |
| B-gbm | S&P 500 PIT | 37% | 42% | +11% | +16.0% | 0.81 | +0.0116 | ❌ |
| A-lambdarank | 2020 PIT (narrow) | 25% | — | −40% | +12.1% | 0.72 | +0.0084 | ❌ loses to S&P |
| A-gbm | 2020 PIT (narrow) | 21% | — | −44% | +11.8% | 0.75 | −0.0198 | ❌ loses to S&P |
| A-mlp | 2020 PIT (narrow) | 14% | — | −48% | +11.5% | 0.74 | +0.0043 | ❌ loses to S&P |
| B-gbm | 2020 PIT (narrow) | 1% | — | −16% | +7.1% | 0.59 | +0.0350 | ❌ (18mo, starved) |
| B-lambdarank | 2020 PIT (narrow) | 0% | — | −18% | +5.6% | 0.48 | +0.0516 | ❌ (18mo, starved) |

## Conclusions

- **Winner: `sp500 / A (momentum-only) / lambdarank`** — beats S&P in 88% of
  resampled histories (73% on the recent 2023→ slice). Clears the 60% bar in and
  out of sample.
- The **narrow 2020 ETF-holdings universe cannot beat the S&P** with any model —
  it is a concentrated mega-cap slice that lagged. Breadth (full S&P 500) is what
  creates the edge.
- The **currently-live model (B / quality screen) does NOT meet the bar** (45%).
  Its quality screen drops too many names on this universe, cutting the
  momentum winners that drive the edge.
- IC stays ~0.004–0.016 everywhere — tiny. The win-rate is driven by the large
  compounded magnitude of a concentrated momentum book in the 2020–2024 bull,
  not by reliable per-name skill. Treat the live forward curve as the real test.
