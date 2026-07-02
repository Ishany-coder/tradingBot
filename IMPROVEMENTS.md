# Win-rate improvement round — PRE-REGISTERED 2026-07-01

Goal: raise the block-bootstrap probability of beating SPY to ≥ 50%, without
repeating the data snooping the ablation just exposed. This file is written and
committed BEFORE the grid is run; the decision rule below is binding.

## The grid (32 structural cells, all literature-anchored)

| Dimension | Values | Basis |
|---|---|---|
| Overlays | both ON · gate only · vol-target only · none | Faber SMA gate; Moreira-Muir vol targeting — which insurance costs least? |
| Sectors held | 3 · 4 | breadth (fundamental law: IR ≈ IC·√breadth) |
| Book size | 15 · 25 | breadth |
| Rank signal | 12-1 momentum z · momentum/vol (risk-adjusted) z | risk-adjusted momentum literature |

Model: naive rank (no ML — retired by ablation). Sizing: inverse-vol, 12% name
cap, 6-per-sector cap, hysteresis + banding as deployed. Costs 5bp.

## Windows

* **Selection: 2012-01 → 2019-12.** Fresh — never used for any decision in this
  project. Contains 2015-16 chop and 2018 Q4 — regimes the burned window lacks.
* **Validation: 2020-01 → now.** Partially burned by earlier decisions —
  treated as secondary confirmation only.

## Binding decision rule

1. Compute win-rate vs SPY (paired block bootstrap, n=2000, block 6) on both
   windows for all 32 cells.
2. Qualify: selection-window win ≥ 55% (buffer over 50% for 32-trial
   multiplicity) AND validation-window win ≥ 50%.
3. Deploy the qualifying cell with the highest MIN(selection, validation).
   Ties → fewer changes vs current config.
4. **If no cell qualifies: deploy nothing**, report honestly, and iterate on
   research-backed signal improvements instead of re-cutting this grid.
5. Winner gets a Deflated-Sharpe check on the selection window (N=32 trials);
   DSR < 0.95 is reported prominently either way.

## Pre-registered caveats

* Pre-2020 has thinner delisted-price coverage → survivorship inflation is
  LARGER in the selection window. Cell-vs-cell comparisons stay informative
  (same bias applies to all); absolute win-rates are optimistic.
* XLRE (2015) and XLC (2018) ETFs don't exist for much of the selection window
  → those sectors are unrankable early; ~9 sectors effectively.
* 32 correlated trials: the max cell is expected to look good by chance; hence
  the 55% buffer, the two-window intersection, and the DSR report.
