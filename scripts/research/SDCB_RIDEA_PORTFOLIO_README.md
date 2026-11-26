# SDCB greedy R-I / R-IDeA portfolio reference

Standalone Rust reference for selecting a design-only portfolio with sequential MMD² representativeness updates.

## Build and run

```bash
rustc --edition=2024 --test scripts/research/sdcb_ridea_portfolio.rs -o /tmp/sdcb-ridea-portfolio
/tmp/sdcb-ridea-portfolio
```

The source uses the sibling R-IDeA score module, `sdcb_ridea.rs`.

## What it does

- Scores remaining candidates against the current selected-design history.
- Selects the highest positive R-I or R-IDeA acquisition score.
- Adds the selected design to history and recalculates every remaining candidate's MMD² ratio.
- Respects total cost and experiment-count limits.
- Stops explicitly when the empirical target is covered, budget is exhausted, no positive score remains, or the pool is empty.

## Important limitation

EIG, current-model prediction and proxy prediction are frozen inputs for the whole portfolio. The planner does not observe results or update the posterior. It is a greedy design-only batch selector, not a fully adaptive sequential R-IDeA agent; it does not calculate joint EIG.

## Frozen example

History {0,3}, target {0,1,2,3}. The planner chooses cover-a (x=1), reducing MMD² from 0.14173847 to 0.03760813, then cover-b (x=2), reducing MMD² to zero. A higher-EIG x=4 extrapolation is not selected.

## Validation boundary

12 Rust tests are authored but not compiled/executed in this environment. The JSON fixture was independently recalculated in JavaScript. No model or physical experiment was run.