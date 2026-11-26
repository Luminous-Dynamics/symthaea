# SDCB robust/Pareto reference harness

Standalone reference for finite ambiguity-set robust experiment design and Pareto/hard-constraint decisions.

## Build and run

```bash
rustc --edition=2024 --test scripts/research/sdcb_robust_pareto.rs -o /tmp/sdcb-robust-pareto
/tmp/sdcb-robust-pareto
```

Uses only the Rust standard library.

## Robust finite-scenario EVSI

- Multiple plausible complete utility/likelihood scenarios are supplied without scenario probabilities.
- The current action maximizes worst-case expected utility.
- Experiment policies map outcomes to actions and are shared across hidden scenarios.
- The planner enumerates deterministic policies and maximizes worst-case post-experiment value.
- Net robust gain subtracts an explicit cost rate.
- It reports every scenario value and abstains if no experiment has positive net robust gain.

Exact-search limits: 8 scenarios, 5 actions, 4 outcomes, 8 candidates.

## Pareto and hard constraints

- Multiple objectives remain separate; every objective is maximized.
- Encode risk as a negative loss or safety margin with a documented direction.
- Hard lower bounds are checked under every hypothesis with nonzero prior, not just on posterior averages.
- Experiment-policy fronts compare expected objective vectors and cost without a weighted scalar score.
- Each outcome-to-action policy is compared with a zero-cost no-experiment lottery using the same marginal action frequencies. This prevents a random measurement outcome from being credited as scientific information.

Exact-search limits: 8 actions, 6 objectives, 4 outcomes, 8 experiment candidates.

## Current coverage

19 Rust test functions are authored. They have not been compiled or executed in this environment because rustc/cargo are unavailable. The repository protocol contains robust and Pareto JSON fixtures plus independently computed expected results.

## Claim limits

Robustness is only over the supplied finite ambiguity set. The reference enumerates pure (not mixed) robust policies. It does not prove physical safety, authenticate evidence, or establish SOTA performance. A Pareto frontier is only as meaningful as its owner-approved objectives and hard constraints.