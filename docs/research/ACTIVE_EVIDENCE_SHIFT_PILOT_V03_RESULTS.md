# Active-evidence shift pilot v0.3 — local synthetic reference results

**Status:** executable local reference experiment; exploratory synthetic evidence only. This summary is committed to the protocol branch; the Python harness and raw CSV artifacts are distributed in the companion bundle, not embedded in this Markdown file.
**Tracking:** issue [#8709](https://github.com/Luminous-Dynamics/symthaea/issues/8709), protocol PR [#8692](https://github.com/Luminous-Dynamics/symthaea/pull/8692), provenance issue [#8691](https://github.com/Luminous-Dynamics/symthaea/issues/8691).

## Experiment

- 8 fixed synthetic scenarios × 16 deterministic seeds × 9 policy arms = 1,152 trajectory summaries.
- Five cost checkpoints per trajectory = 5,760 budget-trace rows.
- Standard-library Python reference implementation; Python 3.13.5.
- Independent result verifier passed; 8/8 unit tests passed, including contract mutation/missing-file rejection, altered-result rejection, and policy/evaluator truth separation.
- Frozen synthetic contract SHA-256: `163c68e5a5d4654a8c57363b84e3c1cb5e6ebd614c5194de7ae4034a94f42b63`.

## Main result

Lower mean regret AUC is better in this particular synthetic environment:

| Arm | Mean regret AUC | Final regret, cost 18 | Mean spend | Mean queries | Abstention rate |
|---|---:|---:|---:|---:|---:|
| Entropy | 0.084594 | 0.089438 | 17.884 | 12.00 | 0.375 |
| Cost-normalized entropy | 0.088028 | 0.109832 | 17.762 | 17.00 | 0.375 |
| Random | 0.110740 | 0.132563 | 17.660 | 11.84 | 0.774 |
| Bounded uncertainty probe → robust selector | 0.118876 | 0.158862 | 12.739 | 10.55 | 0.922 |
| Target-weighted coverage heuristic | 0.122936 | 0.153716 | 17.663 | 13.62 | 0.500 |
| Two-query EVSI lookahead | 0.140280 | 0.178107 | 11.199 | 7.95 | 0.969 |
| Robust minimax-regret selector | 0.143856 | 0.169799 | 7.853 | 5.36 | 0.984 |
| Plug-in one-step EVSI | 0.148816 | 0.172146 | 7.803 | 5.01 | 1.000 |

Entropy is the strongest deployable arm tested on mean regret AUC in this pilot. Cost-normalized entropy is close: its paired difference versus entropy is +0.003433 with a 95% seed-bootstrap percentile interval [−0.003152, +0.010090], which includes zero.

## Failure analysis

- Plug-in EVSI abstained in 100% of trajectories; robust minimax regret abstained in 98.4%; two-query EVSI abstained in 96.9%. The current objective can deadlock when a query cannot immediately flip the downstream action.
- A bounded uncertainty-probe tranche improved the pure robust arm (AUC 0.143856 → 0.118876), but did not beat entropy.
- In the rare-group scenario, robust minimax-regret AUC was 0.0711 versus 0.0080 for entropy.
- Under conditional shift outside the configured ambiguity scale, regret remained high across methods. Robustness is conditional on the declared ambiguity set; this selector provides no arbitrary-shift guarantee.
- The evaluator-only one-step truth-aware arm abstained in every trajectory. It is not a valid multi-round oracle and must not be advertised as a hindsight upper bound.

## Statistical and claim limits

Bootstrap intervals resample 16 paired seed clusters conditional on eight fixed, public synthetic scenarios. They do not establish generalization across a population of tasks or shift mechanisms. The coverage heuristic is not a reproduction of Uncertainty Herding, and no R-IDeA method was implemented. No SOTA or industry superiority claim is supported.

Local verification is separate from repository qualification: the reference verifier and 8 tests passed, but this environment had no `cargo` or `rustc`. No Rust compilation or CI pass is claimed.

## Next research increment

1. Replace the flawed one-step truth-aware arm with an exact finite-horizon Bellman oracle on a tractable state space.
2. Compare planning horizons 1, 2, 4, and 8, and test whether calibration-before-action escapes the myopic abstention trap.
3. Predeclare worst-group regret and add an explicit rare-group protection arm.
4. Implement Uncertainty Herding faithfully under matched budgets before comparing against it; implement R-IDeA only if its objective can be reproduced accurately.
5. Use development/validation/test shift families and report each regime separately.
6. Then port the selected policy logic into Rust and qualify the exact source head.

### Research context

Uncertainty Herding motivates a genuine uncertainty-coverage baseline, not an informal proxy: https://proceedings.iclr.cc/paper_files/paper/2025/hash/0302fb83c62991efbccf0a003e4f5a92-Abstract-Conference.html

R-IDeA motivates testing representativeness, informativeness, and error de-amplification under misspecification, but this pilot does not reproduce it: https://proceedings.mlr.press/v300/tang26d.html

NIST AI 800-3 motivates separating a fixed benchmark result from generalized performance and making statistical assumptions explicit: https://doi.org/10.6028/NIST.AI.800-3