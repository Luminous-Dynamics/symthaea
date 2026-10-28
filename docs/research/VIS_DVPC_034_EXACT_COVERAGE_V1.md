# VIS-DVPC-034 — Exact finite-sample OPE interval coverage v1

**Status:** implementation draft; synthetic exact-enumeration benchmark only.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8781

## Question

VIS-DVPC-028 checks exact bias, variance and RMSE for a finite contextual-bandit DGP. Those metrics do not show whether a reported uncertainty interval contains the target policy value at its nominal rate. VIS-DVPC-034 makes coverage and interval width first-class benchmark outputs.

## Frozen design

The Rust tool `tools/vision-dvpc-coverage-exact.rs` reads `tests/fixtures/vision-dvpc-034-coverage.tsv`. Every supported scenario has six iid rows, binary actions and Bernoulli rewards; enumerating all 4^6 = 4096 action/reward sequences yields exact probabilities with no Monte Carlo error.

Compared intervals are:

- `WALD_95`: normal approximation around the IPS mean using sample variance. It is a measured baseline, not assumed to have nominal finite-sample coverage.
- `HOEFFDING_95`: a two-sided bounded-reward interval using the known maximum target/logging importance ratio and log(2/alpha).
- `EMP_BERNSTEIN_95`: the empirical-Bernstein HCOPE radius documented by SCOPE-RL, using the sample variance and a known upper bound on the importance-weighted reward.

Intervals are intersected with the parameter range [0,1]. If the intersection is empty, the state contributes zero width and fails coverage; it is not turned into a boundary point. Exact coverage and mean width are weighted by the exact state probabilities.

## Assumptions and failure controls

The bounded intervals' formal-coverage comparison is meaningful only when logged rows are iid/stationary, rewards are in [0,1], the logging propensities are known/correct, and the target policy has support. These assumptions are explicit in the fixture. Scenarios with estimated propensities that differ from the true logger are marked `ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY`: their empirical coverage is diagnostic and does not certify the interval.

The zero-support scenario returns `NOT_IDENTIFIED` and `confidence_intervals=SUPPRESSED`; no estimator-model value is allowed to stand in for identification.

## Exact coverage–width frontier

The following values are independently enumerated under the frozen Bernoulli DGP (six rows per sample, true target value 0.5). Mean width is the expected length of the confidence set after intersecting it with [0,1].

| Scenario | Wald coverage | Wald mean width | Hoeffding coverage | Hoeffding mean width | Empirical-Bernstein coverage | Empirical-Bernstein mean width |
|---|---:|---:|---:|---:|---:|---:|
| `healthy_correct` | 0.781250 | 0.716348 | 1.000000 | 0.881179 | 1.000000 | 1.000000 |
| `weak_overlap_010` | 0.449037 | 0.543168 | 0.999462 | 0.999671 | 1.000000 | 1.000000 |
| `weak_overlap_005` | 0.228808 | 0.405637 | 0.999964 | 0.999964 | 1.000000 | 1.000000 |
| `weak_overlap_001` | 0.063024 | 0.269518 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The central negative result is that Wald coverage degrades sharply with weak overlap. At logging propensity 0.99 for action A, a nominal 95% Wald interval covers only about 6.30% of samples with average width about 0.270. Hoeffding is essentially 100% covering in this fixture, but its expected width is essentially the entire [0,1] reward range. Empirical-Bernstein is equally conservative here. The comparison therefore makes the coverage–width tradeoff visible instead of hiding it behind a single score.

These values are exact for this DGP, not external-validity results. The benchmark intentionally reports uncertainty width rather than assuming more conservative is automatically more useful.

## Machine-readable receipt

`--sweep` emits deterministic TSV with per-scenario coverage/width metrics, aggregate minimum coverage over correct-propensity scenarios, and explicit LIMIT records. The workflow runs unit tests, verifies known-answer coverage points, checks the valid-assumption coverage floor for bounded intervals, and hashes source, frozen fixture, output and Rust version into a receipt.

## Reference points

- SCOPE-RL documents Hoeffding, empirical-Bernstein, Student-t and bootstrap high-confidence OPE intervals: https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html
- Off-Policy Confidence Sequences gives non-asymptotic, non-parametric bounds valid uniformly over time and at arbitrary stopping times: https://proceedings.mlr.press/v139/karampatziakis21a.html
- Empirical Likelihood for Contextual Bandits constructs OPE confidence intervals through convex optimization: https://proceedings.neurips.cc/paper/2020/hash/6d34d468ac8876333c4d7173b85efed9-Abstract.html

These are research baselines, not all implemented here. The next layer is a faithful anytime-valid confidence-sequence implementation and stopping-time coverage evaluation. v1 makes no claim of beating SOTA, real-trace calibration, or production safety.