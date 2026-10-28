# Multiplicity-adjusted deadline reliability gate — 28 October 2026

**Evidence class:** Deterministic post-processing of the saved synthetic event model. No new simulation, Rust test, or radio measurement was performed.

## Why adjust the frontier?

The initial policy selector used pointwise Wilson 95% intervals and chose among four competing strategies per scenario/deadline. Pointwise intervals do not provide simultaneous 95% coverage for the selected winner, and sweeping hundreds of candidates creates a broader multiplicity problem.

This analysis uses **Bonferroni-adjusted two-sided Wilson intervals**. Bonferroni does not assume independent comparisons, though it can be conservative.

## Declared confidence families

- **Per-case family-wise:** control family error at 5% across the four strategies within each of 192 scenario/deadline cases (4 channel models × 3 spreading factors × 4 ACK-loss modes × 4 deadlines).
- **Benchmark-wide family-wise:** control family error at 5% across all 768 strategy-by-scenario/deadline outcomes. This is the stricter gate for broad statements about the sweep.
- For each target, a candidate qualifies only if its adjusted Wilson lower bound meets the target. Select minimum mean analytical PHY airtime among eligible candidates; otherwise record `infeasible` and preserve the best-observed candidate for diagnosis.

## Adjusted feasibility

| Reliability lower-bound target | Per-case feasible | Total cases | Benchmark-wide feasible | Total cases |
|---:|---:|---:|---:|---:|
| 90.0% | 84 | 192 | 84 | 192 |
| 95.0% | 76 | 192 | 76 | 192 |
| 99.0% | 41 | 192 | 26 | 192 |
| 99.9% | 0 | 192 | 0 | 192 |

The number of cases is unchanged at 90% and 95% in this sample. At a 99% target, case-wise correction leaves 41/192 cases, while the benchmark-wide family leaves 26/192. No case qualifies for a 99.9% lower-bound target under either family.

These counts are **not radio capability limits**. They reflect the current synthetic model, candidate strategies, scenario matrix and 2,000-trial sample. The 99.9% result means this experiment does not statistically qualify any scenario to that target; it does not prove the protocol could never achieve 99.9%.

## How to use the two families

Case-wise bounds suit selection for one predeclared, already-frozen deployment scenario. Benchmark-wide bounds are safer when publishing a single claim about results over the entire sweep. If the deployment scenario itself is picked after seeing the sweep, that selection also needs to be addressed; the stronger evaluation is to tune on development seeds and then confirm on independent held-out seeds and, later, held-out radio conditions.

## Recommended evaluation protocol

1. Use development seeds to tune policy and protocol parameters; do not report the maximum over candidates as an unbiased result.
2. Freeze the profile, policy, deadline and target, then evaluate on independent held-out seeds and later held-out radio/board conditions.
3. Predeclare the confidence family and publish all scenarios, including infeasible ones.
4. Optimize measured airtime/energy only among strategies that pass the delivery target.
5. Compare cost on common traces using paired bootstrap or randomization tests instead of inferring superiority from overlapping confidence intervals.
6. Report simulation sampling uncertainty separately from model uncertainty and hardware/field variability.

## Limits and next gates

The underlying model assumes a hypothetical 24-byte ACK, fixed gaps/timeout, one bounded retry round, analytical PHY airtime and simplified state-weighted packet-loss probability. It excludes measured TX/RX energy, calibrated SNR/PER, contention, duty-cycle delays, multi-hop and regional transmission policy.

This correction improves the **statistical claim boundary**, not the physical model. It does not make synthetic results a SOTA claim or production guarantee. The associated draft benchmark is [PR #8635](https://github.com/Luminous-Dynamics/symthaea/pull/8635); the event-time ACK-loss scope is [issue #8695](https://github.com/Luminous-Dynamics/symthaea/issues/8695).