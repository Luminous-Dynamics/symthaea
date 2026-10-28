# Aviation Calibration and Split-Integrity Protocol (ACSP-001)

**Status:** split-audit implementation and tests defined on the AECB research branch; Rust tests have not been executed in this environment.  
**Implementation:** `crates/domains/symthaea-operations-research/src/aviation_calibration_protocol.rs`  
**Related benchmark:** [distributionally robust recovery](./AVIATION_DISTRIBUTIONALLY_ROBUST_BENCHMARK.md), [public-data evaluation](./AVIATION_PUBLIC_DATA_EVALUATION_PROTOCOL.md).

## Why this layer exists

A robust decision can look superior for the wrong reason if related disruption episodes leak across partitions, timestamps allow future information into model development, or the uncertainty radius is tuned on final-test outcomes. ACSP-001 adds a fail-closed metadata audit before any robustness claim is scored.

This is a *measurement-integrity control*, not an aviation optimizer. It does not estimate disruption probabilities, calculate conformal quantiles, guarantee exchangeability, or prove that the dataset or supplied metadata is truthful.

## Implemented controls

The Rust module includes:

- Unique row IDs and non-empty event-cluster IDs.
- A cluster-disjointness check: one disruption episode or dependence cluster cannot appear in more than one of Training, Calibration, Validation, Test.
- Per-partition row counts, distinct event-cluster counts and time bounds.
- Minimum distinct-cluster requirements; undersized partitions return Indeterminate instead of silently passing.
- Optional strict chronological separation, requiring each split's latest timestamp to precede the next split's earliest timestamp. This can be disabled for explicitly grouped out-of-distribution holdouts where chronology is not the intended split.
- Missing event-time evidence returns Indeterminate.
- A separate tuning-provenance check rejects any declared parameter/model/threshold selection that used Test, and rejects duplicate/empty artifact IDs or an empty tuning manifest.

The report is based on input metadata. A passing audit does not prove that distinct cluster IDs truly represent independent events or that no one inspected test outputs outside the recorded manifest.

## Preregistered split policy

For a time-generalization experiment, use a chronology-separated layout:

1. **Training:** fit predictive components and feature transformations.
2. **Calibration:** estimate uncertainty and calibrate probability/risk diagnostics.
3. **Validation:** choose among the predeclared radius/model/threshold grid and evaluate development-only trade-offs.
4. **Test:** evaluate the frozen decision rule once on a temporally later, previously untouched block.

For an OOD-family experiment, hold out complete disruption families or clusters; use the grouped split mode and report the temporal composition explicitly. Run time and OOD generalization as separate experiments rather than silently mixing their claims.

Every row should have a stable source-row ID, event-cluster ID, event time, partition, source snapshot reference, and preprocessing/version provenance. The full protocol should also record source snapshot digests and a preregistration/decision receipt outside this module. Digest syntax alone is not proof of data lineage, and the current crate does not cryptographically attest those manifests.

## Current research comparator

Momtaz (2026), *Rolling-Calibrated Conformal Decision Support for Seasonal Airport Taxi-Time Planning under Distribution Shift*, evaluates rolling one-sided conformal calibration on 17 IATA seasons and 722 airports in the EUROCONTROL CODA dataset. The paper distinguishes exchangeability-based conformal guarantees from empirical adaptation under distribution shift and notes the operational trade-off between under-planning and padding. DOI: https://doi.org/10.19139/soic-2310-5070-3782.

That work is a calibration-method comparator, not proof that this split auditor improves prediction intervals. The next fair comparison should reproduce its calibration regime where data licensing permits, then separately test whether a calibrated uncertainty set improves the *downstream recovery decision*.

Other core comparators:
- Zhang et al. (2026), distributionally robust aircraft sequencing using learning-driven arrival/departure predictions and real Xiamen airport data, Omega 2026. DOI: https://doi.org/10.1016/j.omega.2025.103415.
- EUROCONTROL CWBO, network-wide weather/capacity triggers and coordinated actions. Published 14 September 2026: https://www.eurocontrol.int/publication/how-capacity-and-weather-based-operations-cwbo-are-building-network-resilience-capacity.

## Required experiment and metrics

Evaluate at least these arms on the same frozen event clusters and action space when comparable:

1. Nominal estimate-then-optimize, using training-fitted probabilities.
2. Calibrated nominal policy, with radius/quantiles chosen on Calibration/Validation only.
3. TV-DRO policy / gate over a preregistered radius grid.
4. A conventional Wasserstein-DRO or published airport sequencing baseline where the same inputs/action space can be supported.
5. Incumbent/do-nothing and hindsight oracle (upper-bound reference only, never deployable).

Report:
- feature/split audit statuses and every finding;
- per-split rows, event clusters and temporal boundaries;
- nominal expected cost, worst-case expected cost, robustness premium and sign-flip radius;
- predictive interval/coverage calibration by season, airport-size cohort and disruption regime, where applicable;
- mean cost, CVaR / tail regression, maximum actor harm, delay propagation, cancellations, utilization, runtime and abstention/rejection rate;
- confidence intervals clustered by event episode and temporal block, not by row when rows share a disruption;
- all failed/indeterminate runs and all preregistered deviations.

## Proposed pass criteria (targets, not results)

- 100% rejection of injected cross-partition event-cluster leaks and test-tuning declarations.
- 100% status/report invariance under input row permutation.
- 100% non-acceptance for missing time evidence or insufficient cluster counts.
- 100% agreement with independent fixtures for the split and tuning contract.
- No final-test-driven radius or threshold changes.
- Demonstrated reduction in held-out tail cost compared with the best matched baseline, while satisfying a predeclared average-cost and utilization non-inferiority bound.
- For predictive uncertainty, report empirical coverage and width by time block and regime; do not claim formal coverage under arbitrary distribution shift.

## Test status and claim ceiling

Thirteen Rust unit tests are defined in `aviation_calibration_protocol.rs`, but they have not been compiled or run here because the environment lacks `cargo` and `rustc`. Test definitions are not test results. A successful metadata audit is necessary but not sufficient for credible calibration, independent validation, airline savings or operational approval.

Follow-up: [AVIATION-ENTERPRISE-009: calibration and split integrity](https://github.com/Luminous-Dynamics/symthaea/issues/8753) (to be created).
