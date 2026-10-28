# Aviation Online Calibration Freshness Benchmark (AOCF-001)

**Status:** prototype module and test definitions committed to the AECB research branch; no Rust execution is claimed.  
**Implementation:** `crates/domains/symthaea-operations-research/src/aviation_online_drift.rs`  
**Integration fixtures:** `crates/domains/symthaea-operations-research/tests/aviation_online_drift_oracle.rs`  
**Related protocols:** [split integrity](./AVIATION_CALIBRATION_AND_SPLIT_INTEGRITY.md), [distributionally robust recovery](./AVIATION_DISTRIBUTIONALLY_ROBUST_BENCHMARK.md).

## Research hypothesis

A frozen ambiguity radius can become unsafe or economically wasteful after the underlying operational regime changes. A monitor should detect evidence that the frozen nominal scenario model is no longer credible, distinguish a new/out-of-support event from drift within the modeled support, and latch an explicit state that requires investigation/recalibration rather than silently carrying on.

The next intended paradigm is **sequential decision integrity**: every decision is conditioned on both the plan's feasibility evidence and a current calibration-freshness state. An alarm invalidates blind reuse of the old calibration; it does not autonomously choose a safer dispatch action.

## Current implementation

`aviation_online_drift.rs` provides a fixed-reference likelihood-ratio monitor:

- Frozen nominal categorical scenario distribution.
- A predeclared mixture of normalized alternative distributions and mixture weights.
- Log-space mixture likelihood-ratio accumulation, compared to `ln(1/alpha)`.
- Unique event IDs and non-decreasing event times; bad metadata fails without partially updating state.
- Immediate, separate `SupportShiftAlarm` when a category is absent from nominal support or has zero nominal mass.
- Latched drift/support alarms; ingestion cannot continue until an explicitly re-created monitor has a new declared reference and calibration receipt.
- A positive support requirement for alternatives: each alternative assigns non-zero probability to every nominally possible category. This prevents the entire mixture from irreversibly collapsing to zero on a category that was possible under the null.

Under independent observations from the *exact frozen nominal distribution*, each fixed alternative likelihood ratio is a nonnegative martingale; a predeclared convex mixture remains an e-process. Ville's inequality gives `P(sup_t E_t >= 1/alpha) <= alpha`. This statement is conditional on the null/model and independence assumptions. It is not a universal drift detector and not valid by assertion for serially dependent airline episodes.

A scenario outside the frozen nominal support is treated as an immediate support-shift alarm rather than forced into the likelihood-ratio formula.

## Relevant current research

- Kuang & Xia (September 2026), [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536). Addresses repeated use of a fixed calibration reference without assuming independent conformal p-values across repeated uses; the method derives a predictive rank martingale with anytime-valid type-I control under its rank-exchangeability setup. This is an important comparator if our categorical nominal-independence assumption is too strong.
- Liang, Ren & Chen (February 2026), [Optimal training-conditional regret for online conformal prediction](https://arxiv.org/abs/2602.16537). Studies abrupt changes and smooth drift, combining shift detection with adaptive calibration and regret guarantees under the paper's assumptions.
- Huang, Ma & Michailidis (UAI 2026), [Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift](https://proceedings.mlr.press/v337/huang26b.html). Uses martingale PAC-Bayesian certificates and a predict-then-update protocol for temporal dependence and shift.
- Momtaz (June 2026), [Rolling-Calibrated Conformal Decision Support for Seasonal Airport Taxi-Time Planning under Distribution Shift](https://doi.org/10.19139/soic-2310-5070-3782). Evaluates 17 IATA seasons across 722 EUROCONTROL CODA airports; illustrates the operational trade-off between reducing under-planning and increasing buffer/padding, and explicitly limits formal validity claims under drift.

These methods are not interchangeable: a categorical mixture likelihood ratio is exact for its declared null/alternatives, while rank martingales or sequential conformal procedures operate under different assumptions and observables. Fair comparisons must use the same incoming event stream, detection latency budget and escalation cost.

## Required benchmark design

### Frozen detector arms

1. Fixed-reference mixture likelihood-ratio e-process (this prototype).
2. Predeclared Page-Hinkley/CUSUM and EWMA detectors, with thresholds selected on calibration/validation only.
3. Predictive-rank martingale, if rank scores and its exchangeability contract can be validly constructed.
4. Rolling conformal recalibration/abstention baseline.
5. No detector / stale calibrated policy.
6. Oracle change-time detector, diagnostic upper bound only.

Use the same scenario events, shift injection times and downstream decision loss for every detector. Freeze the alternative mixture, `alpha`, baseline period, minimum evidence thresholds, reset protocol and escalation action before looking at final test outcomes.

### Shift families

- Sudden increase in weather/capacity disruption rate.
- Gradual drift over a predeclared ramp duration.
- Seasonal/regime transition.
- Airport or fleet cohort change.
- Compound events and correlated episodes.
- New category outside the modeled support.
- Label/observation delays, missing events and duplicate/stale data.
- Benign non-stationarity that should not cause excessive false alarms.
- Alternatives that poorly represent the actual shift (detector misspecification).

### Metrics

- Anytime false-alarm rate over long nominal streams under the declared null.
- Detection probability and median/90th-percentile detection delay by shift family and size.
- Missed detections and alarms under no-shift controls.
- Time to recalibration, abstention fraction and time spent in latched alarm.
- Additional operating cost from conservative fallback/abstention.
- Post-shift worst-case/upper-tail recovery cost and stakeholder harm.
- Percentage of runs with out-of-support events correctly surfaced.
- Runtime and memory per incoming event; monitor state size.
- Performance under alternative-mixture misspecification and event dependence.

Report uncertainty using independent run seeds and event/time blocks; don't treat serially correlated event observations as independent replicates.

## Proposed acceptance gates (targets, not results)

- 100% deterministic alarm decisions and state under event-order replay.
- 100% explicit alarms for out-of-support scenarios.
- Zero ingestion after a latched alarm without new monitor/reference initialization.
- Empirical false-alarm rate bounded by the predeclared budget in independent nominal simulations; state the simulation design and its uncertainty.
- At least 90% detection on preregistered operationally material shifts within a predeclared maximum delay, *without* unacceptable false-alarm or abstention cost. The precise shift magnitude/delay and cost gate must be fixed before test.
- Outperform fixed-window heuristics on a held-out shift mix in a predeclared utility that penalizes missed shifts, late detection, false alarms and unnecessary recalibration.
- No policy promotion on the monitor alone: feasibility, actor-harm and data-contract gates remain separately necessary.

These are proposed targets and must be revised before the final test if the operationally material shift scale is changed—not after inspecting outcomes.

## Decision-readiness composition

The companion [ADRG-001 gate](./AVIATION_DECISION_READINESS_GATE.md) now composes the feature audit, split audit, independent plan verifier, online monitor status, distributional-risk assessment, network harm assessment and active calibration-receipt ID. Drift alarms, support-shift alarms, stale receipts and any rejected/indeterminate upstream status block promotion. The positive state is only `EligibleForHumanReview`, not dispatch authorization.

The synthetic example now runs the composed readiness gate. Its predeclared compound-weather stream intentionally trips the drift monitor; the resulting run manifest should report a blocked decision even when the feature, split, plan and modeled cost assessments pass. This is a contract fixture, not measured detector performance.

## Integration contract

A future orchestrator should implement an explicit state machine:

- **Fresh:** frozen calibration receipt matches the active policy and no drift alarm is latched.
- **Drift suspected:** monitor crossed its declared boundary; stop treating the old radius as current evidence.
- **Support changed:** an unseen/zero-nominal-mass scenario appeared; reject unsupported probability claims.
- **Recalibration pending:** collect clean, partition-audited Calibration episodes and verify cluster/time integrity.
- **Recalibrated:** new nominal/alternative distributions, family-wise radius envelope and validation receipt are bound to a new configuration digest.

Transitions must be monotone and auditable. Reusing a prior calibration requires explicit justification; do not silently clear an alarm or overwrite the baseline with post-change observations.

## Test status and claim ceiling

Nineteen Rust unit tests and five public-API integration fixtures are defined for the monitor. They have not been compiled or executed because the current environment lacks `cargo`/ `rustc`. Passing synthetic tests would establish behavior only for these contracts and fixtures, not detection performance on real airline data or a guarantee under arbitrary temporal dependence.

Follow-up: [AVIATION-ENTERPRISE-011 #8795](https://github.com/Luminous-Dynamics/symthaea/issues/8795).
