# AOBench — Delayed Outcomes and Staggered Entry

**Status:** Deterministic finite-world v0 implemented; exact-head Rust qualification and Monte Carlo coverage study pending  
**Date:** 2026-11-15  
**Tracking:** [AOBENCH-018 #12251](https://github.com/Luminous-Dynamics/symthaea/issues/12251)  
**Implementation:** `crates/domains/symthaea-organizational-closure/src/delayed_outcomes.rs`

## 1. Research basis

Lindon and Kallus, *Design-Based Anytime-Valid Inference for Randomized Experiments with Delayed Outcomes and Staggered Entry* (ICML 2026), target sample cumulative reward by calendar time among units that have entered the experiment. Treatment can change whether an event occurs, when it occurs and the realized reward.

The key theorem-level distinction is that the overall IPW treatment-effect estimation error is not a martingale under any filtration. The two arm-specific IPW errors are martingales under carefully chosen arm-specific event-time filtrations. The treatment-effect confidence sequence is therefore formed by combining two arm-level sequences with a union bound. The paper also characterizes which arm-specific AIPW augmentations preserve the required martingale property.

References:
- [ICML 2026 paper](https://proceedings.mlr.press/v306/lindon26a.html)
- [arXiv preprint](https://arxiv.org/abs/2603.25971)
- [Authors' reference implementation](https://github.com/michaellindon/icml2026)

## 2. Estimand and event-time semantics

For query q and calendar time t, define the entered population as units with entry time no later than t. The arm-specific target is the sample cumulative reward that would have accrued by t had every entered unit received that arm.

An event after t contributes zero to cumulative reward *by t*. This is not the same as imputing an unknown eventual outcome to zero.

AOBench makes the observability condition explicit:

- An event record contains event time, report time, reward and an independently adjudicated measurement receipt.
- Every entered unit has a follow-up completeness watermark and a separate receipt.
- Absence of an event can be treated as no reward by t only when the watermark proves the event stream complete through t.
- If follow-up/censoring ends before t, the query is Inconclusive.
- If an event occurred by t but is missing before a verified completeness watermark, the input is NotQualified.
- v0 requires event time to equal report time, matching the paper's event-time filtration. A logger or measurement delay after the event is Inconclusive until a valid filtration for that process is implemented.

This avoids turning pending, censored or unavailable measurements into zero outcomes.

## 3. Arm-wise IPW sequence

For unit i, let A_i be its randomized assignment, pi_i the known predictable treatment probability, and Y_i(t) its cumulative reward by calendar time t. The arm-k Horvitz–Thompson cumulative reward estimate uses only observed event increments:

[
widehat R_k(t)=sum_{i:E_ile t}
rac{mathbf 1(A_i=k)}{pi_i(k)}Y_i(t).
]

For a binary treatment:
- pi_i(1) = propensity of treatment;
- pi_i(0) = 1 - pi_i(1).

The reference variance clock increments for an observed reward y in the assigned arm with arm propensity pi by

[
Delta V_i={1-pi_i(k)}left(rac{y_i}{pi_i(k)}ight)^2.
]

AOBench v0 ports the reference normal-mixture boundary:

[
b(V;alpha,eta^2)=
sqrt{left(V+rac{1}{eta^2}ight)
left[log(1+eta^2V)-2log(alpha)ight]}.
]

For Q registered causal queries, family-wise alpha is divided over two arms and all queries:

[
alpha_{	ext{arm}}=rac{alpha_{	ext{FWER}}}{2Q}.
]

The combined treatment-effect interval is the union-bound difference:

[
widehatDelta(t)=widehat R_1(t)-widehat R_0(t),
]

[
CS_Delta(t)=
[widehat R_1(t)-b_1(t)-widehat R_0(t)-b_0(t), widehat R_1(t)+b_1(t)-widehat R_0(t)+b_0(t)].
]

The two arm intervals are combined; AOBench does not assume their difference is itself a martingale.

## 4. Executable contract

The agent-facing scenario contains:
- registered query IDs and decision thresholds;
- unit entry time, realized assignment and treatment propensity;
- randomization, delivery, isolation and follow-up-completeness receipt IDs;
- observation-completeness-through calendar time;
- event records observed by the agent, with event time, report time, reward and measurement receipt;
- a selected calendar-time stopping point and reported effect interval/decision.

The evaluator-only oracle contains:
- each unit's control and treatment potential event time and reward (or NoEvent);
- independently adjudicated dispositions for every evidence receipt.

The oracle and potential outcomes are never part of the agent-facing scenario.

## 5. Qualification rules

- Propensity must be finite and strictly between zero and one.
- Potential event times cannot precede unit entry; all potential rewards must be finite.
- Every entered unit must have verified randomization, delivery, isolation and completeness receipts.
- Every event included by the stop time must have a verified measurement receipt.
- Every entered unit's completeness watermark must reach the selected stop time.
- Event time and report time must coincide in v0.
- A missing event before a verified completeness watermark is a contract failure, not a zero.
- A report interval may not be narrower than the evaluator-recomputed arm-wise sequence.
- Act is permitted only if the reported lower endpoint is above the decision threshold. DoNotAct is permitted only if the upper endpoint is at or below the threshold. Otherwise the decision must defer.
- Rejected evidence blocks the query. Unavailable evidence or incomplete follow-up yields Inconclusive.

A single confidence-sequence coverage miss is possible at the registered alpha. Coverage is therefore a Monte Carlo benchmark metric, not a deterministic per-episode pass criterion.

## 6. Implemented deterministic controls

Fourteen tests cover:
1. Normal-mixture boundary algebra at zero variance.
2. Arm-wise IPW estimates, variance clocks and union-bound interval.
3. Staggered entry population selection.
4. Incomplete follow-up cannot be imputed as zero.
5. Verified no-event-by-time is a valid zero cumulative contribution.
6. Event omitted before a verified watermark is rejected.
7. Reporting lag outside the event-time filtration is inconclusive.
8. Future event leakage is rejected.
9. Unavailable randomization evidence is inconclusive.
10. Rejected intervention delivery blocks the query.
11. Reported interval cannot be narrower than the recomputed sequence.
12. Multiple queries share family-wise alpha allocation.
13. Invalid randomization propensity is rejected.
14. Invalid alpha and mixture parameters fail closed.

These are deterministic unit fixtures, not a Monte Carlo coverage study or SOTA-power claim.

## 7. Baselines and required calibration

- Immediate-outcome bounded Hoeffding confidence sequences from AOBENCH-017 (positive control only).
- Fixed-horizon cumulative reward difference.
- Naive pending-equals-zero and complete-case-only estimators (negative controls).
- Design-based variance upper bound.
- Lindon–Kallus arm-wise IPW normal-mixture sequences.
- Valid arm-specific AIPW augmentations once their martingale conditions are independently implemented.
- Oracle cumulative-reward sequence (upper bound only).

The confirmatory suite should use paired world seeds, frozen treatment propensities, matched entry schedules, outcome process and administrative censoring. Compare empirical simultaneous coverage over all registered queries and calendar-time grids, interval width, time-to-decision, power and unresolved rate.

## 8. Proposed targets (not achieved)

- 100% exact timeline/entry/event/arrival fixture correctness.
- Zero pending/censored/observer-unavailable outcomes silently converted to zero.
- Zero statistical updates from rejected randomization, delivery, isolation or measurement receipts.
- Simultaneous empirical coverage at least 95% at nominal family-wise alpha 0.05, reported with Monte Carlo confidence intervals.
- False decisive-action rate no greater than 5% on null and threshold-straddling controls.
- Report interval width, time-to-decision, number pending, censoring and power at preregistered effect sizes.

## 9. Explicit limitations and next extensions

The v0 model allows at most one reward event per unit and query. It requires event-time/report-time coincidence, and therefore does not claim validity under additional asynchronous logger latency. It is a design-based randomized-experiment evaluator, not an observational identification method or a general time-varying treatment-regime estimator.

Next work:
1. Reproduce the published reference example and compare numerical results on identical seeds.
2. Add Monte Carlo simultaneous-coverage calibration across event-time and reward families.
3. Add recurrent/multiple reward events per unit.
4. Add a valid delayed-measurement filtration rather than treating reporting lag as event time.
5. Implement the paper's valid arm-specific AIPW augmentation class.
6. Add cluster-level dependent outcomes and multiple outcomes, using separate cluster-robust anytime-valid methods rather than reusing the scalar confidence sequence.

## Claim boundary

This is a finite deterministic benchmark evaluator using the published design-based arm-wise IPW confidence-sequence construction. It does not prove validity under observer latency, arbitrary censoring, interference, observational assignment, recurrent events or unregistered queries. Exact-head Rust format/check/test/Clippy and Monte Carlo coverage validation are required before any statistical performance claim.
