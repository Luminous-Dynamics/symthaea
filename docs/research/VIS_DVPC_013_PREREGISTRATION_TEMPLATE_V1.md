# VIS-DVPC-013: Preregistered real-trace comparison template

**Template version:** VIS-DVPC-013-v1  
**Parent protocol issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8572  
**Use:** Copy this file into a dated experiment record, fill every required field, commit it, and obtain independent review before exposing held-out evaluator outcomes. Do not fill results sections after seeing test outcomes.

This is a protocol template, not a benchmark result or legal opinion. Empty fields block confirmatory evaluation. Any change after test outcomes are opened creates an exploratory result or a new preregistration; do not silently revise the original.

## 1. Experiment identity and freezing

- Experiment ID:
- Preregistration commit SHA:
- Policy artifact ID and exact digest:
- Code commit SHA:
- Scorer/verifier source digest:
- Dataset ID and exact immutable revision:
- Dataset snapshot descriptor SHA-256:
- Source manifest SHA-256:
- Group-level split manifest SHA-256:
- Utility specification SHA-256:
- VIS-DVPC-012 bundle lock SHA-256:
- Hardware/device/firmware identity:
- Runtime/toolchain versions:
- Date and independent custodian:
- Confirmatory or exploratory: **must be marked before any outcome access**.

## 2. Rights, provenance, and permitted use

For every data source, record the authoritative license/terms URL, version/date checked, license identifier, allowed use (research, noncommercial, commercial, redistribution), derivative-data restrictions, and who reviewed the interpretation. Record data custody and whether source bytes can be re-hashed locally.

A metadata string is not a rights decision. Keep raw sensor files outside the code repository unless the exact source terms permit redistribution. If the rights or permitted use cannot be established, the source is not admissible for the proposed use.

## 3. Sampling unit and split isolation

- Independent sampling unit: sequence / scene / session / subject / episode / other (justify):
- Why this is the correct independent unit:
- Train groups and manifest digest:
- Calibration/validation groups and manifest digest:
- Final test/private-test groups and manifest digest:
- Minimum independent clusters per primary comparison:
- Power / minimum-detectable-effect analysis and assumptions:
- Predeclared subgroup and domain list:
- Known duplicate, near-duplicate, scene overlap or cross-dataset leakage checks:

All rows in a correlated sequence/session/scene must remain in one split. Frames are not independent replicates when the claim generalizes to scenes or sequences. The final test split must remain inaccessible to policy tuning and calibration.

## 4. Decision target and primary endpoint

- Downstream task and action system:
- Cheap path:
- Expensive/escalated path:
- Evaluator-owned downstream outcome/loss:
- Primary metric name and units:
- Positive direction:
- Normalization formula, if any:
- Practical minimum effect worth deploying:
- Primary comparison:
- Confirmatory hypothesis:
- Loss/harm event definition:
- Missing, invalid, skipped, aborted, unsupported and unaffordable trace handling:
- Safety and tail metrics, reported separately from utility:

Perception-level metrics may be reported as diagnostics, but do not replace the primary downstream decision-value endpoint. Freeze the task utility and harm definitions before test outcomes are observed.

## 5. Comparison family and baselines

List the complete comparison family before evaluation; comparison_family_size must match its entries.

| ID | Candidate/baseline | Frozen artifact/source | Tuning split only | Primary comparison? |
|---|---|---|---|---|
| B0 | All-cheap |  |  | Reference |
| B1 | All-full |  |  | Reference; not assumed optimal |
| B2 | Uniform at matched spend |  |  | Yes/no |
| B3 | Random at matched feasible spend |  |  | Yes/no |
| B4 | Fixed confidence/uncertainty router |  |  | Yes/no |
| B5 | Entropy router |  |  | Yes/no |
| B6 | Perception-quality-gain router |  |  | Yes/no |
| B7 | Lightweight rule-based router |  |  | Yes/no |
| B8 | Strongest available published/official comparator (name/version) |  |  | Yes/no |
| B9 | Symthaea calibrated DVPC router |  |  | Candidate |

State why each comparator is fair and strong. Record unavailable baselines and why they cannot be reproduced. Do not tune the candidate and then compare it with a baseline left at an unreasonable operating point.

The safety/tail-constrained hindsight oracle is an evaluator-only upper reference; it is not a deployable baseline and cannot be an input to the online allocator.

## 6. Resource budgets and measurement

Freeze simultaneous latency and energy ceilings as independent dimensions. Choose the operating points before final evaluation; for example, 5%, 10%, 20% and 30% of the measured full-pass resource reference may be useful, but use the range suitable to the deployment and report absolute units too.

| Operating point | Run latency ceiling (us) | Run energy ceiling (uJ) | Per-action latency bound (us) | Per-action energy bound (uJ) | Predeclared |
|---|---:|---:|---:|---:|---|
| P0 |  |  |  |  | Yes |
| P1 |  |  |  |  | Yes |
| P2 |  |  |  |  | Yes |
| P3 |  |  |  |  | Yes |

Measure and charge all allocator costs: feature extraction, shift detection, decision-value estimation, policy selection, bookkeeping, failed routes and runtime overhead. Measure escalation latency/energy on the same hardware and runtime conditions. Document cold/warm behavior, measurement source, resolution, calibration and outliers. Any unknown cost or overrun remains in the report and invalidates a claim of budget-feasible performance for that run.

Record whether values are measured, conservative bounds or estimates. Do not mix those categories.

## 7. Statistical analysis frozen before test

- Independent cluster count and planned exclusions:
- Primary contrast and effect-size units:
- Cluster-level delta aggregation rule:
- Confidence level and interval method:
- Multiplicity family and correction:
- Sign-flip/randomization method and assumptions, if used:
- Harm-risk bound method and family-wise delta:
- CVaR tail probability and aggregation:
- Planned subgroup/domain analyses:
- Missingness/sensitivity analysis:
- Bootstrap seed and replication count:
- Power/minimum-detectable-effect computation:
- Stopping rule:

Report paired per-cluster results, not just means. Use cluster-level inference. Bootstrap intervals are approximations and sign-flip tests have assumptions; neither is a universal substitute for design-aware inference. Set the sample size to the power requirement rather than choosing it after the observed interval is known.

For the risk gate, preregister the pooled and every required group target. An example target is a one-sided 95% family-wise upper bound at or below 5%, but select a threshold justified by the application before collecting outcomes. Unknown or shifted domains are a separate stratum and do not inherit IID/exchangeability guarantees automatically.

## 8. Negative and falsification controls

- Shuffled-feature / null-value-estimator control:
- Random-at-matched-budget control:
- Cheap-only and full-only reference checks:
- Sign-reversal / utility-direction check:
- Duplicate-group and split-leakage controls:
- Label-leakage control:
- Changed-lock / changed-digest control:
- Budget overrun and missing-meter control:
- Shifted / unknown-domain control:
- Deliberately harmful escalation cases:
- Predeclared falsification criteria:

Falsification outcomes must be retained. Do not weaken gates, suppress groups or redefine harm after observing a failed result.

## 9. Go/no-go rules

Specify thresholds for all of the following before test opening:

- Minimum meaningful primary utility/loss effect versus the strongest budget-matched deployable baseline.
- Family-wise uncertainty criterion for the primary contrast.
- Maximum harmful-escalation upper bound, pooled and per required group.
- CVaR/tail-loss non-regression criterion.
- Safety-violation non-regression criterion.
- Zero-tolerance versus allowed count for each latency/energy overrun.
- Required minimum counts per group/domain.
- Cross-domain replication criterion.
- Conditions that force abstention or mark the run invalid.

An illustrative engineering target is at least a 5% relative reduction in downstream decision loss versus all-cheap while avoiding harm/tail regressions; it must not be used as a universal target without checking the task’s absolute scale and safety implications. If any required gate fails, do not report a general win.

## 10. Required public result bundle

- This preregistration and its commit SHA.
- Exact policy, evaluator, verifier, dataset-manifest, utility and lock hashes.
- Independent custody statement for held-out outcomes.
- Baseline versions/configurations and tuning-split provenance.
- Count of all traces and clusters, plus every exclusion/invalid/aborted/over-budget reason.
- Paired cluster-level deltas and result receipt from VIS-DVPC-014.
- Utility, harm, CVaR, safety, latency, energy, allocator overhead and budget violations as separate metrics.
- Confidence intervals, effect sizes, multiplicity correction and subgroup/domain results.
- Hardware/runtime measurement details and calibration.
- Negative controls and all failed/falsifying results.
- Known limitations, deviations from preregistration and a clear exploratory/confirmatory label.

## Exit gate

A confirmatory claim may be made only after the policy, utility, thresholds, budgets, baselines, comparison family, split manifest and analysis plan are frozen; the exact evidence bundle is verified; the outcomes are held out from the online policy; and the final receipt is reproducible. The protocol does not itself establish SOTA superiority.
