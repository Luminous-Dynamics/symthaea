# SPECTRA Conjunctive Multi-Threshold Feasibility v0.1

**Status:** deterministic Rust classifier and test declarations; not compiled or executed.
**Research parent:** #10776
**Method ID:** spectra-conjunctive-multi-threshold-classification-v1

## 1. Why this exists

Most consequential decisions are conjunctive. A system must meet a performance floor, a safety floor, a maximum latency, a cost cap, and a burden cap at the same time. These constraints should not be collapsed into one weighted reward: otherwise excellent performance can numerically compensate for an unacceptable safety, latency, or dignity failure.

Jiang et al., *Multi-thresholding Good Arm Identification with Bandit Feedback*, ACML 2025, formalize vector-feedback good-arm identification. They seek an arm whose mean vector exceeds a threshold vector componentwise, or return a no-good symbol. Their MultiTUCB algorithm comes with sample-complexity analysis and is the direct matched baseline for the multi-threshold arm problem: https://proceedings.mlr.press/v304/jiang26a.html

## 2. Contract

Freeze:
- the complete action universe;
- the complete constraint universe;
- direction and threshold for every constraint;
- upstream source, target, scenario, provenance and inference scope;
- a jointly simultaneous interval family over every action × constraint pair.

For each constraint with interval [L,U] and threshold τ:

- AtLeast(τ): CertifiedPass iff L ≥ τ; CertifiedViolation iff U < τ; otherwise Unresolved.
- AtMost(τ): CertifiedPass iff U ≤ τ; CertifiedViolation iff L > τ; otherwise Unresolved.

For each action:
- CertifiedFeasible iff every constraint passes;
- CertifiedInfeasible iff at least one constraint is definitely violated;
- Unresolved otherwise.

The action result preserves every constraint outcome and every violation witness. One definite violation is sufficient to rule out an action; other unresolved constraints remain visible.

For the complete action family:
- GoodCandidateSetFound if at least one action is certified feasible;
- CertifiedNoGoodCandidate only if every action is certified infeasible;
- NotEstablishedAtBudget if no action is good and at least one remains unresolved.

Equality follows conservative semantics:
- AtLeast lower=threshold is a pass.
- AtMost upper=threshold is a pass.
- AtLeast upper=threshold with lower below threshold is unresolved.
- AtMost lower=threshold with upper above threshold is unresolved.

## 3. Multiplicity and interpretation

If the upstream intervals are jointly valid over the complete frozen action × constraint grid, the classification is deterministic post-processing of that event. No additional union bound over every action is needed. The base interval family must still receive correct simultaneous/time-uniform multiplicity control.

This module does not estimate intervals, verify source data, authenticate receipts, establish target support, qualify model adequacy, or prove external operational feasibility. Those evidence gates remain mandatory and separate.

## 4. Rust implementation

File: tools/spectra-uvsd-fit/src/multi_threshold.rs

- Frozen action/constraint contract.
- Explicit AtLeast and AtMost directions.
- Complete-grid validation; rejects missing, duplicate, unknown and invalid inputs.
- Per-constraint outcomes and violation/unresolved witnesses.
- Conjunctive per-action dispositions and complete-family good/no-good/unresolved outcome.
- Canonical output ordering by action and constraint IDs.
- Independent deterministic replay that compares the entire contract, witness and disposition output.

## 5. Tests declared

- Complete feasible set, with infeasible and unresolved alternatives retained.
- No-good only when every action has at least one definite violation.
- Any unresolved action prevents no-good when no action is good.
- Both AtLeast and AtMost equality boundaries.
- Input-order invariance.
- Missing, duplicate and unknown action/constraint cells.
- Duplicate frozen action/constraint identities.
- Empty action/constraint universe.
- Nonfinite threshold, nonfinite interval and inverted interval.
- Replay of exact output and rejection of witness/disposition mutation.
- Deterministic 256-campaign partition test: every action belongs to exactly one family outcome.

These are source-level test declarations. No Rust compile, test, formatting, Clippy, CI or performance result is claimed.

## 6. Benchmarks and industry comparison

### Matched research baselines

- MultiTUCB (Jiang et al., ACML 2025) for vector feedback and componentwise threshold feasibility.
- Single-objective APGAI on one-threshold reductions, not as a substitute for the vector guarantee.
- Uniform and independent-per-constraint allocation.
- Objective-wise multi-objective BAI on the distinct best-arm-per-objective task.
- Constrained Pareto PSI on the distinct feasible Pareto-set task.

### Industrial experimentation baselines

Large-scale product experimentation platforms contribute important system-level safeguards that are not equivalent to bandit algorithms:
- Microsoft ExP practice: data-quality metrics (including sample-ratio mismatch), objective/overall evaluation criteria, local diagnostics and guardrail metrics; automated monitoring, alerts and auto-shutdown for severe regressions.
- Uber XP: fixed-horizon A/B/N, sequential probability ratio tests, causal-inference methods and continuous bandit/optimization methods; universal holdouts to estimate longer-term effects.
- Netflix's science-centric platform: shared end-to-end experimentation architecture, causal-inference abstraction and ability to run the same production analysis locally.
- Statsig's documented practice: predeclared hypothesis, primary/secondary/guardrail metric separation, sequential-testing options, CUPED/variance reduction and multiple-comparison controls.

These should be treated as platform capability comparators, not direct algorithmic sample-complexity baselines.

Sources:
- Microsoft trustworthy experimentation, pre-experiment: https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-pre-experiment-stage/
- Microsoft trustworthy experimentation, during-experiment: https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-during-experiment/
- Uber experimentation platform: https://www.uber.com/us/en/blog/xp/
- Netflix science-centric platform (ICSE-SEIP 2020): https://ieeexplore.ieee.org/document/9276550/
- Statsig configuration and sequential-testing documentation: https://docs.statsig.com/statsig-warehouse-native/features/experiment-options

### Locked matrix

Sweep:
- actions K = 4, 8, 16, 32, 64, 128;
- constraints M = 2, 4, 8, 16, 32;
- mixed AtLeast and AtMost constraints;
- all-feasible, no-good, sparse-infeasible and unresolved-at-budget regimes;
- independent, positively correlated, negatively correlated and block-correlated constraints;
- well-separated, near-threshold and exact-boundary means;
- clean data, sample-ratio mismatch, missingness, attrition and exposure contamination;
- static and dynamic contexts, drift, source/target shift and delayed outcomes;
- independent feedback versus shared vector feedback.

Report:
- familywise false-good and false-no-good rates;
- complete good-set precision/recall;
- false inclusion/exclusion;
- familywise coverage;
- unresolved rate and cost-to-good/no-good certificate;
- per-constraint witness correctness;
- SRM / data-quality detection latency;
- guardrail violation detection latency and unsafe exposure;
- p50/p95/p99 allocation, classification and replay time.

## 7. Proposed acceptance targets

- 0 hard-constraint false passes in every deterministic negative control.
- 0 false no-good outcomes while any action remains unresolved.
- At least 95% complete-good-set success in registered in-scope regimes, subject to correct familywise coverage.
- The matched MultiTUCB baseline must be reproduced first; only then assess whether shared vector evidence, interval composition and scoped evidence gates reduce fully charged cost.
- Target at least 15% lower cost-to-valid-good-set/no-good decision than MultiTUCB on preregistered high-correlation or shared-observation regimes, with no increase in false-good/no-good rates. This is a research target, not a result.

No SOTA claim is made by this classifier alone.
