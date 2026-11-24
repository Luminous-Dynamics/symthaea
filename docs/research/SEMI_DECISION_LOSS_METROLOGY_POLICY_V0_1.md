# SEMI-DECISION-LOSS-001 — decision-loss metrology policy v0.1

**Research issue:** #13538  
**Implementation:** `symthaea-semiconductor::benchmark::decision_loss_metrology_policy`  
**Status:** Rust EVSI scorer and exact small-portfolio optimizer with authored tests; exact-head validation pending.

## Paradigm

Choose physical metrology by expected reduction in downstream decision loss, not by predictive variance or interval width alone.

A sample is valuable when knowing its physical result is likely to change an action (accept, hold, rework, scrap, escalate) in a way that avoids a larger expected loss. An uncertain wafer far from a decision boundary may have low EVSI; a moderately uncertain wafer near an asymmetric false-accept boundary may have high EVSI.

The score must include:
- the prior probability of the relevant physical quality states;
- the declared loss of each downstream action under each state;
- the likelihood of each assay result conditional on physical state;
- measurement cost;
- turnaround penalty and decision deadline;
- pre-measurement instrument/calibration readiness;
- group coverage requirements (tool/chamber/recipe strata).

Every cost and likelihood is explicit input. The system does not infer economics from public wafer datasets.

## Expected value of sample information

Let quality state be (S), decision action (a), assay outcome (O), prior (p(s)), loss (L(a,s)), and assay model (p(o\mid s)).

Before measuring:

[
R_{0}=\min_{a}\sum_s p(s)L(a,s).
]

For each possible assay outcome, update the state posterior:

[
p(s\mid o)=\frac{p(s)p(o\mid s)}{\sum_{s'}p(s')p(o\mid s')}.
]

Expected post-measurement decision risk:

[
R_{1}=\sum_o p(o)\min_a\sum_s p(s\mid o)L(a,s).
]

Gross EVSI:

[
\operatorname{EVSI}=R_0-R_1.
]

Net value in the current prototype:

[
V_{net}=\operatorname{EVSI}
-C_{measurement}
-C_{delay}\times T_{turnaround}.
]

The current module uses a caller-declared decision-loss model and assay confusion likelihoods. It does not estimate those inputs from data. The likelihood matrix must include physical assay uncertainty, false-positive/false-negative behaviour and measurement-system limitations. If those likelihoods are misspecified, EVSI is misspecified.

## Readiness gate

A candidate includes a digest-sealed premeasurement readiness statement containing:
- instrument, measurand and unit;
- certificate ID, certificate digest and structured calibration-record digest;
- measurement-system profile digest;
- validity window and effective revocation time;
- expected standard-uncertainty upper bound;
- readiness state QUALIFIED / UNKNOWN / REJECTED.

Only QUALIFIED candidates whose certificate is valid at planned measurement time and whose predicted result can return before the decision deadline are eligible.

This readiness statement is a scheduling precondition, not a postmeasurement traceability certificate. After measurement, the actual label/value must still pass the full #13382 qualification and #13467 numeric-binding gate before it can update a model or calibrator.

## Portfolio optimizer

The v0.1 portfolio problem is exact 0/1 enumeration for at most 20 candidate measurements.

Objective: maximize the sum of candidate net EVSI, subject to:
- measurement budget units;
- predeclared minimum selection counts for tool/chamber/recipe groups;
- candidate readiness and deadline constraints.

The exact optimizer returns INFEASIBLE_COVERAGE rather than silently weakening group minima. It uses deterministic candidate-ID ordering and lexicographic tie-breaking. More than 20 candidates is rejected rather than silently using a heuristic; scalable mixed-integer or dynamic-programming methods are future work and must be independently cross-checked.

A required group-coverage quota may force selection of a negative-net-EVSI candidate. That is an explicit fairness/audit constraint, not an EVSI-positive recommendation. Reports retain per-candidate net values so that distinction is visible.

## Authored tests

Fourteen unit tests currently cover:
1. informative assay reduces Bayes decision risk more than a weak assay;
2. UNKNOWN/REJECTED readiness is never selected;
3. exact portfolio chooses the highest net decision value under a fixed budget;
4. group coverage minima are hard constraints;
5. impossible group minima return explicit INFEASIBLE_COVERAGE;
6. an assay that cannot return before the decision deadline is excluded;
7. calibration revoked before planned measurement is excluded;
8. selection receipt deterministic replay;
9. negative net-value candidate excluded unless required by a quota;
10. duplicate candidate IDs rejected;
11. candidate count above declared exact limit rejected;
12. invalid assay likelihood matrix rejected;
13. tampered readiness evidence digest rejected;
14. candidate input order does not change the portfolio receipt.

Exact-head rustfmt, strict Clippy, unit tests and SEMI-BENCH-001 reference lanes are still required.

## Competitive benchmark

Compare under identical chronological streams, hidden ground-truth labels, equal physical-measurement budgets and equal turnaround models:

- random sampling with stratified randomized audit;
- fixed-frequency / fixed-per-lot;
- maximum predictive variance;
- maximum conformal interval width;
- highest defect probability;
- raw EVSI;
- cost-aware net EVSI (this method);
- robust EVSI over preregistered cost/assay-parameter intervals;
- the existing #13328 adaptive policy and #13336 independent selection receipts;
- the #13387 two-stage screen/confirm planner, where its assumptions match.

Downstream performance metrics:
- realized/expected decision loss per wafer, lot and qualified measurement;
- escaped-defect rate and its one-sided upper confidence bound;
- false holds, unnecessary inspection and rework;
- decision-loss reduction at equal budget;
- measurements per detected excursion;
- time-to-detection and missed decision deadlines;
- subgroup coverage by tool/chamber/recipe;
- selection bias / prevalence estimator bias and interval coverage;
- robust regret to an offline oracle with the same candidate set and budget.

Evidence/operations:
- readiness QUALIFIED / UNKNOWN / REJECTED counts;
- selected and unselected candidate receipt digests;
- assay sensitivity/specificity and uncertainty assumptions;
- capacity utilisation, queue length, turnaround and late results;
- exact portfolio replay and budget/group-accounting checks.

## Proposed targets (hypotheses)

- At equal qualified physical-measurement budget, target ≥10% reduction in realised decision loss versus random/fixed-frequency sampling.
- At equal budget, target ≥5% improvement over max-uncertainty acquisition without worsening the preregistered escaped-defect upper bound.
- 100% exact replay of EVSI inputs, expected loss, selection and budget accounting.
- 100% exclusion of UNKNOWN/REJECTED/revoked/not-valid-at-measurement calibration readiness.
- 100% explicit INFEASIBLE_COVERAGE when group quotas cannot be satisfied.
- No subgroup below its preregistered audit minimum.
- Publish Pareto frontiers for loss, metrology cost, turnaround and escaped-defect risk. Do not hide trade-offs inside one score unless the user-declared utility function is shown.

These targets are hypotheses and must be preregistered on matched chronological data. Synthetic tests do not demonstrate manufacturing savings.

## What would count as beating industry/SOTA

The strongest public operational comparator is the 2026 uncertainty-aware dynamic VM + physical-metrology selection/adaptation study on 114,774 wafers from a 300 mm plasma-etch fab. A fair comparison requires the same dataset, metrology capacity, decision-loss definition and temporal split; our public work has no access to that private fab dataset, so no head-to-head win is claimed.

The GFA-VM+ few-shot benchmark, GRAPE chronological streaming benchmark and NIST SI-for-Manufacturing TIDAL integration/benchmark programme are complementary but measure different tasks. They belong on separate scorecard axes, not one composite leaderboard.

## Claim ceiling

This is a synthetic Bayes decision/EVSI calculator and exact small-portfolio optimizer. It does not establish real fab cost coefficients, assay likelihoods, physical measurement traceability, escaped-defect performance or manufacturing ROI. It does not authorize sampling-plan changes, wafer disposition, APC/R2R, recipe changes or equipment control.


## Independent optimizer oracle extension

A third integration test now independently enumerates every feasible subset on a four-candidate, two-chamber fixture. It recomputes candidate net EVSI from the separate posterior-risk oracle, applies capacity and minimum-group constraints, and compares selected IDs, total net value, budget use and group counts with the exact production optimizer. This tests the optimizer's claimed exactness independently of its bitmask loop.
