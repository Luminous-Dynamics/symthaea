# Observer risk/coverage evaluator

## Status and scope

The Rust module `observer_evaluation` evaluates a frozen observer on one locked-confirmation or external-replication partition. It is a deterministic metrics and data-contract utility, not a model, assay, statistical design generator, appliance controller or release authority.

It expects each physical run to have:
- a unique record in the grouped run manifest;
- a complete item inventory;
- one pass probability for every required item × endpoint;
- one independently adjudicated assay decision for every required item × endpoint, each explicitly bound to the same source physical-run ID;
- one stable model ID, model SHA-256 and calibration-bundle SHA-256 shared across the evaluation.

The supplied outcome records must already have been qualified upstream. An assay-record ID is a traceability reference only. This module cannot authenticate the archive, prove that a method is independent, or prove that the assay result is scientifically correct.

## Partitions and grouping

The evaluator requires development, calibration and locked-confirmation partitions. For external replication it additionally requires an external-replication partition. It delegates group-leakage checks to `validation::audit_grouped_holdout`.

The caller must name holdout axes explicitly, such as machine unit, session, soil lot, water batch or assay laboratory. The audit proves only that the supplied opaque group identifiers do not cross the named partitions. It does not establish statistical independence within the target partition or correct cluster correlation.

Evaluate locked confirmation and external replication separately. Do not choose thresholds, retrain, recalibrate, or select the winning sensor tier after inspecting locked-confirmation outcomes.

## Threshold semantics

Each prediction is the model's calibrated probability that one item/endpoint passes its own criterion. For threshold (t), the run is released for metric purposes only when:

[
min_{i in items, e in endpoints} p_{i,e} ge t
]

This minimum-of-marginals rule is intentionally conservative. It is **not** a joint probability that all items/endpoints pass, and the evaluator does not assume independence between endpoints.

The code sorts thresholds in ascending order and rejects duplicates, non-finite values, or values outside [0, 1].

## Ground-truth aggregation

For a physical run:
- **Conforming:** every required item/endpoint assay is PASS.
- **Nonconforming:** at least one required item/endpoint assay is FAIL.
- **Unresolved:** no endpoint fails, but at least one outcome is INDETERMINATE.

A failed endpoint makes the run nonconforming even if another endpoint is indeterminate. An unresolved run is never promoted to conforming.

Every required item × endpoint must have exactly one prediction and exactly one assay decision. Duplicate, missing, unexpected or invalid rows are rejected rather than imputed. Each assay row carries a source run ID that must match the containing physical run.

## Reported metrics

For every threshold the report includes:
- physical runs, conforming/nonconforming/unresolved truth counts;
- released run count and release coverage, with unresolved runs retained in the denominator;
- false-accept count among known nonconforming challenge runs;
- challenge false-accept rate: false accepts / all known nonconforming runs;
- false-release rate among released runs with adjudicable ground truth;
- unresolved runs that crossed the release threshold (unverifiable releases);
- conforming runs not released and conforming release rate;
- endpoint-specific nonconforming/false-accept/unresolved counts.

The challenge false-accept rate and false-release rate among verified releases answer different questions. The latter depends on the prevalence of failures in the evaluation sample. Operational false-release probability must be estimated on a representative operating-population sample; it must not be inferred from an intentionally enriched challenge set.

## Zero-event bound caveat

The report optionally gives an exact one-sided zero-event Clopper-Pearson upper bound for the challenge false-accept probability:

[
p_U = 1 - alpha^{1/n}
]

where (n) is the number of adjudicable nonconforming challenge runs and (alpha) is the one-sided tail probability.

The bound is emitted only when:
- no false accepts occurred;
- at least one nonconforming challenge run exists; and
- every target-partition run has adjudicable ground truth.

The third condition is deliberately strict: unresolved outcomes could conceal false accepts, so any unresolved target run suppresses the zero-event bound.

**This is an IID diagnostic, not a cluster-adjusted confidence bound.** The grouped partition audit does not make repeated runs on one machine/day/soil lot independent. A valid inferential claim requires a preregistered challenge sampling plan with defensible independent units or a separate cluster-aware method. Do not use items, coupons, images, frames or assay replicates as independent trials.

## Probabilistic scoring and reliability

For each endpoint, the evaluator additionally reports:
- run-balanced Brier score and binary log loss, averaging item-level scores within a physical run before averaging runs;
- run-balanced mean predicted pass probability and observed item pass fraction;
- ten fixed-width reliability bins over each run's mean endpoint probability, with per-bin physical-run counts, mean prediction and mean observed item pass fraction;
- resolved and unresolved run counts and the number of resolved item × endpoint pairs.

An endpoint/run is scored only when every item has an adjudicable PASS or FAIL result for that endpoint. If any item is INDETERMINATE, the whole endpoint/run is excluded from binary probabilistic scoring and counted as unresolved. No missing label is imputed.

The run-balanced weighting prevents larger loads from automatically contributing more statistical weight than single-item runs. Within each run, however, item-level predictions still contribute to the run's mean scoring loss. Reliability bins are descriptive; sparse bins are not strong evidence of calibration.

Brier score and log loss measure probabilistic prediction quality but do not establish a safe false-release bound. The reliability summary is not a substitute for a separate model calibration protocol, locked confirmation, external replication or cluster-aware confidence intervals. The evaluator does not derive calibrated probabilities or recalibrate on the target data.
- cluster-robust confidence intervals or hierarchical models;
- causal comparisons between sensor tiers;
- cost-weighted policy value or adaptive treatment effects;
- subgroup confidence intervals for each material/soil/geometry beyond endpoint-level event counts.

Those must be added only with explicit calibration partitions, estimands, grouping structure and statistical methods. Prediction probabilities used here must be calibrated elsewhere before being described as calibrated. A threshold sweep does not itself calibrate probabilities.

## Test coverage authored

14 source-authored unit tests cover:
- threshold sorting and run-level risk/coverage;
- a false-accepting challenge run;
- unresolved assay truth and unverifiable release;
- suppression of the zero-event bound when any target truth is unresolved;
- one physical run with multiple items remaining one trial;
- complete item × endpoint matrix requirements and assay-to-run ID mismatch rejection;
- group leakage rejection;
- external replication requiring a locked-confirmation partition;
- frozen model/calibration identity;
- invalid probability range;
- run-balanced Brier/log-loss and reliability-bin arithmetic;
- unresolved endpoint results excluded from binary scores and retained in the denominator;
- missing evaluation runs and unsupported target partitions.

These tests are source-authored. Cargo test/fmt/clippy must be run against the exact PR head before merge. No physical observer or early-stop performance is claimed.
