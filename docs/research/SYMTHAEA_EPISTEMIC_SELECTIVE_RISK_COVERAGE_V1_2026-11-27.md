# Symthaea selective-prediction evaluation v1

Status: research implementation; not a production decision policy or benchmark result.

## Purpose

PR #13969 separated predicted confidence from independently adjudicated correctness and added adjudication-only Brier/ECE. This increment addresses the next missing capability: measuring the risk incurred when the system answers, and how that risk changes as it abstains more often.

The evaluator is deliberately separate from the learner. It does not update credibility, choose a threshold, authorize an answer, or treat its own scores as truth.

## Input and admissibility

Each frozen cohort row contains a stable unique ID, predicted binary label, confidence in that predicted label, optional independently adjudicated truth, observed candidate abstention and reason, domain, source family, delayed/censored status and adjudicator.

Only rows with an independent truth label and a finite confidence in [0, 1] enter selective-risk metrics. Pending/censored outcomes are excluded until adjudicated; invalid-probability adjudicated rows are separately counted and excluded. Duplicate and empty IDs fail validation. The evaluator cannot authenticate the independence or competence of a supplied adjudicator; this remains a custody/evaluator responsibility.

The claim-correctness adapter maps a positive claim to predicted label=true and maps the adjudicated correctness value to truth. General binary tasks must provide their actual predicted and true labels.

## Metric definition

The metric version is `symthaea-risk-coverage-v1`.

For an eligible cohort of size N, sort predictions by descending confidence. Let A(t) be all examples with confidence at least threshold t. Coverage is |A|/N. Selective risk is:

`risk(A) = count(predicted_label != adjudicated_truth in A) / |A|`.

Equal confidence values are admitted atomically. No tie is split by input order or record ID. A requested coverage target selects the first complete confidence group whose achieved coverage is at least the target; therefore achieved coverage may exceed 50%, 80%, or 90%.

AURC uses a specified right-step integral: for each complete tie group j, multiply the increase in coverage by the risk of the resulting accepted set, then sum. For group sizes m_j, cumulative accepted n_j and cumulative errors e_j:

`AURC = sum_j (m_j / N) * (e_j / n_j)`.

This convention produces 0 for an all-correct cohort and 1 for an all-wrong cohort. It is tie-invariant under the evaluator's atomic-group rule. Alternative trapezoidal or item-wise conventions must use a different metric version; scores are not interchangeable.

## Baselines and uncertainty

The report includes:
- always-answer risk and its two-sided 95% Wilson score interval;
- random-abstention expected risk at each requested coverage (equal to full-cohort risk in expectation), with the full-cohort Wilson interval;
- inclusive confidence-threshold policies at configured thresholds;
- observed candidate coverage, risk, abstention reasons and decision cost.

Fixed-coverage points report target and achieved coverage, accepted sample count, error count, selective risk and Wilson interval. A score and interval are omitted when the accepted set is smaller than the configured minimum. The default minimum is 20; the small-sample test configuration exists only to verify edge cases.

Decision cost is optional and only emitted when the evaluator supplies nonnegative finite costs for false positives, false negatives and abstentions. The reported value is mean cost per original eligible case. Random-abstention expected cost uses the requested coverage and assumes random selection independent of correctness. Costs are policy/application-specific and should never be inferred from confidence.

Strata are reported independently for domain, source family, delayed/censored status and adjudicator. Small strata are counted but do not receive scores until they meet the minimum sample size. Unknown metadata is retained as an explicit group.

## Interpretation limits

- The evaluator does not create ground truth. Pending records never enter calibration or risk.
- The cohort, labels, policy outputs and costs must be frozen before threshold selection. Tuning and reporting on the same cohort will produce optimistic estimates; promotion requires a held-out cohort.
- Wilson intervals quantify binomial sampling uncertainty under their assumptions; they do not account for clustered task families, repeated measurements, adjudicator dependence or distribution shift. Confirmatory comparisons should bootstrap at the task/family cluster level.
- The random-abstention result is an expectation, not a seeded sample realization. A paired experiment should separately report repeatable randomized-baseline runs.
- AURC and Brier/ECE are not substitutes for task success, scientific quality, evidence integrity, or causal decision value.
- This implementation is an offline metric library. It is not connected to a live Symthaea policy and no SOTA or safety claim is made.

## Acceptance and tests

The source is imported by path in the standalone Rust 1.96 adjudication harness. Tests cover pending/invalid exclusion, ties, AURC edge cohorts, empty and single-sample cohorts, sample-size gating, asymmetric costs, observed abstention reasons, threshold policies, required strata, duplicate IDs and invalid configuration. Exact-head formatting, test and strict-Clippy results must be observed before calling the increment qualified.
