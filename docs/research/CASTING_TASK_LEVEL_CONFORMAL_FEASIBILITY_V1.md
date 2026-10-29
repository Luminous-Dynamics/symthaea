# CASTING: Task-Level Conformal Feasibility Screen v1

**Evidence class:** implementation protocol and synthetic unit tests only. The method is not yet executed by confirmed CI in this change and is not validated against foundry data.

## Motivation

The historical-transfer evaluator currently includes a distance-based abstention policy. Its fixed descriptor-distance threshold is a useful diagnostic but is a heuristic, not a calibrated uncertainty estimate or false-feasibility risk guarantee. CASTING needs a principled way to translate archived prediction errors into a conservative recommendation rule while making the assumptions visible.

This module adds a task-level split-conformal screen for the **frozen finite candidate pool** within a predeclared engineering stratum. It is intended as a fail-closed research tool that can be compared against the existing fixed-threshold abstention policy.

## Algorithm

1. Freeze the policy source/config digests, the ordered candidate pool (candidate IDs and normalized design coordinates), and the calibration/target split before unsealing calibration outcomes. Target prediction manifests contain no observed outcomes.
2. For each independent calibration task, calculate one score: the maximum positive feasibility-margin overprediction across the entire frozen candidate pool:
   score(task) = max_candidate max(0, predicted_margin - observed_margin).
   Candidate points are nested observations and do not inflate the independent task count.
3. Within each stratum, compute the finite-sample one-sided split-conformal order statistic at 1 - alpha: rank = ceil((n + 1) * (1 - alpha)). If the rank exceeds the number of calibration tasks, or the configured minimum task count is not met, the stratum is underpowered and every target candidate in it abstains.
4. For a target task in a calibrated stratum, compute its nearest calibration-task descriptor distance. A target outside the predeclared descriptor-support threshold abstains.
5. Otherwise calculate calibrated_lower_margin = predicted_margin - q. Recommend feasible only when that lower margin is nonnegative. All other candidates abstain rather than being force-labelled feasible or infeasible.
6. Emit a receipt binding exact config, calibration manifest, target prediction manifest, and evaluator source digests. Outcomes for target tasks are absent from the decision path.

## Assumptions and guardrails

- Calibration and target **tasks** must be exchangeable within each declared stratum for the standard split-conformal interpretation. Matching a descriptor threshold does not prove exchangeability.
- A fixed, identical finite candidate pool is required for calibration and target tasks. Pool integrity is checked using a canonical digest over ordered candidate IDs and normalized design coordinates.
- For alpha = 0.05, the minimum task count for a finite order statistic is 19; this is only a mathematical floor, not evidence of calibration quality or sufficient validation for physical safety.
- If task families represent unseen material systems, process routes, geometries, or boundary regimes not supported by the calibration strata, the system must abstain. A descriptor-distance guard does not turn an out-of-distribution task into an exchangeable one.
- The task-level bound concerns the event of any false-feasible recommendation in the predeclared finite pool on a new exchangeable task. It is not an unconditional per-pour guarantee, does not survive arbitrary adaptive changes to the candidate pool, and does not certify casting feasibility or safety.
- Never pool synthetic, public-analytic, simulator-backed, and physical evidence classes. A calibration receipt from one evidence class does not qualify a deployment in another.

## Required benchmark

Compare at matched policy and task-pool digests:

- Low-fidelity-only predictor with no abstention.
- Existing fixed-distance KNN abstention (threshold frozen before outcomes).
- The task-level conformal feasibility screen.
- A naive empirical residual quantile baseline.
- A calibration-shuffled negative control.
- A deliberately OOD stratum to verify fail-closed abstention.

Report task-clustered risk-coverage curves; overall and per-stratum false-feasible counts and denominators; one-sided exact confidence bounds from independent post-decision validation tasks; selective candidate coverage; fraction of tasks with at least one recommendation; objective quality lost to abstention; descriptor distance; calibration sample count; and full evaluation cost. Do not count candidate rows as independent observations. The conformal score quantile is not a substitute for post-decision risk validation.

## Falsification criteria

The hypothesis that task-level conformal screening improves decisions is weakened or falsified if:

- it fails to reduce false-feasible outcomes at comparable coverage on independent, exchangeable tasks;
- the supposed improvement disappears after task-level, rather than candidate-level, aggregation;
- it merely abstains on almost everything without improving covered-set error/utility;
- its calibration or false-feasible risk degrades badly across preregistered shifts;
- the decision rule is sensitive to unreported task/candidate selection or candidate-pool changes.

## Claim ceiling

A receipt reports recommendations and abstentions under a declared task-level calibration protocol. It is not proof of exchangeability, holdout secrecy, calibration under unseen shift, physical feasibility, safety certification, optimizer superiority, or SOTA performance. The module is a comparison candidate, not yet evidence that the approach outperforms the current heuristic or published methods.


## Independent post-decision risk audit

The companion tools/casting-benchmark/task_conformal_validation_audit.py accepts the frozen decision receipt, exact target prediction manifest, a predeclared risk contract, and held-out outcomes unsealed only after the decisions were frozen. It requires the contract to bind the raw decision-receipt and target-manifest digests, then joins candidate IDs and task IDs exactly.

Its primary independent unit is the target task, not the candidate: a task is a false-feasible failure when **any** candidate recommended feasible has an observed negative margin. The scorer computes exact one-sided Clopper–Pearson upper bounds for each declared critical stratum, counts only tasks with at least one feasible recommendation in the risk denominator, and reports candidate-level counts as descriptive secondary metrics.

For a one-sided 95% upper bound at or below 5%, 59 independent recommended tasks with zero false-feasible task failures is the zero-failure planning floor. More validation units may be required when failures occur, when critical strata are heterogeneous, or when a stricter risk target is appropriate. A pooled good result cannot rescue a critical stratum whose bound exceeds the contract. A risk-bound failure takes precedence over an unrelated total-sample shortfall; adequate evidence must satisfy both the family unit floor and recommended-task denominator floor to pass.

This audit is separate from the prediction path so target outcomes cannot influence the recommendation. It still relies on claims that validation units are representative and independent, and that the contract/digests were frozen before outcome unsealing. Those claims are not proven by hashes. A pass means only that the declared research-stage risk gate passed, not physical safety certification.
