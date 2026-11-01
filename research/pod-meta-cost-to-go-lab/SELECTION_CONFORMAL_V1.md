# Selection-Conditioned Proof-Cost Calibration V1

**Status:** selected-policy split-conformal research baseline. Not CAP, not an online false-coverage-statement-rate controller, and not a production scheduler.

Issue: [#9697](https://github.com/Luminous-Dynamics/symthaea/issues/9697).

## 1. Why selection changes the evaluation population

POD-Meta does not execute a uniformly random candidate. It selects a candidate using predicted proof progress, resource cost, risk, frontier context, and policy constraints. The selected-action population is therefore a policy-induced distribution and may differ materially from the full candidate pool.

All-candidate calibration must not be reported as selected-action calibration. The selected candidate's eventual proof cost may be systematically higher or lower than the pool average, especially when the selector chooses actions with unusually optimistic forecasts.

## 2. Selection receipt

Each candidate decision emits an immutable pre-execution `SelectionReceipt` binding:

- episode identity and monotonic decision sequence;
- digest of the complete candidate pool presented to the selector;
- selected operation ID and operation-family label;
- frozen selection-policy digest;
- selected-population / exchangeability-cohort digest;
- authoritative pre-action state and proof-frontier digests.

The predictor receipt separately binds model identity and the same frontier digest. These digests are caller-supplied identifiers in the research simulator; the module does not recompute cryptographic attestations.

## 3. Calibration target

The first estimator is split conformal over independently verified **selected episodes** produced by one frozen selection policy / policy version and one declared selected-population cohort.

For each selected calibration episode, the target label is total remaining resource use from the selected pre-action state until sufficient proof is independently verified. The nonconformity score is the maximum positive normalized residual over all six resource dimensions:

`A_i = max_j max(0, (observed_remaining[i,j] - predicted_point[i,j]) / predicted_scale[i,j])`

The finite-sample rank is `ceil((n + 1) * (1 - alpha))`. When that rank exceeds the selected calibration count, the interval is unbounded. The rank is never clamped to the largest observed residual.

## 4. What the baseline guarantees—and does not

Under a frozen predictor and selection policy, a held-out calibration set and future episode that are exchangeable **after applying the same declared selection mechanism**, and matching model / policy / cohort identities, split conformal targets marginal simultaneous resource-vector coverage for the selected-policy population.

It does not provide:

- conditional coverage for arbitrary selection rules or every candidate subgroup;
- coverage when the policy version changes or adapts using calibration labels;
- false-coverage-statement-rate control across repeated online picks;
- a guarantee that all selected operations will complete proof;
- a censoring-adjusted guarantee for selected episodes whose eventual proof cost is unobserved.

The selected-cohort calibrator requires completed, independently verified proof outcomes. Censored selected episodes are excluded and must remain visible in the survival / IPCW datasets. Completed-case selection must not be generalized to all initiated episodes.

## 5. CAP and online partial-feedback comparators

[CAP: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://www.jmlr.org/beta/papers/v26/24-0452.html) studies post-selection predictive inference and reports exact finite-sample selection-conditional coverage and online false-coverage-statement-rate control under its stated procedures and assumptions. This implementation does not reproduce CAP.

[Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4eb5daabc45b45a9a312aa2c8fca8a74-Abstract-Conference.html) considers partial feedback when labels are observed only in a selection-dependent way and develops long-run coverage results connected to regret. This is relevant when POD-Meta observes outcomes only for selected operations.

[Practical Advantages of Online Conformal Prediction for Data Streams](https://arxiv.org/abs/2610.27740) compares sequential procedures against weighted full-conformal references, reporting coverage, width, and compute. It motivates measuring predictor adaptation and calibration-threshold adaptation separately.

## 6. Benchmark design

Paired candidate pools and outcomes should be reused across policies. Report:

- candidate-pool coverage and selected-cohort coverage separately;
- selected-policy coverage by operation family, task family, frontier complexity, and policy version;
- joint six-resource coverage, marginal coverage, per-dimension width, and envelope volume;
- false-feasible rate and false verified completion;
- risk/coverage and abstention on shifted or unsupported selected cohorts;
- calibration examples, effective cohort size, and selected-family support;
- calibration/selection overhead and decision latency.

Compare at least: candidate-pool split conformal, selected-policy split conformal, fixed-order selection, gain-only selection, risk-adjusted selection, and an independently implemented CAP-style reference when its exact selection assumptions can be met. Preserve native metrics and do not aggregate them into a weighted composite.

## 7. Required next research step

Implement an online selection-aware calibration-after-adaptive-pick method that controls repeated-statement FCR (or another explicitly defined sequential error measure) for POD-Meta's actual adaptive selection protocol. This requires a separate derivation and protected sequential evaluation; the present split-conformal baseline is not a substitute.