# Symthaea fixed-horizon SOWRC risk control v1

Status: research implementation; exact-head Rust qualification pending. This implementation adapts the observable martingale-mixture construction in Bai, Fang and Chen (2026), *Risk-Controlling Predictive Sets for Time-Series Events Under Selective Observation with Finite-Sample Guarantees*, Axioms 15(9), 706, DOI 10.3390/axioms15090706.

## Purpose and composition

The selective-prediction evaluator (#14036) measures risk on independently adjudicated records. The propensity estimator (#14062) supplies inverse-inclusion point estimates when adjudication is selective but has positive support. The stratified SRSWOR audit (#14101) adds design-based finite-population variance.

This SOWRC lane is different: it uses a predictable per-event inclusion probability and a martingale-mixture upper boundary to control a weighted calibration-population risk over a predeclared nested prediction-set grid. It does not treat the fixed-quota SRSWOR marginal pi_i as a substitute for predictable conditional q_t.

## Input and filtration boundary

Each record is chronological and contains a stable ID, sequence number, conditional inclusion probability q_t, whether an independent label was observed, and the nonconformity score for the observed true label.

The candidate's class probabilities and prediction outputs are frozen before labels. The true-label nonconformity is r_t(X_t,Y_t) = 1 - p_t(Y_t | X_t), computed by the evaluator only after an independent label exists. The candidate process never receives the audit membership, hidden label, or resulting true-label score.

The config binds three distinct manifests:
1. calibration protocol: alpha, delta, threshold grid, betting fractions and mixture weights;
2. inclusion design: q_t schedule, q_min, randomization process and custodian custody;
3. prediction artifact: frozen pre-label class probabilities and nonconformity function.

The inclusion probability q_t must be the probability a label is actually observed, conditional on pre-label history. For an independent Bernoulli audit, q_t is the audit probability. If a predictable alert channel is combined with an independent audit channel, q_t can be pi_t + rho_t - pi_t*rho_t; that union formula is valid only under conditional independence. The module accepts q_t as an externally supplied pre-label value and does not generate random numbers or implement an alert channel.

## Assumptions

1. Predictability: the audit decision and q_t are known before the corresponding true label.
2. Conditional ignorability: I_t is independent of Y_t conditional on pre-label history.
3. Positivity: a deterministic q_min > 0 satisfies q_t >= q_min for every horizon event.
4. Bounded loss: nonconformity scores lie in [0,1], and miscoverage is 1{r_t > lambda}.
5. Nested safety: as lambda increases, the prediction set expands and miscoverage cannot increase. At lambda=1 the set is the complete label space and miscoverage is exactly zero.
6. Frozen deterministic design: the horizon, alpha, delta, q_min, threshold grid, betting fractions, mixture weights, prediction artifact and inclusion design are committed before audit labels are inspected.

Conditional ignorability and the integrity of the custodian's randomization are external assumptions; the evaluator cannot prove them from selectively observed labels.

## v1 deterministic envelopes

The implementation deliberately restricts to constant predictable weights a_t = 1. Therefore A_n = n, c_n = 1, and b_n = 1/q_min. Horizon-dependent forgetting and anytime weights are deferred.

## Horvitz-Thompson loss

For a nested set threshold lambda, define L_t(lambda) = 1{r_t(X_t,Y_t) > lambda}. The observed HT loss is:

`Z_t(lambda) = I_t * L_t(lambda) / q_t`.

The observable cumulative estimate is the sum of Z_t. The descriptive HT empirical risk is that sum divided by n. It is not clipped to [0,1] and is not itself a risk certificate.

## Martingale-mixture boundary

For each fixed betting fraction eta, define:

`g(eta) = eta^2 / (2 * (1 - eta/3))`

and

`kappa(eta) = eta - (1/q_min) * g(eta)`.

Admissibility requires 0 < eta < 3 and kappa(eta) > 0. The default betting fractions are fixed fractions of:

`eta_max = 2*q_min / (1 + 2*q_min/3)`.

For fixed positive mixture weights w_l summing to one, the observable mixture at candidate risk r is:

`E_n(lambda,r) = sum_l w_l * exp(kappa(eta_l)*n*r - eta_l*HT_n(lambda))`.

The threshold-specific error allocation is delta_j = delta/J for J grid points. The upper bound is:

`U_n(lambda; delta_j) = min(1, inf{r in [0,1]: E_n(lambda,r) >= 1/delta_j})`.

The map in r is increasing, so the implementation inverts it by bisection in log space to avoid direct exponential overflow.

## Nested threshold selection

The threshold grid is strictly increasing, fixed before labels, and ends at lambda=1. The rule selects the smallest grid threshold whose simultaneous mixture bound is at most alpha. If no threshold qualifies, it returns lambda=1. That fallback is structurally safe: the prediction set contains every label, so its miscoverage is exactly zero. The selected risk upper bound is reported as zero for this structural fallback, separately from the statistical mixture bounds reported for all grid points.

Under the paper's assumptions and theorem, the fixed-horizon mixture bounds hold simultaneously over the grid with probability at least 1-delta, and the selected nested set has calibration-population risk at most alpha with probability at least 1-delta. This implementation encodes the theorem's formula, but does not independently prove the mathematics or verify the input assumptions.

## Numerical and algorithmic details

The module sorts observed nonconformity scores once, then maintains a monotone cursor through the fixed threshold grid. A suffix-weight array produces HT loss totals for all thresholds without rescanning the cohort. Complexity is O(N log N + N + J), where N is the calibration horizon and J is the number of thresholds. Tie semantics follow the strict loss test r > lambda: a score exactly equal to lambda is included in the prediction set and is not a miscoverage.

The fixed-horizon evaluator is not time-uniform. It does not implement multiple-risk, adaptive budget, estimated-propensity or future-drift extensions.

## What this does not guarantee

- It controls the weighted calibration population only. It does not imply future deployment control under unrestricted drift.
- Prospective deployment risk needs an external, independently certified drift envelope with its own failure probability delta_dep.
- It does not authenticate randomization, the audit custodian, the class-score artifact or the independent adjudicator.
- It does not estimate subgroup-conditional risk separately.
- It does not use the fixed-quota SRSWOR variance estimator from #14101; marginal pi_i values are not a substitute for predictable q_t without a separate sequential design derivation.
- A statistical certificate is not evidence that the model is accurate, that a scientific claim is true, or that the underlying research policy is superior.

## Tests authored

The production module includes 15 tests covering default admissibility across audit floors, HT arithmetic, monotone loss and safe terminal behavior, nontrivial bounds, structural fallback, delta allocation, support failures, label/score consistency, sequence and ID integrity, grid/mixture validation, bounded scores, exact Bernoulli audit-mask enumeration with design-probability weights, strict tie semantics and calibration-manifest requirements.

These tests are imported by path into the standalone Rust 1.96 epistemic harness. They are authored, not claimed as executed until an exact-head workflow result is observed.
