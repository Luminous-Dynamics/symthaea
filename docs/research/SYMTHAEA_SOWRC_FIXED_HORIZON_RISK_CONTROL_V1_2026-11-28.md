# Symthaea fixed-horizon SOWRC risk control v1

Status: research implementation; exact-head Rust qualification pending. This implementation adapts the observable martingale-mixture construction in Bai, Fang and Chen (2026), *Risk-Controlling Predictive Sets for Time-Series Events Under Selective Observation with Finite-Sample Guarantees*, Axioms 15(9), 706, DOI 10.3390/axioms15090706.

## Why a separate lane is required

PR #14036 measures risk on independently adjudicated records. PR #14062 estimates population risk from known positive inclusion probabilities. PR #14101 provides a fixed-quota stratified SRSWOR audit and a design-based variance estimator.

Those are not the same as a finite-sample risk-control certificate. The SOWRC construction adds a martingale-mixture upper boundary that can control a weighted calibration-population risk over a predeclared nested prediction-set grid, even when the underlying time-series outcomes are temporally dependent, provided the design assumptions hold.

The current v1 is a fixed-horizon implementation of the core mixture/inversion step. It is not an anytime-valid implementation and does not include the future-deployment drift theorem.

## Input and filtration boundary

Each record is in exact chronological order and contains:
- a stable ID and sequence number;
- the conditional probability q_t that an independent label is observed, known from pre-label history;
- whether the label was actually observed;
- the nonconformity score of the observed true label, when observed.

The candidate's class probabilities and prediction outputs are frozen before labels. The true-label nonconformity is r_t(X_t,Y_t) = 1 - p_t(Y_t | X_t), computed by the evaluator only after an independent label exists. The candidate process never receives the audit membership, the hidden label, or the resulting score.

The inclusion probability q_t must be the probability a label is actually observed, conditional on the pre-label history. For an independent Bernoulli audit only, q_t is the audit probability. If a predictable alert channel is combined with an independent audit channel, q_t may be computed as pi_t + rho_t - pi_t*rho_t; that union formula is valid only under conditional independence of the two channels. The present module takes q_t as an externally supplied, manifest-bound quantity and does not generate random numbers or implement an alert channel.

## Assumptions

1. Predictability: the audit decision and q_t are known before the corresponding true label.
2. Conditional ignorability: I_t is independent of Y_t conditional on pre-label history.
3. Positivity: a deterministic q_min > 0 satisfies q_t >= q_min for every horizon event.
4. Bounded loss: nonconformity scores lie in [0,1], and miscoverage is 1{r_t > lambda}.
5. Nested safety: as lambda increases, the prediction set expands and the miscoverage loss cannot increase. At lambda=1 the set is the complete label space and miscoverage is exactly zero.
6. Frozen deterministic design: the horizon, alpha, delta, q_min, threshold grid, betting fractions, mixture weights, predictor artifact and inclusion design are committed before the audit labels are inspected.

Conditional ignorability and the integrity of the custodian's randomization are external assumptions; the evaluator cannot prove them from the observed selective labels.

## v1 deterministic envelopes

The implementation deliberately restricts to constant predictable weights a_t = 1. Therefore:

- A_n = n;
- c_n = 1;
- b_n = 1/q_min.

This keeps the first implementation auditable. Horizon-dependent exponential forgetting, arbitrary predictable weights and anytime monitoring are deferred.

## Horvitz-Thompson loss

For a nested set threshold lambda, define:

`L_t(lambda) = 1{r_t(X_t,Y_t) > lambda}`.

The observed HT loss is:

`Z_t(lambda) = I_t * L_t(lambda) / q_t`.

The observable cumulative estimate is:

`HT_n(lambda) = sum_t Z_t(lambda)`.

The descriptive HT empirical risk is HT_n(lambda)/n. It is not clipped to [0,1] and is not itself a risk certificate.

## Martingale-mixture boundary

For each fixed betting fraction eta, define:

`g(eta) = eta^2 / (2 * (1 - eta/3))`

and

`kappa(eta) = eta - (1/q_min) * g(eta)`.

The admissibility conditions are 0 < eta < 3 and kappa(eta) > 0. The module's default betting fractions are fixed fractions of:

`eta_max = 2*q_min / (1 + 2*q_min/3)`,

which is strictly inside the positive-kappa region.

For fixed positive mixture weights w_l summing to one, the observable mixture at candidate risk r is:

`E_n(lambda,r) = sum_l w_l * exp(kappa(eta_l)*n*r - eta_l*HT_n(lambda))`.

The threshold-specific error allocation is delta_j = delta/J for J grid points. The upper bound is:

`U_n(lambda; delta_j) = min(1, inf{r in [0,1]: E_n(lambda,r) >= 1/delta_j})`.

The map in r is increasing, so the implementation inverts it by bisection in log space to avoid direct exponential overflow.

## Nested threshold selection

The threshold grid is strictly increasing, fixed before labels, and ends at lambda=1. The rule selects the smallest nonterminal lambda whose risk upper bound is at most alpha. If none qualifies, it returns lambda=1. That fallback is structurally safe: the prediction set contains every label, so its miscoverage is exactly zero. The selected risk upper bound is reported as zero for this structural fallback, separately from the statistical mixture bounds reported for every grid point.

Under the paper's assumptions and theorem, the fixed-horizon mixture bounds hold simultaneously over the grid with probability at least 1-delta, and the selected nested set has calibration-population risk at most alpha with probability at least 1-delta. This implementation encodes the theorem's formula, but does not independently prove the mathematical result or verify the input assumptions.

## Numerical / algorithmic details

The module sorts observed nonconformity scores once, then maintains a monotone cursor through the fixed threshold grid. A suffix-weight array produces HT loss totals for all thresholds without rescanning the cohort. Complexity is O(N log N + N + J), where N is the calibration horizon and J is the number of thresholds. Tie semantics follow the paper's strict loss test r > lambda: a score exactly equal to lambda is included in the prediction set and is not a miscoverage.

## What this does not guarantee

- It controls the weighted calibration population only. It does not imply future deployment control under unrestricted drift.
- A prospective deployment guarantee needs an external, independently certified drift envelope with its own failure probability delta_dep; the total failure budget then includes both delta and delta_dep.
- This fixed-horizon module does not implement the anytime/time-uniform extension, multiple-risk control, adaptive audit budgets, or estimated propensities.
- It does not authenticate the external randomization, audit custodian, class-score artifact or label adjudicator.
- It does not estimate subgroup-conditional risk separately.
- It does not implement the fixed-quota SRSWOR variance estimator from PR #14101; first-order SRSWOR propensities cannot be substituted for predictable q_t without a separate filtration/design derivation.
- A statistical certificate is not evidence that the model is accurate, that a scientific claim is true, or that the underlying research policy is superior.

## Tests authored

The production module includes 15 tests covering default admissibility across audit floors, HT arithmetic, monotone loss and safe terminal behavior, nontrivial bounds, structural fallback, delta allocation, support failures, label/score consistency, sequence and ID integrity, grid/mixture validation, bounded scores, exact Bernoulli audit-mask enumeration with design-probability weights, strict tie semantics and calibration manifest requirements.

These tests are imported by path into the standalone Rust 1.96 epistemic harness. They are authored, not claimed as executed until an exact-head workflow result is observed.
