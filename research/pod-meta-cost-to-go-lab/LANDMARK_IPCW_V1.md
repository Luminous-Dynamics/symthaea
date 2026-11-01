# Landmark-Conditioned IPCW Proof-Completion Calibration V1

**Status:** HAPS-inspired weighted estimating-equation research baseline. The censoring nuisance model is external / cross-fitted; this module consumes its receipts. No exact finite-sample coverage claim.

Issue: [#9690](https://github.com/Luminous-Dynamics/symthaea/issues/9690).

## 1. Target

For a fixed work landmark `tau`, estimate a one-sided simultaneous upper resource envelope for the remaining cost to sufficient independently verified proof, among episodes that are still incomplete and at risk at `tau`. Each episode's forecast is bound to the proof-frontier state and population cohort observed at the landmark.

The event time `T` is total work to sufficient proof. The observed endpoint is `X = min(T, C)`, with proof-event indicator `Delta`, where `C` is a right-censoring time. Episodes with `X <= tau` are not at risk at the landmark.

## 2. History-dependent censoring receipt

For each observed proof event after the landmark, the caller supplies a `CensoringWeightReceipt` with:

- censoring-model digest;
- nuisance-training partition digest, distinct from the calibration partition digest;
- calibration-partition digest;
- observed full-history digest at the proof endpoint;
- declared population-cohort digest;
- `Ghat(T | history_T)`, the estimated probability of remaining uncensored through the proof event conditional on the time-varying execution / frontier history.

The estimator uses weight `1 / Ghat`. It does not fit the censoring model. Cross-fitting / split separation must be performed upstream and evidenced by the receipt identities; this crate checks structural identity but cannot independently prove the upstream fitting procedure.

## 3. Weighted joint calibration equation

The score is the same one-sided joint max-normalized resource residual as the completed-case vector conformal baseline:

`A_i = max_j max(0, (actual_remaining[i,j] - predicted_point[i,j]) / predicted_scale[i,j])`

At landmark `tau`, choose the smallest observed score threshold `theta` satisfying:

`sum_i I(X_i > tau) * Delta_i / Ghat_i * ( I(A_i <= theta) - (1 - alpha) ) >= 0`

Only independently verified proof completions after `tau` contribute a score and IPCW weight. Right-censored episodes after `tau` remain in the at-risk diagnostics but do not contribute an observed score; their effect is addressed by the censoring weights estimated from the training partition. Pre-landmark exits are excluded from the survivor-conditional calibration cohort.

The implementation sorts scores and finds the first cumulative weighted mass at or above `(1-alpha) * total_weight`; this is the smallest threshold satisfying the weighted balance. It reports raw total weight, effective sample size, maximum weight, minimum observed censoring survival probability, at-risk count, observed proof-event count, censored-after-landmark count, and pre-landmark exits.

## 4. Assumptions and claim boundary

- **Conditional independent censoring given recorded history:** after conditioning on the time-varying proof/execution history, censoring must not depend on the unobserved future proof-completion time.
- **Positivity:** `G(T | history_T)` must remain bounded away from zero on the target histories. The caller declares a floor `eta`; a smaller estimate causes an explicit `PositivityViolation`. We do not silently clip weights.
- **Nuisance separation:** the censoring model's training partition and calibration partition must have distinct digests. All event receipts must match the expected model, training partition, calibration partition, history, and population cohort.
- **Independent proof outcome:** observed proof events require a nonzero verifier digest.
- **Competing events:** permission revocation, explicit abort, and policy abandonment are not treated as right-censoring in this version. Any competing terminal after or before the landmark is rejected. A multi-state / competing-risk extension is a separate research problem.
- **Coverage:** this weighted estimating-equation adaptation targets a PAAC/asymptotic survivor-conditional calibration regime under its assumptions. It does not claim exact finite-sample split-conformal validity, and it does not inherit HAPS' theorem automatically. The one-sided six-resource score and POD-Meta event definition require separate derivation.
- **Generalization:** no guarantee is made under arbitrary history shift, lack of overlap, or task families outside the declared cohort.

## 5. Relationship to HAPS

HAPS (Wang, Levis, Yang, and Han, May 2026) develops survivor-conditional prediction sets at a decision time using time-varying covariate histories. It maps a censoring-free calibration condition to observed data through IPCW, and introduces HAPS-DR and AIPCW extensions. Its stated result is probably asymptotically approximately correct survivor-conditional coverage under conditional independent censoring, positivity, and consistent nuisance estimation.

This module adapts the weighted estimating-equation shape to a one-sided joint resource-cost score. It is not a reimplementation of HAPS, HAPS-DR, or HAPS-A.

Reference: [History-Aware Conformal Prediction Sets for Censored Time-to-Event Outcomes](https://arxiv.org/abs/2605.06581).

## 6. Qualification controls

The unit tests cover weighted threshold selection, at-risk filtering, effective sample size, censoring positivity failure, competing-risk rejection, landmark consistency, history / partition binding, shifted predictions, digest requirements, duplicate episodes, and invalid miscoverage. The deterministic CLI matrix emits one weighted survivor calibration plus positivity and competing-risk negative controls.
The public Rust API groups these knobs in `LandmarkIpcwConfig` rather than accepting positional scalar/digest arguments. This makes the landmark, miscoverage, positivity floor, predictor, censoring model, calibration partition, nuisance-training partition, and target cohort explicit at the call site.
