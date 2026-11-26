# GRAPE adapter parity v0.2

Issue: #13608  
Implementation: `symthaea-semiconductor::benchmark::grape_adapter_parity`  
Upstream: `cheonwoolee/GRAPE@cce2d27a092abd9e397d7739e4beb6dfe0ae145e` (Apache-2.0)  
Status: composed adapter-runtime implementation with 17 unit tests and 4 independent reference-vector integration tests authored; exact-head validation pending.

## What this module does—and does not—reproduce

This module now covers the pinned GRAPE **online adapter lifecycle** around a caller-supplied frozen global Ridge prediction:
1. train-only global residuals select adapter feature indices;
2. training targets and residuals set base/outer clipping and temporal activation thresholds;
3. an immutable prediction decomposition records global, feature, hierarchy and temporal components;
4. the selected NLMS or RLS adapter is updated from the clipped target/error rules;
5. the hierarchy receives the post-adapter residual at the declared equipment/chamber context.

The global Ridge fit, official loader/preprocessing, hyperparameter search, source pickle conversion, chronological validation/test/unseen runner and paper metric reproduction are not implemented here. Do not call this a full GRAPE reproduction or claim paper metrics from the synthetic Rust tests.

The official-code-faithful runner (including immediate predict-then-update, mutable state across validation→test→unseen, and the unseen feature-index asymmetry) remains a separate reproduction track from the K1-gated delayed operational adapter in #13478.

## Feature selection and train initialization

- Selects the top (n_a) features by absolute Pearson correlation between training global-model residuals and training features.
- Uses population centering/variance, skips features whose standard deviation is at most (10^{-8}), and falls back to the first (n_a) feature columns when residual variance is below (10^{-8}) or no nonconstant feature exists.
- Resolves exact correlation ties by ascending original feature index. The pinned NumPy `argsort` uses its default sorting implementation, whose tie order is not a portable cross-language contract; this explicit deterministic tie policy is documented rather than pretending to match a platform-dependent tie order.
- Derives training target 1st/99th percentile bounds and expands each by one unit for base clipping.
- Calibrates temporal activation from the declared percentile (default 96th) of absolute training residuals using linear percentile interpolation; the outer label range expands target quantiles by `max(2, 6 × threshold)`.
- Matches the pinned low-target-variance rule: if target standard deviation is below 0.3, NLMS learning rate is raised to at least 0.02 and hierarchical context minimum support to at least 25.

## Feature adapter equations

NLMS (pinned GRAPE equation):

[
w_{t+1}=w_t+eta,e_t\frac{x_t}{x_t^Tx_t+10^{-3}}.
]

RLS (pinned GRAPE equation):

[
p_t=P_tx_t,quad
k_t=\frac{p_t}{\lambda+x_t^Tp_t},
]
[
w_{t+1}=w_t+k_te_t,quad
P_{t+1}=\frac{P_t-k_t(x_t^TP_t)}{\lambda}.
]

The covariance is symmetrized after each update and validated for finite values, symmetry and positive definiteness. Initialization is (P_0=\text{ridge\_init}\,I), matching the pinned upstream implementation. Both update paths are transactional: failed numeric or covariance checks leave the original state unchanged.

RLS requires (O(n_a^2)) storage and arithmetic per update, plus (O(n_a^3)) Cholesky validation in this safety-oriented reference implementation. This extra validation cost must be measured before high-throughput use; it is not an upstream runtime cost.

## Hierarchical context correction

Implements global / equipment / chamber / equipment×chamber residual EMAs:
- equipment/chamber split occurs at the last underscore in the context key;
- updates advance global EMA and every applicable local bucket;
- each level's reliability is (n/(n+n_{min}));
- correction is the normalized weighted mean of available levels;
- a new equipment/chamber combination can use known equipment and chamber levels but cannot read an unrelated local-pair bucket;
- a missing context key does not update the context hierarchy, matching the pinned GrapeModel path.

The four hierarchy levels are separate from the flat context EMA in #13478; do not silently substitute one for the other.

## Temporal correction and clipping

- Training residuals calibrate the absolute-residual activation percentile.
- Online temporal residual EMA uses the pinned alpha (default 0.5).
- Temporal correction is zero below the activation threshold, otherwise it returns the current EMA.
- Prediction computes global + feature-adapter + hierarchy correction, clips to the base label bounds, adds temporal correction and clips to the outer label bounds.
- On update, target is clipped to outer label bounds then base bounds. Temporal error is measured against the clipped base prediction; feature-adapter error follows the same clipped-base target equation as the pinned implementation.
- The hierarchy residual is computed after the feature-adapter update from the base-clipped target minus global prediction and updated feature-adapter prediction; it does not subtract temporal or prior context correction, matching the upstream code.

## Authored test layers

**17 in-module tests** include:
- scalar hand-computed RLS and NLMS updates;
- covariance symmetry / positive-definiteness across bounded update sequences;
- RLS ridge covariance initialization;
- transactional RLS failure;
- four-level reliability weighting and unseen context fallback;
- context counter overflow atomicity;
- temporal threshold interpolation and activation;
- residual-correlation feature ranking;
- train-derived clipping/temporal bounds;
- low target-variance initialization behavior;
- prediction decomposition;
- NLMS update error decomposition and context residual;
- no-context behavior;
- transactional runtime overflow;
- deterministic RLS lifecycle replay.

**4 external integration tests** in `tests/grape_adapter_parity_reference.rs` independently check:
1. 2-D hand-computed RLS gain, weights and covariance;
2. 2-D hand-computed NLMS weights with the upstream (10^{-3}) norm floor;
3. a train→predict→update sequence with hand-computed feature selection, NLMS weight, temporal EMA and context residual;
4. four-level context reliability combination.

These are authored tests, not passed results until the exact-head workflow completes.

## Benchmark protocol and comparison

First reproduce the pinned adapter recurrence against the official Python implementation on static known-answer vectors. Then reproduce the official runner on its pinned derived datasets and chronological splits. Keep two independently reported modes:

- **Official-code-faithful:** same global Ridge/preprocessing, feature indices, NLMS or RLS mode, hierarchy, temporal correction, clipping, same mutable validation→test→unseen lifecycle and immediate predict/update ordering.
- **Corrected evidence-gated delayed mode:** freezes prediction-time features and decompositions, waits until physical label and K1 report availability, requires numeric/revision/event/payload identity, and updates only with qualified labels. This is not covered by the official zero-delay baseline's guarantee unless a separate theorem supports the change.

Benchmark D1 and D2 separately, in their declared target scales. Report RMSE, MAE, (R^2), worst chronological window, worst tool/chamber context, adaptation recovery delay, update latency p50/p95, RLS covariance memory, and qualified-label counts. Paired multi-seed confidence intervals and equal **qualified-label** budgets are required for the delayed operational comparison.

## Claim ceiling

The current module reproduces selected upstream adapter and clipping equations around an external global prediction. It does not authenticate labels, reproduce the full GRAPE paper result, establish physical traceability, prove fab transfer or production savings, or authorize wafer disposition, recipe changes, APC/R2R, rollback or equipment control.
