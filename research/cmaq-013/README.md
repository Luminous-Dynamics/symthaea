# CMAQ-013: Selection-Conditioned Uncertainty Calibration

**Status:** protocol v0.1 + dependency-free Rust synthetic benchmark; no production claim  
**Tracking issue:** [#9127](https://github.com/Luminous-Dynamics/symthaea/issues/9127)  
**Claim ceiling:** synthetic calibration stress test; ACI-style baseline is reduced, not a formal reproduction

## Why this benchmark exists

Calibration measured on random behavior actions is not necessarily calibration on the actions a planner selects. The planner searches candidate actions and selects unusually optimistic predictions. Increasing the candidate-set size changes the selected residual distribution even if the underlying model and random-action calibration data remain unchanged. Policy-induced distribution shift can further invalidate the original calibration.

CMAQ-013 measures this mismatch directly. It uses the same synthetic action pools and predictive model for every calibration method, with search widths 8, 32 and 128, and stationary, abrupt and gradual error-scale regimes.

## Calibration methods

- **Point/no guard:** no uncertainty correction; diagnostic baseline.
- **Fixed Gaussian bound:** assumes a one-sided Gaussian 90% residual bound.
- **Behavior split conformal:** finite-sample residual quantile from random behavior actions, before planner selection.
- **Selection-conditioned split conformal:** finite-sample residual quantile from the top-scoring action selected by the same search width. This calibrates the selected-action distribution in its initial regime, not after future shifts.
- **ACI-style selective feedback:** reduced adaptive-conformal quantile controller initialized from behavior calibration, updated only when an executed selected action's delayed outcome arrives. This is not asserted to reproduce a specific published ACI implementation or its guarantee.
- **ACI-style plus counterfactual audit:** same online update, but pays for a synthetic audit of a rejected candidate every eight steps. The audit stands for an independently permitted simulator/digital-twin query; it is not a physical action safety claim.
- **Rolling quantile/selective feedback:** moving-window selected residual quantile, updated only on observed accepted actions.
- **Oracle selected quantile:** evaluator-only per-regime selected-action calibration upper bound; never deployable.

The benchmark intentionally separates selection-aware calibration from online adaptation. A fixed selected-action calibration can solve the initial search-selection problem yet become stale after dynamics change. Online adaptation can respond, but feedback is delayed and selectively censored by fallback. The audit arm tests whether extra counterfactual information offsets that censoring, after charging its cost.

## Environment

The Rust CLI uses 512 paired episodes per scenario/search width and 96 decisions per episode. Candidate pools contain 8, 32 or 128 actions. Each candidate has a true utility, a model-predicted utility, and a reported uncertainty. True prediction residual scale increases with action complexity; the reported uncertainty is deliberately under-dispersed. The planner selects the candidate with maximum predicted utility. A calibrated lower confidence bound is:

\`LCB = predicted_value - q * reported_sigma\`.

The model action is accepted only when the LCB exceeds the declared utility threshold; otherwise a fallback action is taken. Fallback utility, audit cost and feedback delay are explicit constants in the source.

Scenarios:
- stationary model error;
- abrupt error-scale increase at step 48;
- gradual error-scale increase from steps 24 to 72.

Accepted model actions provide residual feedback after four steps, with 5% feedback loss. Rejected candidates are unobserved unless the ACI-audit arm purchases the counterfactual query. This selection-dependent censoring is intentional.

## Required metrics

Every row reports:
- overall one-sided selected-action coverage and pre-/post-shift coverage;
- mean lower-bound width (sharpness);
- false-safe rate (accepted actions whose realized utility violates the safe threshold);
- fallback rate and realized utility;
- paired utility difference vs point/no-guard baseline with bootstrap 95% interval;
- mean regret vs best candidate in each pool;
- audit count/cost, observed feedback count/drop rate;
- calibration recovery delay after a shift;
- mean selected-action optimism.

The random-behavior calibration quantile and selection-conditioned calibration quantile are included in every row. Do not report random-action coverage as a substitute for selected-action coverage.

## External research connections

- The Physical-AI world-model survey (published 3 September 2026) identifies policy-induced distribution shift, calibration of uncertainty and planner exploitation as central closed-loop tensions. It recommends one- and multi-step metrics, proper scoring rules, closed-loop predictive consistency, NLL/Brier/ECE and control outcomes: https://link.springer.com/article/10.1007/s44163-026-02122-1
- The decision-centric world-model evaluation position (June 2026) recommends counterfactual action fidelity and evaluation under intervention, policy-induced shift and long-horizon rollout: https://arxiv.org/abs/2606.15032
- Adaptive Conformal Inference and the 2024 work on arbitrary distribution shifts provide online adaptation baselines.
- ICLR 2026 Conformal Optimistic Prediction (COP) studies online coverage/regret trade-offs; CAP (2025) studies selection-conditional coverage after adaptive pick. This benchmark does not claim to implement COP or CAP. Future revisions may add faithful reproductions when their update equations and assumptions are pinned.

## Interpretation limits

A green workflow validates code, report schema and synthetic invariants. It does not establish general distribution-free coverage, physical safety, or superiority on real environments. The synthetic uncertainty/error relationship is designed, not learned. The "oracle" is evaluator-only. The ACI-style quantile update is a simplified feedback controller and must not be cited as an ACI theorem.

## Next

Port the same report metrics into CMAQ-012's real paired-result adapter and the CMAQ-011 Gymnasium/MuJoCo and DMC tasks. For real environments, calibrate on development only, preserve an untouched task-parameter qualification split, record raw action-level residuals, and compare selected-action coverage against a direct policy under identical planning budgets.
