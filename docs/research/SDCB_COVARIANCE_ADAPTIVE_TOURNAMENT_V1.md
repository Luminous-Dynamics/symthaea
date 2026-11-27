# SDCB full-covariance and adaptive measurement-design tournament v1

**Status:** preregistration proposal; no external benchmark or physical experiment has been run.

## 1. Research question

At matched candidate measurements, information, training data, and resource budgets, can SDCB distinguish target-relevant scientific information from redundant common-cause evidence, and exploit outcome-adaptive second measurements without violating pathwise constraints?

The tournament must separate three sources of improvement:
- a better measurement/noise model;
- a better selection policy;
- stronger evidence qualification and provenance.

## 2. Task families

| Family | Construction | Required behavior |
|---|---|---|
| FC0 diagonal controls | Independent prior and residual covariances | Reproduce #13848 |
| FC1 correlated priors | Nonzero off-diagonal latent prior covariance | Propagate prior dependence |
| FC2 correlated residuals | Full residual covariance across candidate actions | Avoid over-counting duplicate channels |
| FC3 multiple shared factors | Environment, reference, batch, device and acquisition effects | Select measurements that target the relevant confounders |
| FC4 near-singular covariance | Ill-conditioned but valid covariance matrices over declared scale | Report conditioning/numerical limits; no silent jitter |
| FC5 invalid covariance | Asymmetric, indefinite, non-finite or dimension-mismatched matrices | Fail closed |
| FC6 partial reference ancestry | Some actions share only part of the reference chain | Preserve partial dependence |
| FC7 sequential non-Gaussian | Categorical paired worlds with outcome-dependent next action | Improve on fixed second-stage policies |
| FC8 common-cause transition | Shared state drifts between first and second measurements | Discount stale first-stage calibration |
| FC9 intervention contamination | Candidate intervention changes another causal parent/measurand | Do not credit invalid exclusion assumptions |
| FC10 abstention boundary | Weakly separating or observationally equivalent worlds | Abstain where experiment value is insufficient |
| FC11 mobile short-collocation | Fast sensor compared to slower reference over short, shifting process windows | Include process covariance and unequal averaging-time uncertainty |
| FC12 model discrepancy | Linear Gaussian model misses a nonlinear or hidden residual cause | Detect misspecification or abstain, not overconfidently extrapolate |

FC0–FC7 are suitable for exact finite/matrix references. FC8–FC12 require sequential state models, richer likelihoods, measured process metadata, or controlled counterfactual generators.

## 3. Baselines and matched comparisons

All policies receive identical action sets, cost units, training data, latent registry, covariance assumptions, and terminal utility.

1. Random feasible selection.
2. Fixed-cadence/check-standard allocation.
3. Independent-noise Bayesian fusion.
4. Full-covariance Gaussian conditioning with all observations.
5. D-optimal design (joint log determinant).
6. A-optimal design (trace covariance).
7. c-optimal design (declared scientific coordinate variance).
8. Fisher-information and profile-likelihood design for nonlinear models where appropriate.
9. OU/Kalman filter and Gaussian-process correlated-noise baselines.
10. Hierarchical/random-effects calibration.
11. EVSI/decision-focused acquisition.
12. Robust maximin/distributionally robust selection.
13. Exact full-covariance subset oracle for small instances.
14. Fixed second-stage policy.
15. Exact contingent two-stage policy.
16. Approximate HDC/CfC SDCB policy.

Do not compare their native objective values as though all share a unit. Score every method on the same held-out target variance, realized decision utility/regret, calibration coverage, and cost.

## 4. Mandatory factorial design

A. Keep the covariance/calibration model fixed and change only the action-selection policy.

B. Keep the selection policy fixed and change only the covariance/noise model.

C. Cross model and policy factors to measure their interaction.

D. Compare adaptive and non-adaptive policies with identical per-path budgets and outcome access.

E. Evaluate all models on both correctly specified and deliberately misspecified test families.

## 5. Metrics

- Target posterior variance and fractional variance reduction.
- Joint information gain, kept separate from target-specific utility.
- Interval coverage and sharpness for scientific and nuisance marginals.
- Residual covariance estimation error and calibration coverage.
- False independence / false common-cause rates.
- Near-singular conditioning and numerical failure rate.
- Expected cost and maximum path cost.
- Regret against exact subset and contingent policy references.
- Decision accuracy, abstention rate, selective risk and coverage.
- Drift detection delay and stale-reference cost.
- Short-collocation uncertainty coverage for mismatched averaging windows.
- Receipt binding failures, invalid-evidence acceptance and zero-mutation behavior.
- Runtime, peak memory and energy where measurable.

Report Pareto frontiers. Do not collapse accuracy, cost, assurance, and physical validity into one score.

## 6. Proposed gates

Research targets, not achievements or standards:
- 100% agreement with exact known-answer Gaussian and categorical fixtures.
- 100% pathwise budget compliance.
- Zero acceptance of invalid covariance or malformed likelihoods.
- Zero false independent-confirmation claims in common-cause controls.
- Non-inferiority on independent-error controls.
- At most 2% regret against the exact subset/two-stage oracle on in-cap instances.
- At least 10% lower target variance per equal cost than matched IID/EIG policies on held-out FC2/FC3/FC6 tasks, with paired confidence intervals excluding zero.
- At least 10% decision-value gain over the best fixed second-action comparator on a preregistered adaptive task family, not merely on the frozen toy fixture.
- Nominal 90% and 95% uncertainty coverage by task family.
- Independent reproduction of at least one licensed real-data result before making a real-world improvement claim.

## 7. Real measurement benchmark additions

### Mobile short collocation

Add a dedicated benchmark based on the 2026 work by Kok and Tabandeh. The task should have a fast-reporting mobile sensor, a slower reference with a longer averaging period, a collocation window shorter than the reference averaging period, and a measurand varying during collocation. Compare OLS, distance/generalized-distance regression, the paper's C-GDR method, and Gaussian-process/process-covariance uncertainty handling.

This is a materially different evaluation from ordinary paired-row RMSE. The benchmark must preserve raw timestamps and averaging windows, model the process covariance, and score uncertainty coverage as well as calibration error.

Paper: https://doi.org/10.1002/qre.70319

### Inha indoor calibration

Reproduce S1 reference-efficiency, S2 location transfer, S3 drift, and S4 event-conditioned results before adding SDCB selection. Keep the five locations within one building distinct from independent laboratory replication.

Dataset: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset  
Paper: https://arxiv.org/abs/2610.11236

### UCI gas drift

Use chronological batches for drift and channel robustness only. Continue to treat the licence conflict between narrative research-only wording and structured CC BY 4.0 field as unresolved pending written clarification.

## 8. Literature positioning

The 2026 Qi–Baker work directly studies experimental schedules with IID versus OU-correlated observation noise and finds that noise structure affects optimal measurement times: https://doi.org/10.1016/j.mbs.2025.109571

The 2026 Kok–Tabandeh work treats unequal sensor/reference averaging periods and varying measurands during short collocation. It develops C-GDR and reports circumstances where it outperforms OLS: https://doi.org/10.1002/qre.70319

NIST guidance recommends measurement-assurance programmes and control-chart evidence rather than a universal recalibration interval: https://www.nist.gov/calibrations/recommended-calibration-interval

Consequently, SDCB should not claim to invent correlated measurement modelling, optimal design, or mobile sensor metrology. Its differentiating claim to test is integration of target-scientific utility, exact adaptive reference oracles, paired-world identifiability, shared/local covariance, and evidence-qualified state changes.

## 9. Reproducibility record

Pin the exact dataset/source release, license evidence, raw file digest, timestamp/averaging-window interpretation, candidate action set, covariance matrices, model/policy digest, split manifest, cost budget, all outcomes and actions, exact oracle output, seeds, toolchain, hardware, resource telemetry, failures, abstentions, and receipt references.
