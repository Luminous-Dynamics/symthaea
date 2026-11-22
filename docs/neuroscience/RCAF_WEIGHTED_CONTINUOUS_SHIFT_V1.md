# RCAF Weighted Covariate-Shift Calibration v1

**Status:** executable synthetic weighted-threshold and shift-diagnostic primitive; not empirical fMRI validation.  
**Base:** estimator-response, held-out null, and groupwise null-calibration layers.  
**Purpose:** extend the known-discrete-strata strategy to a continuous covariate-shift setting without hiding the cost of importance weighting or overstating the guarantee.

## 1. Continuous shift is not just another discrete group

A low-/high-motion or single-/multi-echo stratum is a useful first defence, but it cannot guarantee calibration when covariate distributions move continuously within a stratum. Examples include:
- scanner drift over time;
- a gradual shift in motion burden;
- changing physiological-noise amplitude or spectral shape;
- continuous changes in image quality, temporal signal-to-noise or acquisition parameters.

Under a covariate-shift model, source and target covariates differ but the conditional score law is assumed invariant:

`P_source(S \mid X)=P_target(S \mid X)`.

If the target covariate distribution is absolutely continuous with respect to the source distribution, the ideal importance weight is:

`w(x)=dQ_X/dP_X`.

This is a substantive assumption. It does not hold automatically when scanner changes also alter the conditional noise/score law, when an unseen site lies outside source support, or when biological signal/measurement relationships change. A density ratio is not a magic correction for concept shift.

## 2. Weighted upper-tail calibration with the test-point atom

For calibration-null scores `s_i` with source-to-target weights `w_i`, and a particular test point with weight `w_*`, the weighted empirical distribution places the test point's own mass at (+infty).

The target cumulative calibration mass at miscoverage level alpha is:

`(1 - alpha) * (sum_i w_i + w_*)`.

A finite threshold exists only when this target mass is no larger than the calibration weight sum. Otherwise the point mass at infinity is too large and there is no finite upper threshold. RCAF reports that test point as `test_point_mass_exceeds_alpha` and abstains instead of fabricating a finite cutoff.

The threshold is test-point-specific: two evaluation observations with different weights can have different critical values even when they use the same calibration set.

This construction is based on the weighted split-conformal treatment of covariate shift with known likelihood ratios. It is not the same as applying an ordinary unweighted empirical quantile to importance-weighted scores.

## 3. What this module implements—and does not implement

The Rust module `research/rcaf/src/weighted_covariate_shift.rs` accepts:
- calibration null scores and source-to-target density-ratio weights;
- held-out null/signal scores and the test point's target-to-source ratio;
- explicit provenance for known versus estimated ratios;
- separate calibration, evaluation and (when estimated) ratio-model estimation split identifiers;
- caller attestations for source-support overlap and conditional-score invariance.

It does **not** fit a density-ratio model, discover covariates, estimate the support set or certify the scientific truth of either attestation. These must be separately evidenced. The ratio model is a distinct component with its own data, code, calibration, cross-fitting and error report.

The module enforces:
- unique, non-overlapping calibration/evaluation replicate IDs;
- a disjoint estimation split for an estimated ratio model;
- finite non-negative calibration scores and calibration weights;
- strictly positive evaluation test-point weights;
- finite direction cosines;
- optional effective-sample-size floor;
- abstention when support or conditional-score invariance is not attested;
- abstention when a finite test-point-specific threshold does not exist.

Estimated density ratios are labelled `estimated_ratio_diagnostic_only`; they do not receive the exact finite-sample guarantee. Cross-fitting and a separate estimation split reduce obvious leakage, but do not make estimated ratios known.

## 4. Weight-concentration and shift diagnostics

For calibration weights (w_1,\ldots,w_n), the module reports:

`ESS = (sum w_i)^2 / sum(w_i^2)`

and the empirical second-moment / chi-square proxy:

`n * sum(w_i^2) / (sum(w_i)^2) - 1`.

It also reports:
- calibration weight sum and square sum;
- ESS fraction;
- maximum normalized calibration weight;
- maximum-to-mean weight ratio;
- minimum and maximum test-point mass share `w_* / (sum_i w_i + w_*)`.

These are concentration diagnostics, not proof of support overlap or correct ratio estimation. A large maximum weight can be a warning, but the recent asymptotic analysis shows that the second moment / chi-square divergence, rather than simply the supremum of the weights, is central to the process limit and calibration cost. Infinite chi-square divergence can move the asymptotics out of the usual square-root-n regime.

An operational ESS floor is a policy gate for this implementation. It is not a theorem that ESS above the floor makes estimated weights valid. Points that fail the floor are recorded as abstentions. Rates are computed over finite-threshold evaluation points only; users must inspect the abstention fraction and must not interpret those conditional rates as population-wide guarantees without accounting for selection.

## 5. The guarantee boundary

The report sets `finite_sample_guarantee_applicable = true` only when:
1. ratio provenance is explicitly `known_density_ratio`;
2. source-support overlap is attested;
3. conditional score-law invariance is attested.

Even then, the mathematical interpretation is conditional on those assumptions and the exchangeability structure of the calibration observations. The code cannot prove that a supplied ratio is the true likelihood ratio or that the score conditional law is invariant.

For `estimated_density_ratio`, the report always sets the exact guarantee flag to false. It records the estimator identifier, the separate estimation split and whether cross-fitting was used. Its weighted thresholds and intervals are diagnostics only. Ratio-estimation error, model misspecification and residual conditional shift still need independent analysis.

The module intentionally does not clip extreme weights. Clipping changes the effective target distribution and can destroy the nominal guarantee unless a correction or conservative inflation is separately justified. If clipping is studied later, report unclipped and clipped ESS, maximum normalized weight, target-mass loss, and a coverage sensitivity envelope.

## 6. Research frontier: sketched calibration and shift leakage

A promising next paradigm is to calibrate on a compressed covariate representation (Z=T(X)), rather than forcing the ratio estimator to model every source/target difference in the full covariate space.

Recent work on sketched calibration shows that compressing covariates can reduce the chi-square cost, but its coverage loss depends on a leakage quantity measuring whether discarded covariate shift still changes the conditional response/score law given the sketch. A low-dimensional sketch is not automatically sufficient. RCAF should therefore treat the following as a separate research workstream:
1. learn or pre-register candidate sketches on a ratio-estimation/training split;
2. quantify source/target discrepancy retained by each sketch;
3. estimate a held-out conditional-score leakage bound `Delta_T`;
4. compare full-covariate, groupwise and sketched weighted calibration at matched alpha;
5. abstain when leakage cannot be bounded or when ESS collapses.

This v1 does not implement sketch learning or a leakage bound. It only exposes the diagnostics needed to identify when ordinary weighting becomes expensive and when a test point carries too much mass to permit a finite threshold.

## 7. Executable fixture and interpretation

The smoke fixture uses 99 synthetic calibration observations with smoothly varying, analytically supplied weights and 100 target evaluation points. One test point is assigned a larger importance weight such that its mass at (+infty) exceeds the 5% tail budget. All three estimators must report 99 finite thresholds and one abstention each for that point.

The fixture uses a known density ratio within this constructed synthetic example, with support and conditional invariance explicitly attested. It checks:
- finite-sample flag and validity status;
- ESS and chi-square diagnostics;
- all three estimator order/threshold summaries;
- per-test-point mass abstention;
- Wilson interval counts and denominators.

It does not demonstrate actual scanner-shift robustness, empirical covariate-ratio accuracy or biological validity.

## 8. Current research sources

- Mehrdad Pournaderi (2026), *Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency*, arXiv:2611.20052, submitted 15 November 2026. The paper studies the weighted empirical process, the role of chi-square divergence in the second moment, estimated-ratio error and efficiency. https://arxiv.org/abs/2611.20052
- Mehrdad Pournaderi (2026), *Sketched Calibration for Conformal Prediction under Covariate Shift*, arXiv:2610.09208, 6 October 2026. It studies compression of covariates for lower weighted-calibration cost and a conditional-score leakage term for discarded shift. https://arxiv.org/abs/2610.09208
- Aabesh Bhattacharyya and Rina Foygel Barber (2026), *Group-weighted conformal prediction*, Electronic Journal of Statistics 20(1):1171–1199, DOI 10.1214/26-EJS2506. https://doi.org/10.1214/26-EJS2506

## Claim limits

This v1 is an importance-weighted threshold and diagnostic primitive. It consumes externally supplied weights and does not estimate them. Only caller-attested known ratios plus support and conditional-score assumptions are marked as exact-guarantee-applicable; estimated ratios are explicitly diagnostic only. The synthetic fixture does not establish real covariate-shift validity, clinical performance or superiority over current neuroimaging pipelines.


## 9. Direct estimator-response adapters

The weighted module exposes adapters from the existing `FactorialEstimatorResponseReport`:
- `null_observation_from_estimator_response` takes the three nuisance-only connectivity displacements in the pinned estimator order;
- `evaluation_observation_from_estimator_response` takes the same nuisance-only null scores, clean-to-signal-plus-nuisance connectivity displacements, and the connectivity-space directional cosine.

The adapters reject missing or reordered estimator summaries and reject a signal-evaluation record when no target direction is defined. This keeps the score definition shared with the earlier estimator-response layer rather than relying on callers to reconstruct arrays by hand.
