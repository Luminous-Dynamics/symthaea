# RH-006 Inference Estimator Bridge Review — v1

## Purpose

This note records the current scientific comparison of candidate forecast-accuracy inference frameworks against the actual RH-006 estimator and rolling-origin evidence design.

It is a research qualification artifact, not an inference procedure. It does **not** authorize a p-value, confidence interval, or formal hypothesis test.

The current estimator remains:

- **Estimator:** `fixed-ridge-standardized-v1`
- **Training preprocessing:** feature means and scales estimated from each training window only
- **Fit:** deterministic ridge-regularized linear regression in standardized predictor space
- **Rolling estimation:** a fresh fit at each origin
- **Primary comparison:** `RelationalAugmented` vs `NonRelationalContext`
- **Loss:** squared error
- **Held-out design:** fixed-width training windows, a gap, fixed-width disjoint held-out target windows, and a fixed forecast horizon
- **Current formal-inference gate:** `stop-assumption-failure`

The purpose of this review is to determine which published framework is actually closest to the implemented estimator and evidence geometry, and to make every remaining incompatibility explicit rather than treating a citation as a validity proof.

## 1. Important distinction: forecasting method versus nested regression model

The RH-006 comparison is a nested-model comparison, but the scientific object being evaluated is the **complete forecasting method**:

> feature definition + training-window geometry + training-only preprocessing + ridge regularization + deterministic fitting + forecast horizon + rolling-origin schedule.

This matters because different forecast-evaluation frameworks make different asymptotic commitments about how parameter-estimation error enters the forecast.

Giacomini & White (2006) explicitly frame forecast evaluation around predictive methods rather than only population models. Their framework preserves the finite-sample behavior of the forecasting estimators, permits nested and non-nested models, and allows general estimation methods. They also require the estimation window to be bounded by a finite constant; this rules out the standard expanding-window setup and is compatible in principle with fixed-width rolling estimation.

Source: Giacomini, R. & White, H. (2006), *Tests of Conditional Predictive Ability*, Econometrica 74(6), 1545–1578.
https://doi.org/10.1111/j.1468-0262.2006.00718.x

A primary-source reproduction of the paper states that the estimation window size is bounded and describes rolling-window forecasts with the estimator, window size, and observation weights treated as parts of the forecasting method:
https://www.eco.uc3m.es/~jgonzalo/teaching/PhdTimeSeries/GiacominiWhite.pdf

This makes Giacomini–White conceptually closer to RH-006 than an inference framework that assumes a diverging estimation sample and ignores the exact forecasting estimator.

However, conceptual compatibility is not mathematical validation.

## 2. Why Giacomini–White is not approved yet

There are three separate mismatches.

### 2.1 The null is stronger than RH-006 needs

The original Giacomini–White framework is a conditional predictive-ability framework. Its conditional equality concept is stronger than merely asking whether the two forecasting methods have equal **average** out-of-sample loss.

Zhu & Timmermann (2020) show that, under rolling estimation, the conditional-expectation null used by Giacomini–White can fail under broad conditions and can produce substantial size distortions when interpreted as a test of unconditional equal average accuracy.

Source: Zhu, Y. & Timmermann, A. (2020), *Can Two Forecasts Have the Same Conditional Expected Accuracy?*
https://arxiv.org/abs/2006.03238

Therefore, RH-006 should not adopt a Giacomini–White statistic merely because the estimator is fixed-window and rolling.

### 2.2 RH-006 does not currently produce one consecutive forecast per time point

The Giacomini–White construction is naturally expressed as a sequence of out-of-sample forecasts indexed through time. RH-006 instead uses **disjoint held-out target windows** for its rolling-origin qualification.

That design is intentional: every held-out target is scored once, while training windows may overlap across origins.

The current evidence therefore has two dependence layers:

1. serial dependence inside each held-out target window;
2. dependence among origin-level mean loss differentials induced in part by overlapping training sets.

Concatenating all target windows into one long loss sequence would create artificial adjacency across window boundaries.

Conversely, applying a standard single-sequence HAC formula directly to the origin-level means discards the within-origin target-level structure.

A valid inferential adaptation therefore has to state exactly what the statistical unit is:

- individual target forecast;
- origin-level block;
- or a two-level object with a prespecified cluster/block covariance construction.

That choice is currently unresolved.

### 2.3 The estimator includes training-window standardization and ridge regularization

The literature citation alone does not show that the limiting variance or bootstrap calibration of a nested comparison remains valid when each rolling fit:

1. estimates feature means;
2. estimates feature scales;
3. standardizes the training design;
4. applies a fixed positive ridge penalty;
5. solves the resulting penalized system deterministically;
6. repeats the entire operation at each origin.

The applicability artifact therefore remains `not-approved-for-execution`.

## 3. Candidate A — Giacomini–White

### Alignment

- fixed-width rolling estimation: **aligned in concept**;
- general estimation methods: **aligned in concept**;
- nested and non-nested forecasting methods: **aligned in concept**;
- finite estimator-window effects: **aligned in concept**;
- fixed forecast horizon: **compatible in concept**.

### Unresolved conditions

- conditional versus unconditional null: **not resolved**;
- disjoint block scoring versus consecutive forecast sequence: **not resolved**;
- two-level dependence from overlapping training windows: **not resolved**;
- ridge + training-standardization bridge: **not established**;
- appropriate HAC/covariance construction for the actual retained loss object: **not established**.

### Decision

**Candidate only. Do not execute.**

## 4. Candidate B — Clark & McCracken nested-model bootstrap

Clark & McCracken (2015) directly addresses the nonstandard finite-sample behavior of nested forecast comparisons and develops bootstrap-based inference. Their framework allows recursive or rolling forecast estimation.

Source:
Clark, T. E. & McCracken, M. W. (2015), *Nested forecast model comparisons: A new approach to testing equal accuracy*, Journal of Econometrics 186(1), 160–177.
https://doi.org/10.1016/j.jeconom.2014.06.016

### Strength

This is the strongest conceptual match to the **nested** RH-006 comparison.

### Problem

The published theoretical/Monte-Carlo framework is tied to particular regression/nesting structures and assumptions about the predictive regressors and forecast-error process. RH-006's estimator is a regularized, standardized regression procedure and the additional relational channels are tested inside that procedure.

A citation to a nested OLS bootstrap cannot be promoted to a proof that the exact ridge estimator has the same null distribution.

### Decision

**Candidate only. Not approved for execution.**

## 5. Candidate C — Doko Tchatoka & Haque hybrid bootstrap

Doko Tchatoka & Haque (2023) address finite-sample problems in nested forecast testing and propose a hybrid bootstrap combining a moving-block component with a residual-based component. Their construction is a bootstrap data-generating process with model refitting; it is not simply “resample the retained loss-differential vector.”

Source:
Doko Tchatoka, F. & Haque, Q. (2023), *On bootstrapping tests of equal forecast accuracy for nested models*, Journal of Forecasting 42(7), 1844–1864.
https://doi.org/10.1002/for.2987

### Strength

This is currently the clearest literature source for why RH-006 must not apply an IID bootstrap to the retained loss differential.

### Problem

The actual estimator compatibility is still unresolved. The bootstrap DGP and refitting algorithm would have to be rewritten explicitly for:

- fixed ridge regularization;
- training-window feature standardization;
- the two nested feature sets;
- fixed-width rolling windows;
- the RH-006 gap;
- disjoint target blocks;
- overlapping training windows.

The block-length rule and residual construction must also be independent of the realized held-out effect.

### Decision

**Candidate only. Not approved for execution.**

## 6. Candidate D — average forecast accuracy under possible instability

Harvey, Leybourne & Zu (2025) study equal **average** forecast accuracy when the mean of the loss-differential process may vary over time. They show that the standard Diebold–Mariano long-run variance estimator can behave badly in such settings and propose local demeaning before long-run variance estimation.

Source:
Harvey, D. I., Leybourne, S. J. & Zu, Y. (2025), *Testing for Equal Average Forecast Accuracy in Possibly Unstable Environments*, Journal of Business & Economic Statistics 43(3), 643–656.
https://doi.org/10.1080/07350015.2024.2418835

### Why this is relevant

RH-006 intentionally retains the full per-origin improvement vector and does not assume that a relational effect is stationary across the entire experiment.

An eventual inference framework should distinguish:

- equal accuracy everywhere;
- equal accuracy on average;
- time-varying advantage with zero average;
- a persistent positive average advantage.

This paper provides useful guidance on that scientific distinction.

### Problem

Its statistic is built around a contiguous loss-differential sequence and a modified long-run variance estimator. RH-006 currently has disjoint target blocks plus an origin-level dependence layer, so the published variance result does not transfer automatically.

### Decision

**Useful conceptual candidate for the estimand; implementation compatibility unresolved.**

## 7. Candidate E — sub-sampling for rolling-window average accuracy

Zhu & Timmermann propose a sub-sampling alternative after showing problems with the original Giacomini–White interpretation for equal unconditional average accuracy under rolling estimation.

This direction is especially important because it treats the forecasting method's finite-window estimation error as part of the comparison instead of assuming it disappears asymptotically.

However, a sub-sampling procedure still needs a declared statistical unit and a dependence-preserving implementation for the RH-006 disjoint-block design.

### Decision

**Candidate for later investigation; no execution path yet.**

## 8. The strongest present conclusion

There is currently no literature procedure that can be adopted by citation alone without changing or extending the RH-006 design.

The closest conceptual decomposition is:

| Question | Best current literature direction | RH-006 status |
|---|---|---|
| Nested forecast comparison | Clark–McCracken | candidate |
| Finite-width forecasting method | Giacomini–White | candidate |
| Nested bootstrap with dependence | Doko Tchatoka–Haque | candidate |
| Equal accuracy on average under instability | Harvey–Leybourne–Zu | candidate |
| Rolling-window average-accuracy robustness | Zhu–Timmermann sub-sampling | candidate |
| Exact fixed-ridge + training-standardization bridge | none established by the cited papers | **hard stop** |
| Disjoint test blocks + overlapping training windows | no direct published match identified here | **hard stop** |

This is a narrowing result, not a failure of the RH-006 architecture.

## 9. Recommended architecture for the next bridge

Do **not** add a p-value implementation yet.

Instead, choose one of two explicit research paths.

### Path 1 — change the forecast evidence geometry

Add an inference-only evaluator that produces a forecast for every eligible target time using the exact fixed-width ridge method.

Then the statistical object becomes a conventional ordered loss-differential series:

`d_t = L(y_t, f_rel,t) - L(y_t, f_ctx,t)`

with the estimator, standardization, window width, horizon, and loss fully frozen.

This would make the mapping to rolling-window forecast-evaluation literature substantially cleaner.

The current disjoint-origin qualification can remain unchanged as the primary descriptive qualification artifact.

### Path 2 — preserve RH-006 block geometry

Keep the disjoint target windows and treat each origin as an explicit block.

Then derive a hierarchical/block inferential procedure that specifies:

- the estimand;
- the null;
- the resampling unit;
- within-origin dependence treatment;
- across-origin dependence treatment;
- how overlapping training windows affect dependence;
- how ridge and standardization are refit in every resample;
- how the gap and fixed horizon enter the bootstrap DGP;
- block-length selection;
- small-sample behavior;
- exact rejection/interval rule.

This is scientifically more faithful to the existing evidence artifact, but it is also a new methodological derivation rather than an off-the-shelf test.

## 10. Required simulation before approval

Before changing the applicability artifact from `not-approved-for-execution`, the repository should have deterministic simulation checks for at least:

1. **Null size:** relational and non-relational methods have equal predictive ability.
2. **Local alternative:** a small known incremental relational effect.
3. **Moderate alternative:** a clearly positive incremental effect.
4. **Serial dependence:** AR-type errors or another declared dependent DGP.
5. **Time-varying mean:** the true loss differential changes over time while average effect is zero.
6. **Ridge sensitivity:** several prespecified fixed ridge coefficients, including the production candidate.
7. **Standardization sensitivity:** verify that every bootstrap/refit recalculates training means/scales only from the resampled training window.
8. **Rolling overlap:** verify the resampler reproduces the dependence induced by overlapping training windows.
9. **Fixed horizon:** verify forecast target alignment under the declared horizon.
10. **Near-singularity:** verify deterministic failure rather than silent numerical degradation.
11. **Small samples:** explicitly measure empirical type-I error at the smallest scientifically admissible sample sizes.
12. **Reproducibility:** same source + same manifest + same seed/algorithm produces byte-identical inference artifacts.

The approval criterion must be prospective. A method does not become “approved” because it happens to give a desirable result on the confirmatory dataset.

## 11. Hard-stop invariant

Until one of the two paths above is completed and independently validated:

- `ForecastInferenceSelectionPath` must continue to resolve to `stop-assumption-failure`;
- the applicability artifact must remain `not-approved-for-execution`;
- no p-value or confidence interval may be emitted by the RH-006 inference path;
- dependence characterization remains descriptive;
- surrogate exceedance fractions remain empirical diagnostics;
- the nested MSE difference remains an estimand/diagnostic, not a formal significance result.

## 12. Research conclusion

The RH-006 estimator is not “obviously incompatible” with forecast-accuracy inference. The stronger conclusion is:

> **The exact fixed-ridge, training-standardized, rolling-origin forecasting method is not yet covered by a sufficiently specific, validated inferential bridge.**

The next useful work is therefore methodological derivation and null-size simulation, not additional receipt plumbing.

That is the boundary this note freezes.
