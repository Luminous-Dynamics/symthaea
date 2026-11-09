# Model-dependent detection in open-world experiment design

Date: 2026-11-09  
Status: executable reference model / synthetic qualification only  
Owner: [UAP-MNAR-001 #11247](https://github.com/Luminous-Dynamics/symthaea/issues/11247)  
Parent: [UAP-ACTIVE-001 #11187](https://github.com/Luminous-Dynamics/symthaea/issues/11187) and [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Research question

When candidate physical models predict different probabilities that a sensor will produce a usable detection, can Symthaea correctly account for the information in both detection and non-detection, without confusing that with information in the measured value after detection?

The paradigm change is **the observation process is part of the experiment model**. “No measurement” is not automatically missing-at-random, and it is not automatically a physical zero. A sensor that executes and returns no detection may have produced an informative binary outcome. A sensor that failed to execute or whose data path failed is a different event.

This extension reuses the existing `symthaea-science-evidence::model_discrimination` scalar Gaussian EIG engine. It does not create a competing EIG implementation and does not silently change the existing uninformative-availability API.

## Existing and extended channels

The original `CandidateExperiment::availability_probability` is retained as the model-independent probability that the instrument is successfully executed and returns a usable data opportunity. It is not the probability of detecting the physical signal.

The new `DetectionAwareCandidateExperiment` adds one model-specific Bernoulli detection probability:

    q_i = P(D = 1 | M_i, execution succeeds)

For prior model probabilities `pi_i`, let

    q_bar = sum_i pi_i q_i

The information in the detection/no-detection indicator is

    I(M;D) = H_B(q_bar) - sum_i pi_i H_B(q_i)

where `H_B` is binary entropy.

If a detection occurs, the posterior model prior is

    pi_i^(D=1) = pi_i q_i / q_bar

The existing Gaussian value channel is then evaluated under this detection-conditioned prior:

    I(M;D,Y) = I(M;D) + q_bar I(M;Y | D=1)

Finally, the existing model-independent execution probability scales the expected information:

    EIG_operational = P(execution succeeds) I(M;D,Y)

This is the chain rule for mutual information for the declared two-stage observation channel. It is not a heuristic score. The reference implementation computes the detection term analytically and reuses the existing deterministic Gaussian quadrature for the measured-value term.

### Boundary cases

- If all models share the same detection probability q, `I(M;D)=0` and the result reduces to the existing `q * conditional_EIG` availability result.
- If all models have q=0, detection never occurs and there is no value-channel term.
- If all models have q=1, detection is certain and supplies no model information; the value-channel prior remains unchanged.
- Zero-prior models do not affect `q_bar`, the detection information or the detection-conditioned prior.
- If the experiment is not executed, its expected information is zero under this v0.1 operational model.
- An open-world `OTHER_UNRESOLVED` hypothesis remains outside the named numeric likelihood set; the report preserves the open-world exclusion flag.

## Implemented API

In `crates/domains/symthaea-science-evidence/src/model_discrimination.rs`:

- `ModelDetectionProbability` binds a named model to `P(D=1 | M, executed)` and a non-empty derivation/calibration reference.
- `DetectionAwareCandidateExperiment` composes the existing candidate experiment and exactly one detection record per named model.
- `rank_detection_aware_model_discriminating_experiments` returns separate detection-channel EIG, conditional value-channel EIG, total EIG conditional on execution, execution-adjusted total EIG, expected posterior entropy, and quadrature error for each prior scenario.
- Prediction and detection records are canonicalized by model ID, not caller-provided vector position.
- Invalid probabilities, missing/duplicate/unexpected model rows and missing detection derivations fail closed.
- Non-comparable physical predictions retain the existing typed blockers.

The example `model_detection_reference` contrasts (1) detection plus informative values, (2) detection-only information with identical value distributions, and (3) model-independent detection that reduces to the prior API.

## Qualification fixtures

The exact-reference tests require:

1. Equal q across models reproduces the legacy uninformative-availability EIG.
2. Different q with identical Gaussian value distributions yields positive detection-indicator EIG and approximately zero conditional value EIG.
3. The detected-value EIG uses the prior conditioned on detection.
4. q=0, q=1, q_bar=0, execution probability 0 and zero-prior models are defined without division by zero.
5. An independent direct integration of the joint Bernoulli/Gaussian outcome likelihood agrees with the chain-rule implementation.
6. Reordering prediction rows or detection-probability rows does not change scores.
7. Missing, duplicate, unexpected and invalid detection rows are rejected.
8. Quadrature convergence, model comparability and open-world blockers remain explicit.

The benchmark reports prior entropy, expected detection probability, detection-indicator EIG, conditional value EIG, total EIG, operationally adjusted EIG and expected posterior entropy separately. The quadrature gate is applied to the unweighted conditional detected-value integral; the report separately gives its smaller operational error bound after multiplying by detection and execution probabilities. A low operational probability must never mask a numerically under-resolved conditional likelihood. The metrics must not be collapsed into one opaque score.

## What this does not model yet

This first implementation assumes:
- one binary detection indicator;
- one constant model-specific detection probability per experiment;
- model-independent execution availability;
- a Gaussian value distribution conditional on detection;
- no false-alarm process distinct from model-dependent detection;
- no numerical value when D=0;
- no latent nuisance integration in q_i.

Real sensors often require a richer channel:

    execution / sensor-health outcome
      -> detection / non-detection / false alarm / saturation / censored value
      -> measurement value conditional on the relevant outcome

A future extension should represent these outcomes as a typed categorical observation process with separate likelihoods. Examples include:
- thresholded SNR-dependent probability of detection;
- false-alarm rate and clutter/background models;
- missed detections caused by occlusion or field-of-view limits;
- saturation and interval-censored values;
- sensor-health, transmission, timing and calibration failures;
- time-varying detection probability integrated over target-state and calibration uncertainty.

Do not use the v0.1 Bernoulli model to turn “not seen” into “not present” or to infer object identity. A non-detection can be physically informative only when the detection process is calibrated or derived under the stated instrument, environment and geometry.

## SOTA and scientific positioning

This is not a new result in missing-data statistics or Bayesian optimal experimental design. Informative missingness, selection models and informative censoring are mature statistical subjects. Little's review, *Missing Data Assumptions*, Annual Review of Statistics and Its Application 8 (2021), distinguishes ignorable missingness from informative missingness and discusses the assumptions required to use a missingness mechanism safely: https://doi.org/10.1146/annurev-statistics-040720-031104

For physical experiment design, Heinrich et al., *Designing physics experiments with artificial intelligence*, Nature 657, 47–58 (published 2026-09-02), emphasizes the joint challenges of expressive hardware search spaces, reliable simulators, scientific objectives and feasible optimization: https://doi.org/10.1038/s41586-026-10898-6

A 2026 model-discrimination OED study in Mathematical Biosciences 399 (September 2026), *Optimal experiment design for practical parameter identifiability and model discrimination*, explicitly distinguishes practical parameter identifiability from model discrimination and discusses profile likelihoods as a complement to local FIM diagnostics: https://doi.org/10.1016/j.mbs.2026.109710

Symthaea's proposed contribution is narrower: preserve the detection mechanism as part of the observation contract, compose it with open-world model prediction coverage and budgeted experiment design, and attach exact-head reproducibility evidence. It should be compared against standard two-stage likelihood/selection-model baselines and direct joint-likelihood integration, not claimed as a novel missing-data theory.

## Future benchmark gates

Before using real instrument data, require:
- calibration data for q_i across signal-strength, geometry, background, threshold and sensor-health regimes;
- held-out detection calibration metrics (Brier score, log score, reliability curves, calibration slope/intercept and uncertainty);
- false-alarm and missed-detection rates with confidence intervals;
- detection-probability sensitivity to plausible calibration/pose/environment nuisance variables;
- model-prior sensitivity and open-world adequacy reporting;
- matched comparisons between uninformative availability, explicit Bernoulli selection, and a richer categorical/censored outcome likelihood;
- independent implementation of the joint observation likelihood;
- sealed regimes not used to calibrate q_i or tune the decision rule.

A synthetic pass validates only the declared Bernoulli/Gaussian observation channel. It does not validate real detection probabilities, establish model-set completeness, identify a UAP, or prove any physical theory.
