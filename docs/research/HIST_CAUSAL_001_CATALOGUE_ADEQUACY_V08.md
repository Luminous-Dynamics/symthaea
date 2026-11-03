# HIST-CAUSAL-001 v0.8: candidate-catalogue adequacy without OOD likelihood

## Objective

v0.7 used an oracle pairwise likelihood-ratio test that knew the hidden out-of-family mixture and its nearest in-catalogue candidate. v0.8 removes that advantage: the detector knows the six declared candidate likelihoods and the current recipe parameters, but it is not given the OOD mixture likelihood.

The detector computes the maximum log-likelihood across the six candidates. It calibrates a lower-tail rejection threshold by parametric bootstrap under every in-family model, then uses the most conservative (minimum) threshold. If even the best declared candidate fits unusually poorly, it flags model_misspecified.

This is still a finite known-parameter adequacy baseline. It does not solve unknown parameter estimation or general OOD discovery.

## Run design

- 12 public development recipes.
- 200 calibration replicates per in-family candidate, recipe and sample-count level.
- 200 evaluation replicates per candidate model and one OOD mixture, recipe and sample-count level.
- Six probe types; n independent observations per probe.
- Target false-alarm level alpha = 5%.
- Candidate likelihoods and recipe parameters are known; OOD mixture likelihood is not available to the detector.

## Results

| Samples per probe | OOD detection | Pooled in-family false OOD | Worst per-model false OOD |
|---:|---:|---:|---:|
| 1 | 7.79% | 0.55% | 3.29% |
| 2 | 27.75% | 1.48% | 3.79% |
| 5 | 63.58% | 1.44% | 6.58% |
| 10 | 88.75% | 0.99% | 5.33% |
| 20 | 99.54% | 0.90% | 5.17% |

The worst false-alarm candidate was actuator non-compliance at all tested sample counts. Its observed rate slightly exceeded nominal 5% at n=5, 10 and 20. This is a calibration-precision warning, not something to average away: the bootstrap used only 200 samples per model and recipe, and the development suite is small. Increase calibration replicates, use held-out calibration seeds, and report per-model confidence intervals before claiming controlled false-alarm rates.

The pooled rate can look excellent while one mechanism class is under-calibrated. All model-conditional false-alarm rates remain first-class metrics.

## Interpretation

This is a more realistic open-world comparator than the v0.7 oracle because it does not know the OOD likelihood. It still assumes known point parameters for all candidate models and knows the candidate model catalogue. It is not a real scientific agent and does not demonstrate physical discovery.

The result suggests that the hidden mixture is detectably different from the declared catalogue as evidence accumulates: detection rises from 7.8% at one observation per probe to 88.8% at ten and 99.5% at twenty. However, the residual model-specific false-alarm excess shows calibration still needs work.

## Next calibration improvements

1. Increase bootstrap calibration replicates to at least 1,000 per candidate/recipe/condition.
2. Use separate calibration and evaluation seed sets.
3. Add nuisance-parameter estimation and integrate over parameter uncertainty instead of assuming known values.
4. Add multiple OOD families, not only a fixed sensor/actuator mixture.
5. Test correlation between repeated observations and nonstationarity.
6. Score calibration, false misspecification, missed OOD, abstention and compute separately.

## Claim ceiling

v0.8 is a synthetic, known-parameter, finite-catalogue bootstrap baseline. It does not establish general model adequacy, physical mechanism discovery, morphic resonance, Chain Theory's physical interpretation or SOTA superiority.