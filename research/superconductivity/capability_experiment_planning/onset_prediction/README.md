# SC-ONSET-PREDICTION-AUDIT-0.1

**Status:** synthetic, censoring-aware evaluation contract; not an onset predictor and not a reproduction of published scores.  
**Research issue:** [#13011](https://github.com/Luminous-Dynamics/symthaea/issues/13011)  
**Spatial reference stack:** [PR #13005](https://github.com/Luminous-Dynamics/symthaea/pull/13005)

## Why this benchmark matters

Walters et al., “Predicting solid-state reaction onset temperatures from high-throughput pairwise synthesis” (posted 2026-11-17; https://arxiv.org/abs/2611.23527), report 1,035 precursor-pair trials and 1,000 analyzable outcomes. At the Tammann-rule temperature, only 25.3% formed inter-precursor products. Their three-feature log-normal accelerated-failure-time (AFT) model used censored binary dwell outcomes and achieved 88 °C MAE on 26 held-out in-situ measurements, compared with 131 °C for Tammann's rule. The 32-feature extension did not materially improve held-out accuracy.

The key semantics are easy to get wrong:
- A verified inter-precursor product at dwell temperature T establishes onset at or below T (left-censored).
- A verified absence of inter-precursor product at T establishes onset above T (right-censored).
- A lower-temperature non-reaction and higher-temperature reaction bracket onset (interval-censored).
- An in-situ measurement can establish an exact onset or an interval according to its instrument resolution.
- A transformed-only precursor or mechanically uncharacterizable sample is not evidence that the intended inter-precursor reaction occurred.

Do not replace one-sided bounds with point labels. That invents information and biases both training and evaluation.

## Contract and scoring

The fixture schema uses absolute temperature in kelvin because the log-normal AFT likelihood operates on positive absolute temperatures. Each pair carries precursor IDs, evidence IDs and a raw-data digest. Each model carries an artifact digest, explicit uncertainty method, training pair IDs and one predictive log-normal distribution per evaluation pair.

For a predicted onset X distributed log-normally with median m and log-scale sigma:
- left-censored evidence scores P(X <= upper bound);
- right-censored evidence scores P(X > lower bound);
- interval-censored evidence scores P(lower < X <= upper);
- exact evidence scores the log-normal density at the measured onset.

The main metric is mean negative log-likelihood across the stated evidence classes. This rewards calibrated distributions, not just plausible point estimates. Exact-only MAE, RMSE, 90% interval coverage and mean interval width are reported separately and never include censored rows. All predictions require finite, positive Kelvin medians and positive log-scale uncertainty.

The auditor rejects:
- malformed or inconsistent censoring bounds;
- invalid evidence or model artifact digests;
- duplicate evaluation pairs, model IDs or predictions;
- missing/extra predictions;
- any evaluation pair listed in the model's training pair IDs.

The bundled data is synthetic plumbing only. Digests are synthetic placeholders, and the fixture's predeclared uncertainty scales are not fitted to real material data.

## Research programme and SOTA ladder

1. **Reproduce before competing.** Obtain the original Precursor Genome and literature data, model code and 26-pair in-situ validation table. Verify release identity, data license, source digests, pair exclusions and preprocessing before computing any new score.
2. **Exact reproduction.** Reproduce the published Tammann 131 °C and compact AFT 88 °C MAE figures on the same 26 held-out reactions, and reproduce the reported interval coverage. Treat the paper's MAE and LOPO censored log-score as different evaluation tasks.
3. **Leakage-resistant generalization.** Preserve the authors' split, then add precursor-held-out, chemistry-family-held-out and time-forward tests. A random row split is insufficient when precursor identity and repeated pairs cross folds.
4. **Uncertainty quality.** Report censored negative log-likelihood, exact-only MAE/RMSE, 50/80/90/95% coverage and interval width, calibration curves, tail misses, and per-precursor error slices.
5. **Decision utility.** Compare experiment selection policies under equal experiment, time, material and characterization budgets. Measure expected information gain, onset interval contraction, calibration after update, decision regret and failed/ambiguous evidence rates.
6. **Integrate physics only where measured.** Couple the onset model to the thermochemical network, transport-correlation auditor and spatial reaction-diffusion reference. Preserve uncertainty and provenance across every boundary; never convert a modelled onset into an observed phase outcome.

## Stretch targets (not achieved results)

- Reproduce the published baseline exactly before claiming any improvement.
- Reduce exact-onset MAE below 70 °C while maintaining at least 90% empirical coverage for nominal 90% intervals on a larger independent validation cohort.
- Improve censored negative log-likelihood over the three-feature AFT under precursor- and chemistry-family-held-out evaluation.
- Reduce experiment count to reach a declared onset interval width by at least 2x versus fixed-temperature grids, at equal safety and characterization constraints.
- Demonstrate that any gain survives paired bootstrap intervals and prospective validation; report cost per correctly resolved onset and not merely a leaderboard score.

These are research goals. The current synthetic fixture provides no evidence that any target has been met.

## Literature comparison

- Karan et al., *Nature Materials* (2026), “Ion correlations explain kinetic selectivity in diffusion-limited solid-state synthesis reactions”: https://www.nature.com/articles/s41563-026-02596-5. It integrates calculated ion transport and cross-ion correlations with thermodynamics and a spatial cellular framework, validated against Ba–Ti–O experiments. The Symthaea finite-volume reference is a simplified 1D PDE and is not a reproduction.
- Schlesinger et al., *Materials Horizons* (2026), “Thermodynamic assessment of machine learning models for solid-state synthesis prediction”: https://doi.org/10.1039/D6MH00209A. It finds synthesizability models can overpredict likelihood and evaluates predictions against thermodynamic feasibility and reaction selectivity.
- Walters et al., arXiv:2611.23527 (2026-11-17): https://arxiv.org/abs/2611.23527. It establishes a contemporary quantitative baseline for censored onset learning.

## Execution

```bash
cargo test --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_onset_prediction_audit
cargo run --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_onset_prediction_audit -- \
  --schema data/schemas/superconductivity_onset_prediction_v0_1.schema.json \
  --fixture tools/superconductivity-discovery-harness/fixtures/onset_prediction_smoke_v0_1.json
```

## Claim ceiling

This is a synthetic onset-evaluation plumbing check only. It establishes no real reaction onset, model accuracy, phase yield, material synthesizability, superconductivity or laboratory outcome.
