# Black-Hole Scientific Reasoning Contract

**Purpose:** Define what scientifically competent black-hole reasoning means for Symthaea, and make clear which calculations belong to Symtropy, evidence/provenance to Mycelix, and observation projection to Sol Atlas.

## 1. Capability boundary

The existing `symthaea-core/src/physics/general_relativity.rs` is an exploratory HDC semantic representation. It can support concept association and retrieval, but a hypervector similarity is not a metric component, geodesic, curvature invariant, waveform, or physical proof.

Symthaea's black-hole capability should therefore have two connected but non-interchangeable lanes:

- **Semantic lane:** retrieve, associate, compare and organize physical concepts and literature.
- **Verified physics lane:** consume typed, versioned outputs from Symtropy or independently maintained reference implementations.

Every response and machine-readable result must identify which lane produced it. HDC must never silently substitute for physical calculation.

## 2. Typed scientific objects

Symthaea should reason over:
- physical model (GR, Kerr, Schwarzschild, perturbation model, BOB, GRMHD, etc.);
- coordinate chart and conventions;
- analytic derivation;
- numerical simulation;
- detector observation / strain;
- reconstructed waveform component;
- inferred parameter posterior;
- measurement-space residual;
- hypothesis and counter-hypothesis;
- computational reproduction;
- independent replication;
- null / failed / inconclusive result.

These are distinct ontological categories. Example: an EHT reconstructed image is not a raw telescope visibility; a GW posterior is not a detector measurement; a filtered component is not automatically a separate physical emission channel.

## 3. Required output contract

For a material scientific analysis, emit or persist a `BlackHoleAnalysisV1`-equivalent record containing:

- analysis ID and exact code revision;
- question and preregistered hypothesis;
- physical model and alternatives;
- assumptions, parameter ranges and applicability domain;
- equations and units;
- coordinate chart, signature, frame and time convention;
- source/data release and file digests;
- software versions and configuration digest;
- result values and units;
- statistical, numerical, calibration, data-quality and model-form uncertainty separately;
- null controls and negative-control outcomes;
- independent verification status;
- claim ceiling and unresolved issues.

A textual answer without a reproducibility record may educate, but it must not be reported as a completed scientific result.

## 4. GW250114 direct-wave model-discrimination programme

The direct-wave dispute is a high-value scientific reasoning task because the data permit reproduction and the interpretation has multiple independent competitors.

### Published evidence paths

- Nature 2026-06-24, DOI 10.1038/s41586-026-10696-0: reports an extracted GW250114 component interpreted near 2*Omega_H with damping characterized by kappa.
- Kankani & McWilliams, arXiv:2607.02380: SXS counteranalysis; reports no general frequency/horizon or damping/surface-gravity association over comparable-mass spin sweep.
- Kankani & McWilliams, PRD 114, 084025 (2026), DOI 10.1103/kjjd-nk8f: BOB analysis; reports direct-wave frequency follows a different spin dependence from horizon frequency and tracks News frequency at peak News amplitude.
- Dyer, Chung & Moore, PRL accepted 2026-10-27, DOI 10.1103/xftt-2n6t: direct wave extracted over different start times, but frequency deviates from the horizon value.
- Suzuki et al., PRD 114, 044099 (2026), DOI 10.1103/6x5z-q9ml: orthonormalized QNM analysis; retain its mode basis and posterior definition separately from non-orthogonal fits.

### Competing models

Symthaea should coordinate, not pre-choose, among:

1. QNM / overtone / mode-mixing residual.
2. Fixed complex-frequency damped sinusoid.
3. Freely evolving complex-frequency residual.
4. Kerr horizon-tied 2*Omega_H and kappa model.
5. News-frequency-at-peak-News-amplitude predictor.
6. Backwards-One-Body / prompt-plunge dynamics.
7. Flexible detector transient and noise alternatives.

### Exact SXS counteranalysis cases

| SXS case | Final spin |
|---|---:|
| SXS:BBH:2089 | 0.4942 |
| SXS:BBH:2088 | 0.5478 |
| SXS:BBH:3919 | 0.6215 |
| SXS:BBH:4434 | 0.6864 |
| SXS:BBH:0305 | 0.6921 |
| SXS:BBH:2095 | 0.7447 |
| SXS:BBH:3927 | 0.8029 |
| SXS:BBH:2099 | 0.8574 |
| SXS:BBH:3897 | 0.9075 |
| SXS:BBH:0178 | 0.9499 |

The counteranalysis used a (2,2) strain filter with 7 prograde QNMs, 2 retrograde QNMs, and an additional prograde (3,2,0) mode to address spherical-spheroidal mixing. Transcribe the exact four fitting intervals from its supplement before running comparisons.

### Required reasoning behavior

- Treat "residual can be extracted", "frequency/damping estimated", "horizon properties measured", and "area law tested" as four independent claims.
- Build posterior predictive checks and blind injection recovery.
- Use off-source/noise-only segments, time/phase scrambles, wrong remnant parameters, and randomized spin labels as negative controls.
- Compare the predictor models with matched data conditioning, priors and likelihoods.
- Explain identifiability and degeneracies; abstain when one event cannot distinguish models.
- Do not convert a coincidental crossing near chi_f≈0.69 into a universal law.
- Keep the EMRI regime separate from comparable-mass claims.

The exact source and data manifest is maintained in Symtropy's `direct_wave_sxs_spin_sweep.json` and Mycelix's evidence-package schema.

## 5. Scientific surrogate reasoning

Current waveform surrogates are already highly capable. Symthaea should not simply declare one model faster or better.

It should reason over:
- NRSur7dq4 NN (arXiv:2607.24960);
- NRSur7dq4v2 (arXiv:2609.07873);
- SEOBNRv5PHM_NNSur7dq10 (PRD 114, 084042);
- AESur3dq8 (PRD 114, 083047);
- eccentric NRSurE_q4NoSpin_22 (PR Research 8, 023362);
- SpEC and SXS high-resolution references;
- LALSimulation/Bilby inference baselines.

A model may be fast but poorly calibrated outside its training domain. Symthaea should compare mismatch distributions, phase/amplitude residuals, posterior coverage, remnant mass/spin recovery, domain gaps and cross-model disagreement.

### Active scientific computation selection

Choose new expensive NR simulations where they have maximum expected value:
- highest cross-model prediction disagreement;
- high posterior probability / high observational relevance;
- poor uncertainty calibration;
- underrepresented parameter-space boundary;
- ability to distinguish competing physical models.

Compare active selection to random and space-filling selection with equal simulation budgets. Preserve an untouched validation set.

## 6. Benchmarking scientific reasoning

### Machine-checkable tasks

- Derive Schwarzschild horizon, photon sphere, ISCO, lapse and redshift.
- Derive Kerr inner/outer horizons, ergosurface, Omega_H and kappa in declared units.
- Distinguish coordinate singularity from curvature singularity.
- Explain which quantities are coordinate-dependent and which are scalar invariants.
- Detect wrong metric signature, index order, units, chart domain and sign convention.
- Reproduce selected metric and connection fixtures from Symtropy.
- Classify raw strain, posterior samples, simulation waveform, reconstructed component, and physical interpretation.
- Compare direct-wave competing models with appropriate abstention.
- Detect when a surrogate is outside its validated parameter domain.
- Recognize when a single event cannot identify a physical mechanism.

### Metrics

- exact derivation and dimensional-check accuracy;
- wrong-unit / wrong-sign / chart-domain error detection;
- evidence-source and version accuracy;
- uncertainty calibration and coverage;
- false-positive rate on adversarial and null prompts;
- correct abstention on underdetermined or out-of-domain cases;
- independent reproduction rate;
- calibration of model-selection decisions on held-out injections.

A fluent explanation, self-consistency score or HDC similarity score is not evidence of physical correctness.

## 7. Cross-system interface

- **Symtropy:** supplies analytic metrics, connection, future curvature/geodesics, remnant horizon observables and validated numerical outputs.
- **Mycelix:** supplies canonical claims, evidence manifests, signed reproduction/critique/replication events and provenance.
- **Sol Atlas:** supplies source-faithful views of strain, posteriors, simulation waveforms, News quantities, reconstructed components and model predictions.
- **Symthaea:** coordinates hypotheses, model comparisons, experiment selection, uncertainty interpretation and falsification.

## 8. Acceptance gates

1. Existing GR HDC module is explicitly labelled semantic/experimental.
2. Physical reasoning calls typed validated providers, retaining exact units/chart/model metadata.
3. Held-out question set includes analytical, observational, simulation and epistemic-status tasks.
4. Direct-wave model comparison includes horizon, News-peak and BOB/plunge predictors plus null controls.
5. Surrogate evaluation includes domain detection and posterior-coverage checks.
6. Results are saved with Mycelix-compatible evidence manifests.
7. No scientific discovery claim is emitted without an independently reproduced prediction and a documented claim ceiling.
