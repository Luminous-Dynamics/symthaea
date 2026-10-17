# Cosmology and Astronomy Research Roadmap (2026)

**Status:** research plan and benchmark contract; proposed targets are not achieved results.  
**Companion implementation:** `ExpansionHistory` in `crates/core/symthaea-core/src/physics/cosmology.rs`.  
**Scope:** Symthaea numerical cosmology first; Symtropy simulation integration and Mycelix evidence capture are later, explicitly gated steps.

## Thesis: from scientific vocabulary to executable, falsifiable knowledge

The paradigm shift is not to add more cosmology terms to a knowledge encoder. It is to make every scientific claim traceable through a complete loop:

1. **Observation:** frozen dataset release, masks, units, covariance, selection function, calibration, and provenance.
2. **Model:** equations, parameterization, prior, approximation domain, and independently tested numerical solver.
3. **Prediction:** observables generated before looking at evaluation/holdout data.
4. **Inference:** likelihood, nuisance parameters, posterior samples, convergence checks, and model-complexity accounting.
5. **Falsification:** negative controls, simulated-data recovery, predictive checks, and robustness to defensible alternate assumptions.
6. **Evidence:** exact source revision, configuration, data checksums, executable commands, output hashes, and independent reproduction.

The output should be a **reproducible scientific claim**, not a free-floating answer or vector similarity score.

## Role boundaries

- **Symthaea:** scientific model representation, equation execution, uncertainty-aware reasoning, model comparison, anomaly triage, and experiment selection. HDC similarity is a navigation aid, not a substitute for physical units, likelihoods, or numerical validation.
- **Symtropy:** simulation only where its integrator, force model, conservation properties, and numerical error have been validated for the target problem. A rigid-body simulator is not automatically a cosmological N-body code.
- **Mycelix:** attestations and tamper-evident provenance for data, configurations, outputs, reviews, and replications. Cryptographic integrity and community agreement do not establish physical truth.
- **External scientific software:** use established codes as independent reference oracles and cross-checks. Do not reimplement a mature Boltzmann or likelihood stack merely for architectural purity.

## Immediate code increment

The companion implementation adds a validated FLRW background expansion model with constant dark-energy equation of state, radiation, matter, curvature, fixed-grid and tolerance-driven adaptive Simpson integrations for line-of-sight comoving distance and lookback time, curvature-adjusted transverse distance, luminosity distance, angular-diameter distance, and standard BAO observables D_M/r_d, D_H/r_d, and D_V/r_d with an explicit caller-supplied sound horizon. Analytic tests cover Einstein–de Sitter, radiation-only, and de Sitter limits, distance reciprocity, curvature direction, invalid inputs, non-finite expansion rates, adaptive-convergence failure, and tolerance validation. Adaptive integration has a bounded recursion depth and fails closed instead of silently returning an under-resolved answer.

It deliberately does **not** claim to provide a CMB Boltzmann solver, perturbation evolution, nonlinear structure formation, survey likelihoods, parameter inference, or precision Planck constraints. The default helper is an explicitly radiation-free flat-ΛCDM approximation for continuity with existing code, not a precision parameter release.

## Proposed scientific benchmark ladder

### B0 — analytic numerical oracles (must pass before external-data claims)

Fixtures must cover:

- Einstein–de Sitter: Ωm=1, Ωr=Ωk=Ωde=0. Test E(z)=(1+z)^(3/2) and χ(z)=2c/H0 × (1-1/√(1+z)).
- de Sitter: Ωde=1, w=-1. Test constant H(z) and χ(z)=cz/H0.
- Open and closed FLRW models. Test the sign of the transverse-distance correction against radial distance.
- Distance reciprocity: DL=(1+z)^2 DA for a transparent metric theory on the same background.
- Lookback time in Einstein–de Sitter and de Sitter limits; H0-to-Gyr conversion must be explicit and tested.
- BAO distance ratios D_M/r_d, D_H/r_d, and D_V/r_d with explicit r_d input; this helper must not imply it models recombination or infers r_d.
- Invalid/non-finite inputs, closure failures, pathological expansion rates, and invalid integration interval counts must fail closed.
- Adaptive integration must honour declared absolute/relative tolerances or return an explicit non-convergence error; recursion depth and arithmetic must be bounded.

**Acceptance:** analytic expansion-rate identities at tight floating-point tolerance; integrated distance relative error ≤ 1e-9 against analytic cases at the committed integration resolution; no NaN/Infinity silently emitted; every failure path has a deterministic test. Tighten the tolerance only when it is justified by a reference comparison and CI platform behavior.

### B1 — cross-implementation background comparison

Compare the Rust outputs against at least one independent, established implementation over a frozen grid in redshift and cosmological parameter space. Include flat/non-flat models, matter/radiation/dark-energy domination, and boundary cases. Record the external code name, release/version, settings, input fixtures, output digest, max absolute/relative error, and known domain differences.

**Acceptance:** predeclared error limits; grid-wide coverage (not selected cherry-picked examples); independently recomputable fixtures. The first cross-check can use background distances only; do not conflate it with CMB likelihood validation.

### B2 — synthetic recovery and calibrated uncertainty

Generate synthetic observations from known parameters; run the complete intended inference pipeline; compare recovered posterior intervals with the known generator parameters across many fixed seeds.

**Acceptance:**
- 68% and 95% interval coverage reported with binomial uncertainty across repeated simulations.
- Rank/PIT or equivalent posterior calibration diagnostics.
- Convergence diagnostics and effective sample sizes for sampling methods where applicable.
- Prior sensitivity and parameter-degeneracy analysis.
- A simple, independently verified baseline retained as a negative control.

No single successful seed is sufficient evidence of calibrated inference.

### B3 — real-data reproduction before novel cosmology

Recommended order:

1. Reproduce a small published BAO distance-likelihood example with frozen covariance and documented nuisance assumptions.
2. Reproduce released DESI DR2 BAO cosmology-chain/posterior summaries for identical models and priors.
3. Treat the July 2026 DESI DR2 Lyman-α full-shape result as a separate, harder reproduction target. DESI's documentation states that the underlying DR2 spectra/redshifts are not yet released, although BAO chains and posterior-maximization products are available. Never claim a full-shape reproduction from summary products alone.
4. Add Type Ia supernova distance moduli and their covariance as an independent probe.
5. Only then add joint inference and investigate whether conclusions move under alternate systematics, priors, and model extensions.
6. Add CMB constraints only after validating the necessary transfer functions, spectra, likelihoods, masks, beam/noise treatment, and nuisance parameters against established reference outputs.

**Acceptance:** under identical inputs and assumptions, posterior summaries agree with the reference within predeclared numerical/statistical tolerance; mismatches are explained and retained in an audit note rather than tuned away. Report data provenance, covariance, likelihood normalization, prior bounds, chains/seeds, and convergence checks.

### B4 — observation selection and active research

Once the forward model and likelihoods pass B0–B3, measure whether Symthaea can choose the next observation or analysis that best discriminates among surviving hypotheses. Score candidate actions using expected information gain or expected decision value, while accounting for cost, selection effects, telescope access, and model uncertainty.

**Acceptance:** beat random and common heuristic baselines on preregistered simulated tasks; show calibration on held-out generators; test adversarial cases where the current model family is wrong. An acquisition policy must be able to answer “none of our candidate models explains these data” instead of always selecting a winner.

### B5 — transient astronomy / alert triage

Rubin's first public scientific alerts began on 25 February 2026; the observatory reported 800,000 alerts on the first night and expects the system to scale toward millions nightly. The first practical target is not autonomous discovery claims: it is calibrated classification, cross-matching, duplicate handling, uncertainty-aware anomaly triage, and useful follow-up prioritization.

**Acceptance:** evaluate on time-separated and source-separated holdouts; report precision/recall by event class, calibration error, false-alert burden, latency, drift, and selection effects. Ensure the test split prevents the same astrophysical source from leaking across train/test.

### B6 — scientific claim reproducibility

A machine-readable evidence manifest should bind each claim to:

- immutable dataset/version and checksums;
- exact git commit and build/configuration;
- model equations and parameter units;
- priors, likelihood/covariance, random seeds, and sampler diagnostics;
- input and output artifact hashes;
- expected versus actual benchmark values and tolerance;
- independent verifier identity/version and explicit reproduction status.

Use states such as `proposed`, `locally-verified`, `reproduced-independent`, `reference-mismatch`, and `not-run`. Never translate queued CI into a pass, and never call a result independently verified based only on its author-run artifact.

## Paradigm-shifting research directions after the gates

1. **Cross-probe consistency as a first-class object.** Represent which parameter combinations are constrained by BAO, supernovae, CMB, lensing, clustering, and local distance ladders. Identify whether tensions persist when likelihoods and systematics are modeled consistently.
2. **Predict-before-fit evaluation.** Reserve time- or source-based holdouts and publish predictive scores. Use parameter recovery on simulated data to expose bugs that ordinary fit quality cannot detect.
3. **Robust discovery rather than forced model selection.** Include out-of-distribution tests, posterior predictive checks, unknown-systematic models, and explicit “none of the above” outcomes.
4. **Active experimental design.** Select analyses or follow-up measurements for expected information gain only after posterior calibration has passed. Distinguish information gain inside a misspecified model family from true model discrimination.
5. **Differentiable/hybrid numerical methods as optional acceleration.** Compare gradients, emulators, and learned surrogates against exact/reference solvers; propagate emulator error into the likelihood. The scientific result must remain reproducible without trusting an opaque surrogate.
6. **A null-result ledger.** Preserve failed fits, convergence failures, negative results, and model-dependent results as versioned evidence, not as expendable logs.

## 2026 external reference points

- DESI DR2 key publications and data status: https://data.desi.lbl.gov/doc/papers/ and https://data.desi.lbl.gov/doc/papers/dr2/
- DESI 30 July 2026 Lyman-α full-shape result: https://www.desi.lbl.gov/2026/07/30/new-desi-dr2-lyman-alpha-results-shed-light-on-dark-energy/
- Planck public-release likelihood, spectra, parameter grids, and covariance products: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/
- Rubin Observatory first scientific alerts (25 February 2026): https://rubinobs.org/news/first-alerts
- Euclid current and planned data-release timeline: https://euclid.caltech.edu/page/data-release-timeline
- Cobaya cosmological inference and external-likelihood documentation: https://cobaya.readthedocs.io/en/latest/

These are candidate reference sources, not a statement that this repository currently reads those releases or reproduces their results.

## Operational rules

- Keep this work on a review branch until the numerical tests and code review are complete.
- Run focused Rust tests and deterministic checks without waiting for long workspace CI.
- CI status is always exact-head-specific; queued means not verified.
- Do not weaken negative controls or thresholds to make a benchmark green.
- Make no claims of new cosmology until the full inference, uncertainty, independent reproduction, and peer-review gates are met.
