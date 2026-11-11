# OLI comparative abiotic baselines across related environments v0

**Status:** empirical-Bayes reference module added; Rust validation pending  
**Owner:** Symthaea / `symthaea-alife`  
**Related:** [OLI experiment-selection issue #11446](https://github.com/Luminous-Dynamics/symthaea/issues/11446)  
**Code:** `crates/domains/symthaea-alife/src/comparative_baseline.rs`  
**Manifest:** `benchmarks/oli/comparative-baseline-v0.json`

## Research basis and industry comparison

Constantinou, Shorttle, Cranmer and Rimmer (2026), *Comparative biosignatures with systemic retrievals*, MNRAS 547, stag316, DOI [10.1093/mnras/stag316](https://doi.org/10.1093/mnras/stag316), propose a comparative multiplanet method for constructing an empirical abiotic baseline from related planets. Their method uses a hierarchical Bayesian model, leave-one-planet-out predictive assessment, and model-relative anomalies. It explicitly preserves an **unknown unknown** when neither abiotic nor biotic model adequately predicts the data.

NASA's 2026 Exoplanet Exploration Program Science Gap List, Rev I, identifies SCI-16 as building the inventory of remotely observable exoplanet biosignatures and their false positives. It asks for research into novel and agnostic biosignatures, planetary processes that can mimic or alter them, false negatives, and statistical frameworks that use planetary context. The document is a tactical program-level gap list for the next 3–5 years, not a universal classifier or a substitute for a National Academies strategy.

OLI's opportunity is to make comparative baselines **evidence-carrying and substrate-general**:
- within one planetary system, compare related worlds with shared formation/star-history context;
- within one world, compare geological regions, ocean plumes, mineral settings and repeated time windows;
- across laboratory preparations, compare matched abiotic and biological reference families;
- across systems, construct a cautious supersystem only after harmonizing latent context and testing transfer.

More observations are not automatically more information. Pooling across systems without the right covariates can introduce confounding and erase meaningful structure.

## Executable reference model

The initial Rust module implements a compact **empirical-Bayes random-intercept baseline**, not a full hierarchical Bayesian retrieval.

For qualified abiotic controls (i) in system (s), it fits:

[
y_{si}=\beta x_{si}+b_s+\epsilon_{si},
]

where (x) is one declared environmental context covariate, (\beta) is a common within-system slope, and (b_s) is a system-specific abiotic offset with a normal random-intercept approximation.

The implementation:
1. fits the shared context slope by within-system centered least squares;
2. estimates within-system residual variance after accounting for declared measurement uncertainty;
3. estimates between-system variance by a method-of-moments correction;
4. partially pools each system intercept toward the across-system center;
5. evaluates a held-out target against its own system's predictive distribution.

The system-level prior center gives each system one contribution rather than allowing a large sample-count system to dominate the between-system estimate.

## Evidence and anti-leakage rules

Only observations tagged `QualifiedAbioticControl` enter the fit. Each must carry:
- a system ID;
- a unique physical independence group;
- a qualification evidence reference;
- context and signal values with declared measurement uncertainty.

`Candidate` and `Unresolved` observations are excluded even if they are present in the same input slice. Every candidate is evaluated after fitting, so it cannot leak into its own baseline.

A candidate whose context lies outside the observed qualified-control range is rejected as unsupported extrapolation. An unseen system with no qualified controls is uncalibrated and must abstain. Neither case is converted into a biological anomaly.

## Frozen benchmark cases

- **Shared system offset:** a single pooled global line would make a system-B target look anomalous even though it exactly follows system B's local abiotic trend.
- **Within-system anomaly:** a target deviates from its own system's baseline while remaining inside the control context range.
- **Unresolved-control exclusion:** a high-value observation with unresolved provenance must not contaminate the baseline.
- **Context extrapolation:** no confident score outside qualified control support.
- **Unseen system:** abstain without a calibrated system baseline.
- **Unequal control counts:** the system-level prior center must not be dominated by systems with more sampled observations.

## What this does and does not claim

The result `ComparativeAnomaly` means only that a sample is an outlier relative to the declared qualified abiotic baseline under this model. It does not establish biology or life.

This initial reference is deliberately limited:
- one context covariate;
- normal random-intercept/residual approximations;
- empirical-Bayes moment estimates rather than full HBM posterior sampling;
- no explicit biotic evolutionary model;
- no multi-planet atmosphere retrieval or spectral forward model.

The next increment should replace the scalar context with a multivariate, instrument-aware observation operator, incorporate uncertainty in the context covariates, add an explicit biotic alternative model, and retain a third unknown-unknown disposition when both competing models have poor predictive adequacy.

## Qualification targets (proposed, not achieved)

- Zero candidate-to-baseline leakage.
- Zero qualified-control admission without source/qualification evidence.
- Zero duplicate physical independence groups in the control set.
- Correct system-conditioned prediction under held-out-world tests.
- Correct comparative-anomaly response for the registered outlier case.
- Abstention on unseen systems and context extrapolation.
- Report leave-one-world-out and leave-one-system-out performance separately.
- Report calibration and error rates per system, not only pooled.
- Compare against a global pooled baseline, no-pooling per-system baselines, partial pooling, and a full HBM reference when available.
- Never promote a comparative anomaly to life without independent biotic-vs-abiotic model comparison and evidence qualification.

## Industry/SOTA next step

The academic systemic-retrieval approach currently has a more sophisticated HBM/evolutionary-inference formulation than this small reference. OLI should not claim it is already better. The competitive engineering goal is to combine:
- local/systemic baselines;
- explicit instrument and observability models;
- open-world/model-inadequacy handling;
- adversarial false-positive and false-negative tests;
- cost-aware experiment selection;
- immutable evidence receipts and independent replay.

The differentiator is the complete scientific-assurance loop across domains, not a claim that this first reference beats a full atmospheric retrieval stack.
