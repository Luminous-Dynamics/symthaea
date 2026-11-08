# DE-001 — Cosmological Acceleration Anomaly Localization

**Date:** 2026-09-18  
**Status:** preregistered research design; this document reports no scientific result.  
**Initial implementation:** `symthaea-cosmology-research`

## Research question

Is there a reproducible feature in current late-time cosmological observations that
is inadequately explained by the preregistered ΛCDM baseline and that survives
reasonable changes in representation, inference method, dataset composition,
and systematic-error treatment?

DE-001 does **not** begin with the claim that dark energy evolves.

The first claim DE-001 can license is only:

> A qualified observational anomaly exists relative to the preregistered baseline.

A claim of phenomenological dark-energy dynamics requires a separate
qualification lineage. A physical-mechanism claim requires another independent
lineage after that.

## Why this experiment is needed

DESI DR2 analyses have made the late-time acceleration question unusually
informative but also unusually easy to overstate.

- DESI's extended DR2 dark-energy analysis reports consistent low-redshift
  trends across several parametric and non-parametric reconstructions.
- DESI's 2026 DR2 Lyman-alpha full-shape result moved toward ΛCDM relative to
  the earlier BAO-only point and explicitly leaves open both a fading
  dynamical-dark-energy hint and model insufficiency.
- Current public tooling already exposes DESI DR2 BAO likelihoods, so the
  first tranche should reproduce and stress-test published inference rather
  than replace mature numerical cosmology software.

External anchors:

- DESI DR2 publications: https://data.desi.lbl.gov/doc/papers/dr2/
- DESI 2026 Lyman-alpha full-shape release:
  https://www.desi.lbl.gov/2026/07/30/new-desi-dr2-lyman-alpha-results-shed-light-on-dark-energy/
- DESI extended dark-energy analysis: https://arxiv.org/abs/2503.14743
- Cobaya DESI DR2 BAO likelihood documentation:
  https://cobaya.readthedocs.io/en/stable/likelihood_bao.html
- desilike: https://github.com/cosmodesi/desilike

These references motivate the experiment; they are not imported as Symthaea
evidence.

## Baseline

The initial null is spatially flat ΛCDM with constant dark-energy density.

For the direct-density lane define

[
X(z) = \frac{\rho_{DE}(z)}{\rho_{DE}(0)}.
]

ΛCDM predicts

[
X(z)=1.
]

The experiment must not infer a global model-comparison result from isolated
pointwise deviations.

## Evidence cube

Every result must identify all of the following axes:

1. **Dataset/probe**
2. **Representation**
3. **Inference lane**
4. **Systematics treatment**

The initial representation families are:

- CPL `w0-wa`;
- direct `rho_DE(z)`;
- binned `w(z)`;
- spline `w(z)`;
- Gaussian-process `w(z)`;
- PCA/eigenmode reconstructions;
- later physical models.

The initial inference lanes are:

- frequentist/profile likelihood;
- Bayesian posterior;
- Bayesian evidence;
- posterior predictive checks;
- simulation calibration.

No single cell of the cube can establish representation invariance.

## Claim layers

The implementation must keep three claim types distinct:

1. **ObservationalAnomaly** — something reproducible is difficult for the
   baseline.
2. **PhenomenologicalDynamics** — independent reconstructions support
   non-constant effective dark-energy behaviour.
3. **PhysicalMechanism** — a concrete physical theory survives stability,
   consistency, and out-of-sample prediction tests.

Qualification of one layer only permits testing the next layer. It never
qualifies the next layer automatically.

## DE-001 gate sequence

### DE-001A — Exact reproduction

Reproduce selected published DESI DR2 baseline constraints with frozen
versions of the numerical backend, likelihood data, configuration, and
environment.

Required evidence identity:

- subject commit;
- protocol version;
- dataset digest;
- configuration digest;
- result digest;
- backend/environment identity.

A failure to reproduce is a result and blocks confirmatory interpretation.

### DE-001B — Independent inference

Run logically distinct inference lanes over the same frozen scientific subject.

At minimum the program should include profile/frequentist and Bayesian lanes;
simulation calibration should be added before any headline anomaly claim.

### DE-001C — Representation invariance

Test whether the anomaly survives materially different representations.

The primary question is not whether fitted parameter values agree exactly. It
is whether a common feature survives without requiring one privileged
functional form.

### DE-001D — Dataset localization

Perform leave-one-probe-out and, where supported, leave-one-tracer/bin-out
analyses.

The output must identify which data contribute to the anomaly rather than
compressing all attribution into one headline significance.

### DE-001E — Systematics adversary

Search preregistered plausible perturbations including calibration,
covariance, selection, redshift, nuisance-model, and prior sensitivity.

The adversary is successful if it can produce an observationally similar
feature without the proposed new physics.

### DE-001F — ΛCDM counterfeit challenge

Generate ΛCDM synthetic universes and pass them through the same complete
analysis path.

Measure empirical false-positive behaviour for the *pipeline*, not merely an
analytic asymptotic statistic.

### DE-001G — Signal injection

Inject several dynamic histories and measure recovery power and failure modes.

The injection family must include signals that are not exactly represented by
the preferred fitting basis.

### DE-001H — Holdout unblinding

Development partitions and confirmatory holdouts must be disjoint before this
gate.

Any model, threshold, systematic treatment, or analysis-path change after
confirmatory holdout exposure marks that lineage as contaminated for
confirmatory use. A new holdout is then required.

### DE-001I — Independent replication

Repeat the surviving result with a second implementation whose numerical and
analysis dependencies overlap as little as practical.

### DE-001Q — Qualification

A PASS can license only an `ObservationalAnomaly` claim.

DE-001Q cannot by itself license:

- "dark energy is dynamic";
- "w crossed -1";
- "modified gravity is preferred";
- "a scalar field has been discovered";
- any discovery-level particle/field ontology.

## Primary falsification criteria

DE-001 is weakened or falsified as an anomaly program if one or more of the
following occurs:

- the effect fails exact reproduction;
- it exists only under one reasonable representation;
- it is localized to one dataset/probe and is plausibly reproduced by that
  probe's systematic model;
- ΛCDM counterfeit simulations generate comparable features at an
  unacceptably high rate;
- independent inference or independent implementation does not recover it;
- confirmatory holdout results require post-unblinding tuning.

Null and negative results remain first-class evidence.

## Numerical architecture boundary

Symthaea should not initially implement its own Boltzmann solver or standard
cosmological sampler.

Preferred architecture:

[
\text{Symthaea scientific controller}
\rightarrow
\{\text{Cobaya},\text{desilike}\}
\rightarrow
\{\text{CLASS},\text{CAMB},\ldots\}.
]

The Symthaea layer owns experiment identity, hypothesis control, evidence
lineage, adversarial analysis, cross-representation comparison, and claim
licensing.

## First implementation scope

The first code tranche intentionally implements only:

- evidence-cube coordinate types;
- explicit claim layers;
- fail-closed adjacent-layer promotion;
- DE-001 gate ordering;
- evidence receipts with provenance identity;
- post-unblinding contamination tracking;
- direct `X(z)` point validation.

It does **not** contain data, likelihoods, fitting, significance calculation,
or a scientific verdict.

## Next tranche after this preregistration lands

1. Frozen backend manifest for Cobaya/desilike/CLASS-or-CAMB.
2. DESI DR2 BAO reproduction adapter.
3. Machine-readable DE-001 run manifest.
4. Receipt writer that binds exact backend/data/config/result identities.
5. Synthetic ΛCDM fixture generator for pipeline self-tests.
6. CI lane that validates the research harness without claiming scientific
   qualification.

Only after exact reproduction passes should representation and systematics
search be expanded.
