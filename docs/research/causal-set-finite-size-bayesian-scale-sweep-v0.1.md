# Finite-size Bayesian causal-set dimension scale sweep v0.1

Date: 2026-11-09  
Status: executable calibration/qualification benchmark; not physical dimension-reduction evidence  
Parent: [CAUSAL-DIM-BAYES-001 #11191](https://github.com/Luminous-Dynamics/symthaea/issues/11191)  
Programme: [CTGP draft PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Purpose

Test whether a finite-size Bayesian dimension estimator remains calibrated as the number of events changes, and how often it falsely assigns high posterior mass to D≤3 when the planted truth is 4D Minkowski.

The existing empirical KDE likelihood is correctly conditioned on an exact event count N. It calibrates the distribution of the whole-causet Myrheim–Meyer ordering fraction for each candidate dimension. It does not assume pair-indicator independence. This is a good first likelihood, but it cannot be transferred unchanged from one N to another.

The new executable `bayesian_dimension_scale_sweep` therefore performs **separate calibration at every event count** and tests each calibration on disjoint, deterministic, whole-sprinkling qualification samples.

## Frozen pilot matrix

| Event count N | Calibration replicates / dimension | Qualification replicates / truth dimension |
|---:|---:|---:|
| 64 | 192 | 24 |
| 128 | 192 | 24 |
| 256 | 128 | 24 |
| 512 | 64 | 24 |
| 1,024 | 24 | 24 |

Candidate dimensions are D=2,3,4. Proper time and Alexandrov interval family are held fixed. Three priors are evaluated at each N:
- uniform: [1/3, 1/3, 1/3];
- low-dimension-favouring: [0.60, 0.30, 0.10];
- high-dimension-favouring: [0.10, 0.30, 0.60].

Calibration and qualification use disjoint SplitMix64 seed schedules. The event count is encoded into the seed derivation. The replicate allocation is a compute-bounded pilot, not a power analysis or final promotion threshold. In particular, 24 calibration replicates at N=1024 may produce wide or unstable KDE bandwidths; those effects must remain visible.

## Output metrics

For every (N, planted dimension, prior) tuple, report:
- qualification sample count;
- calibration-support fraction and outside-support count;
- 95% discrete credible-set coverage, conditional on a calibrated posterior;
- Wilson 95% interval for that empirical coverage;
- MAP dimension accuracy, conditional on a calibrated posterior;
- mean posterior mass on the planted dimension;
- mean log loss and multiclass Brier score, conditional on support;
- for planted D=4, count of cases with posterior P(D≤3)≥0.95;
- 95th percentile of P(D≤3) among supported D=4 qualification cases.

Out-of-support results are abstentions. They are not assigned P(D≤3)=1, are not scored as false dimension-reduction evidence, and do not enter posterior-accuracy denominators. The overall support fraction remains a separate result; high coverage conditional on support is not meaningful without it.

The false-reduction threshold of 0.95 is a descriptive pilot marker. The final threshold and acceptable false-positive rate must be preregistered using power and decision-cost analysis before sealed qualification.

## Why the baseline matters

Meyer (2026), *Evidence for and against dimensional reduction at the smallest scale of causal sets*, Physical Review D, DOI 10.1103/klxp-f94q, develops a Bayesian approach to small-scale causal-set dimension inference. Its stated result is that well-approximated Minkowski causal sets do not show reduction to three or fewer dimensions.

The Symthaea benchmark is not a reproduction of that paper's full analysis. It provides a transparent, deterministic finite-N Minkowski calibration baseline and a measurable false-reduction control. The scientific question is whether any apparent shift survives:
- event-count conditioning;
- held-out whole-causet calibration;
- prior sensitivity;
- calibration-sample uncertainty;
- posterior coverage and predictive scoring;
- model-family criticism.

A future comparison must pin the published estimator, priors, summary statistics, data-access assumptions, and exact sample-size regime before claiming reproduction.

## Important statistical constraints

1. **Whole sprinklings are the independent units.** Pairwise relations within one causal set are dependent through shared events and transitivity.
2. **Each N is calibrated separately.** Do not reuse a finite-N likelihood at another cardinality unless a scale-transfer model is derived and qualified.
3. **Calibration/qualification separation is mandatory.** No qualification seeds may tune the KDE bandwidth or priors.
4. **Prior sensitivity is explicit.** A high posterior for D≤3 under one prior is not a prior-free dimension result.
5. **Abstention is scored separately.** An out-of-support observation is evidence of model inadequacy, not a dimension estimate.
6. **The benchmark is open-world only to the extent of explicit controls.** Curved, non-Poisson and non-manifold-like causal sets are not automatically assigned a Minkowski dimension.
7. **A fixed-N sweep is not a scale-dependent quantum geometry.** Varying event count changes sample size, not necessarily the physical scale of the underlying manifold.

## Next qualification stages

### Stage 1 — posterior reliability
- Increase calibration/qualification repetitions based on Wilson coverage precision targets.
- Plot reliability diagrams and posterior rank histograms.
- Compare prior predictive and posterior predictive distributions.
- Add bootstrap uncertainty for KDE bandwidth and calibration corpus choice.
- Test coverage and scoring separately by dimension and N.

### Stage 2 — out-of-family controls
- Curved 1+1D conformal-FRW samples.
- Finite-cone and product-window sampling bias.
- Non-Poisson and clustered-event controls.
- Antichain and total-order degeneracies.
- Report typed OutsideCalibrationSupport / Unresolved rather than coercing the control into D=2,3,4.

### Stage 3 — window-scale analysis
A scale-dependent-dimension experiment must use nested Alexandrov subintervals or another explicitly defined physical window. Windows drawn from the same sprinkling are correlated; do not count them as independent trials. Calibrate the joint multi-window statistic on whole causal sets, preserve window nesting and event reuse, and use held-out *whole sprinklings* as the qualification unit.

### Stage 4 — joint signatures
Compare:
- ordering-fraction KDE;
- whole-sprinkling joint Gaussian synthetic likelihood over interval-profile plus dimension-conditioned longest-chain features;
- nonparametric or mixture likelihoods only after calibration and coverage validation.

The interval profile already contains ordering fraction information; do not append ordering fraction or the BDG linear bracket as independent features without modeling their dependence. Any covariance regularization and likelihood family must be included in the model-uncertainty study.

## Acceptance gates

- Deterministic replay from calibration and qualification seed manifests.
- Exact posterior normalization and finite-likelihood checks.
- Wilson intervals are reported for credible-set coverage.
- The D=4 false P(D≤3) rate is reported with uncertainty on sealed seeds.
- Event-ID permutation and Lorentz-boost invariance preserve the posterior within numerical tolerance.
- Out-of-family controls abstain or report explicit support failure.
- All priors, bandwidth policy, dimensions, N, replicate counts, source SHA and result digests are recorded.
- No claim of dimensional reduction, fundamental dimension, quantum gravity or continuum-limit recovery from this synthetic benchmark.

## Claim ceiling

A pass establishes only finite-N posterior behaviour for the calibrated fixed-cardinality Minkowski sprinkling family. It does not establish the dimension of physical spacetime. The correct conclusion can be: **the observation is outside the calibrated model family; no dimension should be assigned**.
