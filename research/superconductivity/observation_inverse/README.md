# Specimen-Linked Multimodal Inverse Modelling

**Contract:** SC-OBSERVATION-INVERSE-0.1  
**Status:** synthetic observation/split auditor; no inverse solution or material parameter posterior is claimed  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12811

## Why this exists

A coarse-grained superconductivity model should not simply fit one curve. The calibration observations should constrain parameters, and the model should then predict an independent measurement modality or condition from the same specimen.

This workstream connects:
- #12677: pairing / coherence evidence and same-specimen observation semantics;
- #12755: fixed-amplitude XY phase networks;
- #12779: quantum phase-only rotor;
- #12793: complex amplitude-phase GL model;
- #12651: application and engineering translation.

The data contract is intentionally stricter than a generic table: the top-level material/specimen/batch/state/region identity binds each observation to the same specimen context. Process history is separate from the measurement conditions. Each observation retains its own temperature, field, pressure, doping, strain, geometry, time, method, calibration, uncertainty, source and raw-data digest.

## Observable separation

Distinct observation modalities include:
- amplitude-sensitive spectroscopy;
- shot-noise pair-transfer;
- phase correlations and Josephson response;
- helicity modulus / superfluid density;
- vortex imaging;
- heat capacity;
- resistivity;
- magnetic and thermodynamic transitions.

A coarse-grained GL value \(|\psi|\) is not automatically an STM gap, a shot-noise pair-transfer energy, or a bulk superfluid density. Any mapping from a model variable to a measured quantity must be defined in a versioned observation operator with units, nuisance parameters and calibration uncertainty.

The raw-data digest may be shared by multiple derived observations. Each observation has a unique identity even when several analyses originate from one instrument acquisition. Geometry IDs are observation-specific; the top-level specimen region provides the sample/region link.

## Leakage-resistant calibration / holdout

The first split contract supports `leave_one_modality_out`. It requires:
1. a non-empty calibration set;
2. a non-empty held-out set;
3. every observation assigned exactly once;
4. no duplicate IDs or calibration/holdout overlap;
5. exactly one held-out modality;
6. that modality absent from calibration observations.

A later fitter must train only on the calibration observations, preserve the pre-registered split, and score the held-out modality with a predictive distribution. A good in-sample fit is not evidence of predictive validity.

Other split modes (leave-one-condition-slice-out, leave-one-method-out, custom) are enumerated in the schema but are not admitted by this first auditor yet. They require their own explicit leakage semantics before support is added.

## Executable input contract

The Rust auditor:
- validates the versioned JSON schema;
- requires a synthetic-only fixture for this initial stage;
- checks unique observation identities;
- checks finite numerical values and non-negative standard uncertainties;
- validates raw-data SHA-256 syntax while permitting shared digests;
- records condition variants rather than merging measurements across temperature/field/pressure;
- requires UTC timestamps and verifies that split pre-registration precedes every observation;
- validates a complete disjoint calibration/holdout partition;
- rejects unknown, duplicate or overlapping observation IDs;
- rejects a held-out modality that appears in calibration;
- validates the declared two-parameter observation-operator row;
- records exact schema and fixture SHA-256 values in the receipt.

The split auditor is an evidence / leakage check only. A separate conjugate Gaussian fitter now performs the first inverse update below.

## Benchmark ladder

### A. Synthetic parameter recovery
Generate GL observation datasets from frozen known parameters, then infer those parameters with an independent observation operator. Add declared measurement noise and nuisance calibration uncertainty. Evaluate parameter recovery, nominal interval coverage, simulation-based calibration rank histograms, posterior correlations, Fisher-information singular values and posterior predictive residuals.

### B. Same-specimen modality holdout
Pre-register a calibration subset and hold out another modality or field/temperature slice. Fit only on calibration. Evaluate held-out log predictive density, calibrated interval coverage and standardized residuals.

### C. Modality ablations
Compare amplitude-only, stiffness-only, amplitude+stiffness, amplitude+stiffness+vortex and full multimodal calibration. Measure the incremental improvement in held-out predictive performance and parameter identifiability.

### D. Model comparison
Compare fixed-amplitude XY, full amplitude-phase GL and quantum phase-only QXY on data where their assumptions apply. Permit model discrepancy and abstention rather than forcing a bad fit.

### E. Value of information
Rank candidate next measurements by their expected reduction in posterior uncertainty and expected improvement in held-out prediction. Validate those rankings against simulated follow-up acquisitions.

## Proposed targets (research targets, not results)

- Nominal 90% posterior intervals attain 85–95% coverage on frozen synthetic parameter-recovery cases.
- Held-out modality prediction is calibrated and beats a declared simpler baseline on the same specimen cohort.
- Leave-specimen, leave-batch, leave-lab and leave-family tests are reported; random row splits alone are insufficient.
- Every parameter is labelled identifiable, weakly identifiable or non-identifiable, with posterior width/correlation and nuisance sensitivity shown.
- Missing observations remain missing; local probes cannot be promoted into bulk evidence.
- Every posterior/prediction is bound to source observations, raw data digests, split manifest, priors, model version, observation operators and exact code/configuration.

## Claim ceiling

This is a specimen-linked data contract and split auditor. No Bayesian inverse solution, parameter posterior, calibrated material property, Tc, bulk superconductivity or experimental conclusion is established. It is not a synthesis protocol.


## First executable inverse baseline: conjugate linear Gaussian

The second binary, `sc_observation_inverse_fit`, is a transparent two-parameter inverse baseline. It is intentionally not yet a nonlinear lattice-GL forward solver.

For a parameter vector \(\theta\in\mathbb R^2\), prior \(\theta\sim\mathcal N(\mu_0,P_0^{-1})\), and declared linear observation operator \(y_i=h_i^T\theta+\epsilon_i\), with independent known Gaussian errors \(\epsilon_i\sim\mathcal N(0,\sigma_i^2)\), the calibration-only posterior is

\[
P_n=P_0+\sum_{i\in\mathcal C}\frac{h_i h_i^T}{\sigma_i^2},
\qquad
\mu_n=P_n^{-1}\left(P_0\mu_0+
\sum_{i\in\mathcal C}\frac{h_i y_i}{\sigma_i^2}\right).
\]

For each held-out observation \(h_*\), the predictive distribution is

\[
y_*\mid\mathcal C\sim
\mathcal N\left(h_*^T\mu_n,
\sigma_*^2+h_*^T P_n^{-1}h_*\right).
\]

The receipt reports posterior mean/covariance, marginal normal intervals, held-out predictive distributions, standardized residuals and total log predictive density. It binds the calculation to the exact schema and fixture hashes and records that held-out observations were not used in the posterior.

### Synthetic fixture example

The synthetic parameter names are `amplitude_scale` and `phase_stiffness_scale`. The calibration split contains amplitude-sensitive spectroscopy and heat capacity; the held-out modality is vortex imaging. The fixture's observation-operator rows are declared in the data, not inferred or silently chosen by the fitter.

With the current synthetic numbers, the calibration-only posterior mean is approximately

\[
\mu_n\approx(0.91981,\;0.11016),
\]

and the held-out vortex proxy prediction is approximately \(0.07348\), with predictive standard deviation approximately \(0.02746\). These are synthetic test outputs from a deliberately simple two-parameter linear model—not physical estimates of a superconducting material.

A key negative-control test changes the held-out vortex value: the posterior mean and covariance must remain bit-for-bit unchanged, while the held-out predictive score changes. A separate test changes a calibration observation and requires the posterior to change.

### Explicit limitations

This Gaussian baseline assumes independent observations, known fixed standard uncertainties and a linear observation operator. It does not yet model shared calibration nuisance parameters, cross-modal covariance, nonlinear observation maps, posterior multimodality, model discrepancy, censoring, instrument resolution kernels, or material-specific GL simulation outputs. It is a mathematical calibration/leakage baseline, not a validated physical inverse model.

The next inference gates are synthetic simulation-based calibration, nominal 90% interval coverage, prior/likelihood misspecification tests, and then replacing the synthetic linear operators with forward observables generated from independently qualified GL simulation ensembles.
