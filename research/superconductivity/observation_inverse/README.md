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
- validates a complete disjoint calibration/holdout partition;
- rejects unknown, duplicate or overlapping observation IDs;
- rejects a held-out modality that appears in calibration;
- records exact schema and fixture SHA-256 values in the receipt.

The receipt is an evidence / split audit only. It does not perform a Bayesian fit, return a posterior, or infer any material parameter.

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
