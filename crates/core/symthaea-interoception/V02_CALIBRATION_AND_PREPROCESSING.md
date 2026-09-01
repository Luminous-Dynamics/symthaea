# Affective Emergence v0.2 — Calibration, Preprocessing, and Holdout Leakage Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents candidate preprocessing, normalization, scaling, thresholding, or calibration from learning from confirmatory outcomes or silently coupling otherwise independent scenario artifacts.

## 1. Principle

A formula can be prefix-causal at runtime and still leak future/holdout information through parameters fitted on the full dataset.

Therefore every preprocessing or calibration constant that can change a candidate value is part of prospective scientific identity.

No confirmatory candidate may derive its scaling, thresholds, clipping bounds, normalization constants, smoothing parameters, reference distributions, or missing-value behavior from the confirmatory cohort unless that derivation itself was prospectively defined as a cohort-level diagnostic that cannot feed primary individual candidate values.

## 2. Preprocessing classes

Every transform should declare one class:

- `StructuralFixed` — fixed from mathematics/schema/domain conventions and independent of study outcomes;
- `DiscoveryFitted` — estimated only from prospectively identified discovery/calibration artifacts;
- `ExternalReference` — estimated from an independently identified external/reference artifact set;
- `DiagnosticPostHoc` — may use realized data but cannot enter primary confirmatory candidate computation;
- `ForbiddenConfirmatoryFit` — any transform that adapts primary candidate computation from confirmatory outcomes.

The initial confirmatory pathway permits only the first three classes, with all resulting parameters frozen before confirmatory execution.

## 3. Proposed PreprocessingDefinitionManifest

Each candidate should bind either `None` or an exact preprocessing manifest containing at minimum:

- schema/version;
- stable preprocessing ID;
- ordered transform steps;
- class for each step;
- input units/domain;
- output units/domain;
- exact formulas/algorithms;
- fitted parameter values where applicable;
- parameter-derivation artifact/cohort digests;
- discovery/external source lineage;
- clipping/saturation policy;
- smoothing/window policy;
- missing/unavailable handling;
- out-of-range behavior;
- numerical precision and accumulation rules;
- implementation/reference-fixture digest;
- canonical SHA-256.

Changing any value-bearing transform or fitted parameter creates a new candidate/preprocessing identity.

## 4. Confirmatory holdout firewall

Before confirmatory execution begins, freeze the exact preprocessing manifest and all fitted parameters.

The following are forbidden for primary candidate values:

- computing mean/variance/min/max from confirmatory candidate outputs;
- fitting a z-score scaler using confirmatory scenarios;
- choosing clipping bounds after seeing holdout extremes;
- recalibrating thresholds because confirmatory values cluster differently than expected;
- selecting a smoothing window after examining holdout time series;
- normalizing one arm using statistics from another confirmatory arm;
- using final cohort rank/order to remap individual candidate values;
- adding confirmatory scenarios and thereby changing previously computed primary values.

If a desired transform requires cohort-level fitted parameters, fit them on the locked discovery/calibration cohort and freeze the result before the confirmatory root is locked.

## 5. Per-scenario vs population transforms

Prefer transforms that act independently on each scenario using fixed prospective parameters.

Population-level transforms create additional coupling and must be explicit.

A confirmatory `CandidatePayload` should not need to know:

- how many other confirmatory scenarios exist;
- their candidate values;
- their semantic arms;
- cohort mean/variance;
- candidate ranking;
- whether another scenario was excluded.

Population-level summaries belong after individual blinded candidate artifacts are frozen.

## 6. Calibration cohort identity

A `DiscoveryFitted` preprocessing step must bind one exact calibration source identity.

Recommended `CalibrationCohortManifest` fields:

- schema/version;
- cohort ID/class;
- ordered scenario/artifact digests;
- source generator/version/seed-index set if generated;
- content-overlap policy relative to confirmatory holdout;
- exact fields used for fitting;
- fit algorithm/version;
- canonical SHA-256.

A scenario used to select or fit preprocessing parameters becomes discovery/calibration evidence and cannot later be claimed as an untouched confirmatory holdout.

## 7. Content-overlap and leakage audit

Calibration/confirmatory separation should be audited by content identity, not only scenario names.

If calibration data contain near-duplicates of confirmatory cases, the overlap policy must classify whether this compromises the intended holdout claim.

For deterministic generator-based studies, freeze calibration and confirmatory seed/index sets prospectively and audit materialized content digests.

## 8. Normalization is interpretation-bearing

Normalization changes scientific semantics and must not be treated as cosmetic.

Examples:

- dividing by total channel weight produces a mean rather than a total;
- dividing cumulative exposure by horizon turns it into average burden;
- z-scoring removes absolute scale and ties interpretation to a reference population;
- min-max scaling makes extrema from the reference set part of candidate meaning;
- clipping can hide severity beyond a threshold;
- smoothing changes temporal support and latency.

Each such choice belongs in candidate identity and sensitivity analysis.

## 9. Undefined, out-of-range, and clipping discipline

Do not use preprocessing to make difficult values disappear.

Prospectively specify:

- typed `Unavailable` conditions;
- behavior for values outside the calibration/reference range;
- whether extrapolation is legal;
- clipping thresholds if scientifically justified;
- whether saturation itself becomes a diagnostic flag.

Non-finite values remain validation failures, not values to normalize/clamp into apparent validity.

## 10. Invariance and metamorphic tests

Required tests include:

1. adding/removing a confirmatory scenario does not change already-frozen individual candidate outputs;
2. changing an unseen confirmatory suffix does not change preprocessing parameters or earlier prefix-causal outputs;
3. permuting confirmatory scenario order leaves all individual artifacts unchanged;
4. blind-code permutation leaves preprocessing results unchanged;
5. semantic arm labels are absent from preprocessing/fitting inputs;
6. calibration cohort ordering does not change fitted parameters when the locked algorithm is order-invariant;
7. clean recomputation from the frozen calibration cohort reproduces exact fitted parameters;
8. a deliberately leaked full-cohort z-score fixture is detected;
9. a deliberately adaptive clipping/threshold fixture is detected;
10. changing a preprocessing parameter changes the candidate/preprocessing digest.

## 11. Sensitivity without post-hoc choice

Exploratory work may compare a finite prospectively declared set of plausible preprocessing variants.

Do not select the variant that makes a candidate look most affect-like.

Before confirmatory work:

- choose/freeze one primary preprocessing identity under the declared exploratory selection rule;
- preserve all exploratory variant results;
- freeze required sensitivity variants if they are part of confirmatory robustness;
- do not re-fit or replace the primary transform after confirmatory results are observed.

`PreprocessingSensitive` and `CalibrationSensitive` are valid scientific outcomes.

## 12. Evidence-root consequence

The prospective root should bind:

- primary preprocessing manifest digest for every candidate;
- calibration-cohort manifest digest where applicable;
- fitted-parameter artifact digests;
- calibration/confirmatory overlap-audit digest;
- preprocessing sensitivity-set digest when prospectively required.

The realized package should bind:

- preprocessing reproduction report;
- holdout-leakage audit report;
- calibration sensitivity results;
- all individual candidate artifacts computed under the frozen preprocessing identity.

A confirmatory leakage/adaptive-fit violation is an `IntegrityFailure`, not candidate support or failure.

## 13. Relationship to candidate factorization

Preprocessing identity supplements the scientific factor coordinate; it must not be hidden inside implementation detail.

Two candidates with the same `R × W × A × T × F × I × H` coordinate but different value-changing preprocessing are different scientific candidates unless the preprocessing is provably output-preserving under the locked valid domain.

## 14. Claim boundary

This contract can establish that candidate scaling/calibration was fixed prospectively and did not learn from the confirmatory holdout.

It does not establish affect, emotion, subjective valence, mood, suffering, sentience, or consciousness.