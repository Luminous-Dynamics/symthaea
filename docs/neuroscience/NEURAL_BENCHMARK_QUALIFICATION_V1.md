# Neural Benchmark Qualification v1

Status: **qualification boundary / historical benchmark quarantine**

This document defines the scientific authority boundary for the historical
`symthaea-psych-bench::benchmarks::neural_validation` experiments.

The purpose of quarantine is not to discard useful computational work. It is to
prevent a model-behavior experiment, synthetic fixture, transform sanity check,
or self-authored hypothesis from being reported as external neural evidence.

## Core invariant

> A benchmark name, correlation coefficient, neuroscience citation, or brain-like
> representation does not grant neural-evidence authority.

A public external-neural benchmark must establish, at minimum:

1. explicit evidence authority (synthetic, simulated, external surrogate, or observed),
2. immutable source/model/dataset identity and version,
3. native coordinate system,
4. reviewed and digest-bound coordinate transforms for every spatial remapping,
5. no silent synthetic fallback,
6. an analysis whose conclusion is not built into its own synthetic construction,
7. claim scope no stronger than the admitted evidence use,
8. mechanism-execution integrity independent of evidence provenance.

The intended future chain is:

`source artifact -> validated provenance -> admitted evidence use -> coordinate lineage -> mechanism integrity -> statistics -> bounded claim`

## Current public qualification state

**No historical neural-validation benchmark is currently qualified as public
external-neural evidence.**

The implementations remain crate-private for regression, inspection, and repair.

## Historical benchmark matrix

| Benchmark | Current authority | Why it is quarantined | Promotion requirement |
|---|---|---|---|
| `CorticalSimilarity` | synthetic/model behavior by default; legacy external schema unqualified | `load_or_generate_predictions()` silently falls back to hand-generated 12-region pseudo-TRIBE predictions; the simulated response uses a related hand-authored regional profile; legacy `region_activations` have no qualified native-surface transform lineage | remove implicit fallback; ingest provenance-qualified external surrogate data; preserve `fsaverage5`; apply reviewed `fsaverage5 -> Glasser360 -> Symthaea12` mapping; require explicit `SurrogateAlignment` admission |
| `TemporalDynamics` | simulated transform/mechanism test | constructs its own 31 Hz cortical sequence, injects visual/auditory events, then checks the response after applying the canonical HRF; despite historical prose, it does not compare against independent TRIBE temporal data | move/reclassify as HRF transform qualification, or compare against independently sourced temporal neural data with explicit timing/provenance |
| `BidirectionalValidation` | synthetic model behavior | historical prose says TRIBE predicts fMRI from Symthaea output, but implementation calls a keyword-driven `simulate_tribe_prediction_for_text()` heuristic | replace pseudo-TRIBE function with admitted external surrogate inference; bind exact generated text/stimulus identity and model revision; preserve native output coordinates |
| `SubstrateComparison` | synthetic hypothesis exploration | biological/silicon/neuromorphic/quantum activation biases are hand-authored, then evaluated against predictions that may themselves be synthetic; expected substrate behavior is partly encoded by construction | keep only as explicitly synthetic sensitivity analysis, or design an independent substrate experiment; it cannot support substrate-consciousness ranking |
| `ParcellationRobustness` | invalid as an atlas-robustness test | loops across Glasser/DK/Schaefer labels but computes the same 12-region simulated and prediction maps for every atlas; atlas identity changes metric naming/bootstrap seed, not the data transform | implement actual atlas-specific source mappings and aggregate independently before comparison; bind each atlas artifact and mapping digest |
| `EvidenceUpgrade` | forbidden authority transition | historically attempted to convert cortical similarity into substrate-consciousness evidence; PR3 forbids this transition at the core framework | do not restore as a scientific benchmark; replace with invariant tests proving neural resemblance cannot authorize consciousness/substrate evidence |
| `EegValidation` | synthetic transform consistency | converts both sides from the same kind of 12-region activation representation through `activations_to_eeg()` and correlates the derived values; no observed or independently predicted EEG is consumed | reclassify as EEG-transform unit/behavior test, or ingest observed/admitted EEG in sensor space with preprocessing provenance |
| `HybridSubstrate` | synthetic hypothesis exploration | `hybrid_optimal` and `hybrid_inverse` profiles are hand-designed according to the hypothesis being evaluated; test then checks that the preferred profile performs at least as well | retain only as explicitly synthetic parameter exploration; independent substrate evidence would require externally justified profiles and non-circular outcomes |

## Additional legacy hazards

### Implicit synthetic fallback

`load_or_generate_predictions()` makes absence of external data look like a
successful benchmark run by generating five synthetic regional predictions.
This is acceptable only for an explicitly named synthetic fixture experiment.
It is not acceptable for a benchmark presented as comparison with external TRIBE
predictions.

A future external benchmark must fail closed or return an explicit
`UNQUALIFIED_NO_EXTERNAL_EVIDENCE` result when qualified data are unavailable.

### Source labels are not provenance

Legacy `CorticalActivationMap::ActivationSource` values such as `FmriPredicted`
and `FmriObserved` are descriptive labels on a low-level representation. They do
not establish model identity, dataset identity, coordinate lineage, source digest,
or evidence-use admission.

The intended design is to keep `CorticalActivationMap` as a representation
payload and carry scientific authority outside it, e.g.:

`NeuralObservation<CorticalActivationMap>`

followed by an explicit evidence-use admission capability.

### Atlas names are not transformations

A result is not cross-parcellation evidence merely because the same values are
reported under several atlas names. Every atlas comparison must actually derive
its payload through the declared atlas/mapping and bind the exact mapping source.

### Neural similarity is not consciousness evidence

Representational similarity may eventually support a bounded neural-alignment
claim. It does not establish subjective experience, substrate independence, or
consciousness. No Pearson/cosine/RSA threshold may directly upgrade those claims.

## Promotion gates

A historical benchmark may return to the public qualified surface only when all
applicable gates are satisfied.

### NBQ-001 — Explicit evidence authority

Every analyzed neural artifact has validated authority. Synthetic fixtures and
simulated model states cannot enter surrogate or empirical analyses.

### NBQ-002 — No implicit fallback

A request for external or empirical evidence fails closed when that evidence is
unavailable. Synthetic data require an explicit synthetic experiment mode/name.

### NBQ-003 — Native coordinates preserved

External neural data remain in their native coordinate system until an explicit
reviewed transform is applied.

### NBQ-004 — Transform lineage complete

Every spatial remapping carries source/target coordinate systems, transform id,
version, and cryptographic mapping/atlas digest. The chain resolves exactly from
native to current coordinates.

### NBQ-005 — Independent target

The benchmark target must not be generated from the same hand-authored priors,
heuristics, transformation, or preferred profile whose success is being tested.

### NBQ-006 — Modality truth

fMRI, EEG, MEG, and iEEG claims require evidence in the stated modality. A
heuristic conversion from a synthetic regional activation map is a transform
experiment, not observed modality validation.

### NBQ-007 — Evidence-use admission

The consumer requires an admission capability appropriate to the claim:
model behavior, surrogate alignment, or empirical neural analysis. No neural
artifact receives automatic consciousness-inference authority.

### NBQ-008 — Mechanism integrity

The experiment independently proves that the declared computational mechanism
actually executed, using the Symthaea Evidence Plane or equivalent hard-failing
instrumentation.

### NBQ-009 — Claim bounded by evidence

Result names, notes, provenance, dashboards, and papers must distinguish model
behavior, external-surrogate alignment, and observed empirical analysis.

### NBQ-010 — Deterministic qualification

Given exact source artifacts, mapping artifacts, code revision, configuration,
and seeds, qualification/rejection is reproducible.

## Recommended repair order

1. Preserve this quarantine while PR2's canonical Rust provenance/admission types qualify.
2. Remove implicit synthetic fallback from the external cortical-similarity path.
3. Implement provenance-aware TRIBE artifact ingestion in native `fsaverage5`.
4. Implement and qualify a real `fsaverage5 -> Glasser360` mapping.
5. Implement and qualify `Glasser360 -> Symthaea12` only as a declared lossy regional projection.
6. Restore a bounded external-surrogate similarity experiment under `SurrogateAlignment` authority.
7. Move HRF and activation-to-EEG logic into explicit transform/mechanism qualification tests.
8. Add observed human fMRI/EEG experiments separately; never treat surrogate predictions as observations.
9. Only then add representational-similarity analysis between Symthaea/HDC geometry and human neural geometry.

## Non-goals

This profile does not claim that the quarantined experiments are useless, that
TRIBE-style surrogate comparison is invalid in principle, or that Symthaea
cannot model aspects of human cognition. It only prevents current synthetic or
coordinate-unqualified experiments from carrying stronger scientific authority
than their construction supports.
