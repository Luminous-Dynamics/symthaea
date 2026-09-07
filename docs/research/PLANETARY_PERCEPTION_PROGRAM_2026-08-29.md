# Symthaea Planetary Perception Research Program

**Status:** proposed / evidence-first  
**Date:** 2026-08-29  
**Scope:** Earth observation, multimodal planetary inference, semantic downlink, and carefully bounded subsurface reasoning

## 1. Research question

Can Symthaea combine heterogeneous remote and in-situ observations into a useful, uncertainty-aware model of physical regions while preserving the distinction between:

1. what an instrument directly observed;
2. what deterministic processing derived;
3. what a model hypothesized or inferred;
4. what later evidence verified or refuted?

A secondary question is whether a compact semantic representation can reduce transmission/storage cost **without reducing mission-relevant information below an explicit bound**.

A third question is whether multimodal observations can improve inference about hidden subsurface state. This is an inverse problem, not a license to describe every radar or geophysical anomaly as direct underground imaging.

## 2. Load-bearing epistemic invariants

These rules apply before any model architecture or benchmark score.

### PP-I1 — observation is not inference

A calibrated observation may support a hypothesis. It does not become the hypothesis merely because the correlation is strong.

### PP-I2 — indirect is not direct

Surface deformation, thermal response, vegetation stress, gravity anomalies, magnetic anomalies, resistivity structure, and most satellite-radar signatures may support subsurface inference. They must not be labelled direct subsurface observation unless the acquisition method has an explicit, validated direct-penetration bound for the claimed depth and conditions.

### PP-I3 — no universal radar penetration depth

C-, L-, P-, or another radar band does not imply a fixed penetration depth. Medium, moisture, conductivity, wavelength, incidence geometry, calibration, processing, and validation all matter.

### PP-I4 — contradictions survive

Conflicting observations are retained as `EvidenceConflict`-like objects. Do not average disagreement away before asking whether it reflects sensor failure, temporal mismatch, scale mismatch, model failure, or genuine physical complexity.

### PP-I5 — confidence is not truth

Model confidence, posterior probability, similarity, HDC distance, or policy score is never promoted to ground truth without independent verification.

### PP-I6 — no autonomous coercion from orbital evidence

Remote observations may support alerts, prioritization, scientific inquiry, maintenance, and requests for local verification. They do not by themselves justify sanctions, deprivation of rights, policing, or other coercive action.

### PP-I7 — every derived result carries lineage

A derived result must be traceable to source products, acquisition time, footprint, calibration/processing steps, software/version, parameters or parameter digest, masks/quality, and uncertainty.

### PP-I8 — learned components cannot redefine evidence semantics

Models may score, rank, retrieve, predict, or propose hypotheses. They cannot silently change whether an observation is direct/indirect, its units, its acquisition identity, or its provenance.

## 3. Layered architecture

```text
providers / instruments
        |
        v
provider bridges
        |
        v
Earth-observation contracts
        |
        +---------------------+
        |                     |
        v                     v
deterministic features      content payloads
        |                     |
        +----------+----------+
                   |
                   v
         evidence-bearing state
                   |
        +----------+-----------+
        |          |           |
        v          v           v
     Morphos    HDC/SVMP    domain models
        |          |           |
        +----------+-----------+
                   |
                   v
          hypotheses/inference
                   |
        +----------+-----------+
        |                      |
        v                      v
 next-best observation     Mycelix evidence
        |                      |
        v                      v
 instruments / field      distributed review
 verification
```

Provider endpoints, credentials, and HTTP semantics do not belong in domain science crates. Large imagery/raster payloads do not belong in Mycelix entries; evidence records reference content-addressed payloads.

## 4. Initial sensor matrix

| Modality | Primary direct information | Potential hidden-state value | Default subsurface claim mode |
| --- | --- | --- | --- |
| Optical / multispectral | surface reflectance | vegetation/soil/mineral proxies | indirect only |
| Hyperspectral | surface spectral response | mineral/chemical proxies | indirect only |
| Thermal IR | surface temperature/radiance | shallow thermal/process proxies | indirect only |
| C/L/P-band SAR | calibrated backscatter / phase-derived products | moisture, structure, deformation; conditional shallow penetration in validated cases | surface by default; indirect unless specifically validated |
| InSAR | relative surface line-of-sight deformation | subsidence, uplift, underground extraction/injection/processes | indirect |
| Lidar / DEM | surface geometry | collapse/subsidence/structure proxies | indirect |
| Gravity | gravitational field anomaly | density contrasts | indirect inverse problem |
| Magnetics | magnetic field anomaly | magnetization/geology | indirect inverse problem |
| Airborne EM | electromagnetic response | conductivity/resistivity structure | indirect inverse problem |
| Electrical resistivity | electrical response | resistivity structure | indirect inverse problem |
| GPR | reflected radar response | shallow interfaces/objects where penetration is validated | direct only within validated acquisition bounds |
| Seismic | wavefield response | subsurface velocity/reflectivity structure | interpretation/inversion required |
| Borehole / excavation / local instrument | local physical measurement | high-value verification | direct within measurement support |

The table is a default reasoning policy, not a substitute for per-acquisition calibration.

## 5. Phase A — evidence contracts

### A1. `symthaea-earth-observation`

- provider-neutral identity
- geographic support
- modality
- calibrated sensitivity
- uncertainty
- processing lineage
- evidence-stage taxonomy
- contradictions
- direct-vs-indirect subsurface guard

### A2. deterministic feature math

Start with low-ambiguity, unit-tested transformations:

- NDVI
- explicitly named NDWI formulations
- NBR
- SAR linear/dB conversion
- explicitly typed polarimetric ratios/differences
- mask propagation
- quality propagation

Do not embed provider calibration assumptions in these functions.

## 6. Phase B — offline-first Sentinel witness

Build the live path only after a frozen path exists.

1. frozen Sentinel-1 GRD metadata/product fixture;
2. frozen Sentinel-2 L2A metadata/product fixture;
3. product digests and source manifest;
4. deterministic AOI/window extraction;
5. Sentinel-2 feature extraction;
6. Sentinel-1 calibrated backscatter feature extraction;
7. paired Wetland Watch observation;
8. complete offline replay.

CI must not require credentials or Internet access.

### Wetland Watch witness

The first end-to-end witness should combine:

- Sentinel-2 vegetation/water indices;
- Sentinel-1 SAR water/flood evidence;
- explicit cloud/quality masks;
- local/in-situ verification where available;
- an evidence report that distinguishes observation, feature, hypothesis, and verification.

Success is not visual attractiveness. Success is reproducible, quantitatively correct evidence with known uncertainty.

## 7. Phase C — visual memory and semantic downlink

Reconcile the preserved visual-compression research lineage before integrating it. Do not build new satellite claims on the public Alpha.8 crate while a newer preserved lineage exists outside `main`.

Compare at least three arms:

- **A:** conventional compression only;
- **B:** conventional compression + simple cloud/change/ROI policy;
- **C:** B + Symthaea semantic/HDC sidecar and utility-aware prioritization.

C must materially beat B to justify the cognitive layer.

Primary metric:

> mission-relevant information delivered per transmitted byte and per joule

Secondary metrics include latency to first useful information, peak RAM, CPU/energy, reconstruction error, index error, anomaly recall, false alarms, and query-without-decode performance.

## 8. Phase D — subsurface synthetic benchmark

Do not begin with uncontrolled archaeological or geological anecdotes. Begin where ground truth is exactly known.

### D1. Hidden world

Generate a 2D/3D world containing configurable:

- stratigraphic layers;
- cavities/voids;
- aquifers;
- conductive/saline regions;
- density anomalies;
- magnetic bodies;
- faults;
- buried channels;
- infrastructure-like objects;
- time-varying extraction/injection/subsidence processes.

The inference system never receives the hidden state directly.

### D2. Forward observations

Generate physically documented synthetic observations for selected modalities. Each forward model gets an independent validation fixture before being used to judge inference.

Initial order:

1. simple gravity anomaly benchmark;
2. simple magnetic anomaly benchmark;
3. surface deformation from a known hidden process;
4. simplified resistivity/EM benchmark;
5. only then more complex radar/seismic cases.

### D3. Baselines

Before HDC or a cognitive model is credited, compare against conventional baselines appropriate to the task:

- direct threshold/rule baselines;
- least-squares/regularized inversion where applicable;
- conventional Bayesian/probabilistic model;
- simple feature fusion;
- random/uninformed next-sensor policy.

### D4. Evaluation

Measure separately:

- target presence/absence;
- localization error;
- depth error;
- geometry/volume error;
- uncertainty calibration;
- false-positive rate;
- confusion among alternative explanations;
- information gain from added modalities;
- next-best-observation efficiency;
- robustness to corrupted/missing/contradictory sensors.

A model that localizes a target but is severely overconfident has not passed.

## 9. Phase E — active scientific investigation

Once passive inference is calibrated, evaluate whether Symthaea can choose useful next observations.

Given competing hypotheses such as:

```text
H1: cavity
H2: groundwater change
H3: anthropogenic excavation
H4: processing artefact
```

the system should rank candidate measurements by expected reduction in uncertainty subject to cost, latency, accessibility, and safety.

Potential actions are observation requests, not autonomous excavation or intervention.

Evaluation compares information gained per unit cost against fixed and random sensor-selection policies.

## 10. Phase F — real-world subsurface validation

Promote only after synthetic and controlled tests pass.

Preferred progression:

1. published/open datasets with known ground truth;
2. controlled test sites with known buried targets;
3. sites with borehole/GPR/geological verification;
4. prospective blinded study where predictions are frozen before field verification.

A retrospective fit to a known site is exploratory evidence, not prospective validation.

## 11. Mycelix integration

Mycelix should hold the distributed evidence/review graph, not raw imagery.

A future adapter should map the specialized Symthaea observation into the shared Mycelix World observation envelope once that shared type lineage is on the live target branch.

Store/reference:

- subject/region;
- phenomenon;
- observed and recorded times;
- source/provider/product identity;
- uncertainty and quality;
- evidence links;
- content digests/payload references;
- review/verification/contradiction state.

Do not create a second universal observation ontology if Mycelix World already owns that shared interoperability layer.

## 12. DTN / orbital integration

Only after the Earth witness is useful should the same evidence be scheduled across constrained links.

Progressive classes:

- L0 alert;
- L1 semantic/evidence synopsis;
- L2 preview;
- L3 region of interest;
- L4 complete compressed product;
- L5 archival/raw product.

Under disruption, the receiver must know what was received, what remains absent, what dependencies are missing, and whether a claim is still justified.

Test long partitions, reordering, duplicates, expiry, storage pressure, restarts, corruption, and clock uncertainty.

## 13. Promotion and kill gates

### Promote Earth-observation contracts when

- construction/validation invariants are unit-tested;
- provider-independent replay passes;
- units/masks/uncertainty are explicit;
- no known route upgrades indirect subsurface evidence to direct evidence.

### Promote Sentinel witness when

- frozen S1/S2 products replay byte-for-byte from a manifest;
- deterministic features match independent reference calculations;
- masks and missing data do not become numeric placeholders;
- Wetland Watch produces reproducible evidence outputs.

### Promote semantic downlink when

- cognitive arm beats codec + simple ROI baseline on held-out scenes;
- scientific utility remains within preregistered error bounds;
- compute/energy cost does not erase transmission savings.

Otherwise keep conventional compression and kill or narrow the cognitive codec claim.

### Promote subsurface inference when

- it beats conventional baselines on held-out hidden worlds;
- uncertainty is calibrated;
- alternative explanations remain visible;
- performance survives missing/corrupted/contradictory sensors;
- prospective real-world validation is eventually successful.

Otherwise retain the system as a hypothesis generator, not a detector.

## 14. Planned PR train

```text
PP-01  Earth-observation evidence contracts
PP-02  deterministic optical feature semantics
PP-03  calibrated SAR feature semantics
PP-04  offline-first Sentinel catalogue/evidence bridge
PP-05  frozen S1/S2 fixtures + source manifest
PP-06  deterministic raster/window pipeline
PP-07  Wetland Watch reference witness
PP-08  Morphos Bioregion Steward adapter
PP-09  Mycelix World evidence adapter
PP-10  visual-compression lineage reconciliation
PP-11  multiband semantic visual-memory adapter
PP-12  codec/ROI/cognitive benchmark harness
PP-13  progressive DTN evidence profile
PP-14  hidden-ground-truth subsurface benchmark
PP-15  conventional geophysical baselines
PP-16  multimodal inference + contradiction tests
PP-17  next-best-observation experiment
PP-18  controlled/open real-world validation
PP-19  orbital contact/storage/energy integration
PP-20  preregistered evidence campaign
```

Each PR should be independently reviewable and should avoid silently importing claims from later stages.

## 15. Non-goals

This program does not begin by claiming:

- general underground vision from orbit;
- a universal radar penetration model;
- autonomous geological truth discovery;
- state-of-the-art image compression;
- operational flight qualification;
- coercive decision automation from satellite evidence;
- that HDC must outperform conventional methods.

The program is successful if it discovers **where these methods help and where they do not**, with a reproducible evidence trail either way.
