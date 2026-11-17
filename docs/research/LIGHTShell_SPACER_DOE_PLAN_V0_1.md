# Lightshell spacer textile design-of-experiments v0.1

**Issue:** Symthaea [LIGHTSHELL-SPACER-DOE-001 #12724](https://github.com/Luminous-Dynamics/symthaea/issues/12724)  
**Parent:** [ROBOT-LIGHTSHELL-001 #12688](https://github.com/Luminous-Dynamics/symthaea/issues/12688)  
**Owner:** apply existing [TEXTILE-INVERSE-DESIGN-001 #11746](https://github.com/Luminous-Dynamics/symthaea/issues/11746) and [SCI-EXP-001 #6226](https://github.com/Luminous-Dynamics/symthaea/issues/6226)  
**Status:** proposed, capability check and sample procurement pending  
**Claim ceiling:** experimental design only; no samples, physical tests or protective rating claimed

## 1. Why the design has to be multi-objective

Yan, Wang and Yu (6 January 2026) investigated 12 inlaid weft-knitted spacer fabrics. Their paper reports that square-wave silicone inlays can increase compression energy absorption and force reduction, whereas some higher-frequency spacer-course arrangements offer better vibration isolation but lower impact-force absorption. The square-wave samples also introduced additional resonance peaks. The paper did not test laundering or durability.

Source: https://doi.org/10.3390/polym18020151

Therefore Lightshell must not seek one scalar "best spacer". We should design a **functional material portfolio**:
- compression/impact-oriented local pads;
- vibration-isolation-oriented sections;
- breathable/thermal sections;
- low-bulk articulation panels;
- optional local hybrid reinforcement.

A result from one zone cannot be extrapolated to another without matching the load, frequency, curvature, mounting and environmental conditions.

## 2. Experimental stages

### Stage 0 — capability and control preflight

Before the design matrix is frozen:
- confirm the actual knitting/weaving equipment, gauge, available feeders/attachments and machine-program semantics;
- confirm whether elastomeric tubes/inlays can be fed, secured and inspected on the available equipment;
- establish process capability and repeatability for the base spacer;
- select real sample lots and preserve fibre/yarn identity;
- define the intended target zone and its use profile;
- freeze specimen dimensions, conditioning, test fixtures, calibration and uncertainty.

The cited paper used a specific V-bed knitting machine, polyester textured yarn, spandex, polyester monofilament and hollow silicone tube. Those details are a **reproduction comparator**, not an assumed Luminous machine recipe. No ownership or access to the cited machine is assumed.

### Stage 1 — matched controls

| ID | Control | Comparison purpose |
|---|---|---|
| C0 | Plain spacer textile, no inlay | Structural baseline |
| C1 | Outer textile face/liner without spacer | Textile-only baseline |
| C2-T | Conventional foam/elastomer pad, equal thickness | Material-thickness matched control |
| C2-M | Conventional foam/elastomer pad, equal areal mass | Lightweighting matched control |
| C3 | Available commercial spacer textile, if an ODD-matched article can be sourced | Commercial construction baseline |

C2-T and C2-M are separate comparisons. Equal thickness does not imply equal mass; equal mass does not imply equal thickness. Report both when the controls can be produced.

### Stage 2 — structural screening matrix

Use four inlay architectures at three spacer-connection frequencies, holding transition geometry constant for the square-wave families. This yields 12 screening structures:

| Architecture | Low spacer frequency | Reference frequency | High spacer frequency |
|---|---|---|---|
| A0 — no inlay | A0-L | A0-R | A0-H |
| A1 — float/continuous inlay | A1-L | A1-R | A1-H |
| A2 — square-wave inlay | A2-L | A2-R | A2-H |
| A3 — square-wave + anchoring tuck | A3-L | A3-R | A3-H |

The level definitions (machine courses per unit length, spacer monofilament count/density, tuck/needle transition spacing) must be explicit and equipment-capability checked before manufacture. "Low", "reference" and "high" are categorical placeholders until the actual machine, gauge and achievable process window are admitted.

If a construction is unsupported by the installed equipment, mark it `NOT_MANUFACTURABLE`; do not silently substitute a different stitch or process.

For A2 and A3 in the screening set, hold transition geometry at the reference setting (two-needle spacing in the symbolic fixture) across all three spacer-frequency levels. This prevents transition spacing and spacer frequency from being confounded.

### Stage 2B — transition-geometry follow-up

Only after the screen identifies viable square-wave families, compare one-, two- and three-needle transitions at the **same reference spacer frequency**:
- A2-T-one/two/three: square-wave without the extra anchoring tuck;
- A3-T-one/two/three: square-wave with the extra anchoring tuck.

This adds six follow-up structures. The full proposal therefore contains 12 screen conditions and 6 conditional transition-refinement conditions (18 total candidate definitions). If the knitting machine cannot reliably make a geometry, mark it not-manufacturable rather than changing the recipe silently.

The 12 screen structures are an exploratory screening set, not a statistically complete characterization of every yarn, material, finish, thickness and process parameter. A follow-on response-surface or sequential design should vary the most influential factors after the screening stage.

### Stage 3 — blocking, replication and holdout

- Block by manufacturing lot, machine setup, operator/shift where relevant, and material lot.
- Randomize specimen manufacture order within each feasible block.
- Randomize test order within fixture constraints.
- Use separate matched specimens for destructive impact tests and for the repeated-flex/cleaning campaign.
- Determine confirmatory sample size from pilot variance and a preregistered minimum effect/precision requirement; do not infer a reliable effect from technical repeats of one specimen.
- Retain a held-out confirmatory manufacturing lot or batch. Do not use it to fit the surrogate or choose the winning design.

For early feasibility screening, technical replicates may estimate measurement repeatability, but they do not establish manufacturing-lot generalization. The final protocol must distinguish specimen, panel, manufacturing lot and repeated measurement as separate units.

## 3. Measurement domains and separate protocols

### A. Physical dimensions and identity
Measure sample thickness, areal mass, stitch/course/wale density, fibre/yarn identity, inlay geometry, surface orientation, relaxed dimensions and moisture conditioning. Capture both pre- and post-conditioning state.

### B. Compression response
Record the complete stress–strain loading and unloading curves, with fixture geometry, platen alignment, rate, preconditioning and cycle history:
- initial/tangent stiffness over preregistered strain bands;
- energy absorbed per specimen volume and per areal mass;
- hysteresis/loss;
- recovery and permanent compression set;
- buckling, lateral bulging, frictional locking and structural collapse.

Do not report "energy absorption" without stating whether it is work to a selected strain, unloading loss, per-area, per-volume or per-mass.

### C. Dynamic impact/compression
Record impactor mass, velocity, shape, contact geometry, support, specimen condition and calibrated measurement boundary. Measure transmitted peak force, impulse, pulse duration, displacement, energy transfer and damage.

The January 2026 paper's ball-drop force-reduction percentage is specific to its fixture. Reproduction can support a research comparison, but it is not a robot impact-protection standard and cannot establish safe human contact.

### D. Vibration
Measure frequency-response/transmissibility using a declared preload, mounting, excitation and frequency range. Record resonant peaks, attenuation bands, damping and amplitude dependence. Add a separate robot-relevant frequency profile after the exact actuator/structure excitation spectrum is measured. Do not assume the paper's hand-arm fixture is equivalent to a robot joint.

### E. Airflow and thermal behavior
Measure air permeability/resistance and relevant heat/moisture transfer. For robot fit, evaluate how the installed garment changes motor/drive temperature under matched duty cycles; coupon breathability alone is not a thermal qualification.

### F. Durability and cleaning
The 2026 spacer study left washability and durability untested. Close this gap explicitly:
- repeated compression and flex;
- abrasion, seam and inlay retention;
- elastomer cracking, migration, leakage or detachment;
- laundering only where domestic wash is the intended care profile;
- industrial cleaning only with exact chemicals, pressure, temperature and exposure declared;
- dimensional, mass, airflow and mechanical changes after each exposure;
- remeasure compression, impact and vibration after aging.

Use a declared lifecycle profile and preregister checkpoints. A sample that performs well when new but degrades after cleaning/flex is not a successful robotic product candidate.

## 4. Specimen allocation — do not reuse destructive coupons

Assign matched sample pieces before testing:
- non-destructive physical/airflow measurements;
- compression curve and recovery;
- vibration sweep;
- impact/ball-drop (potentially destructive);
- flex/compression aging followed by repeat measurements;
- cleaning/aging followed by repeat measurements.

If one test permanently conditions or damages a specimen, it must not silently become the baseline for another response. All splits must retain parent panel/lot lineage.

## 5. Analysis and active experiment selection

The initial screening should use blocked/factorial comparisons and confidence intervals. If interactions matter (inlay × spacer frequency, inlay × face construction), preserve them rather than fitting a one-factor-at-a-time story.

Then use the existing MAT-ACTIVE/MAT-UQ/TEXTILE-INVERSE-DESIGN capabilities to propose the next experiment by **decision value**:
- Does the next test distinguish impact-oriented from vibration-oriented families?
- Is an inlay's performance gain large enough to justify extra mass, process complexity, repair and recycling burden?
- Is the apparent benefit robust across lots and after the intended cleaning/flex profile?
- Is a candidate outside the measured domain and therefore requiring new physical coupons?

### Adaptive fabric comparison (AVT-Fabric research comparator)

The AVT-Fabric project demonstrates an RGB-first, tactile-when-needed fabric comparison approach, reporting 98% accuracy on 400 held-out comparisons with 1.60 of five observation stages on average. Its real-robot demonstration reports 78.1% pairwise ranking accuracy and 7/8 correct final selections. The project's metrics are scoped to seen-fabric question answering and its robot demonstrations, with model-side latency excluding physical acquisition.

Source: https://zhuochenn.github.io/AVT-project/

Lightshell can investigate the same **adaptive evidence principle**:
1. record a rights-cleared image and exact sample identity;
2. request a property comparison (e.g. surface texture, perceived structure) only as a triage signal;
3. acquire calibrated tactile/mechanical observations when the decision is uncertain or the property needs measurement;
4. store the actual force/displacement/response curve, not only the model's label;
5. stop when the next observation is unlikely to change a predeclared design decision, or abstain when outside domain.

Do not treat appearance as a measurement of impact attenuation, compression stiffness, air permeability or durability. Physical observations remain the reference evidence.

## 6. Decision policy

First apply hard feasibility constraints:
- machinability and stable process capability;
- no prohibited material/chemical exposure;
- sufficient repeatability and traceable sample identity;
- no detaching or migrating inlay that can create a robot snag/foreign-object hazard;
- no unacceptable thermal, fit, ROM, sensor or service regression when integrated.

Then maintain a **role-specific Pareto set**, not a single winner. A sample may be optimal for vibration attenuation and poor for impact response; that is useful information for zone allocation.

Report failures, inconclusive samples, unavailable process paths and uncertainty. Never remove a disappointing candidate after seeing the outcome.

## 7. Comparison to industry and SOTA

Run two different comparisons:
1. **Textile science comparison:** reproduce the paper's structural families as closely as actual equipment and materials allow, reporting every deviation.
2. **Application comparison:** compare against equal-thickness and equal-mass foam/elastomer, commercial spacer fabric, textile-only and Lightshell hybrid modules under an identical robot-zone profile.

The paper's reported >78% force reduction is a paper-specific result, not our target. A better claim would be a matched, held-out, lot-replicated improvement in force transmission per areal mass, while retaining vibration, airflow, durability and service performance.

## 8. Data contract

The accompanying machine-readable seed manifest lists 12 proposed architectures and five controls. Each result must bind:
- sample, specimen, parent panel and material-lot identity;
- yarn/fibre, knit/weave construction and finish;
- machine, gauge, program and process revision;
- process preflight result and deviations;
- specimen conditioning, orientation and geometry;
- test fixture, protocol, instrument and current calibration;
- raw-data digest, units, uncertainty and analysis-code version;
- randomized/block/replicate IDs;
- acceptance decision and reviewer.

The seed manifest remains `proposed`: no machine, sample, material lot, instrument, calibration, observation or pass is filled in.

## 9. Claim boundary

This protocol does not establish an impact protection rating, human-contact safety, PPE compliance, robot collision safety, washability, flame resistance, ingress protection or integrated robot qualification. It only defines a proposed research campaign.

## 10. Executable matrix admission

The Lightshell-specific structural validator is implemented in:
`crates/domains/symthaea-engineering/src/lightshell_spacer_doe.rs`

It is exposed as:
`symthaea_engineering::lightshell_spacer_doe::parse_and_validate`

Example:
```rust
use symthaea_engineering::lightshell_spacer_doe::parse_and_validate;

let report = parse_and_validate(manifest_json)?;
if !report.is_valid() || !report.matrix_complete {
    return Err("Lightshell spacer DoE matrix rejected".into());
}
```

The validator checks:
- all five named controls exist and have unique identities;
- required inlay, spacer-frequency and transition-geometry factor definitions exist;
- the 12 screening cells cover every inlay architecture × spacer-frequency combination once;
- screen-stage transition geometry is not applicable for no-inlay/float and fixed at two needles for both square-wave families;
- six transition-refinement cells cover both square-wave families × one/two/three needle spacing;
- transition-refinement frequency remains fixed at the reference level;
- candidate identities are unique and candidate values are declared by their factor levels;
- destructive testing is allocated to separate specimens;
- preregistration requires qualified process capability, a fixed randomization seed, blocking variables and a confirmatory holdout;
- the proposed seed cannot claim material lots or process revisions.

Eight Rust unit tests are authored, including mutations that deliberately reintroduce factor confounding, remove a matrix cell, duplicate a candidate, omit a factor or assign material/process evidence to the proposed fixture. These tests are not yet executed. Focused command:
`cargo test -p symthaea-engineering lightshell_spacer_doe -- --nocapture`

This is not a general DOE engine or a statistical analysis engine. It validates the Lightshell v0.1 experiment-design contract and delegates design generation, randomization, inference and active experiment choice to the existing SCI-EXP/MAT-ACTIVE/MAT-UQ/TEXTILE-INVERSE-DESIGN owners.
