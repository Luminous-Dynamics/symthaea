# ENG-SEMI-REF-001A — Reference-device benchmark constitution

Parent: #5868 `ENG-SEMI-REF-001`

Status: **design/evidence constitution only**

Frozen source base for this generation:

`2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9`

This document does not establish semiconductor simulation, fabrication, device characterization, or physical measurement capability. It freezes the identities, evidence ladder, mismatch rules, and adversarial cases that later implementation must satisfy.

## Purpose

Create one bounded semiconductor reference campaign in which every epistemic transition is explicit:

```text
exact commercial device/article identity
    -> exact source-document snapshot
    -> analytical semiconductor estimate
    -> TCAD numerical prediction
    -> model-discrepancy record
    -> compact-model extraction
    -> circuit-level prediction
    -> physical low-voltage observation
    -> prediction/observation residual
```

Core theorem:

```text
analytical agreement
!= TCAD validity
!= compact-model validity
!= circuit validity
!= datasheet truth for one article
!= physical measurement
!= fabrication capability
```

## Ownership and reuse

REF-001A introduces no second quantity, component, provenance, observation, calibration, simulation-execution, or authority ontology.

Reuse, when qualified and available:

- ENG-CATALOG #5675 for generic component/article/source-document identity;
- SE-SEM/FIELD/SE-OBS for quantities, observations, calibration, environment, and time semantics;
- solver-input closure and ENG-EXEC for external numerical execution identity;
- `symthaea-circuits` / ngspice for circuit-level computation;
- ENG-SEMI #5671 for semiconductor-device semantics;
- ENG-DEVICE #5670 for generic instrument/power/thermal interfaces;
- ETK/currentness owners for evidence admission/currentness consequences.

Local reference labels are adapters/placeholders only and may not become competing canonical identities.

## Benchmark generation identity

Every campaign generation binds a complete semantic input set.

Conceptually:

```text
SemiconductorReferenceGenerationV1 {
    generation_id,
    parent_issue,
    source_base_commit,
    device_subject_ref,
    physical_article_ref?,
    source_document_refs,
    analytical_profile_ref,
    tcad_profile_ref?,
    compact_model_profile_ref?,
    circuit_profile_ref?,
    measurement_profile_ref?,
    environment_profile_ref,
    quantity_profile_ref,
    held_out_policy_ref,
    authority_profile_ref,
}
```

Changing a claim-relevant input creates a new generation or explicit applicability review. Friendly labels do not preserve identity when source bytes, model assumptions, article identity, environment, calibration, or fitting policy changes.

## Evidence rung contract

```text
ReferenceSubjectDeclared
SourceDocumentSnapshot
AnalyticalDeviceEstimate
TCADNumericalPrediction
CompactModelFit
CircuitPrediction
PhysicalObservation
ResidualOrDiscrepancy
```

No evidence object can promote itself into a stronger rung.

```text
ReferenceSubjectDeclared != SourceDocumentSnapshot
SourceDocumentSnapshot != PhysicalObservation
AnalyticalDeviceEstimate != TCADNumericalPrediction
TCADNumericalPrediction != CompactModelFit
CompactModelFit != CircuitPrediction
CircuitPrediction != PhysicalObservation
ResidualOrDiscrepancy != corrected physical truth
```

A downstream evidence object references its ancestors; it never rewrites them.

## Reference-device selection gate

REF-001A freezes the selection profile, not a specific manufacturer part.

The first physical article family must satisfy all of the following before exact identity is frozen:

- commercially available silicon junction device;
- ordinary low-voltage/low-energy bench characterization is sufficient for the primary profile;
- public manufacturer documentation can be snapshotted exactly;
- no wafer processing or custom fabrication is required;
- no avalanche/breakdown characterization is required;
- no high-current/high-power operation is required;
- no hazardous optical, chemical, thermal, vacuum, or plasma exposure is required;
- the primary observable can be a simple electrical response such as bounded I-V behavior;
- optional optical, C-V, thermal, or dynamic profiles are separate generations/experiments.

Preferred first family: a simple silicon diode-style subject. A photodiode is acceptable only if optical characterization remains optional and dark electrical behavior is independently testable.

A specific part number is frozen only after exact source-document and article-availability audit. REF-001A therefore does not canonize a convenient example part prematurely.

## Exact subject identity

Distinguish at least:

```text
DeviceDesignSubject
CatalogComponentSubject
PurchasedArticle
InstalledOrTestedArticle
```

Preserve when available:

- manufacturer;
- exact ordering code / variant;
- package variant;
- source-document identity/revision/content digest;
- lot/date code/serial/marking for the tested article;
- procurement/listing snapshot separately from engineering identity;
- fixture/socket/carrier identity when it can affect measurement.

```text
same marketing family
!= same exact component
!= same article
```

## Analytical profile

REF-001B will implement independently checkable analytical relations. Every analytical profile binds:

- equation/model family identity;
- declared assumptions;
- independent-variable domain;
- environmental state, especially temperature where material;
- material/parameter source refs;
- numerical constants and units;
- applicability limits;
- derived outputs;
- unresolved terms/assumptions.

The analytical layer must be intentionally simple enough to check independently.

```text
analytical equation fits measured curve
!= physical mechanism fully identified
```

A fitted ideality factor or saturation-like parameter is profile-relative, not a universal property of every article of that part number.

## TCAD profile

REF-001C may introduce DEVSIM behind ordinary external-solver evidence boundaries.

A TCAD request must bind, where applicable:

- exact simulator/version/executable/environment;
- geometry and dimensionality;
- mesh identity/generation;
- material models and parameter sources;
- doping/profile assumptions;
- contact/interface models;
- mobility model identity;
- generation/recombination model identity;
- thermal assumption/model;
- boundary conditions;
- bias/sweep profile;
- nonlinear solver settings;
- convergence diagnostics;
- raw output identity;
- parser/normalizer identity.

```text
solver converged
!= discretization adequate
!= model-family adequate
!= physical device validated
```

Mesh or solver-setting changes that materially change prediction are discrepancy evidence and cannot be hidden by selecting the preferred run.

## Compact-model extraction

REF-001D treats compact-model construction as a projection under a declared use envelope.

Bind:

- source evidence used for fitting/extraction;
- exact fit domain;
- held-out domain;
- parameterization/model family;
- constraints/priors if any;
- optimizer/tool identity;
- residuals on fit data;
- residuals on held-out data;
- explicit unsupported/extrapolated region.

Forbidden shortcut:

```text
fit entire physical curve
-> report same curve as held-out validation
```

If TCAD and physical evidence are both used during fitting, mixed provenance must be explicit. A good compact-model fit cannot retroactively validate the TCAD mechanism.

## Circuit projection

Circuit prediction is a new evidence rung. Bind:

- exact compact-model identity;
- exact netlist/circuit topology;
- simulator/version/execution environment;
- source/load values and conditions;
- analysis mode;
- convergence/warning state;
- raw output/parser identity.

```text
ngspice converged
!= compact model physically valid
!= tested circuit article exists
```

## Physical measurement profile

The primary physical profile is deliberately benign and narrow.

The first measurement campaign should require only low-voltage, low-energy electrical characterization through existing instrumentation/authority paths.

Bind:

- exact tested article;
- fixture/carrier/socket/wiring revision;
- instrument identity;
- calibration/currentness evidence;
- measurement range/resolution/uncertainty where available;
- source/limit configuration as evidence, not authority;
- environmental state;
- raw samples;
- processing/derivation steps;
- excluded/unavailable region;
- timestamp/session/generation.

The benchmark must not require semiconductor breakdown testing, high-power operation, wafer probing, chemical processing, or custom fab equipment.

```text
configured instrument range
!= achieved physical stimulus
!= measured device response
```

## Held-out evidence policy

Distinguish:

```text
DevelopmentEvidence
FitEvidence
HeldOutValidationEvidence
ExploratoryEvidence
```

A measurement point/session/article exposed during model fitting cannot later be relabeled held-out confirmation under the same generation.

Where sample count is too small for meaningful held-out physical inference, report that limitation directly rather than inventing statistical independence.

## Discrepancy object

Disagreement is first-class output.

Conceptually:

```text
SemiconductorModelDiscrepancyV1 {
    discrepancy_id,
    left_evidence_ref,
    right_evidence_ref,
    comparison_profile_ref,
    residual_summary,
    applicability_overlap,
    candidate_explanations,
    unresolved_causes,
}
```

Candidate explanations may include parameter-source mismatch, article variation, environment mismatch, geometry assumption error, contact/interface inadequacy, mobility/recombination inadequacy, numerical sensitivity, compact-model inadequacy, circuit-context effect, measurement/fixture error, or unknown.

A discrepancy may propose follow-up experiments; it cannot select a causal explanation without evidence.

## Negative and missing evidence

Missing evidence is never an optimistic default.

```text
no current calibration -> strong physical claim unavailable
no exact datasheet snapshot -> source claim incomplete
no geometry evidence -> TCAD geometry remains assumption
no reverse-region measurement -> reverse-region physical observation unavailable
no held-out data -> no held-out validation claim
```

An unavailable region is not silently filled from simulation, datasheet typical curves, or extrapolation.

## Adversarial corpus

Machine-readable fixtures are frozen in:

`docs/engineering/semiconductor/fixtures/eng-semi-ref-001a-adversarial-v1.json`

The corpus is normative for REF-001A semantics. Changing the expected disposition of a frozen case creates a new corpus generation with rationale.

## Promotion gates

### REF-001B — analytical oracle

May begin after this constitution is reviewable. It must implement known-answer analytical fixtures without needing DEVSIM or physical hardware.

### REF-001C — TCAD adapter

Requires external-solver closure/execution provenance and must implement this TCAD evidence contract rather than defining an independent one.

### REF-001D — compact model / ngspice

Requires explicit fit-vs-held-out semantics and exact circuit projection.

### REF-001E — physical measurement

Requires an exact device/article selection, exact instrument/calibration identity, and existing physical authority paths. REF-001A itself grants no hardware authority.

### REF-001F — discrepancy/redesign study

May only compare immutable historical evidence with fresh generations; it cannot rewrite an earlier run after observing the answer.

## Authority boundary

This benchmark does not authorize:

- purchase of hardware;
- energization of instrumentation;
- semiconductor processing;
- wafer handling;
- high-voltage or breakdown tests;
- high-power tests;
- optical exposure;
- autonomous experiment execution;
- foundry submission.

All physical action remains outside the benchmark and must use existing authority/safety paths.

## Acceptance for REF-001A

REF-001A is complete when:

- every evidence rung has an explicit identity and claim ceiling;
- exact source/article/model/measurement lineage is representable without duplicate ontologies;
- the first reference-device selection profile is bounded and benign;
- fit and held-out evidence cannot be silently conflated;
- discrepancies and missing evidence remain first class;
- the adversarial corpus has deterministic expected dispositions;
- later analytical/TCAD/SPICE/measurement children can implement against this constitution without broadening authority.

## Claim ceiling

REF-001A establishes only an implementation-ready evidence constitution for a semiconductor reference-device benchmark.

It does **not** establish a selected reference part, correct semiconductor physics implementation, DEVSIM integration, SPICE model quality, physical device measurements, custom transistor/device design, PDK/tapeout readiness, semiconductor fabrication, production yield, reliability, or local semiconductor independence.
