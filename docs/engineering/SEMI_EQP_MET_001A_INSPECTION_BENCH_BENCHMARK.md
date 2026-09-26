# SEMI-EQP-MET-001A — Inspection Bench Benchmark Constitution

Parent: SEMI-EQP-MET-001 #5912
Issue: #5914

## Purpose

Freeze the first benchmark semantics for a bounded semiconductor inspection, positioning, and calibration research bench before any physical hardware design, vendor selection, or commissioning campaign.

This tranche is representation/data only. It does not select a camera, motion stage, microscope, illumination source, electrical probe, controller, or vendor. It contains no hardware dimensions or operating parameters.

## Core theorem

```text
sharp image
!= calibrated dimensional observation

commanded coordinate
!= observed/reference coordinate

repeatability
!= absolute accuracy

local field observation
!= whole-sample claim

software enhancement
!= new physical evidence

commissioned inspection bench
!= semiconductor process capability
```

## Ownership

This benchmark composes existing owners rather than creating new generic systems:

- SEMI-EQP #5889/#5911 — exact instrument/configuration and equipment evidence;
- SEMI-MET #5891 — semiconductor metrology roles and calibration composition;
- SENSE-DESIGN #5820 — task-relative sensing comparisons;
- PHOT-ENG #5668 — optical engineering where later required;
- FIELD/SE-OBS — physical observations, calibration and uncertainty;
- Mycelix — real article/evidence provenance;
- CIV-BOOT — local/import/maintenance/calibration-renewal closure.

No new physical-observation ontology, generic calibration system, machine-control framework, or semiconductor process engine is introduced here.

## Benchmark planes

### P1 — positioning evidence

Keep distinct:

- commanded-position reference;
- independently observed/reference-position evidence;
- revisit/repeatability evidence;
- remount transform identity;
- residual/discrepancy evidence.

A command is not an observation. Repeatability around an unreferenced coordinate is not absolute accuracy.

### P2 — optical / sensing evidence

Keep distinct:

- raw observation identity;
- reference/scale artifact identity;
- calibration identity and currentness;
- coverage / field profile;
- focus/alignment context where relevant;
- derived image/measurement identity;
- raw-to-derived lineage.

A derived image or scalar cannot create physical evidence that is absent from the raw observation.

### P3 — system / maintenance evidence

Keep independent:

- exact instrument/configuration identity;
- calibration burden and renewal state;
- remount sensitivity;
- maintainability/repairability refs;
- local/import dependency refs;
- evidence maturity.

There is no scalar `inspection_bench_quality` score.

## V1 dispositions and errors

Reference vocabulary:

- `QualitativeOnly`
- `RepeatabilityOnly`
- `CandidateMeasurement`
- `MeasurementQualifiedUnderProfile`
- `CalibrationStaleOrMissing`
- `CoverageLimited`
- `DiscrepancyObserved`
- `EvidenceIncomplete`
- `LineageInvalid`
- `ExecutionNotAuthorized`

Additional reference outcomes in the corpus include explicit identity, history, productive-closure, process-capability, and authority relations. A future implementation may refine type names, but it must preserve these distinctions and differentially qualify against the exact frozen corpus.

## Frozen synthetic corpus

Path:

`docs/release/evidence/semi-eqp-met-001a-synthetic-corpus-v1.json`

Canonical SHA-256:

`969508f750aedf3650b3466f463ed501b4120c7bd763a79f4157129bc05d2d02`

The corpus contains exactly 16 benign synthetic cases:

1. sharp image with no scale reference;
2. current scale/calibration with candidate dimensional evidence;
3. commanded-vs-observed position discrepancy;
4. repeatability without an absolute reference;
5. stale scale calibration;
6. center-field evidence with edge behavior uncharacterized;
7. remount-induced transform identity change;
8. software enhancement preserving the raw physical source;
9. local-region evidence unable to imply a whole-sample claim;
10. derived measurement with missing raw observation;
11. canonicalization of explicitly unordered evidence refs;
12. duplicate evidence-reference rejection;
13. recalibration preserving stale-calibration history;
14. imported subsystems preserving import dependence;
15. commissioned inspection bench not implying semiconductor-process capability;
16. benchmark results minting zero physical execution authority.

## Qualification rules

A future independent reference validator must:

1. hard-bind the exact corpus SHA-256;
2. reject schema/authority/count/case-ID drift;
3. derive expected outcomes from the evidence relationships, rather than simply echoing `expected_*` fields;
4. preserve command-vs-observation and repeatability-vs-accuracy distinctions;
5. preserve raw-to-derived lineage;
6. reject whole-sample claims from local-only coverage;
7. preserve historical calibration evidence after recalibration;
8. preserve operational-vs-productive-closure distinction;
9. reject semiconductor-process capability inference from bench commissioning;
10. mint no physical execution authority.

## Physical-campaign gate

The physical bench campaign must not begin as a claim-bearing qualification campaign until:

- this corpus is frozen and independently qualified;
- the equipment evidence contract #5911 is stable enough for exact configuration identity;
- SEMI-MET ownership for calibration/currentness is reused rather than duplicated;
- physical reference artifacts and their traceability/currentness are separately identified.

A physical prototype may be explored earlier, but exploratory observations must remain exploratory and cannot be promoted retroactively into preregistered qualification evidence.

## Prohibited content

This V1 contract contains no:

- hardware dimensions;
- optical powers;
- motion speeds, forces, or travel ranges;
- voltages, currents, frequencies, or controller commands;
- semiconductor process parameters;
- hazardous materials/process instructions;
- lithography/deposition/etch recipes;
- autonomous physical authority.

## Claim ceiling

Even a complete software PASS establishes only that the frozen synthetic benchmark semantics are reproduced faithfully.

It does not establish:

- real dimensional accuracy or spatial resolution;
- calibrated physical hardware;
- commissioned semiconductor metrology equipment;
- wafer-fab inspection performance;
- lithography or other process capability;
- semiconductor yield;
- fabrication capability;
- physical execution authority.
