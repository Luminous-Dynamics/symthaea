# QCOMP-RT-001A — room-temperature quantum evidence contract

Status: source/evidence contract only; no physical-backend or room-temperature-quantum-computer claim is made.

Tracks: #5806, #5807, #5809.

## Purpose

Freeze the terminology, source identity and evidence planes required before Symthaea models or proposes room-temperature quantum hardware.

The contract exists to prevent these common inference errors:

```text
ambient qubit host
!= fully ambient system

coherence
!= gate
!= multipartite entanglement
!= logical qubit
!= scalable computer

quantum sensor
!= quantum computer

simulation agreement
!= physical hardware validation
```

## Existing repository ownership

`crates/domains/symthaea-quantum-comp` remains the canonical local research scaffold for quantum and quantum-inspired probes, including conservative claim boundaries, statistics, provenance, receipts, experiment manifests, reporting, replay, QASM export and audit helpers.

This program must reuse those surfaces where they fit.

The existing crate explicitly does **not** claim quantum advantage, physical entanglement execution or validated hardware backends. QCOMP-RT preserves that boundary.

PHOT-ENG remains the owner of ordinary photonics/laser engineering. Materials/device owners remain authoritative for physical material evidence. CIV-BOOT owns productive-capability dependency closure rather than quantum physics.

## Evidence record principles

Every literature/experimental row must preserve:

- source ID;
- publication/version date;
- DOI or stable URL;
- platform family;
- physical device/register identity if disclosed;
- measured vs simulated distinction;
- exact temperature profile by subsystem;
- qubit/register size;
- control/readout method;
- directly reported observables;
- reported uncertainty/error bars where present;
- extraction state;
- limitations;
- replication/device-to-device status;
- claim ceiling.

Missing values remain missing.

## Temperature profile

Never encode one `room_temperature=true` field.

At minimum preserve separately:

- qubit-host/material temperature;
- control-electronics temperature;
- optical source temperature;
- detector temperature;
- resonator/cavity temperature where applicable;
- magnetic subsystem requirements;
- vacuum/trap apparatus temperature where applicable;
- readout-chain temperature;
- any cryogenic subsystem.

Suggested disposition vocabulary:

- `AmbientQubitHostObserved`;
- `AmbientRegisterControlObserved`;
- `AmbientMultipartiteEntanglementObserved`;
- `PartiallyAmbientSystem`;
- `CryogenicSubsystemRequired`;
- `FullyAmbientProfileNotEstablished`.

Do not create `RoomTemperatureQuantumComputerEstablished` from a component-level result.

## Evidence planes

Preserve independently:

1. state initialization;
2. state readout;
3. coherence/dephasing;
4. single-qubit control;
5. two-qubit control;
6. multipartite entanglement;
7. register-level benchmark;
8. device-to-device repeatability;
9. error-correction/logical-qubit evidence;
10. algorithm execution;
11. strongest classical comparison;
12. scalability/integration evidence;
13. fabrication/device-placement maturity;
14. system-level thermal dependence;
15. useful sensing evidence;
16. useful networking evidence.

Evidence in one plane does not automatically populate another.

## Frozen 2025–2026 anchor registry

### RTNV-2025-12-QV8

Source:

Tom Jäger et al., "Modeling quantum volume using randomized benchmarking of Room-Temperature NV center quantum registers", npj Quantum Information, published 29 December 2025; volume 12, article 6 (2026).

DOI/source:

- https://doi.org/10.1038/s41534-025-01164-0
- https://www.nature.com/articles/s41534-025-01164-0

Reported claim surface to freeze:

- NV-center diamond quantum register operating at room temperature;
- experimentally calibrated error model;
- register connectivity and single-/multi-qubit performance characterization;
- modeled/estimated quantum volume of 8.

Required claim ceiling:

```text
room-temperature register benchmark evidence
!= useful quantum advantage
!= logical qubit
!= scalable fault-tolerant computer
```

### RTNV-2026-09-GHZ4

Source:

Joseph D. Minnella et al., "Single-gate, multipartite entanglement on a room-temperature quantum register", Nature Nanotechnology, published 14 September 2026.

DOI/source:

- https://doi.org/10.1038/s41565-026-02254-6
- https://www.nature.com/articles/s41565-026-02254-6

Reported experiment/profile fields to freeze:

- nitrogen-vacancy center in diamond;
- central electron plus nearby 13C nuclear spins;
- ambient/room-temperature operation;
- parallelized multi-qubit entangling control using dynamical-decoupling sequences;
- four-qubit GHZ state;
- reported parallel four-qubit gate duration: 14.8 microseconds;
- reported parallel four-qubit gate fidelity: 0.92(4);
- sequential comparison reported at lower fidelity;
- entanglement verification through multiple-quantum-coherence measurements rather than unrestricted full-state tomography.

Required claim ceiling:

```text
verified four-qubit room-temperature multipartite entanglement
!= logical qubit
!= fault tolerance
!= scalable quantum computer
```

### RTPHOT-2026-CV-REVIEW

Source:

Rachel N. Clark et al., "Integrated photonics for continuous-variable quantum optics", Nature Photonics 20, 489–503 (2026), published 27 April 2026.

DOI/source:

- https://doi.org/10.1038/s41566-026-01864-9
- https://www.nature.com/articles/s41566-026-01864-9

Use only as a review/evidence map for integrated continuous-variable quantum photonics. The review highlights the attraction of room-temperature deterministic sources and high-efficiency detectors for CV state generation/measurement, but review-level platform promise is not a fabricated Symthaea device claim.

### RTPHOT-2026-PROGRAMMABLE-REVIEW

Source:

Igor Aharonovich, Kenneth B. Crozier and Dragomir Neshev, "Programmable integrated quantum photonics", Nature Photonics 20, 254–265 (2026), published 19 February 2026.

DOI/source:

- https://doi.org/10.1038/s41566-025-01830-x
- https://www.nature.com/articles/s41566-025-01830-x

Use as a programmable-photonics architecture/research reference, not as evidence that all source/detector/control dependencies can operate at room temperature simultaneously.

### MEM-HDC-2026

Cross-program research anchor only:

Yi Huang et al., "Hyperdimensional in-memory computing with analogue memristive crossbar arrays", Nature Communications 17, 9162 (2026), published 28 July 2026.

DOI/source:

- https://doi.org/10.1038/s41467-026-76067-5
- https://www.nature.com/articles/s41467-026-76067-5

This source belongs primarily to NEURO-SILICON rather than QCOMP-RT. It is retained here only to prevent accidental conflation of analogue/memristive HDC with quantum computation.

```text
analogue in-memory computing
!= quantum computing
```

## Platform families

### RT-Q1 — NV diamond

Track separately:

- electron-spin qubit;
- nuclear-spin memory/register qubits;
- hyperfine coupling profile;
- optical initialization/readout;
- microwave/control-pulse requirements;
- permanent/electromagnetic field profile;
- photon collection/readout efficiency;
- defect/register selection and placement;
- device-to-device reproducibility.

### RT-Q2 — SiC and other solid-state spin defects

No transfer from NV diamond is automatic. Material, defect, optical transition, spin coherence, gate/control and fabrication evidence remain source/profile-specific.

### RT-Q3 — integrated continuous-variable photonics

Track separately:

- source/squeezing mechanism;
- interferometer/network;
- modulation/control;
- detector;
- feed-forward;
- loss budget;
- phase/frequency stabilization;
- packaging/integration;
- subsystem temperatures.

### RT-Q4 — integrated discrete-variable photonics

Track single-photon/source, interference, switching, detector, feed-forward and loss/error budgets independently from CV systems.

### RT-Q5 — room-temperature-apparatus trapped-ion or related systems

Use subsystem-explicit terminology. A room-temperature vacuum chamber/electrode package does not imply that every relevant physical degree of freedom or component is an ordinary ambient-temperature subsystem.

### RT-Q6 — emerging molecular/organic/other spin platforms

Research registry only until reproducible multi-qubit control evidence justifies a stronger class.

## Extraction states

Every numeric/qualitative datum should declare one of:

- `DirectlyReported`;
- `DerivedFromReportedValues`;
- `DigitizedFromFigure`;
- `ReconstructedFromDescription`;
- `ReviewSummaryOnly`;
- `Missing`;
- `ConflictingAcrossSources`.

Derived or digitized values must never be serialized as though directly measured by the source.

## Model-admission rule for #5809

An NV simulation/model subject is admissible only when all parameters used in the claimed comparison are assigned a provenance class:

1. directly reported;
2. derived from reported values;
3. bounded nuisance parameter;
4. external independent source;
5. unknown.

Unknown values cannot silently become best-fit parameters.

If agreement requires unconstrained hidden parameters, the correct result is `ModelUnderdetermined`.

## Comparison dispositions

Recommended model/literature comparison outcomes:

- `ReproducedWithinDeclaredTolerance`;
- `PartiallyReproduced`;
- `MismatchObserved`;
- `InsufficientPublishedParameters`;
- `ModelUnderdetermined`;
- `OutOfProfile`.

Do not reduce a multi-observable paper to one `reproduced=true` bit.

## Contradictions and later evidence

A later paper does not rewrite an older source row.

Instead:

```text
source lineage A
+ source lineage B
+ differing device/profile/result
-> preserve both
-> compare applicability/profile
-> rerun affected model subject
```

Conflicting evidence is first-class.

## Quantum-advantage boundary

Quantum advantage requires a separate claim with:

- exact computational task;
- exact physical hardware/profile;
- exact quantum runtime/resources;
- strongest credible classical comparison;
- end-to-end I/O/control/readout overhead;
- uncertainty;
- reproducibility.

Neither coherence time, entanglement, quantum volume nor algorithm novelty alone establishes useful advantage.

## CIV-BOOT projection

Room-temperature quantum remains optional advanced research infrastructure.

Possible explicit capability dependencies include:

- high-purity diamond/SiC or photonic substrate;
- defect creation/implantation/annealing;
- nanofabrication;
- lasers and precision optics;
- microwave/RF generation and control;
- magnetic-field generation/alignment;
- detectors/readout;
- vacuum where relevant;
- precision timing/synchronization;
- conventional control compute;
- metrology/calibration.

```text
quantum model local
!= quantum device local
!= quantum fabrication local
```

## Advancement gates

### Gate A — source matrix qualified

Requires:

- exact source identities;
- explicit temperature/subsystem fields;
- evidence-plane separation;
- missing/unknown preservation;
- extraction-state labeling;
- claim ceilings.

### Gate B — calibrated NV model

Owned by #5809.

Requires reproduction attempts against at least the frozen RTNV anchors without unconstrained post-hoc fitting.

### Gate C — design proposal

Only after Gate B may Symthaea propose improved pulse/control/register/readout candidates against the calibrated model.

### Gate D — external hardware observation

Requires an independent physical backend/laboratory evidence path. Simulation cannot satisfy it.

### Gate E — useful-computation comparison

Requires exact task and strong classical baseline; small-register entanglement alone is insufficient.

## Stop conditions

Pause or demote a claim when:

- a key subsystem temperature is unknown;
- model agreement depends on unconstrained hidden parameters;
- one observable matches while another source-bound observable materially disagrees;
- source/device identities are mixed across incompatible registers;
- scaling is inferred only from idealized simulation;
- readout/control overhead is omitted from useful-computation claims;
- a sensing result is being used to imply computing performance.

## Nonclaims

This document does not establish that a practical fault-tolerant room-temperature quantum computer exists, that Symthaea has physical quantum hardware, that quantum computation benefits Symthaea cognition, or that any listed platform has quantum advantage for a useful task.

It freezes the evidence vocabulary and source anchors needed before narrower claims can be tested.