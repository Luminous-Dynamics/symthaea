# SENSE-DESIGN-000 — Perceptual Sensor Co-Design Architecture

Status: architecture / evidence contract only

Tracks: #5820, #5821, #5822, #5823

## Objective

Define how Symthaea may design and compare physical perceptual sensing systems without duplicating the canonical observation, calibration, quantity, fusion, active-perception, device-engineering, robotics, or manufacturing layers.

The program asks a bounded question:

> For a declared task and operating profile, can a changed sensing system produce better physical evidence and/or better task outcomes under explicit resource, calibration, manufacturing, lifecycle and robustness constraints?

It does not begin from the premise that a new sensor architecture is better.

## Ownership

SENSE-DESIGN owns sensor-system **co-design questions and comparison profiles**.

It does not own canonical physical observations.

Existing owners remain authoritative where applicable:

- FIELD / SE-OBS: physical observation/evidence;
- SE-SEM: quantity, unit, frame and time semantics;
- ENG-DEVICE: reusable device/component infrastructure and sensor-to-FIELD projection;
- ENG-SEMI: semiconductor/MEMS/device physics and future custom sensing devices;
- PHOT-ENG: optical source/detector/imaging engineering;
- ROB-INSTR / ROB-EXP / ROB-REALIZE: robotics instrumentation, experiment and as-built evidence;
- spatial sensing work: likelihood calibration, time alignment, extrinsics and dependence-aware fusion;
- active perception #607: sensing-action proposal and predicted-vs-realized information gain;
- ENG-EVID-INDEP: common-mode and evidence-independence assessment;
- NEURO-SILICON: sensor-near/event-driven acceleration;
- CIV-BOOT / PIE metrology closure: repair, productive closure and calibration continuity.

The downstream cognitive perception stack consumes admitted sensor evidence; it does not define the physical truth of the sensor.

## Required separation

The following are different propositions:

```text
sensor design exists
!= fabricated sensor exists
!= installed sensor exists
!= calibration executed
!= calibration current
!= physical likelihood/error model qualified
!= useful task improvement observed
!= improvement reproduced
!= local productive closure
```

Likewise:

```text
higher sensitivity
!= higher accuracy
!= higher information value
!= better task performance
```

and:

```text
more sensors/modalities
!= more independent evidence
```

## Sensor-system stack

Represent the complete sensing path as independently inspectable layers:

```text
physical measurand / interrogation
        ↓
transducer
        ↓
analog / optical / mechanical front end
        ↓
ADC / readout / event generation
        ↓
timing + calibration + self-test
        ↓
raw observation
        ↓
likelihood / uncertainty projection
        ↓
near-sensor representation / compression
        ↓
existing fusion / perception
        ↓
declared task outcome
```

Mechanical or optical preprocessing is part of the sensor design when it materially shapes the signal before transduction.

## Three evidence planes

### P1 — physical characterization

Examples:

- bias / calibration residual;
- sensitivity / responsivity;
- noise;
- dynamic range;
- resolution;
- bandwidth;
- latency / timing uncertainty;
- hysteresis;
- drift;
- cross-sensitivity;
- saturation / recovery;
- environmental applicability.

### P2 — task evidence

Examples:

- contact/slip detection;
- contact localization;
- motion/obstacle timing;
- state estimation;
- object/material discrimination;
- fault discrimination.

### P3 — system/resource evidence

Examples:

- power / energy;
- data volume;
- compute / memory;
- size / mass;
- calibration burden;
- manufacturing burden;
- repairability;
- metrology/reference dependence.

One plane may improve while another regresses. No plane may silently substitute for another.

## Comparison rule

SENSE-DESIGN-001 #5823 owns the generic comparison constitution.

Preferred dispositions are profile-relative and preserve raw dimensions, for example:

```text
ParetoImprovedUnderProfile
TradeoffUnderProfile
EquivalentWithinDeclaredTolerance
RegressionOnProtectedAxis
EvidenceInsufficient
ComparisonInvalid
OutOfProfile
```

No canonical universal `sensor_quality_score` exists.

## Fair baselines

A candidate sensing system must be compared against a credible incumbent under comparable conditions.

Bind where relevant:

- task / scene / contact population;
- optics / mechanical placement;
- calibration effort;
- software / preprocessing opportunities;
- resource measurement boundary;
- timing boundary;
- hardware and firmware identities;
- held-out conditions.

Candidate-only tuning or a deliberately weak baseline invalidates a broad superiority claim.

## Held-out discipline

Keep development, calibration, design-search and confirmatory evidence distinct where the claim requires independence.

After confirmatory reveal, changing any material design element begins a new generation, including:

- transducer geometry/material;
- package mechanics;
- optics;
- analog front end;
- thresholds;
- calibration model;
- preprocessing;
- task model.

Repeated samples from one physical episode are not automatically independent trials.

## First family — tactile / e-skin

SENSE-TAC-001 #5821 is the first physical design benchmark.

It is deliberately chosen because useful prototypes can exercise the complete loop without custom semiconductor fabrication:

```text
mechanical/electrode design
-> fabricated patch
-> electronics/readout
-> calibration
-> normal/shear/slip evidence
-> held-out task
-> repair / remount / drift evidence
```

Initial channels remain separate:

- normal/contact;
- shear/slip;
- vibration/dynamic cues;
- optional temperature;
- optional proximity;
- passive mechanical preprocessing.

A successful robotics-bench patch does not establish human-contact safety.

## Second family — event / HDR vision

SENSE-EVENT-001 #5822 begins with COTS event/frame/HDR cameras rather than custom sensor silicon.

It tests whether asynchronous sensing helps under profiles such as:

- fast motion;
- high dynamic range;
- low-latency obstacle/motion cues;
- sparse-change monitoring;
- tightly synchronized IMU fusion.

Negative-control profiles must include static texture/color tasks where conventional frame cameras may remain preferable.

Custom pixel/readout design is permitted only after COTS evidence identifies a hardware-limited bottleneck that cannot reasonably be removed through optics, calibration, timing or software alone.

## Active sensing

Physical sensing actions compose #607.

Examples include:

- viewpoint changes;
- illumination changes;
- focus/exposure changes;
- ultrasound excitation;
- coded optical/RF interrogation.

SENSE-DESIGN may propose or model such configurations. It does not grant physical execution authority.

```text
predicted information gain
!= authorized sensing action
!= executed sensing action
!= realized information gain
```

## Near-sensor HDC / neuromorphic compute

Sensor-near computation composes NEURO-SILICON #5804.

Possible profiles include:

- event extraction;
- sparse filtering;
- HDC encoding;
- associative classification;
- local temporal/CfC processing.

The full measurement boundary must include sensor readout, transfer, preprocessing, accelerator and downstream task.

```text
accelerator core energy reduction
!= sensing-system energy reduction
```

Derived HDC/event representations remain descendants of the original physical evidence and cannot become independent physical observations.

## Manufacturing and civilization bootstrap

For every sensor family separately track closure of:

- sensing material/transducer;
- electrodes/substrate;
- package/mechanics/optics;
- analog front end;
- ADC/readout;
- MCU/FPGA/ASIC;
- connectors/cables;
- calibration fixtures/reference standards;
- test/inspection equipment;
- firmware/driver;
- repair/replacement processes.

Useful partial closure is allowed and should be reported honestly.

```text
locally fabricated tactile patch
+ imported ADC/MCU
= useful local sensor-production capability
!= complete local sensor reproduction
```

Calibration continuity is part of productive capability: an operating sensor whose calibration route cannot be renewed may remain operational while losing evidence-qualified sensing capability.

## Candidate future families

After the tactile and event/HDR profiles prove the common integrity layer, expand selectively into:

- proprioceptive position / force / torque sensing;
- depth / ranging;
- acoustic / ultrasonic arrays;
- RF / mmWave radar;
- environmental / chemical sensing;
- photonic scientific sensors;
- magnetic/electric-field sensing.

Do not expand merely because a modality is interesting. New families should exercise a genuinely new architectural dependency or a high-value product/robotics need.

## Research invariants

1. Missing evidence remains missing.
2. Unknown calibration uncertainty does not become zero.
3. Same raw observation transformed twice does not become two independent witnesses.
4. Shared reference/ADC/clock/extrinsic failures remain visible.
5. Simulation does not mint physical sensor evidence.
6. Task labels do not become physical measurands.
7. Task performance cannot hide protected metrology/resource regressions.
8. Calibration data and held-out confirmation do not silently overlap.
9. Design improvements remain profile-relative.
10. No sensor-design artifact creates actuation authority.

## Program order

```text
SENSE-DESIGN-000 architecture
        ↓
SENSE-DESIGN-001 comparison constitution
        ↓
SENSE-TAC-001 tactile/e-skin benchmark
        +
SENSE-EVENT-001 event/HDR benchmark
        ↓
cross-domain conformance
        ↓
selected third sensing family
        ↓
custom semiconductor/device work only where evidence justifies it
```

## Claim ceiling

This architecture can support later claims such as:

> Under exact task, calibration, environment, resource and held-out profiles, candidate sensor system B showed a declared physical/task/resource improvement over baseline A without the protected regressions specified by the comparison profile.

It cannot support an unqualified claim that Symthaea designed a universally superior sensor.

## Nonclaims

This document establishes no sensor performance, manufacturing capability, calibration, physical experiment, custom semiconductor, human-contact safety, or autonomous sensing authority. It freezes ownership and evidence boundaries for later executable work.