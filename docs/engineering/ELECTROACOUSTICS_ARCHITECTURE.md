# Symthaea Electro-Acoustic Engineering Architecture

**Status:** EAC-000 architecture contract  
**Issue:** #5630  
**Scope:** architecture and authority only; no claim of product qualification or physical superiority

## 1. Purpose

Electro-acoustics is the first deliberately coupled physical-product pilot for Symthaea's engineering stack.

The target is not a standalone "speaker AI" and not a new simulation kernel. The target is a reusable evidence-driven design loop:

```text
requirements
    -> analytical model
    -> numerical simulation
    -> constrained multi-objective search
    -> manufacturable design
    -> calibrated physical measurement
    -> prediction/observation residual
    -> digital-twin update
    -> revised design hypothesis
```

A loudspeaker is a useful pilot because it couples electrical circuits, electromagnetics, mechanics, structural vibration, thermal behavior, acoustic radiation, room interaction, DSP/control, manufacturing, measurement, and human perception.

If the architecture works here, the same generic engineering substrate can later support motors, actuators, magnetics, power electronics, microphones, ultrasonic transducers, sensors, antennas, and other physical subsystems.

## 2. Existing ownership that must not be duplicated

Electro-acoustic work MUST reuse the existing workspace boundaries.

| Concern | Existing owner | Electro-acoustic rule |
| --- | --- | --- |
| Applied physical acoustics | `symthaea-acoustics` | Reuse; do not fork equations or material/room models. |
| Continuum acoustics/electrodynamics | `symthaea-continuum-physics` | Reuse where its fidelity is appropriate. |
| Exploratory semantic physics/HDC | core physics/HDC layers | May support analogy/search; never substitute for numerical or measured authority. |
| Closed-form circuit checks | `symthaea-circuits` | Use for analytical sanity checks and reduced models. |
| Circuit simulation | `symthaea-ngspice-bridge` through `symthaea-sim-bridge` | Adapter remains solver authority boundary. |
| Generic FEM/multiphysics | `symthaea-sim-bridge` + adapter crates | Add/extend adapters; do not put a FEM solver in electro-acoustics. |
| Engineering requirements/review | `symthaea-engineering` | Reuse requirements, evidence, review and safety workflows. |
| Digital twins | `symthaea-digital-twin` | Reuse for measured asset state and prediction residuals. |
| Geometry/fabrication | `symthaea-fabrication-kernel` | Remains the internal geometry/fabrication foundation. |
| Physical field observation | FIELD series | FIELD owns calibrated observations, raw evidence, clocks, units, frames and calibration provenance. |
| Audio/DSP realization | `symthaea-muse` and DSP layers | Reuse signal-processing primitives; electro-acoustics owns the physical transducer/system model. |
| Manufacturing provenance | Mycelix Manufacturing / Supply Chain | Extend existing BOM/work-order/provenance structures; do not create a parallel product ledger. |

### 2.1 New domain owner

A future `crates/domains/symthaea-electroacoustics` crate owns only the domain-specific coupling between those foundations.

It MAY own:

- loudspeaker/transducer parameter models;
- electro-mechanical-acoustic equivalent models;
- enclosure and acoustic-load models;
- large-signal transducer state;
- active loudspeaker/system composition;
- domain-specific constraints/objectives;
- construction of generic `SimulationRequest` / `MultiPhysicsRequest` values;
- interpretation of solver outputs into electro-acoustic domain evidence;
- mappings from calibrated FIELD observations into model-identification and discrepancy workflows.

It MUST NOT own:

- raw sensor ingestion;
- generic field observation schemas;
- generic optimization algorithms;
- CAD kernels or generic mesh formats;
- SPICE/FEM implementations;
- generic digital-twin storage;
- human-listening authority;
- hardware actuation authority.

## 3. Authority lattice

The core rule is that evidence classes are not interchangeable.

```text
DesignIntent
    |
    v
AnalyticalPrediction
    |
    v
NumericalPrediction
    |
    v
PrototypePrediction

BenchMeasurement -----------+
                            |
AcousticMeasurement --------+--> physical evidence

HumanPerceptualEvidence --------> separate perceptual axis
```

The arrows above indicate increasing physical contact with the subject, not automatic logical implication.

Required nonclaims:

```text
AnalyticalPrediction != NumericalPrediction
SurrogatePrediction != NumericalSolverExecution
NumericalPrediction != PhysicalMeasurement
SimulationConvergence != ModelValidity
AnechoicOrFieldMeasurement != InRoomResult
FlatOnAxisResponse != ControlledDirectivity
AcousticMeasurement != HumanPreference
HumanPreferenceSample != UniversalPreference
OnePrototype != ManufacturingCapability
DigitalTwinFit != CausalMechanismEstablished
```

Every result used for promotion MUST preserve:

- source class;
- exact subject/design revision;
- model/solver identity and version when relevant;
- input parameter provenance;
- uncertainty;
- convergence/validity state;
- calibration identity for physical observations;
- environment and coordinate/frame information where physically meaningful.

Unknown values MUST NOT silently become zero or nominal values.

## 4. Initial transducer model

EAC-001 should introduce a typed `TransducerModel` that is useful before any external solver is available.

The first model should support values with units, provenance and optional uncertainty for at least:

### 4.1 Electrical

- DC voice-coil resistance `Re`;
- inductance representation `Le`;
- nominal/minimum impedance metadata;
- voltage/current/power constraints;
- reference temperature for temperature-dependent quantities.

### 4.2 Motor

- force factor `Bl`;
- optional flux/motor geometry references;
- excursion/current operating envelope;
- future nonlinear `Bl(x, i)` representation.

### 4.3 Mechanical

- moving mass `Mms`;
- compliance `Cms` and/or stiffness `Kms` with explicit conversion;
- mechanical resistance `Rms`;
- one-way excursion and mechanical-clearance constraints;
- future nonlinear `Cms(x)` / `Kms(x)` / `Rms(v)` representations.

### 4.4 Acoustic

- effective piston area `Sd`;
- reference radiation/load model identity;
- front/rear acoustic-boundary references;
- optional measured or predicted sensitivity metadata.

### 4.5 Thermal

- voice-coil reference temperature;
- resistance temperature coefficient where known;
- thermal limits;
- future lumped thermal-network references.

### 4.6 Provenance

Every parameter carries a source classification such as:

```text
Assumed
Datasheet
AnalyticalDerived
NumericalDerived
Measured
FittedFromMeasurement
```

`FittedFromMeasurement` MUST retain the measurement evidence reference and fitting-method identity. A fitted value is not rewritten as directly measured truth.

## 5. Fidelity ladder

Electro-acoustic design should use a multi-fidelity ladder rather than dispatch every candidate to expensive multiphysics simulation.

### Tier 0 — invariant/validation

- units;
- ranges;
- sign conventions;
- geometry references;
- provenance completeness;
- impossible/undefined parameter combinations.

### Tier 1 — analytical/reduced-order

Examples:

- Thiele-Small/equivalent-circuit response;
- sealed/vented/passive-radiator approximations;
- piston radiation approximations;
- simple thermal networks;
- closed-form electrical filters.

Purpose: reject obviously poor candidates cheaply and provide inspectable baselines.

### Tier 2 — numerical single-domain

Examples:

- ngspice circuit simulation;
- structural/modal FEM;
- magnetostatic or magnetodynamic motor analysis;
- acoustic FEM/BEM or other field solver;
- thermal FEM.

### Tier 3 — coupled multiphysics

Examples:

- electromagnetic -> force -> structural motion -> acoustic radiation;
- electrical/thermal iteration with coil-resistance drift;
- structural/acoustic coupling;
- controller + physical plant co-simulation.

Use existing `MultiPhysicsRequest` and coupling semantics rather than introducing an EAC-specific orchestration protocol.

### Tier 4 — physical prototype

- electrical impedance;
- transfer function / impulse response;
- polar/spatial radiation;
- vibration/motion;
- nonlinear distortion/compression;
- thermal state;
- tolerance/unit variation.

Physical observations MUST enter through FIELD-compatible calibrated evidence.

### Tier 5 — human perceptual study

This is a separate axis. Human listening evidence may compare defined populations/protocols but MUST NOT be treated as a universal aesthetic truth.

## 6. Solver strategy

External solvers remain numerical authorities.

### 6.1 Circuit

`ENG-SPICE-001` should complete the existing ngspice bridge by parsing real solver output and recording exact solver/netlist provenance.

Process exit success MUST NOT imply:

- convergence;
- requested metric availability;
- physically valid operating point;
- promotion PASS.

### 6.2 FEM/multiphysics

Add a generic Elmer adapter as `ENG-FEM-001` rather than coupling `symthaea-electroacoustics` directly to a specific FEM library.

Initial qualification fixtures should be generic engineering fixtures, for example:

- steady heat conduction;
- simple elastic/modal case;
- simple magnetostatic field case.

Only after the adapter is independently qualified should EAC depend on it for driver-motor or vibro-acoustic studies.

### 6.3 Geometry and meshing

The fabrication kernel remains Symthaea's internal geometry/fabrication owner.

A generic mesh/provenance adapter may use a tool such as Gmsh, but electro-acoustics should consume solver-neutral geometry/mesh identities rather than embedding mesher-specific semantics in its canonical domain types.

## 7. Optimization contract

Do not define a single `hifi_score`.

The generic engineering layer should represent:

- design variables and domains;
- hard constraints;
- soft objectives;
- evaluation fidelity;
- evaluation cost;
- uncertainty;
- infeasible candidates;
- Pareto dominance;
- robust/tolerance-aware objectives.

Example electro-acoustic objective families include:

- listening-window response;
- directivity consistency;
- distortion/intermodulation;
- dynamic compression;
- excess group delay/ringing;
- maximum SPL and excursion margin;
- thermal margin;
- amplifier current demand;
- cabinet radiation;
- seat-to-seat variation;
- component/manufacturing tolerance sensitivity;
- mass/volume;
- BOM/manufacturing cost;
- energy efficiency.

The system should produce a Pareto set, for example:

```text
A: strongest reference-axis accuracy
B: wider listening area
C: smaller enclosure with bounded degradation
D: higher output headroom
E: lowest cost satisfying qualification thresholds
```

Humans or higher-level requirements choose among those tradeoffs.

## 8. Measurement and FIELD integration

EAC MUST NOT create another measurement envelope.

FIELD-003 is the intended observation path for acoustic/vibration/ultrasound evidence. Electro-acoustic measurement profiles should extend that architecture for:

- voltage/current frames;
- impedance sweeps;
- microphone pressure response;
- accelerometer/vibration observations;
- transducer motion/velocity observations;
- spectral and bandpower summaries;
- distortion/compression observations;
- spatial position/orientation metadata.

The EAC measurement layer begins only after the relevant FIELD observation contracts are qualified.

The important derived artifact is the model discrepancy:

```text
prediction(subject, model, parameters)
    - observation(subject, calibration, environment)
    = residual with both lineages preserved
```

Residuals are evidence. They MUST NOT be hidden by retuning the model without preserving the old prediction and the update provenance.

## 9. Digital-twin direction

A physical loudspeaker can eventually maintain a bounded digital twin using measured electrical/thermal/motional state.

Potential observations include:

- terminal voltage/current;
- inferred or measured coil temperature;
- driver velocity/displacement;
- enclosure vibration;
- room transfer-function changes.

Potential residual hypotheses include:

- thermal drift;
- suspension aging;
- driver tolerance;
- enclosure leakage;
- mechanical damage;
- sensor/calibration failure;
- model inadequacy.

The twin may trigger inspection, remeasurement, or a new simulation request. It MUST NOT gain unreviewed authority to energize or overdrive hardware.

## 10. First physical pilot

The first product-level pilot should use characterized commodity drivers and amplification.

This deliberately asks a narrower question before custom component R&D:

> Can Symthaea produce an evidence-backed high-performance active reference monitor by co-designing the system around known components?

Suggested pilot sequence:

1. freeze room/use/budget/size/output requirements;
2. build a provenance-bound component dataset;
3. search architecture/driver-count/crossover/directivity candidates;
4. simulate enclosure/structure/radiation;
5. design active crossover and protection/control;
6. size electrical/thermal path;
7. execute tolerance/robustness campaign;
8. generate fabrication/BOM package;
9. build one physical prototype;
10. measure through calibrated FIELD evidence;
11. publish prediction-vs-prototype discrepancy;
12. run separately governed human evaluation;
13. adjudicate bounded claims.

Only after this pilot demonstrates an end-to-end loop should Symthaea attempt a custom transducer motor, suspension, amplifier topology, or acoustic metamaterial as a promoted product claim.

## 11. Planned PR series

### Reusable engineering prerequisites

- **ENG-SPICE-001** — real ngspice output parser + provenance/convergence evidence.
- **ENG-FEM-001** — generic Elmer FEM/multiphysics adapter boundary.
- **ENG-MESH-001** — solver-neutral mesh provenance/identity bridge.
- **ENG-OPT-001** — generic engineering design-space / constraints / objectives / Pareto contract.
- **ENG-OPT-002** — multi-fidelity evaluation scheduling with surrogate authority separation.
- **ENG-UQ-001** — tolerance/robust-design campaign representation.

### Electro-acoustic foundations

- **EAC-000** — this authority/ownership architecture.
- **EAC-001** — canonical typed transducer parameter model.
- **EAC-002** — linear lumped electro-mechanical-acoustic model.
- **EAC-003** — enclosure and acoustic-load models.
- **EAC-004** — large-signal/nonlinear transducer state.
- **EAC-005** — geometry/structural/radiation bindings.
- **EAC-006** — coupled electro-vibro-acoustic `MultiPhysicsRequest` construction.

### Measurement

- **EAC-MEAS-001** — FIELD electro-acoustic observation profile.
- **EAC-MEAS-002** — impedance/model-identification evidence.
- **EAC-MEAS-003** — calibrated transfer-function evidence.
- **EAC-MEAS-004** — nonlinear/compression evidence.
- **EAC-MEAS-005** — spatial radiation evidence.
- **EAC-MEAS-006** — model-discrepancy evidence.

### Design

- **EAC-DES-001** — domain objective/constraint profiles.
- **EAC-DES-002** — crossover/directivity co-design.
- **EAC-DES-003** — robust commodity-component selection.
- **EAC-DES-004** — enclosure inverse design.
- **EAC-DES-005** — acoustic absorber/metamaterial inverse design.
- **EAC-DES-006** — motor magnetic design.
- **EAC-DES-007** — diaphragm/suspension structural design.
- **EAC-DES-008** — large-signal/thermal co-optimization.

### Pilot and room/twin follow-ons

- **EAC-PILOT-001A..L** — commodity-component active reference-monitor qualification chain.
- **EAC-ROOM-001..006** — room plant identification, distributed bass, constrained MIMO and adaptive room twin.
- **EAC-TWIN-001..006** — electrical/thermal/motional state estimation, bounded feedback and aging/anomaly residuals.

## 12. Mycelix integration boundary

Do not create a new Mycelix product just for audio.

Extend Mycelix Manufacturing/Supply Chain with generic physical-product evidence such as:

- immutable design revision identity;
- CAD/PCB/DSP/firmware digests;
- component lot/substitution provenance;
- serialised physical-unit identity;
- assembly/work-order binding;
- instrument/calibration identity;
- qualification receipt;
- repair/replacement/recalibration history;
- independent replication submissions.

Independent submissions do not automatically become canonical evidence. Their protocol, calibration, subject and provenance remain explicit.

## 13. EAC-000 exit gate

EAC-000 is complete when this architecture is accepted and all of the following remain true:

- no new numerical physics kernel was introduced;
- no duplicate FIELD observation schema was introduced;
- no duplicate CAD or generic optimization subsystem was introduced;
- simulation, surrogate prediction, measurement and human evidence remain non-interchangeable;
- physical measurement is not inferred from simulation success;
- the first implementation tranche can add typed transducer parameters with no external-solver dependency;
- the path to the first physical pilot is explicit;
- the same underlying engineering improvements remain reusable outside audio.

The next semantic implementation tranche should be **EAC-001**, while **ENG-SPICE-001** and **ENG-FEM-001** proceed independently as reusable engineering foundations.
