# Symthaea Systems Engineering FOSS Integration Profile v1

Status: architecture profile only. No runtime authority is established by this document.

Tracking: #3677, #3681, #3685, #3686

## 1. Purpose

Symthaea already has many bridge crates and engineering-domain crates. The next step is not to maximize the number of adapters. The goal is to make external FOSS tools and internal Symthaea subsystems compose through one evidence-bounded engineering contract.

Canonical laws:

```text
external tool execution
!= engineering evidence
!= requirement satisfaction
!= qualification
```

```text
internal Symthaea hypothesis
!= engineering relation
!= causal fact
!= authority
```

```text
more bridge crates
!= better interoperability
```

## 2. Existing substrate to reuse

The repository already contains integration boundaries including:

- `symthaea-sim-bridge`;
- `symthaea-mujoco-bridge`;
- `symthaea-gazebo-bridge`;
- `symthaea-ros-bridge`;
- `symthaea-opensees-bridge`;
- `symthaea-ngspice-bridge`;
- `symthaea-openfoam-bridge`;
- `symthaea-lean-bridge`;
- `symthaea-proof-audit`;
- `symthaea-control`;
- `symthaea-physics-bridge`;
- `symthaea-telemetry-grpc`;
- Mycelix, UI, web, and visualization bridges.

`symthaea-sim-bridge` is the preferred starting substrate for external engineering execution because it already models normalized requests/results, uncertainty, multi-physics staging, execution provenance, command execution limits, and backend registration.

Do not replace it with a second generic solver framework.

## 3. Canonical integration pipeline

```text
Symthaea subsystem or external FOSS tool
        |
        v
Typed projection / bridge adapter
        |
        v
Engineering artifact envelope
        |
        v
SE semantic graph + exact model/configuration snapshot
        |
        +--> proposal / computation / observation / candidate evidence
        |
        v
ETK admission + currentness + assurance
        |
        v
Separately authorized downstream use
```

## 4. Representation before execution

For external engineering formats and model packages, qualify representation/inspection before granting execution.

Canonical ordering:

```text
package bytes
-> identity / archive safety
-> schema / syntax inspection
-> conservative semantic projection
-> unsupported/lossy semantics report
-> qualified representation contract
-> separately qualified execution path
-> candidate evidence integration
```

Apply this ordering to at least:

- SysML v2 / KerML;
- ReqIF;
- SACM;
- RAAML;
- FMI / FMUs;
- SSP packages;
- AADL;
- STEP/CAD artifacts;
- KiCad/EDA artifacts;
- mesh/model packages.

Freeze:

```text
parser success
!= semantic completeness
!= executable safety
!= engineering correctness
```

## 5. Engineering artifact envelope

A shared artifact envelope should eventually bind:

- stable producer kind and producer ID;
- adapter version;
- tool version;
- executable/package/environment identity where applicable;
- exact SE graph snapshot;
- exact configuration/twin/model snapshot;
- input manifest and content digests;
- output manifest and content digests;
- schema, units, coordinate frame, and clock/time-base conventions;
- execution mode and determinism class;
- assumptions and declared validity envelope;
- uncertainty representation where meaningful;
- parser/normalizer identity;
- resource/execution policy;
- authority class.

Initial authority classes are descriptive only:

```text
Proposal
Computation
Observation
CandidateEvidence
```

Adapters MUST NOT mint `Verified`, `Qualified`, `Certified`, `Safe`, or equivalent authority states.

## 6. Standards projection vs internal semantics

Symthaea may use stronger internal semantics than external interchange standards, but should project to mature standards wherever possible rather than creating proprietary interchange unnecessarily.

Freeze:

```text
internal artifact envelope
!= new interoperability standard
```

Internal semantics may preserve ETK-specific authority/currentness details that external formats cannot express. Projection loss must be explicit.

## 7. Bridge classes

Prefer a small number of reusable bridge classes over one custom execution model per tool.

### 7.1 Model/interchange bridge

For standards and modeling environments such as SysML v2, ReqIF, SACM, RAAML, AADL, Capella, STEP, and related formats.

Responsibilities:

- preserve source identity;
- preserve unsupported semantics explicitly;
- preserve lexical/unit/provenance information when possible;
- make lossy projections machine-visible;
- never treat parser success as semantic correctness.

### 7.2 Simulation/co-simulation bridge

Build on `symthaea-sim-bridge`.

Responsibilities:

- normalized requests/results;
- exact input/output identity;
- solver/adapter identity;
- bounded execution;
- uncertainty and convergence metadata;
- no simulation-to-authority shortcut.

### 7.3 External execution capsule

Use only for explicitly registered engineering tools. It is not a general shell-execution facility.

A future `ExecutionCapsule` should bind at least:

- registered capability ID;
- executable/package identity and digest;
- Nix store path/derivation/lock identity where applicable;
- container/image identity where applicable;
- arguments;
- explicit environment allowlist;
- working-directory policy;
- network policy;
- input manifest;
- CPU/memory/time/output policy;
- stdout/stderr/raw-output manifests;
- parser/normalizer identity;
- preflight identity;
- postflight identity.

The existing `CommandSolver` timeout/output protections are useful substrate, but engineering evidence needs stronger environment/executable/cwd/input identity than a generic command launch.

### 7.4 Optimization/UQ bridge

For tools such as OpenMDAO and Dakota.

Responsibilities:

- design variable/objective/constraint identity;
- seed/sampling/driver/optimizer identity;
- exact evaluated configuration per design point;
- sensitivity/UQ/calibration provenance;
- explicit distinction between numerical optimum and validated design.

Detailed semantics are tracked separately in #3686.

### 7.5 Formal verification bridge

For Lean, Z3/SMT-LIB, TLA+/TLC, and later symbolic model checkers.

Responsibilities:

- exact property/model/tool identity;
- explicit backend vs verdict semantics;
- bounds/assumptions;
- counterexample/proof artifact retention;
- no formal-result-to-engineering-authority shortcut.

Detailed semantics are tracked separately in #3685.

### 7.6 Operational observation bridge

For digital twins, fabrication/metrology, ROS, real hardware, sensors, and inspection systems.

Responsibilities:

- source/device identity;
- configuration identity;
- timestamps/time base;
- calibration metadata;
- observation uncertainty;
- raw data identity;
- no observation-to-evidence shortcut.

## 8. Standards-first priority

Tool-specific integrations should be delayed when a mature standard provides the interoperability surface.

Priority standards:

1. SysML v2 / KerML API for system semantics;
2. ReqIF for requirements interchange;
3. SACM for assurance representation;
4. RAAML for risk-analysis representation;
5. FMI 3.0.2 for model exchange, co-simulation, and scheduled execution;
6. SSP 2.0.1 for composite simulation architectures and parameterization;
7. SSP-LS-Traceability 1.0.0 for simulation/process traceability and credibility metadata projection;
8. STEP/other open geometry interchange for CAD lineage where practical.

### 8.1 FMI implementation direction

Prefer a native Rust inspection path before native-code FMU execution.

A suitable implementation order is:

```text
FMU bytes
-> digest/archive validation
-> modelDescription inspection
-> FMI version/capability inventory
-> units/variables/clocks/causality projection
-> unsupported-semantics report
-> frozen reference/conformance fixtures
-> later isolated execution
```

Evaluate the current Rust `fmi` / `fmi-schema` ecosystem for FMI 2/3 parsing/import rather than writing XML schemas and FMU loading from scratch.

Native FMU code loading is a materially stronger privilege than modelDescription inspection and requires a separate execution qualification gate.

### 8.2 SSP implementation direction

SSP 2.0.1 should be the target package baseline for initial structural inspection.

Split implementation rather than one large adapter:

```text
SE-FOSS-004A  FMI package/modelDescription inspection
SE-FOSS-004B  SSP 2.0.1 package/structure/parameter inspection
SE-FOSS-004C  SSP-LS-Traceability projection
SE-FOSS-004D  interchange adversarial/conformance corpus
SE-FOSS-004E  later isolated FMI/SSP execution/co-simulation
```

Use SSP layered-standard traceability externally where it fits. Do not weaken richer internal ETK/currentness semantics to match the interchange projection.

## 9. FOSS integration priorities

### Priority 1 — high leverage

#### OpenModelica

Use primarily as a multi-domain model environment and FMI producer/consumer.

Prefer FMI/SSP integration. Add a direct OpenModelica adapter only where the standards path cannot expose needed capabilities.

Do not assume full FMI 3 behavior from the tool merely because Symthaea's representation layer supports FMI 3; tool capability/version support must be discovered and recorded explicitly.

#### OSATE / AADL

Use for analyzable cyber-physical architecture: processors, software deployment, buses, modes, timing, resources, and error models.

Prefer existing SysML v2/AADL mappings where they faithfully cover semantics. Unsupported mappings such as flows/modes in an incomplete external mapping must remain explicit rather than being guessed.

Results are computations/diagnostics until ETK admits relevant evidence.

#### OpenMDAO

Use for multidisciplinary design analysis and optimization.

Symthaea may propose variables/objectives/constraints. OpenMDAO performs numerical coupling and optimization. Every evaluated point must bind to an exact engineering configuration.

#### Dakota

Use for uncertainty quantification, calibration, sensitivity analysis, reliability, and optimization.

Sampling policy, distributions, seeds, solver identity, and exact model input must be captured. Experimental structured-input paths must be version-bound and separately qualified.

#### FreeCAD + open geometry formats

Use for parametric CAD and geometry interchange.

CAD validity does not imply manufacturability or design qualification.

#### Gmsh / SALOME

Use for reproducible meshing and pre/post-processing.

Mesh lineage must retain geometry identity, mesher/version, parameters, groups/boundaries, and quality metrics.

#### KiCad

Use for schematic/PCB design interoperability and compose it with the existing ngspice path.

ERC, DRC, schematic correctness, PCB correctness, and SPICE analysis remain separate claims.

#### Capella / Arcadia

Support read-only interoperability for organizations using Capella.

Capella semantics must not be silently collapsed into SysML semantics or Symthaea's internal ontology.

### Priority 2 — domain infrastructure

Only after the common bridge spine qualifies:

- Code_Aster / Salome-Meca for richer structural and thermomechanical analysis;
- EnergyPlus / OpenStudio for building energy systems;
- EPANET for water-distribution networks;
- open power-system analysis tools for electrical distribution/grid studies;
- additional specialized tools only where they add analysis coverage rather than duplicate an already qualified path.

### Priority 3 — system-of-systems federation

After model/traceability/execution contracts qualify, evaluate federation frameworks such as HELICS for large coupled infrastructure/system-of-systems studies.

Federation success remains simulation computation, not cross-domain system validation.

## 10. Internal Symthaea projections

The SE graph should become the integration plane for Symthaea itself.

### 10.1 Proposal/cognition producers

The following may propose graph relationships, requirements, assumptions, analyses, or review targets, but may not directly establish engineering truth:

- Broca;
- HDC;
- LTC/CfC temporal cognition;
- causal reasoning;
- semantic/episodic memory;
- global workspace/attention.

Examples:

```text
Broca text -> candidate Requirement
HDC analogy -> candidate DependsOn / common-cause hypothesis
LTC trajectory -> candidate degradation hypothesis
causal model -> candidate causal dependency
memory recall -> prior-episode reference with provenance
```

### 10.2 Native analysis producers

These should eventually emit engineering artifacts bound to exact SE graph/configuration snapshots:

- materials;
- structural;
- thermofluids;
- grid physics;
- control theory;
- circuits;
- acoustics;
- optics;
- DSP;
- operations research;
- Lean / proof-audit outputs.

Native calculations are not external-solver simulations and should preserve their own computation/evidence class.

### 10.3 Observation producers

These should emit provenance-bound observations:

- digital twins;
- fabrication/metrology;
- ROS/Gazebo/MuJoCo operational paths;
- perception/vision inspection where applicable;
- telemetry gRPC.

### 10.4 Collaboration/distribution producers

Swarm and Mycelix integration may carry:

- proposals;
- reviews;
- replicated artifacts;
- signatures;
- provenance;
- distributed observations.

Consensus or popularity MUST NOT automatically increase epistemic authority.

### 10.5 Adjacent Luminous infrastructure

Where deliberately integrated:

- Nixward may provide reproducible execution/environment identity and policy;
- Mycelix may carry replicated/signed/reviewed engineering artifacts;
- Xenia-style transcript/provenance/signing mechanisms may attest transport/process origin and integrity.

Freeze:

```text
reproducible environment
!= solver correctness

valid signature
!= content truth

consensus
!= engineering validity
```

### 10.6 Authority boundary

ETK remains the authority layer for evidence admission/currentness/assurance consequences.

The SE graph may say:

```text
this model changed
this requirement may be impacted
this evidence may require applicability review
```

It may not independently conclude:

```text
this evidence is invalid
this requirement is satisfied
this design is qualified
```

unless those conclusions are returned by the appropriate ETK/authority boundary.

## 11. `symthaea-engineering` migration direction

The existing `symthaea-engineering` facade currently owns and calls many subsystems directly. Do not rewrite it wholesale.

Migrate interaction-by-interaction toward three producer classes:

```text
ProposalProducer
AnalysisProducer
ObservationProducer
```

and one common artifact path:

```text
EngineeringManager
    |
    +--> typed proposal request
    +--> typed analysis request
    +--> typed observation
    +--> typed engineering artifact
    |
    v
SE graph / execution registry / ETK boundary
```

The facade should become an orchestrator over typed contracts rather than a location where authority semantics and domain implementations accumulate.

## 12. Execution environment and reproducibility

Where practical, external FOSS tool executions should be bound to reproducible package/environment identity.

For Nix-hosted execution, prefer binding the exact store path / derivation and relevant lock/environment identity into the engineering artifact.

Environment identity does not prove solver correctness. It proves which environment ran.

## 13. Suggested PR sequence

```text
SE-FOSS-000  integration profile (this document)
SE-FOSS-001  EngineeringArtifact envelope + capability descriptor
SE-FOSS-002  ExecutionCapsule / CommandSolver provenance hardening
SE-FOSS-003  internal Proposal/Analysis/Observation producer traits
SE-FOSS-004A FMI 3.0.2 package/modelDescription inspection
SE-FOSS-004B SSP 2.0.1 package/structure/parameter inspection
SE-FOSS-004C SSP-LS-Traceability 1.0.0 projection
SE-FOSS-004D interchange adversarial/conformance corpus
SE-FOSS-004E later isolated FMI/SSP execution
SE-FOSS-005  OpenModelica via FMI-first path
SE-FOSS-006  OpenMDAO trade-space adapter (aligned with #3686)
SE-FOSS-007  Dakota UQ/calibration adapter (aligned with #3686)
SE-FOSS-008  FreeCAD/STEP + Gmsh geometry/mesh lineage
SE-FOSS-009  KiCad + ngspice design/analysis composition
```

Formal verification follows #3685 rather than becoming another generic FOSS adapter family.

Runtime PRs that depend on the SE semantic graph remain blocked until the required lower-layer SE contracts have executable qualification.

## 14. Adapter qualification minimum

Every external engineering adapter should demonstrate:

```text
exact tool identity
+ exact adapter identity
+ exact input identity
+ bounded execution where execution exists
+ raw output identity
+ normalized output reproducibility
+ explicit units/schema
+ explicit unsupported/lossy semantics
+ negative/error corpus
+ immutable postflight
```

For package-inspection-only adapters, prove that native code is not executed as part of inspection.

For nondeterministic tools, record and test the source of nondeterminism rather than asserting determinism.

## 15. Non-goals

- no arbitrary-command execution API for Symthaea;
- no solver-output-to-ETK shortcut;
- no HDC-similarity-to-graph-authority shortcut;
- no CAD-to-fabrication-authority shortcut;
- no optimization-winner-to-design-qualification shortcut;
- no duplicated in-house solver solely to avoid integrating a mature FOSS tool;
- no proprietary replacement for mature interchange standards;
- no bridge-count metric as a measure of engineering maturity.

## 16. Closure criterion

The integration program succeeds when one exact engineering subject can move through:

```text
system model
-> native/external analysis
-> optimization/UQ
-> simulation/co-simulation
-> physical/operational observation
-> change-impact review
-> ETK evidence review
```

while preserving a machine-checkable answer to:

1. which model/configuration was analyzed;
2. which tool/version/environment executed;
3. which exact inputs and outputs were used;
4. what assumptions and uncertainty applied;
5. which semantics were unsupported or projected lossily;
6. what changed later;
7. which evidence remains applicable;
8. and what authority the resulting claim actually has.
