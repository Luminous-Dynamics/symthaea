# Symthaea Engineering Roadmap

**Status date:** 2026-05-07

## Honest Assessment

Symthaea already has several primitives that map well to engineering work:

| Capability | Current code anchor | Engineering value |
| --- | --- | --- |
| Topology-aware synthesis and sheaf checks | `crates/symthaea-geodesic` | Concept topology, load-path reasoning, verified control/software generation |
| HDC semantic physics | `crates/symthaea-physics-bridge`, `crates/symthaea-physics-catalog` | Equation analogy, dimensional/symmetry search, cross-domain hypothesis generation |
| Continuous-time dynamics | `symthaea-core`, `crates/symthaea-broca`, robotics crates | Long-horizon dynamics, controls, fatigue/degradation trajectories |
| Free-energy active inference | `crates/symthaea-fep`, platform crates | Digital-twin surprise, inspection triggers, adaptive control |
| Materials and fabrication | `crates/symthaea-materials`, `crates/symthaea-fabrication-kernel` | Aging, manufacturability, geometry/toolpath loop |
| Infrastructure and robotics embodiment | `crates/symthaea-infrastructure`, `crates/symthaea-manipulator`, `crates/symthaea-vehicle`, `crates/symthaea-auv`, flight crates | Domain-specific sensorimotor loops |
| Formal bridge | `crates/symthaea-lean-bridge`, `tla/`, `lean-proofs/` | Safety cases, proof obligations, verified subsystem claims |
| CAD/geometry foundation | `crates/symthaea-fabrication-kernel` | STEP parsing, NURBS, mesh validation, STL/3MF export, toolpaths |

The main gap is not another internal physics engine. The gap is a disciplined interface layer that lets Symthaea ask for simulations, track assumptions, maintain live twins, and require proof/evidence before acting.

## Crate Setup

The engineering setup is intentionally split into four small crates:

| Crate | Tier | Purpose |
| --- | --- | --- |
| `symthaea-sim-bridge` | Tier 2 foundation | Solver-agnostic requests/results and backend traits for FEA, CFD, multibody, process, and circuit tools |
| `symthaea-digital-twin` | Tier 2 foundation | Asset state, telemetry ingestion, health, and free-energy trend tracking |
| `symthaea-formal-safety` | Tier 2 foundation | Safety cases, proof obligations, evidence status, proof-assistant-neutral interchange |
| `symthaea-engineering` | Tier 3 facade | Requirements, concepts, simulation requests, twins, and safety gates in one review package |
| `symthaea-mujoco-bridge` | Adapter | Generic MuJoCo `SimulationBackend` boundary for robotics and multibody requests |
| `symthaea-opensees-bridge` | Adapter | OpenSees `SimulationBackend` boundary for civil/structural requests |
| `symthaea-ngspice-bridge` | Adapter | ngspice `SimulationBackend` boundary for circuit and power-electronics requests |
| `symthaea-openfoam-bridge` | Adapter | OpenFOAM `SimulationBackend` boundary for CFD requests |

Root crate features:

| Feature | Purpose |
| --- | --- |
| `engineering` | Lightweight facade only, suitable for default development loops |
| `engineering-foundations` | Facade plus direct root access to sim bridge, digital twin, and formal safety crates |
| `engineering-adapters` | Dry-run adapter crates for MuJoCo, OpenSees, ngspice, and OpenFOAM |
| `engineering-full` | Engineering foundations, adapters, Geodesic synthesis, physics bridge, materials, fabrication, and infrastructure crates |

External solver adapters are separate crates so native solver dependencies never enter default builds:

| Adapter crate | Backend target | Current status |
| --- | --- |
| `symthaea-mujoco-bridge` | MuJoCo multibody dynamics | Dry-run backend; existing concrete MuJoCo work remains in robotics crates |
| `symthaea-opensees-bridge` | OpenSees structural analysis | Dry-run backend; real OpenSees process/FFI adapter pending |
| `symthaea-ngspice-bridge` | SPICE circuit simulation | Dry-run backend; C API/process adapter pending |
| `symthaea-openfoam-bridge` | OpenFOAM CFD | Dry-run backend; case-directory/process adapter pending |
| `symthaea-freecad-bridge` | FreeCAD CAD/parametric geometry | Planned; leverage existing fabrication kernel STEP/NURBS/mesh work first |
| `symthaea-elmer-bridge` | Elmer FEM multiphysics | Planned after OpenSees/OpenFOAM direction is validated |

Keep adapters out of default builds. They will carry heavy native dependencies and should compile only behind explicit features or package selection.

## Industry Tool Priorities

| Rank | Tool | Domain | Why this order |
| --- | --- | --- | --- |
| 1 | MuJoCo | Robotics and multibody dynamics | Already integrated in `symthaea-multirotor` and `symthaea-humanoid`; strongest fit for FEP/embodiment |
| 2 | OpenSees | Civil and structural engineering | High public value, especially seismic and infrastructure safety |
| 3 | ngspice | Electronics and circuits | Clean adapter path via C API or process mode; enables circuit/control co-design |
| 4 | OpenFOAM | CFD | High leverage for aerospace, energy, environmental flows; heavier integration surface |
| 5 | FreeCAD | CAD and parametric geometry | Critical for geometry workflows; best approached after stabilizing STEP/NURBS/mesh interfaces |
| 6 | Gazebo/Harmonic | Robotics and ROS ecosystems | Useful complement to MuJoCo after the generic robotics adapter pattern is proven |

## Current CAD/Geometry Work

Symthaea already has a practical geometry foundation in `symthaea-fabrication-kernel`:

- `step_import.rs`: basic ISO 10303-21 STEP parser with `CARTESIAN_POINT`, B-spline curve, and B-spline surface support.
- `nurbs.rs`: NURBS curve/surface representation, evaluation, tessellation, and line constructors.
- `mesh.rs`, `validate.rs`: triangle mesh representation and validation.
- `export.rs`: STL/3MF export.
- `toolpath.rs`, `slicer.rs`, `infill.rs`: fabrication-oriented geometry-to-machine pipeline.
- `analytical.rs`, `generative.rs`: lightweight structural/manufacturing analysis hooks.

Near-term CAD strategy: use the fabrication kernel as Symthaea's internal geometry model, then add a FreeCAD adapter for parametric CAD import/export and a later OCCT/OpenCascade route only if direct geometry-kernel access becomes necessary.

## 18-Month Roadmap

### Phase 0: Foundation, Weeks 0-4

Goal: make engineering work a first-class, testable workflow without pulling in external solvers.

- Land the four foundation crate skeletons plus dry-run adapter boundaries.
- Add examples for a bridge-span concept, a mechanism concept, a live microgrid twin, and a multi-physics request.
- Define canonical units and provenance conventions for simulation requests.
- Add `cargo check -p symthaea-engineering -p symthaea-sim-bridge -p symthaea-digital-twin -p symthaea-formal-safety -p symthaea-mujoco-bridge -p symthaea-opensees-bridge -p symthaea-ngspice-bridge -p symthaea-openfoam-bridge` to local verification.

Exit criteria:

- A concept can produce requirements, proof obligations, simulation requests, and a blocking/nonblocking review decision.
- A multi-physics request can represent ordered solver stages and coupling mode.
- No external solver is required for unit tests.

### Phase 1: Simulation Adapters, Months 1-4

Goal: make Symthaea a reasoning layer over trusted numerical tools.

- Turn the dry-run MuJoCo bridge into a shared adapter over existing multirotor/humanoid MuJoCo patterns.
- Turn the dry-run OpenSees bridge into a narrow structural adapter for frame/truss/beam cases.
- Add normalized metrics for stress, displacement, modal frequency, contact impulse, thermal load, and convergence.
- Add golden-file fixtures for solver output parsing.

Exit criteria:

- A simple truss/beam concept dispatches to an FEA adapter and returns normalized metrics.
- A mechanism concept dispatches to a multibody adapter and returns trajectory/control metrics.

### Phase 2: Digital Twin Loop, Months 4-7

Goal: close the loop between telemetry, surprise, and simulation requests.

- Connect `symthaea-digital-twin` to `symthaea-fep` and existing observability patterns.
- Add telemetry schemas for civil strain/vibration, mechanical temperature/vibration, and electrical load/power quality.
- Trigger simulation or inspection requests when free-energy trend crosses thresholds.
- Store assumptions and interventions in an auditable event log.

Exit criteria:

- A twin can ingest telemetry, detect rising surprise, propose an intervention, and request a targeted simulation.

### Phase 3: Formal Safety Bridge, Months 7-10

Goal: turn engineering claims into explicit evidence obligations.

- Map blocking requirements to `ProofObligation` records.
- Maintain domain templates for civil, mechanical, robotics, electrical, aerospace, process, nuclear, materials, environmental, and systems cases.
- Add Lean/TLA+/SMT adapter boundaries without hard-coding one proof assistant into the engineering facade.
- Define safety-case templates for civil structure, mechanism/control, circuit, and system-of-systems reviews.
- Require evidence references before a deployment review can pass.

Exit criteria:

- A safety-critical concept cannot pass review with open blocking obligations.
- Simulation, proof, test, telemetry, and standards evidence can be attached independently.

### Phase 4: Domain Depth, Months 10-14

Goal: move from scaffolding to credible domain pilots.

- Civil pilot: low-carbon pedestrian bridge, load path alternatives, FEA evidence, standards evidence.
- Mechanical pilot: linkage or manipulator end-effector, MuJoCo trajectory evidence, manufacturability evidence.
- Electrical pilot: simple power electronics or microgrid dispatch, SPICE/grid evidence.
- Systems pilot: cross-domain safety case tying software, sensors, control, and physical plant.

Exit criteria:

- At least two domain pilots run end-to-end from requirements to evidence-backed review.

### Phase 5: Engineering Copilot, Months 14-18

Goal: package the workflow as a serious engineering reasoning system.

- Add CLI/API surfaces for concept creation, simulation dispatch, and safety review.
- Add benchmark tasks: FMEA quality, requirement traceability, solver result interpretation, and uncertainty flags.
- Add documentation for adapter authors.
- Publish an engineering capability report with measured limits.

Exit criteria:

- Symthaea can honestly say: "I can propose concepts, request the right simulations, track uncertainty, and block unsafe deployment until evidence exists."

## Near-Term Priority Order

1. Finish the lightweight crate APIs and examples.
2. Pick exactly one FEA backend and one multibody backend.
3. Build small, reproducible pilots before broad domain coverage.
4. Treat formal safety and proof obligations as workflow gates, not decoration.
5. Defer CAD/PLM integrations until simulation and evidence loops are stable.

## Non-Goals

- Do not build a replacement FEA, CFD, SPICE, or CAD kernel inside Symthaea.
- Do not place heavy solver dependencies in default workspace builds.
- Do not allow generated engineering claims to pass review without evidence references.
- Do not optimize for impressive demos before unit, provenance, and safety-case discipline are in place.

## Public Infrastructure Positioning

Symthaea's engineering layer should become open public infrastructure for solver orchestration and safety reasoning:

- Rust-native, auditable adapter traits and data models.
- Solver-agnostic uncertainty and evidence records.
- Domain safety templates that make review obligations explicit.
- Multi-physics request objects that describe coupling before execution.
- CAD/geometry interchange based first on existing STEP/NURBS/mesh work, then FreeCAD integration.
- Strict separation between reasoning crates and native solver dependencies.
