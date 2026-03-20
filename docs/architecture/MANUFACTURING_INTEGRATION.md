# Manufacturing Integration Architecture

## Overview

Three systems compose consciousness-aware manufacturing: the **fabrication-kernel** sub-crate (`symthaea-fabrication-kernel`, v0.5.0) provides HDC-encoded digital twins, CSG geometry, mesh export, and multi-horizon CfC prediction; the **Mycelix Fabrication hApp** (7 zomes, `mycelix-workspace/happs/fabrication/`) manages print jobs, materials, designs, printers, verification, Symthaea bridge, and cross-cluster bridge on the DHT; and the **Mycelix Supply Chain** cluster (`mycelix-supplychain/`, 8 zomes) provides provenance tracking, inventory, logistics, and payments. The `FabricationManager` (interval 47) inside Symthaea's cognitive loop translates sensor events and twin readings into neuromodulatory signals, consciousness-level feedback, and telemetry for the Pulse dashboard.

## Data Flow

```
Cincinnati Sensors --> CincinnatiMonitor --> ManufacturingReading
                                                  |
                                        ManufacturingTwin.step()
                                                  |
                                        FabricationEvent
                                                  |
                                        FabricationManager.process()  [interval 47]
                                            |              |
                                  SubsystemOutput    PendingInjection/Baseline
                                       |                    |
                              OutputCollector         apply_fabrication_neuromod()
                                     |                       |
                              Consciousness Level       Neuromod Bath (DA/NE/5-HT/OT)
                                                  |
                                        CycleMetadata --> Pulse Dashboard
```

`ManufacturingTwin.step()` encodes 4 observables (tolerance, surface quality, throughput, energy cost) into a 16,384-D `ContinuousHV`, evolves a CfC neuron across 5 prediction horizons (0.1s tool pass to 1-day shift) using O(1) closed-form temporal jumps, and computes free energy against a reference state via an FEP agent. The agent selects one of 5 actions (Maintain, AdjustTooling, RecalibrateProcess, ReduceSpeed, EmergencyHalt) based on free-energy thresholds.

`DesignLoopTwin` provides an analogous design-to-manufacture feedback loop with its own free-energy tracking.

## Neuromod Pathways

| # | Trigger | Transmitter | Effect | Citation |
|---|---------|-------------|--------|----------|
| 1 | Cincinnati anomaly (severity > 0.5) | Norepinephrine (phasic, dose 0.06, half-life 15 cycles) | Attentional reorientation, negative reward | Aston-Jones & Cohen 2005 |
| 2 | Print job completed (quality > 0.5) | Dopamine (phasic, dose 0.08, half-life 25 cycles) | Reward prediction confirmation, confidence +0.02 | Schultz 1997 |
| 3 | Safety level Red | NE (phasic, 0.10, 30 cycles) + 5-HT (baseline -0.05) | Emergency halt, VETO_ACTION flag, arousal +0.15, reward -1.0 | Sapolsky 2004 |
| 4 | Sustained quality trend (PoGF EMA > 0.7) | Serotonin (baseline +0.03) | Well-being, prosocial manufacturing behavior | Crockett 2009 |
| 5 | High PoGF score (> 0.7) | Oxytocin (phasic, 0.03, 40 cycles) | Prosocial trust/bonding reward | Zak 2012 |

Additional modulation: poor quality (< 0.5) dampens learning rate to 0.8x (Friston 2010); very poor quality (< 0.3) boosts exploration by +0.1 (Berlyne 1960).

## Cross-Cluster Bridges

All cross-cluster calls use `CallTargetCell::OtherRole` within the unified hApp.

| Bridge | Direction | Mechanism | Purpose |
|--------|-----------|-----------|---------|
| Fabrication <-> Supply Chain | Bidirectional | `SupplyChainLinkEntry` (material hash + supplier DID + optional SC item hash) | Material provenance, sourcing verification |
| Fabrication <-> Identity | Fab -> Identity | `OtherRole("mycelix-identity")` | Consciousness gating (4D profile check, 5-min cache TTL) |
| Fabrication <-> Commons | Fab -> Commons | `OtherRole("mycelix-commons")` | Property digital twin wear prediction, anticipatory repair loop |
| Fabrication <-> Knowledge | Fab -> Knowledge | `OtherRole("mycelix-knowledge")` via verification zome | Safety verification of designs and materials |
| Fabrication -> Marketplace | Fab -> Marketplace | Bridge zome | Design trading |

Consciousness gating: `require_fabrication_consciousness()` checks the caller's identity hApp tier before state-changing operations. Graceful fallback allows operations if the identity hApp is unreachable. Rate limiting: 100 ops / 60s per agent on all state-changing endpoints.

## Manufacturing Cluster (Mycelix Fabrication hApp)

7 zomes (each with integrity + coordinator pair):

| Zome | Purpose |
|------|---------|
| `designs` | Design entries, versioning, search |
| `printers` | Printer registration, capabilities, status |
| `prints` | Print job lifecycle, PoGF scoring, Cincinnati quality monitoring, MYCELIUM (CIV) reputation |
| `materials` | Material entries with supply chain hash links |
| `verification` | Safety verification via Knowledge hApp bridge |
| `bridge` | Cross-hApp integration (anticipatory repair, supply chain links, marketplace, audit trail) |
| `symthaea` | Symthaea consciousness bridge (consciousness state sync, twin data relay) |

**PoGF (Proof of Grounded Fabrication) formula:**
```
PoGF = (E_renewable * w_e) + (M_circular * w_m) + (Q_verified * w_q) + (L_local * w_l)
```
Weights loaded from DNA properties via `FabricationConfig`.

**MRP engine flow:** Design entry -> material requirements -> supply chain sourcing (via bridge) -> printer assignment -> print job creation -> Cincinnati monitoring -> quality verification -> PoGF scoring -> MYCELIUM reputation reward.

## Fabrication Kernel Modules

| Module | Purpose | Tests |
|--------|---------|-------|
| `manufacturing` | ManufacturingTwin, HDC encoder (4 observables), CfC predictor (5 horizons), FEP agent, safety levels | 12 |
| `design_loop` | DesignLoopTwin, design-manufacture feedback, FEP design agent | 12 |
| `slicer` | Mesh slicing, contour extraction, layer generation, `SliceConfig` | 26 |
| `bsp` | BSP tree, CSG intersect/subtract operations | 12 |
| `building` | Building system primitives, structural analysis | 12 |
| `mesh` | `TriangleMesh`, vertex/face operations, normals | 8 |
| `primitives` | Geometric primitives (sphere, cube, cylinder, etc.) as HDC-encoded shapes | 5 |
| `csg` | `CSGNode`, `BooleanOp`, `Primitive`, `Transform3D` | 4 |
| `export` | STL (binary/ASCII) and 3MF mesh export | 4 |
| `import` | STL parsing (binary + ASCII), `StlError` | 10 |
| `validate` | Mesh validation, `ValidationReport` (manifold, normals, degenerates) | 15 |
| `thought` | `GeometricThought` — HDC encoding of geometric intent | 7 |
| `simulator` | `PhysicsBackend` trait, `SimState`, `ForceHV` | 7 |
| `analytical` | Analytical geometry (feature-gated: `analytical`) | 11 |
| `generative` | Generative design (feature-gated: `analytical`) | 12 |
| **Total** | | **157** |

Plus 17 unit tests in `FabricationManager`, 2 threshold ordering tests, and 9 integration tests = **185 total** across the manufacturing integration.

## Feature Flags

| Flag | Crate | Gate |
|------|-------|------|
| `advanced-manufacturing` | `symthaea` (main) | Enables `dep:symthaea-fabrication-kernel`, `FabricationManager` in cognitive loop, integration tests |
| `building-systems` | `symthaea` (main) | Enables `dep:symthaea-fabrication-kernel` for building/structural analysis |
| `design-production` | `symthaea` (main) | Enables `dep:symthaea-fabrication-kernel` for design loop integration |
| `analytical` | `symthaea-fabrication-kernel` | Enables `analytical` + `generative` modules (default on) |

The `slicer` and `nurbs` capabilities are built into the kernel crate unconditionally (no feature gate). Printer control logic lives in the Mycelix Fabrication hApp's `printers` zome, not in the kernel.

All three `advanced-manufacturing`, `building-systems`, and `design-production` are members of the `genesis` feature group in the main crate's Cargo.toml.

## Verification Commands

```bash
# Fabrication kernel (all modules, 157 tests)
cargo test -p symthaea-fabrication-kernel --all-features

# FabricationManager unit tests (17 tests)
cargo test -p symthaea --lib fabrication_manager -- --features advanced-manufacturing

# Fabrication threshold tests (2 tests)
cargo test -p symthaea --lib thresholds::fabrication

# Integration tests (9 tests, requires feature flag)
cargo test -p symthaea --test fabrication_integration --features advanced-manufacturing

# Mycelix Fabrication hApp (unit tests)
cd mycelix-workspace/happs/fabrication && cargo test --workspace

# Mycelix Fabrication sweettests (requires conductor)
cd mycelix-workspace/happs/fabrication && cargo test --release --test sweettest -- --test-threads=1

# Mycelix Supply Chain
cd mycelix-supplychain/holochain && cargo test --workspace
```
