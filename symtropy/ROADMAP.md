# Symtropy Roadmap

## North Star

Dual-track engine:

- **A. Research hero** — best-in-class **N-dimensional, Φ-coupled physics with deterministic replay** as a first-class physical law. No other engine does this. 2D/3D/4D rigid-body dynamics with integration metrics (Φ, harmony, energy) that meaningfully modulate forces, impulses, and friction at the solver level.
- **B. Generalist adoption** — a production-grade Rust simulation framework on Bevy, with first-class hooks for ANY per-body metric (health, trust, skill, wealth, …). Consciousness is the reference implementation; the coupling framework is generic.

If a feature helps both tracks, it's a core priority. If it helps only B at the cost of A's focus (soft-body, triangle terrain, GPU broadphase), it lives as an **optional ecosystem crate**, not in the core.

## Current State (honest, 2026-04)

- 61K LOC Rust · 677 tests · 23 crates · 8 published on crates.io
- Rigid-body ND physics real and benchmarked
- Φ coupling load-bearing (5 channels, ThermodynamicLedger, J/Φ ledger)
- Mycelix headless governance harness production-ready (11 scenarios)
- Robotics bridge is a skeleton (platform types + FEP; no joint fidelity)
- Symthaea integration: FEP + Consciousness Equation wired; full CognitiveLoopService not yet stepped inside entities
- 3D rendering pipeline missing (2D sprite projection only today)
- Licensed AGPL-3.0-or-later across all crates (adoption blocker for general use)

---

## Phase 0 — Stabilize & Unblock Adoption (4–6 weeks)

Cheap moves that unlock every later phase.

- **License split** (partially done, 2026-04-17):
  - ✅ **Apache-2.0 OR MIT** (no AGPL deps, publishable now): `symtropy-math`, `symtropy-physics`, `symtropy-render-bridge`
  - ⏳ **AGPL today, permissive variant arrives in Phase 0.5**: `symtropy-bevy`, `symtropy-robotics-bridge`, `symtropy-net` — each has required AGPL deps that must be feature-gated before the permissive claim is honest. See [LICENSING.md](./LICENSING.md) Note 1.
  - ✅ **AGPL-3.0-or-later** (research / integration): `symtropy-consciousness-physics`, `symtropy-sim-bridge`, `symtropy-world`, `symtropy-holochain-relay`, `symtropy-lightyear`, `symthaea-bevy-brain`, game crates

### Phase 0.5 — Bevy / net / robotics-bridge split (inserted)

Before the three currently-AGPL adoption crates can go permissive, each needs a `-core` extraction. Work items:

- **`symtropy-bevy-core`** (new, Apache/MIT) — physics plugin, gizmos, transforms, input, schedules. No consciousness-physics. Feature-gate `biometrics`, `macro_bridge`, `debug.rs:SafetyTier` usage in `symtropy-bevy` (remains AGPL).
- **`symtropy-net-core`** (new, Apache/MIT) — spatial authority partitioning, lockstep protocol, `SyncableState` trait. `symtropy-net` re-exports `-core` + adds Holochain (remains AGPL).
- **`symtropy-robotics-bridge-core`** (new, Apache/MIT) — `PlatformType`, `RoboticAgent` trait, physics-body spawning. `symtropy-robotics-bridge` adds FEP + consciousness-equation (remains AGPL).

Each split is a 1–2 day refactor. All three can ship in Phase 0.5 parallel to Phase 1 integration wins.
- **Publish remaining crates** to crates.io (physics, render-bridge, robotics-bridge, net) with docs.rs metadata, `README.md` per crate, and `html_logo_url`.
- **The Symtropy Book** (mdBook): generic state-coupling in 50 LOC, Bevy integration, 4D tutorial, determinism contract, `PhysicsCallback` guide, 6-platform robotics quickstart.
- **CI matrix**: Linux / macOS / Windows × x86_64 / aarch64 × stable / beta / MSRV. Currently Linux/X11 only.
- **Determinism contract** — document the *actual* guarantee (same-CPU float, Morton integer broadphase, BTreeMap iteration). Add regression tests that lock down invariants across Rust versions.
- **Wayland re-enabled** on Linux; verified Windows launch; verified macOS launch.

**Gate to Phase 1:** crates.io downloads > 100/mo on a core crate; one external user opens a non-trivial issue or PR.

---

## Phase 1 — Integration Wins + Rapier3D Bridge (Q2 2026, 6–8 weeks)

Ship the two test harnesses that are already 90% there, plus the Rapier3D bridge unblocking high-fidelity robotics.

### Mycelix CI harness productionized
- `symtropy-sim-bridge::headless_test` wired into monorepo CI as `symtropy-governance-verify`.
- Per-cluster invariant sets: governance (veto limits, emergency caps, override thresholds), commons (reciprocity), civic (justice), finance (demurrage, Gini bounds).
- Seed-based reproducibility; tyranny-300-ticks scenario runs every PR.

### Robotics bridge — real wiring
- `symtropy-robotics-bridge::RoboticAgent` wires to Symthaea's `EmbodimentBridge::step(thought_hv, dt, phi)`.
- Replace scalar `motor_gain` with per-platform state/command vectors.
- Launch platforms: **humanoid (72D state / 21D cmd)**, **quadruped**, **manipulator (21D / 8D)**.
- Safety tier (NRC 4-tier) gates motor authority as today; now at per-joint resolution.

### NEW: `symtropy-rapier3d-bridge` (opt-in, Apache/MIT)
- **Framing:** a *bridge*, not a replacement. The native ND solver remains the research path.
- Feature-flagged (`rapier3d`). When enabled, robotics platforms can choose backend at spawn:
  ```rust
  spawn_robot_rapier3d(&mut world, PlatformType::Humanoid, ...);  // high-fidelity 3D
  spawn_robot_native(&mut world, PlatformType::Humanoid, ...);    // ND research path
  ```
- Rapier3D handles joint chains, contact, articulation for 3D use cases where fidelity beats the ND research framing.
- `PhysicsCallback` bridged so Φ-coupling works identically on either backend.

### Symthaea full CognitiveLoopService stepped inside entities
- Upgrade from FEP-only to the 38-field consciousness/memory/behavior substructs.
- This is the moment Symtropy becomes a real end-to-end Symthaea test bench.
- Unlocks: reasoning engine, meta-cognition, thalamic router, prediction error → motor precision closed-loop at full fidelity.

**Gate to Phase 2:** At least one robot platform walks/flies/manipulates reliably in-engine with Φ-gated motor authority; Mycelix CI catches a governance invariant violation on a PR.

---

## Phase 2 — Visual & Asset Pipeline (Q3 2026, ~12 weeks)

The biggest current gap. Without this, "best OSS engine" is not reachable.

- **3D rendering pipeline** via Bevy's StandardMaterial — PBR, dynamic lighting, shadows, skybox, bloom, SSAO. Bevy handles most; we wire it cleanly into the Symtropy scene flow.
- **4D → 3D cross-section renderer** — the research hero made visible. Miegakure-style slicing shader, with slice plane controllable from UI. First-class ND debug gizmos.
- **Debug gizmos**: wireframe colliders for any D, contact manifold visualization, joint rendering, Φ heatmap overlay, harmony-field isosurfaces, energy-budget gauges.
- **Scene format** — Bevy Scene + a `.sym` extension for Φ-coupling parameters, safety tiers, harmony sources. Hot reload via Bevy AssetServer.
- **Asset imports**: glTF 2.0 (bevy_gltf), png/jpg/ktx2, (stretch) OpenUSD-Lite.
- **Editor integration** — not a bespoke editor. Ship a `bevy_inspector_egui` plugin set:
  - `SymtropyInspectorPlugin`: Φ inspector, live coupling-channel tuning, replay scrubber, determinism checker, thermodynamic ledger dashboard.

---

## Phase 3 — Animation, Audio, Scripting (Q4 2026, ~10 weeks)

- **Skeletal animation** — Bevy's animation plugin + inverse kinematics solvers for humanoid, quadruped, manipulator. Φ-gated motor output feeds IK targets, closing the loop: cognition → animation → physics → sensory feedback.
- **State-driven audio** — productionize `live-audio`: Φ / harmony modulate synthesis parameters via `symthaea-muse`. Optional Muse biometric bridge for player-driven audio. Feature-flagged; must not regress hot path.
- **Scripting layer**:
  - Rhai first (pure Rust, ergonomic, no unsafe).
  - WASM plugin host second (sandboxed user mods via wasmtime).
  - Expose `PhysicsCallback` and state-coupling API to scripts so non-Rust users can prototype couplings.
- **Animation state machines** — simple transition graph (idle / walk / run / fall), Φ-gated transitions.

---

## Phase 4 — Networking, Platforms, XR/VR (Q1 2027, ~10 weeks)

- **Productionize Lightyear wrap** — verified multiplayer demo: 8 players, rollback, interest management, deterministic rollback leveraging existing Morton + BTreeMap determinism work.
- **Spatial authority integration test** — `symtropy-net` wired to Lightyear, validated on 3 peers over LAN, with a chaos-test harness that injects packet loss/reorder.
- **Holochain persistence** — DHT-backed governance votes and consciousness profiles (via `symtropy-holochain-relay`). Ships as optional feature.
- **WASM target** — browser-runnable experiments. 63 consciousness examples accessible from a web gallery. Huge educational win.
- **Windows / macOS verified** — audited not just built; gamepad, audio, filesystem, window resizing all exercised.
- **XR/VR** — `bevy_openxr` integration for native OpenXR. WebXR via wasm target. First-class hero demo: **visualize 4D cross-sections in a headset** (consciousness-physics made spatially legible). *Note: this is drop-in if bevy_openxr is mature enough; can slip earlier in schedule if a concrete XR use case appears.*

---

## Phase 5 — Ecosystem Crates (ongoing from Q2 2027)

Previously "Will NOT Do" — reframed as welcomed contributions outside the core hot path.

- `symtropy-physics-gpu` — GPU-accelerated broadphase (Jolt-style). Core targets <1000 bodies; GPU crate opens >10K.
- `symtropy-soft` — Soft-body / cloth / XPBD. Hooks into `PhysicsCallback`. Community-led.
- `symtropy-mesh` — Triangle mesh collider. BVH-based narrowphase.
- `symtropy-terrain` — Heightfield + LOD streaming. Integrates with existing Bevy terrain ecosystem crates rather than reinventing.
- `symtropy-fluid` — SPH or FLIP optional crate for fluid dynamics.

**Policy:** The Symtropy team does not commit to maintaining these. We commit to (a) keeping the `PhysicsCallback` / `Shape<D>` / `Constraint<D>` traits stable enough to host them, (b) accepting PRs that extend extensibility, (c) publishing a "How to extend Symtropy" guide in the Book.

---

## Continuing Physics Work (preserved from original roadmap, re-prioritized)

### High priority (unchanged)
- **Prismatic joint** — slide along one axis (suspension, sliders)
- **Motor/drive support** — PD controllers on hinge + prismatic joints (critical for Phase 1 robotics)
- **Island formation** — Union-Find over contact/constraint graphs; skip sleeping islands
- **Composite shapes** — union of shapes on one body (capsule torso + sphere head)

### Medium priority (unchanged)
- **Incremental broadphase** — don't rebuild LBVH every frame
- **Curvature feedback** — entities source conformal curvature proportional to Φ (`consciousness-curvature` feature)
- **ND-generic ActiveInference** — generalize from 2D to const-generic D
- **Proper variational free energy** — replace FEP heuristic with real VFE computation
- **Cross-platform determinism** — float quantization or constrained libm strategy

### Promoted (was "lower priority" — now Phase 0)
- ~~crates.io publication~~ → Phase 0
- ~~Bevy integration tutorial~~ → Phase 0 (Symtropy Book)

### Specialized shapes (community-welcome)
- Cone / Cylinder — specialized support functions
- Triangle mesh — via `symtropy-mesh` optional crate

---

## In Progress (unchanged)

- HalfSpace analytical contact dispatch in `world.rs` narrowphase (bypass GJK for common cases)
- Multi-contact generation from EPA (perpendicular direction sampling)

---

## Completed

### Physics Foundation
- GJK collision detection (any D, stack-allocated simplex)
- EPA penetration depth (2D edge-based, 3D face-based, ND-generic facet-based)
- Semi-implicit Euler integrator with bivector angular dynamics
- LBVH broadphase with Morton encoding (integer quantization for determinism)
- Collision groups/masks (bitmask filtering) and sensor/trigger bodies
- Body sleeping (velocity threshold + tick counter)
- O(1) body handle index (HashMap lookup replacing O(n) scan)

### Collision Shapes (all ND, const-generic)
- `Sphere<D>` — O(1) support
- `Capsule<D>` — O(1) support, aligned to any axis
- `HyperBox<D>` — O(D) support (3.5× faster than ConvexHull for 4D)
- `HalfSpace<D>` — analytical contacts for sphere/capsule/box
- `ConvexHull<D>` — O(vertices) support, quickhull construction

### Joint Types (all ND)
- `DistanceConstraint<D>` — fixed distance, PBD + impulse hybrid
- `BallJoint<D>` — removes D translational DOF
- `FixedJoint<D>` — rigid attachment (zero DOF)
- `HingeJoint<D>` — constrains rotation to one bivector plane

### Contact System
- Multi-point contact manifold (ArrayVec, zero-heap)
- Warm-starting with proximity-based contact matching (BTreeMap cache)
- Baumgarte position correction with configurable slop

### Advanced Features
- Continuous collision detection (swept sphere-sphere, sphere-halfspace)
- Ray casting (analytical ray-sphere, ND-generic)
- Deterministic replay (NetId, ReplayTape, WorldSnapshot, replay-cli binary)

### Φ-Physics Coupling (5 channels)
- Channel 1: Φ → Force (NRC 4-tier safety: Green/Yellow/Orange/Red)
- Channel 2: Φ → Energy (thermodynamic budget, Landauer bound, J/Φ metric)
- Channel 3: Harmony → Impulse (sanctuary zones dampen collisions)
- Channel 4: Harmony → Friction (CEMI-inspired 1/r^(D-1) fields)
- Channel 5: Collision → Φ (prediction error reduces motor precision)
- Φ-gravity (integration-weighted attraction between entities)
- Temperature → Φ feedback (heat stress reduces integration inputs)
- Dimensional leakage coupled to entity EnergyBudget

### Research & Validation
- 63 experiments (all compiling; cooperation, scaling, phase transitions, economics)
- Criterion benchmark suite (GJK per shape, EPA, raycast, step at 10/100/500 bodies)
- Key result: 81.3% tighter clustering under thermodynamic enforcement
- Key result: J/Φ converges to stable substrate-characteristic value

### Infrastructure
- Cargo workspace for core engine crates
- Terminology precision pass (Φ = integrated information, not a claim about experience)
- README.md, ARCHITECTURE.md, FORMAL_SPECIFICATION.md, ENGINE.md

---

## Success Metrics

Each phase should move these. "Now" figures are 2026-04.

| Metric | Now | End Phase 2 | End Phase 4 |
|---|---|---|---|
| crates.io /mo downloads (core) | — | 1 K+ | 10 K+ |
| External contributors | ~0 | 5 | 25 |
| Non-research games shipped on Symtropy | 0 | 1 | 5 |
| Academic citations | 0 | 3 | 15 |
| Symthaea + Mycelix CI coverage | ~0 % | 40 % | 80 % |
| Robot platforms running in-engine | 0 | 3 | 6 |
| CI matrix (OS × arch × toolchain) | 1 × 1 × 1 | 6+ | 12+ |

---

## What We Will NOT Do (in core)

These remain outside the core engine's scope. Reframed from "never" to "not here":

- **GPU-accelerated broadphase** — welcome as `symtropy-physics-gpu` optional crate; core targets <1000 bodies.
- **Soft-body / cloth** — welcome as `symtropy-soft`; complex to stabilize, owned by the community.
- **Jacobian-based solver** — current PBD + impulse works at target scale; switching costs > benefit.
- **HeightField terrain** — integrate with Bevy's terrain ecosystem via `symtropy-terrain`, don't reinvent.
- **Bespoke editor** — use `bevy_inspector_egui` + scene format, not a standalone tool.
- **Custom wgpu renderer** — Bevy owns this; we layer on top.

---

## Contributions That Move the Needle

- Anything that strengthens determinism guarantees (cross-platform float, ordered iteration, invariant tests)
- ND-first features (debug rendering ergonomics for D ≠ 3, 4D visualizations)
- New experiments that test the formal specification's predictions
- Tests that lock down invariants (ordering, floating-point edge cases, determinism)
- Documentation improvements, tutorials, and Book chapters
- Ecosystem crates (GPU broadphase, soft-body, mesh, terrain, fluid) — we maintain the trait stability; you maintain the crate
- Language bindings (Rhai bridges, Python via PyO3 for research use)

---

## Decided Scoping (from roadmap review, 2026-04)

1. **Dual-track framing** — kept: research hero + generalist adoption.
2. **Bevy is the permanent foundation** — no fork, no custom wgpu.
3. **License split** — core Apache-2.0 OR MIT; consciousness + Mycelix AGPL; commercial licensing via `COMMERCIAL_LICENSE.md`.
4. **XR/VR** — Phase 4 (`bevy_openxr` + WebXR); hero demo is 4D cross-section visualization in headset.
5. **Rapier3D** — Phase 1 as `symtropy-rapier3d-bridge` (opt-in), for high-fidelity 3D robotics; native ND solver remains research path. Backend chosen at spawn time; `PhysicsCallback` works identically on both.
6. **Terrain + Soft-body** — Phase 5 ecosystem crates, not core engine builds.
