# Symtropy Roadmap

## North Star

**Component-level gap analysis:** see [`docs/GAME_ENGINE_COMPONENTS.md`](docs/GAME_ENGINE_COMPONENTS.md) for a per-component matrix of what's covered by Symtropy vs Bevy vs Symthaea/Mycelix vs community crates vs still-missing, with concrete crate recommendations for each gap. The doc complements this ROADMAP: the ROADMAP is phase-oriented; the components doc is a living checklist.

Dual-track engine:

- **A. Research hero** — best-in-class **N-dimensional, Φ-coupled physics with deterministic replay** as a first-class physical law. No other engine does this. 2D/3D/4D rigid-body dynamics with integration metrics (Φ, harmony, energy) that meaningfully modulate forces, impulses, and friction at the solver level.
- **B. Generalist adoption** — a production-grade Rust simulation framework on Bevy, with first-class hooks for ANY per-body metric (health, trust, skill, wealth, …). Consciousness is the reference implementation; the coupling framework is generic.

If a feature helps both tracks, it's a core priority. If it helps only B at the cost of A's focus (soft-body, triangle terrain, GPU broadphase), it lives as an **optional ecosystem crate**, not in the core.

## Current State (honest, 2026-04)

- 29 crate manifests in this workspace · 734 Rust test/proptest/bench markers counted locally · 8 published on crates.io
- Rigid-body ND physics real and benchmarked
- Φ coupling load-bearing (5 channels, ThermodynamicLedger, J/Φ ledger)
- Mycelix headless governance harness production-ready (11 scenarios)
- Robotics bridge is a skeleton (platform types + FEP; no joint fidelity)
- Symthaea integration: FEP + Consciousness Equation wired; full CognitiveLoopService not yet stepped inside entities
- 3D rendering pipeline missing (2D sprite projection only today)
- License split is in progress: core crates are Apache-2.0 OR MIT; research/game/integration crates remain AGPL-3.0-or-later

---

## Phase 0.1 — Correctness & Claim Reconciliation (immediate, 1-2 weeks)

Before adoption work, make the code, docs, licensing metadata, and tests agree. This is not feature work; it protects the central claims.

- **PhysicsCallback contract:** `step_with_callback` must exercise all documented channels. Channel 1 (`Metric/Φ -> Force`) is covered by a regression test that verifies zero-gain callbacks block intended motor force while unit-gain callbacks preserve it.
- **Coupling accounting:** callback implementations must not duplicate or bypass their accounting-aware inherent methods. Sanctuary impulse absorption should reach `ThermodynamicLedger` on the same path used by `PhysicsWorld`.
- **License scanner clean:** Cargo license fields, SPDX headers, README, Book, and `LICENSING.md` must tell the same story. Permissive crates cannot contain AGPL SPDX headers; permissive meta-crates cannot require AGPL dependencies by default.
- **Docs state sync:** crate counts, test counts, published versions, and roadmap completion markers should be generated or periodically checked, not hand-maintained in multiple places.
- **Verification surface:** add one canonical `just verify-core` command for fmt, clippy/check, core tests, replay determinism, and license/doc checks.

**Gate to Phase 0.5:** central claims in README/Book/ROADMAP are mechanically backed by code or explicitly scoped as planned.

## Phase 0 — Stabilize & Unblock Adoption (4–6 weeks)

Cheap moves that unlock every later phase.

- **License split** (partially done, 2026-04-17):
  - ✅ **Apache-2.0 OR MIT** (no AGPL deps, publishable now): `symtropy-math`, `symtropy-physics`, `symtropy-render-bridge`
  - ⏳ **AGPL today, permissive variant arrives in Phase 0.5**: `symtropy-bevy`, `symtropy-robotics-bridge`, `symtropy-net` — each has required AGPL deps that must be feature-gated before the permissive claim is honest. See [LICENSING.md](./LICENSING.md) Note 1.
  - ✅ **AGPL-3.0-or-later** (research / integration): `symtropy-consciousness-physics`, `symtropy-sim-bridge`, `symtropy-world`, `symtropy-holochain-relay`, `symtropy-lightyear`, `symthaea-bevy-brain`, game crates

### Phase 0.5 — Bevy / net / robotics-bridge split (partially done, 2026-04-17)

Before the three currently-AGPL adoption crates can go permissive, each needs a `-core` extraction.

- ✅ **`symtropy-bevy-core` 0.1.0** — PUBLISHED Apache/MIT. Generic over `PhysicsCallback` via `step_physics<D, C>`; ships `BevyPhysics<D>`, `PhysicsBody`, `NoCouplingResource`, `BevyPhysicsPlugin<D>`, `BevyPhysicsCorePlugin<D>`. 12 tests. Zero AGPL deps.
- ⏳ **`symtropy-net-core`** — spatial authority partitioning, lockstep protocol, `SyncableState` trait. `symtropy-net` re-exports `-core` + adds Holochain (remains AGPL). Ships in Phase 4 alongside Lightyear productionisation.
- ⏳ **`symtropy-robotics-bridge-core`** — `PlatformType`, `RoboticAgent` trait, physics-body spawning. `symtropy-robotics-bridge` adds FEP + consciousness-equation (remains AGPL). Ships in Phase 1 once the `EmbodimentBridge` wiring stabilises (premature to publish before that churn).

Also queued: refactor existing `symtropy-bevy` (AGPL) to re-export `symtropy-bevy-core` + add `ConsciousnessField` integration on top, eliminating duplication.
- **Publish remaining crates** to crates.io (physics, render-bridge, robotics-bridge, net) with docs.rs metadata, `README.md` per crate, and `html_logo_url`.
- **The Symtropy Book** (mdBook): generic state-coupling in 50 LOC, Bevy integration, 4D tutorial, determinism contract, `PhysicsCallback` guide, 6-platform robotics quickstart.
- **CI matrix**: Linux / macOS / Windows × x86_64 / aarch64 × stable / beta / MSRV. Currently Linux/X11 only.
- **Determinism contract** — document the *actual* guarantee (same-CPU float, Morton integer broadphase, BTreeMap iteration). Add regression tests that lock down invariants across Rust versions.
- **Wayland re-enabled** on Linux; verified Windows launch; verified macOS launch.

### Phase 0.6 — Demo & Visibility (2-4 weeks)

Give external users one obvious path from curiosity to success.

- **Hero demo:** one polished replayable 4D/state-coupled scene with visible Φ/metric controls, contact/energy overlays, and screenshot capture.
- **Starter template:** `cargo generate` or `symtropy new` flow for a small Bevy app using `symtropy-bevy-core` by default, with AGPL Φ/Mycelix sections as explicit opt-ins.
- **Recipe docs:** short guides for character controllers, input mapping, pathfinding, save/load, VFX, profiling, and Mycelix cloud-save/identity opt-in. These should live under `docs/recipes/` and reflect `docs/GAME_ENGINE_COMPONENTS.md`.
- **Comparative benchmarks:** one public Criterion report comparing native Symtropy, Rapier3D bridge targets, and Bevy ecosystem physics on a few representative workloads.
- **README adoption story:** surface why Symtropy + Symthaea + Mycelix is more than a physics crate without forcing that stack on permissive-core users.

### Phase 0.7 — Mk0 Bootstrapper Protocol (immediate operational path)

The North Star city stack should not be mistaken for the first deployable slice. The immediate executable path is a **Mk0 bootstrapper loop** built from off-the-shelf hardware plus Symtropy/Symthaea/Mycelix software. This is the minimum viable metabolism that can fabricate, assemble, move, and coordinate its own next increment.

**Why this matters**
- It turns the roadmap from "future city taxonomy" into a testable warehouse-scale protocol.
- It provides a practical local starting point: one room, one energy loop, one material loop, one compute cluster, one logistics rover.
- It creates the seed conditions for producing the more specialized Phase 1 and Phase 2 platforms rather than waiting for them fully formed.

**Mk0 bootstrapper roster**

| Mk0 role | Hardware posture | Symthaea / Symtropy mapping | Immediate purpose |
|---|---|---|---|
| `mk0-seed-node` | rack server / SBC / workstation mesh | Mycelix DHT + Symtropy runtime + scenario orchestration; conceptually the precursor to `symthaea-archive` rather than a new `cortex` crate | local compute, coordination, replay, policy distribution |
| `mk0-helios` | solar + LiFePO4 + smart inverter microgrid | precursor to `symthaea-infrastructure` / `symthaea-plexus` | autonomous power for compute and fabrication |
| `mk0-detritivore` | plastic shredder + filament extruder | early material-recovery profile of `symthaea-scavenger`, not a separate top-level crate yet | close the minimum plastic loop for local fabrication |
| `mk0-fabricator` | FDM print farm + light CNC | precursor to `symthaea-fabricator` | print structural parts, fixtures, chassis, adapters |
| `mk0-manipulator` | bench arm / cobot | `symthaea-manipulator` | insert fasteners, assemble joints, tend printers/workcells |
| `mk0-vector` | warehouse rover | `symthaea-vehicle` or future `vector` logistics profile | move stock, parts, tools, and finished modules between cells |

**Phase 0 success condition**
- One-room bootstrapper can recycle feedstock, print useful parts, assemble at least one repeatable subassembly, move materials autonomously, and remain energy-aware under local microgrid constraints.

**Strategic note**
- The Mk0 bootstrapper is an **operational phase**, not a permanent parallel naming universe. Where possible, Mk0 work should feed the real platform crates above rather than create duplicate crate taxonomies.

**Gate to Phase 1:** crates.io downloads > 100/mo on a core crate; one external user opens a non-trivial issue or PR.

---

## Phase 0.8 — The Mk0 Scrap-Stack Expansion (immediate)

Beyond the basic stations, the Mk0 environment needs "Scrap-Stack" utilities that leverage existing debris and cheap electronics to establish the facility's sensory and metabolic perimeter.

### 1. Logistics & Movement
- **`mk0-gantry` (Sky-Hook):** Ceiling-mounted pulley grid for 2D coordinate movement of 1–5kg loads (spools, tools).
- **`mk0-scout` (Cable-Crawler):** Sky-wire camera carriage for top-down God-view mapping and human activity heatmaps.

### 2. Primary Energy & Thermal
- **`mk0-thermal` (Bio-Furnace):** Captures heat from composting organic waste via liquid-loop exchangers.
- **`mk0-chimney` (Solar Draft):** Passive cooling via solar-heated updraft pipes with FEP-controlled butterfly valves.
- **`mk0-aeolus` (Wind Harvester):** Salvaged alternators with 3D-printed blades for atmospheric kinetic capture.
- **`mk0-kinetic` (Gravity Engine):** Weight-drop battery using scrap-filled crates to store surplus solar energy for night use.
- **`mk0-geothermal` (Earth-Heat Tap):** Passive cooling loops inserted into basement crawl spaces or local mine shafts.

### 3. Sub-Surface & Sensing
- **`mk0-augur` (Soil-Probe):** Automated drill for mapping soil density and identifying bioremediation injection sites.
- **`mk0-signal` (Semaphore Relay):** Visual gossip system (servos + flags) for safety overrides during total data blackouts.

---

## Phase 0.9 — The Mk0.5 Tooling & Bootstrap Gap (Q2 2026)

The chasm between Mk0 (scrap) and Mk1 (precision) is a manufacturing gap. Mk0.5 is the **Tooling Phase**: using the scrap-stack to build the machines that build the high-fidelity city.

- **`mk0.5-mill` (The Precision Escalator):** A sloppy, 3D-printed CNC router that uses FEP software to compensate for backlash/slop, eventually milling the high-precision aluminum parts for its own successor.
- **`mk0.5-loom` (The Motor Winder):** Automated jig for winding copper wire into custom, high-efficiency BLDC motor stators.
- **`mk0.5-spark` (The PCB Assembler):** Gantry-based pick-and-place machine and reflow hotplate for local circuit board assembly.
- **`mk0.5-forge` (The Scrap-to-Filament Loop):** Extrusion line that turns `mk0-detritivore` shred into high-tolerance 1.75mm filament.

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

### Track B: `symtropy-mycelix-bridge` — Mycelix as a Bevy Resource (preferred architectural path)

Bevy systems and NPCs call Mycelix zomes via a `MycelixClient` Resource that speaks JSON over a subprocess pipe to `mycelix-conductor-bridge`. Turns every Bevy agent into a Mycelix user — a stress test no headless harness can match.

**Architecture:** subprocess IPC, not in-process linking. Holochain's `holochain_client 0.6.0` exact-pins `serde = 1.0.203` while Bevy 0.18 requires `serde_core >= 1.0.221` (unresolvable in one Rust compilation unit). Running the Holochain client as a separate child process sidesteps the conflict permanently and adds process-isolation for free.

- **M1 SHIPPED (2026-04-17, `91408552cd`):** subprocess plumbing, `MycelixClient` Resource, `MycelixResponse` Message, `BevyMycelixPlugin`, `GetActiveProposals` zome call, 9 unit tests, `hello_mycelix` example runs end-to-end.
- **M2 SHIPPED (2026-04-17, `506156c594`):** `SubmitProposal`, `CastVote`, `QueryTendBalance` zome calls wired through `mycelix-conductor-bridge`. `QueryTendBalance` routes to `finance/finance_bridge` instead of governance. Wire protocol upgraded from FIFO to `request_id: u64` correlation (named to avoid colliding with `ProposalInput.id` on serde-flatten). `AtomicU64` counter + `HashMap<u64, Pending>` correlation map. 12 unit tests green.
- **M3 (next):** scenario harness (`ScenarioConfig`, `ScenarioReport`, `run_scenario`) + CI job (`symtropy-mycelix-verify`). 5-entity integration scenario against a live conductor.
- **M4:** 50-NPC visual village demo at 60fps.

**Also shipped 2026-04-17:**
- `symtropy-bevy 0.2.0` refactored to layer on top of `symtropy-bevy-core` — re-exports the permissive types (`PhysicsBody`, `BevyPhysics`, both plugin flavours), adds consciousness integration on top. Commit `8a8963bd6f`.
- `symtropy-robotics-bridge` gains `tick_motor_commands()` returning platform-aware `Vec<f64>` (humanoid → 21 per-joint commands) + `MotorPlanner` trait for future per-joint intelligence. `UniformGainPlanner` ships as baseline. Deliberately does NOT depend on `symthaea-core`; Symthaea `EmbodimentBridge` adapters live in user/platform code. Commit `ba47296649`. 16 tests green.
- **Cross-cutting with** Phase 2 Track C (below) and Phase 4 Track A. All three tracks share this crate's `MycelixClient`.

WebView-embed of real Leptos apps (Track A's native variant) is explicitly NOT pursued. `wry` doesn't support offscreen rendering ([tauri/wry#391](https://github.com/tauri-apps/wry/issues/391), open since 2021); Servo-as-library and CEF are out of scope. See Phase 4 Track A for the tractable browser-overlay alternative.

**Gate to Phase 2:** At least one robot platform walks/flies/manipulates reliably in-engine with Φ-gated motor authority; Mycelix CI catches a governance invariant violation on a PR; at least one `MycelixClient`-driven test scenario runs green with ≥5 NPC-users.

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

### Track C: `symtropy-ui-mycelix` — native Bevy UI for Mycelix screens

Re-implement the 3–5 most-important Mycelix interactions (governance vote card, Praxis lesson panel, Craft profile, TEND balance, commons resource request) as native `bevy_egui` / `bevy_ui` screens. Each renders on a 3D plane with the holographic shader built above. Zero duplicated logic — UI reads and writes through Track B's `MycelixClient`.

- **Tradeoff:** duplicated UI code vs the Leptos web apps. Cost scales with screen count, so we keep the list short and cover only what a player inside the game world would touch.
- **Why bevy_egui over bevy_ui:** faster iteration for data-heavy panels, immediate-mode is a better fit for reactive Mycelix state than Bevy's retained UI today. Wrap behind a feature so consumers aren't forced onto egui if they prefer pure bevy_ui.

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
- **Track A: CSS-3D iframe overlay of real Leptos apps.** Once Bevy runs in WASM, the hero "holographic Mycelix screen" demo becomes trivial: Leptos app lives in an iframe, CSS 3D transforms match a Bevy 3D quad's screen-space projection, postMessage forwards input. Zero duplicated UI code, zero WebView dependency. Browser-only — complements Phase 2 Track C which covers the native binary.
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

## Platform Expansion Program (Regime-Based)

The platform roster is organized by **Physics & Metabolic Regimes**. Each regime introduces distinct physical laws, networking constraints, and civilizational roles.

### 1. Terrestrial Regime (Ground & Low-Atmosphere)
*The foundation: mobility, stewardship, and local assembly.*

| Platform | Why it matters | Physics challenge |
|---|---|---|
| `symthaea-subterranean` | Subsurface autonomy | Dense friction, spoil, heat, occlusion, pressure |
| `symthaea-infrastructure` | Buildings as agents | Thermal balancing, stationary actuation |
| `symthaea-scavenger` | Material recycling | Fracture, stress, teardown, sorting |
| `symthaea-agribot` | Living soil stewardship | Soil/water/light feedback, seasonal budgets |
| `symthaea-terra` | Land remediation | Toxic dust, heavy-metal injection |

### 2. Oceanic Regime (Buoyancy & Macro-Inertia)
*The global bloodstream: moving mass with minimum energy.*

| Platform | Role | Physics / Systems challenge |
|---|---|---|
| `symthaea-leviathan` | Intercontinental cargo | Extreme macro-inertia, hull resistance, weather routing |
| `symthaea-pilot` | Harbor precision tug | Swarm coordination, fluid coupling, multi-agent docking |
| `symthaea-stevedore` | Port gantry transfer | Pendulum dynamics, wind shear compensation, heavy-cable tension |
| `symthaea-abyssal` | Deep-ocean stewardship | Extreme pressure, low-visibility sensing, high-latency comms |

### 3. Arterial Regime (Macro-Throughput & Transit)
*The city's circulatory system: continuous flow and orthogonal separation.*

| Platform | Role | Physics / Systems challenge |
|---|---|---|
| `symthaea-meridian` | Continental spine (Maglev) | Virtual magnetic tethers, electrodynamic suspension, aerodynamic slipstreaming |
| `symthaea-flux` | Urban pod swarm | Swarm fluid dynamics, bumper-to-bumper laminar flow, zero-wait merging |
| `symthaea-stratum` | Adaptive transit surface | Load-aware damping, induction transfer, right-of-way signaling |

### 4. Vacuum Regime (Orbital & Lunar Metabolism)
*Expansion beyond the gravity well: assembly and primary extraction.*

| Platform | Role | Physics / Systems challenge |
|---|---|---|
| `symthaea-cycler` | Orbital tug | Keplerian mechanics, low-thrust Hall-effect trajectories, Oberth effect |
| `symthaea-spindle` | Zero-G assembler | Inertial homeostasis, 300°C thermal swings, vacuum welding prevention |
| `symthaea-regolith` | Lunar mining | Electrostatic dust management, microwave sintering, 1/6th gravity excavation |
| `symthaea-astrolabe` | Optical comms mesh | Point-to-point laser links, orbital debris avoidance, planetary relay continuity |

### 5. Interplanetary Regime (Deep Space Expansion)
*A multi-planetary species: autonomy under extreme latency.*

| Platform | Role | Physics / Systems challenge |
|---|---|---|
| `symthaea-beacon` | Deep space relay | Epoch-batching, high-latency HDC gossip, optical transceivers |
| `symthaea-vault` | Lava-tube steward | Subterranean sealing, geopolymer extrusion, atmospheric bleed prevention |
| `symthaea-isotope` | Fission power heart | Neutronic homeostasis, Stirling-engine heat balancing, radiation shielding |
| `symthaea-torch` | Interplanetary hauler | Nuclear electric propulsion, skeletal macro-logistics, deep-space thermal |

---

## Type 1 Civilization Metabolism (Foundational Loops)

A Type 1 Civilization requires absolute mastery of a planet's energy and chemical cycles. These foundational requirements are integrated into the **Metabolic Infrastructure** and **Civic** platforms.

- **Nitrogen/Phosphorus Cycle (Sanitation):** Localized bioreactor systems that break down biological waste into usable nutrients for the `agribot` layer. No ocean dumping.
- **Grid-Scale Energy Storage:** Beyond chemical batteries; requires macro-storage platforms (phase-change thermal, pumped hydro, or compressed air).
- **Caloric Decoupling:** Shifting from land-use agriculture to precision fermentation and automated bio-reactors for dense calorie production.
- **Atmospheric Engineering:** Localized carbon capture and chemical scrubbing to produce bioplastics and structural materials from airborne carbon.

---

### Metabolic infrastructure layer — the city's body

These are not just "more robots." They are stationary or semi-stationary platforms that make the city itself an embodied system: transit surfaces, utility routing, material refinement, assembly, and shared semantic memory.

| Platform | Role | Why it matters | Physics / systems challenge | Networking / civic challenge |
|---|---|---|---|---|
| `symthaea-stratum` | adaptive transit surface | makes roads/floors active participants in traction, charging, and safety | load-aware damping, induction transfer, fall-softening, traffic-aware surface control | right-of-way signaling, flow prediction, public-safety coordination |
| `symthaea-plexus` | utility manifold for water/power/data | turns passive pipes/cables into self-healing civic metabolism | pressure equilibrium, voltage stability, rerouting, leak isolation, thermal load | utility prioritization, fault isolation, neighborhood resilience |
| `symthaea-foundry` | material synthesis / metallurgy | provides the raw-to-component refinement layer without overloading general robotics crates | thermal induction, phase-transition control, quality sensing, cooling curves | production scheduling, energy budgeting, resource allocation |
| `symthaea-fabricator` | component and robot assembly | bridges refined material into finished parts, modules, and civic hardware | assembly sequencing, tolerance management, fixture coordination, QA gates | workcell dispatch, recipe/version propagation, supply-chain handoff |
| `symthaea-archive` | semantic city memory / knowledge mining | allows the city to learn from its own operations and propagate improvements across platforms | telemetry ingestion, retrieval, semantic clustering, replay/benchmark indexing | fleet-wide update gossip, knowledge retention, governance over learned policies |

**Naming hygiene**
- `symthaea-foundry` is preferred over `symthaea-crucible` because `symthaea-crucible` already exists in-tree as an ethics/stress-testing crate.
- `symthaea-archive` is preferred over `symthaea-cortex` because `cortex` is already heavily overloaded across the Symthaea codebase.

### Civic / habitat platforms — the city's soul

These platforms govern co-habitation rather than raw throughput. They are where the roadmap stops being "automation" and starts being a plausible lived environment for humans, animals, and shared civic life.

| Platform | Role | Why it matters | Physics / systems challenge | Ethical / civic constraint |
|---|---|---|---|---|
| `symthaea-clime` | atmosphere, HVAC, air quality, circadian lighting | makes indoor life healthy and comfortable instead of merely habitable | thermal control, humidity, air-quality sensing, lighting schedules, occupancy response | comfort must not override safety or public-health constraints |
| `symthaea-biota` | interspecies bridge for pets and wildlife | makes animal welfare and right-of-way first-class in the simulation | low-speed mobile or embedded sensing, stress inference, sanctuary-zone signaling | animal movement and distress should be treated as protected civic signals |
| `symthaea-hearth` | adaptive architecture and shared living space | turns homes and communal space into responsive habitat rather than static shell | occupancy-aware reconfiguration, furniture transforms, acoustic/privacy control | must preserve human dignity, consent, accessibility, and predictability |
| `symthaea-vita` | preventive biosensing and advisory care | introduces biological homeostasis as part of city operations | non-invasive sensing, health-vector tracking, early-warning inference, triage handoff | advisory/monitoring first; no autonomous diagnosis or treatment claims without strict validation and governance |

### Aerial platform clarification

The current `symthaea-multirotor` crate is the **multirotor** platform line, not the whole aerial stack. It already covers quadrotor-style hover, swarm, and scenario work well, but it does **not** constitute fixed-wing or passenger-aircraft support.

**Recommended aerial taxonomy**

| Platform | Role | Why it should exist separately | Near-term status |
|---|---|---|---|
| `symthaea-multirotor` | hover, inspection, local logistics, SAR, swarm interception | rotor thrust allocation, hover control, aggressive close-quarters maneuvering | exists today |
| `symthaea-evtol` | urban air mobility, air taxi, lift+cruise, vertiport operations | transition flight, battery reserve logic, passenger safety envelopes, fleet dispatch economics | missing |
| `symthaea-fixedwing` | long-range mapping, cargo relay, endurance cruise, passenger-aircraft foundations | lift/drag curves, stall envelopes, runway or catapult constraints, efficient cruise autonomy | missing |
| `symthaea-aerostat` | persistent relay / observation / mesh anchor | buoyancy and passive station-keeping are a different regime from powered flight | planned in Phase 2 |

**Naming recommendation**
- Package rename is now in place as `symthaea-multirotor`.
- Rust imports should use `symthaea_multirotor`.
- Add `symthaea-evtol` and `symthaea-fixedwing` as separate future crates rather than overloading the multirotor stack.

**Passenger-plane verdict**
- There is no real passenger-plane platform in-tree today.
- Passenger aircraft should not be modeled as a variant of the current quadrotor stack.
- The right path is either:
  1. `symthaea-fixedwing` first, then a passenger-aircraft configuration on top of it, or
  2. `symthaea-evtol` first if the near-term goal is decentralized urban air mobility rather than conventional airline simulation.

### Phase 3 additions — research / prestige platforms

These remain valuable, but they should not outrank the infrastructure/stewardship platforms above.

| Platform | Why it matters | Physics challenge | Networking challenge | Demo value |
|---|---|---|---|---|
| `symthaea-astrolabe` | Practical orbital backbone for relay, debris awareness, and LEO logistics | orbital mechanics, tethered maneuvering, power budgeting, debris interception envelopes | planetary relay continuity, sparse-asset coordination, solar-weather / hazard broadcast | very high |
| `symthaea-softbot` | New morphology/control research line | compliance, deformation, contact-rich locomotion | distributed control across many soft DOF | high |
| `symthaea-abyssal` | Deep-ocean stewardship and resilience | pressure, buoyancy extremes, low-visibility sensing | extreme latency/isolation, sparse relays | high |
| `symthaea-tesseract` | ND/4D prestige probe for the research story | 4D navigation and visualization | nonstandard spatial coordination semantics | very high |

**Orbital note**
- Existing orbital experimentation should converge toward `symthaea-astrolabe` if the goal is an operational platform rather than a prestige stub.

### Recommended platform sequence

1. Finish the current Phase 1 roster properly: manipulator, quadruped, humanoid, flight, vehicle.
2. Add `symthaea-subterranean` next; it is the single best new regime opener.
3. Add `symthaea-infrastructure` and `symthaea-scavenger` before more prestige mobility platforms.
4. Add `symthaea-agribot`, then `symthaea-terra`, to prove the stack can both steward living systems and repair damaged land.
5. Treat `symthaea-multirotor` as the hover/rotorcraft foundation, then add `symthaea-evtol` or `symthaea-fixedwing` when the aerial roadmap needs true passenger/cargo coverage.
6. Add the metabolic infrastructure layer next: `symthaea-stratum`, `symthaea-plexus`, `symthaea-foundry`, `symthaea-fabricator`, `symthaea-archive`.
7. Add the first civic/habitat platforms after that: `symthaea-clime` and `symthaea-biota`, then `symthaea-hearth`.
8. Keep `symthaea-vita` behind a stricter ethical and validation gate than other civic platforms.
9. Only then prioritize aerostat / weaver / brachiator as differentiators, with `weaver` carrying the macro-construction story.
10. Introduce `symthaea-astrolabe` once the ground, atmosphere, and civic stack are credible enough that an orbital backbone has something meaningful to serve.
11. Keep softbot / abyssal / tesseract as research heroes, not near-term schedule drivers.

**Near-term flagship thesis:** once subterranean + infrastructure + scavenger + agribot exist, the platform roster stops feeling like a robotics catalog and starts feeling like a civilization stack.

**Extended flagship thesis:** once the metabolic infrastructure and civic/habitat layers exist, and terra/remediation work closes the damaged-land loop, the stack stops feeling like an industrial robotics program and starts feeling like a plausible city substrate.

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
| Robot platforms running in-engine | 0 | 3 | 10 |
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
7. **Mycelix-in-Bevy** — three tracks across phases, *preferred architectural path is Track B*. Track A (WASM + CSS-3D iframe) covers browser-only hero demo. Track C (native Bevy UI screens) covers desktop-native player interaction. Track B (direct Holochain client as Bevy Resource) is the architectural foundation all three share. WebView-embed of real Leptos apps on native desktop is **not pursued** due to wry offscreen-rendering limitations.


