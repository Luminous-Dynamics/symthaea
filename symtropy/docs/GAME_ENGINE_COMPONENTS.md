# Game Engine Components — Coverage & Gap Analysis

**Purpose.** Map what a production-grade game engine needs, what Symtropy covers today, what the Luminous Dynamics ecosystem (Symthaea + Mycelix) *already* solves, what Bevy provides transitively, and what genuinely remains as a gap with a recommended community crate.

**Why this exists.** The [ROADMAP](../ROADMAP.md) is phase-oriented and philosophy-first ("Bevy is the permanent foundation"). That framing hides a strategic truth: Symthaea + Mycelix already cover 8–10 components that traditional engines farm out — identity, AI, decentralized persistence, economy, attribution, telemetry. Naming that explicitly turns a licensing-split architecture into an *adoption story*.

**Scope.** This document is a living gap checklist; it doesn't replace the ROADMAP. When gaps close, prune rows. When new components are requested, add rows.

---

## Ownership legend

| Label | Meaning |
|---|---|
| `Symtropy` | Covered by a Symtropy core crate (permissive) |
| `Symthaea` | Covered by a Symthaea crate (AGPL — consciousness-coupled) |
| `Mycelix` | Covered by a Mycelix cluster zome/hApp (AGPL — decentralized) |
| `Bevy` | Covered by Bevy itself (recommended: use it) |
| `Community` | Missing from Luminous Dynamics; recommended Bevy-community crate |
| `Missing` | Genuinely missing, no clear community option |
| `Planned/PN` | In the ROADMAP under Phase N |

---

## Coverage matrix

### Core systems

| Component | Status | Notes |
|---|---|---|
| ECS / scene graph | `Bevy` | Foundation |
| 2D rendering | `Bevy` | Sprite pipeline |
| 3D rendering (PBR, shadows) | `Planned/P2` | Via Bevy `StandardMaterial` |
| ND rendering (4D cross-section) | `Planned/P2` | Miegakure-style slicer — research differentiator |
| Windowing | `Bevy` | |
| Audio (basic) | `Bevy` | `bevy_audio` / `bevy_kira_audio` |
| 3D spatial audio | `Community` | `bevy_kira_audio` with spatial feature |
| State-coupled audio | `Symthaea` / `Planned/P3` | `live-audio` crate, Muse integration |
| UI (retained) | `Bevy` | `bevy_ui` |
| UI (immediate mode) | `Community` | `bevy_egui` (recommended for data-heavy panels) |
| Asset pipeline + hot reload | `Bevy` | `AssetServer` |
| Scene format | `Planned/P2` | `.sym` extension for Φ-coupling params |

### Physics & motion

| Component | Status | Notes |
|---|---|---|
| Rigid body dynamics | `Symtropy` | `symtropy-physics` (ND) |
| GJK/EPA collision | `Symtropy` | Const-generic over D |
| Broadphase | `Symtropy` | LBVH with Morton encoding |
| Distance / Ball / Fixed / Hinge joints | `Symtropy` | All ND |
| Prismatic joint | `Planned/P1` | High-priority item |
| Motor/drive support (PD) | `Planned/P1` | Critical for robotics |
| Continuous collision | `Symtropy` | Swept sphere-sphere, sphere-halfspace |
| Ray casting | `Symtropy` | Analytical ray-sphere, ND-generic |
| Rapier3D bridge (high-fidelity 3D) | `Planned/P1` | Feature-flagged opt-in |
| Soft body / cloth | `Planned/P5` | Ecosystem crate `symtropy-soft` |
| Fluids | `Planned/P5` | Ecosystem crate `symtropy-fluid` |
| Triangle mesh collider | `Planned/P5` | Ecosystem crate `symtropy-mesh` |
| **Character controller** | **`Community` — GAP** | See detail below |
| Ragdolls / generic IK | `Symthaea` (partial) | DLS IK in `symthaea-manipulator` (7-DOF arm) |
| Vehicles | `Symthaea` (partial) | `symthaea-vehicle` bicycle model — not a generic controller |

### Animation

| Component | Status | Notes |
|---|---|---|
| Skeletal animation | `Bevy` / `Planned/P3` | `bevy_animation` |
| IK solvers (humanoid) | `Planned/P3` | Symthaea humanoid has one; needs generalization |
| Blend trees / state machines | `Planned/P3` | Φ-gated transitions |
| Morph targets | `Community` | `bevy_morph_targets` |
| Animation retargeting | `Missing` | Uncommon need in research games |

### AI / behavior

| Component | Status | Notes |
|---|---|---|
| **Consciousness-driven AI** | `Symthaea` | **Differentiator** — `symthaea-bevy-brain` integrates the cognitive loop as a Bevy component |
| FEP / active inference agents | `Symthaea` | `symthaea-fep` |
| Moral algebra / ethics gating | `Symthaea` | 16 duties, 92.9% Hendrycks ETHICS |
| Behavior trees (lightweight NPC) | **`Community` — GAP** | Recommend `big-brain` |
| GOAP | `Community` | `bonsai-bt` or custom |
| Pathfinding / navmesh | **`Community` — GAP** | See detail below |
| Flocking / boids | `Missing` | Trivial to add as example, not a crate |
| Crowd simulation | `Community` | `rvo2` for ORCA |

### Networking

| Component | Status | Notes |
|---|---|---|
| Basic client/server | `Planned/P4` | Lightyear wrap |
| Rollback / lockstep | `Planned/P4` | Leverages Morton + BTreeMap determinism |
| Interest management | `Planned/P4` | |
| **Decentralized P2P (DHT-backed)** | `Mycelix` | **Differentiator** — Holochain shared conductor |
| **Identity / DIDs / player accounts** | `Mycelix` | **Differentiator** — `mycelix-identity` W3C DIDs + ZKP |
| Cross-cluster state sync | `Mycelix` | `CallTargetCell::OtherRole` dispatch |

### Persistence

| Component | Status | Notes |
|---|---|---|
| Deterministic replay | `Symtropy` | `WorldSnapshot` + `ReplayTape` + `replay-cli` |
| **Save/load (arbitrary game state)** | **`Community` — GAP** | Recommend `bevy_save` |
| Cloud save (DID-authed) | `Mycelix` | **Differentiator** — personal-cluster vault |
| Save versioning / migration | `Community` | Pattern-level, not crate-level |

### Scripting / modding

| Component | Status | Notes |
|---|---|---|
| Scripting runtime | `Planned/P3` | Rhai first |
| WASM plugin host | `Planned/P3` | `wasmtime` sandbox |
| **Content attribution / provenance** | `Mycelix` | **Differentiator** — `mycelix-attribution` zome |
| **UGC marketplace** | `Mycelix` | **Differentiator** — `mycelix-craft` (talent + skills + reputation) |
| Mod management / versioning | `Missing` | Could layer on top of Mycelix attribution |

### Tools

| Component | Status | Notes |
|---|---|---|
| Scene editor | `Planned/P2` | `bevy_inspector_egui` — not a bespoke tool |
| Debug gizmos | `Planned/P2` | Wireframe colliders, contacts, joints, Φ heatmap |
| Replay scrubber | `Planned/P2` | Built on `symtropy-physics::replay` |
| **Profiler integration (Tracy)** | **`Community` — GAP** | Recommend `tracy-client` + Bevy tracing |
| **Observability (Prometheus)** | `Symthaea` | **Differentiator** — `symthaea/api/metrics` registry pattern |

## Art, assets, and production pipeline

Symtropy should treat art assets as a first-class production system, not as an afterthought.

The goal is not to replace human art direction. The goal is to automate the repetitive steps between concept art, free/public assets, Blender cleanup, Bevy import, and playable scenes.

### Recommended production rule

Use **CC0/public-domain assets** aggressively for prototypes and greybox replacement.

Do not use random “free” assets without license verification.

Every external asset must have provenance metadata even when attribution is not legally required.

### Asset pipeline ownership

| Component                    | Status              | Notes                                                                  |
| ---------------------------- | ------------------- | ---------------------------------------------------------------------- |
| Asset loading                | `Bevy`              | `AssetServer`, glTF/glb, image/audio loading                           |
| Hot reload                   | `Bevy`              | Use during local iteration                                             |
| glTF scene import            | `Bevy`              | Preferred runtime format for 3D assets                                 |
| Blender cleanup/export       | `Community/Tooling` | Use Blender Python scripts for batch conversion and cleanup            |
| Asset license manifest       | `Missing` — GAP     | Add `LICENSES.assets.toml` or `assets_manifest.toml`                   |
| Asset provenance checker     | `Missing` — GAP     | CI script should fail on unlicensed external assets                    |
| Concept-art-to-asset brief   | `Missing` — GAP     | Generate short asset briefs from concept art folders                   |
| Greybox replacement workflow | `Planned/P2`        | Replace primitive cubes with tagged prefabs                            |
| Material library             | `Planned/P2`        | CC0 PBR concrete, metal, glass, water, warning paint                   |
| Prop library                 | `Planned/P2`        | Pipes, valves, tanks, consoles, cables, crates, drones                 |
| Texture optimization         | `Community/Tooling` | Batch resize/compress textures for target quality tiers                |
| Thumbnail generation         | `Community/Tooling` | Generate previews for imported assets                                  |
| Bevy prefab metadata         | `Planned/P2`        | `.ron`/`.toml` metadata for tags, scale, collider hints, gameplay role |

### Recommended public asset sources

Prefer CC0/public-domain sources:

| Source     | Best use                                                             |
| ---------- | -------------------------------------------------------------------- |
| Kenney     | placeholder props, icons, UI, simple game assets, development kits   |
| Quaternius | low-poly 3D props, characters, buildings, vehicles, environment kits |
| ambientCG  | PBR materials, concrete, metal, ground, corrosion, HDRIs             |
| Poly Haven | HDRIs, high-quality PBR materials, selected models                   |

### Import policy

External assets should enter the repository through an explicit import folder:

```text
assets/external/cc0/<source>/<pack_or_asset_name>/
```

Symtropy-authored assets should live separately:

```text
assets/symtropy/<area_or_feature>/
```

Example:

```text
assets/symtropy/old_waterworks/
  greybox/
  imported_cc0/
  materials/
  prefabs/
  manifests/
```

### Required manifest fields

Every external asset should have a manifest entry:

```toml
[[asset]]
id = "ambientcg_wet_concrete_01"
source = "ambientCG"
license = "CC0"
original_url = "https://..."
imported_at = "2026-06-13"
imported_by = "agent"
local_path = "assets/external/cc0/ambientcg/wet_concrete_01/"
used_for = "Old Waterworks floor material"
notes = "Prototype material; may be replaced by Symtropy-authored final art."
```

### Automation targets

The art pipeline should automate:

* downloading from approved source lists
* verifying license metadata
* generating thumbnails
* normalizing file names
* converting models to `.glb`
* resizing textures
* generating material variants
* creating collider hints
* creating Bevy prefab metadata
* reporting triangle counts
* reporting missing textures
* reporting missing license fields

### Do not automate

Do not automate away art direction.

The following require human review:

* faction visual grammar
* Field Deck silhouette
* Null Ecology corruption language
* Archive Witness iconography
* Old Waterworks hero composition
* final key art
* any asset that becomes strongly associated with Symtropy’s identity

### Old Waterworks first asset target

The first asset automation target should be the Old Waterworks room.

Minimum useful asset list:

```text
wet concrete material
rusted metal material
pipe segment
valve wheel
industrial console
warning stripe decal
water tank
cable bundle
broken duct
amber glass/emissive screen material
```

Acceptance criteria:

* assets live under a clear folder
* every external asset has license metadata
* all runtime assets are loadable by Bevy
* greybox cubes can be replaced one at a time
* no unlicensed asset enters the repo
* no large asset dump is committed without curation

### Strategic principle

Free assets are scaffolding.

Symtropy’s identity must remain authored.

Use public assets to make the room exist sooner.

Use art direction to make the room unmistakably Symtropy.


### VFX

| Component | Status | Notes |
|---|---|---|
| **Particle systems** | **`Community` — GAP** | Recommend `bevy_hanabi` |
| Decals | `Community` | `bevy_mod_decals` |
| Trails / beams | `Community` | Part of `bevy_hanabi` |
| Post-processing | `Bevy` | Bloom, SSAO built-in |
| Screen-space effects | `Bevy` | |

### Input

| Component | Status | Notes |
|---|---|---|
| Keyboard / mouse (raw) | `Bevy` | `Input<KeyCode>`, `Input<MouseButton>` |
| Gamepad | `Bevy` | `bevy_gilrs` |
| **Action mapping / rebindable** | **`Community` — GAP** | Recommend `leafwing-input-manager` |
| HDC perception (high-level input) | `Symthaea` | Unusual pattern — bundles raw input into HDC vectors |
| Muse biometric bridge | `Symthaea` | `symthaea-muse` — optional feature |
| Accessibility (screen reader, colorblind) | `Missing` | Not in ROADMAP |

### Platform

| Component | Status | Notes |
|---|---|---|
| Linux | `Symtropy` | Wayland re-enable planned in Phase 0 |
| Windows | `Planned/P0` | Verify launch |
| macOS | `Planned/P0` | Verify launch |
| WASM | `Planned/P4` | Browser-runnable experiments |
| XR (OpenXR) | `Planned/P4` | `bevy_openxr` — hero demo is 4D in VR |
| WebXR | `Planned/P4` | Via WASM target |
| Mobile (iOS / Android) | `Missing` | Not in ROADMAP — should flag as "Will NOT Do" or add |
| Console | `Missing` | Certification out of scope |

### Developer experience

| Component | Status | Notes |
|---|---|---|
| Starter template | **`Missing` — GAP** | `cargo generate symtropy-starter` |
| Example gallery | `Planned/P0` | The Symtropy Book |
| Error messages | `Missing` | No explicit audit |
| Compile times | `Missing` | Unmeasured |
| Benchmarks (vs Rapier/xpbd) | **`Missing` — GAP** | Phase 0 has Criterion suite but no comparative data |
| Hot reload (assets) | `Bevy` | |
| Hot reload (scripting) | `Planned/P3` | Part of Rhai / WASM plugin story |
| Documentation | `Planned/P0` | The Symtropy Book (~22 stubs) |

---

## Key differentiators (the adoption story)

Traditional game engines treat these as platform-farm-out problems. Symthaea + Mycelix treat them as first-class engine components.

### 1. `Symthaea` — Consciousness-driven AI
`symthaea-bevy-brain` wires Symthaea's full cognitive loop (HDC perception, CfC temporal dynamics, predictive processing, Phi metrics, moral algebra) as a Bevy `Component`. This is **not a behavior tree** — it's a sophisticated decision-making substrate that traditional engines simply don't have.

**When to reach for Symthaea vs a traditional BT:**
- **Symthaea**: NPCs that should exhibit adaptive, ethics-aware, prediction-error-driven behavior. 12-region actor brain, ~31 Hz cycle. Heavy but principled.
- **`big-brain` (Community)**: Generic utility-theory AI — orcs patrolling, turrets aiming, resource-gatherers. Lightweight. Use when Symthaea is overkill.

### 2. `Mycelix` — Identity, persistence, economy, governance
Turns normally-platform-dependent systems into engine primitives:

| Traditional need | Mycelix primitive |
|---|---|
| Player account + auth | `mycelix-identity` W3C DIDs (`did:mycelix:...`) + ZKP selective disclosure (eIDAS 2.0) |
| Cloud save | `mycelix-personal` cluster (identity vault + health vault + credential wallet) |
| In-game economy | `mycelix-finance` (TEND time-exchange, demurrage, treasury) |
| Matchmaking / social graph | `mycelix-hearth` (kinship, gratitude, care, decisions) |
| UGC + royalties | `mycelix-craft` (talent marketplace, living credentials, endorsements) |
| Moderation / governance | `mycelix-governance` (proposals, voting, DKG threshold-signing) |
| Mod attribution | `mycelix-attribution` (dependency registry, usage receipts, reciprocity) |

No Holochain → none of this is reachable. But consumers who opt into the AGPL Symtropy + Mycelix stack get these for free. This is a **radically differentiated adoption argument** and should be surfaced in README material, not buried in the ROADMAP.

### 3. `Symtropy` + `Symthaea` — Observability done right
Symthaea's `MetricsRegistry` (Prometheus-compatible, in `symthaea/src/api/metrics.rs`) + `CycleMetadata` telemetry pattern is more sophisticated than what most Bevy crates ship. Documenting this as the recommended observability pattern for Symtropy games gives us a clean answer to "how do I see what my sim is doing?".

---

## The real gaps — concrete crate recommendations

For each `GAP` above, here is the concrete landing:

### Character controller
- **Recommend: `bevy_tnua`** — kinematic-on-physics abstraction, well-maintained, integrates with both Rapier and `bevy_xpbd`. Once Rapier3D bridge lands (Phase 1), `bevy_tnua` + `symtropy-rapier3d-bridge` is the default answer.
- For bipedal/physics-based characters, use `symthaea-humanoid` directly.
- **Action item:** add `docs/recipes/CHARACTER_CONTROLLERS.md` pointing to both.

### Pathfinding / navmesh
- **Recommend: `oxidized_navigation`** — active, Bevy-integrated navmesh, dynamic agent path planning. Alternative: `bevy_nav_mesh`, less active.
- For general A*/Dijkstra (not navmesh), the `pathfinding` crate (non-Bevy, but trivial to integrate).
- Symthaea's `commons_mesh-time` is about scheduling, not pathfinding — don't confuse them.
- **Action item:** add a recipe; no core crate needed.

### Behavior trees / lightweight AI
- **Recommend: `big-brain`** — utility-theory AI, most popular Bevy crate for this. Composes well with Symthaea: use `big-brain` for the low-Phi NPCs, reserve `symthaea-bevy-brain` for the high-consciousness ones.
- Alternative: `bonsai-bt` for pure BT semantics.
- **Action item:** doc recipe covering when to use each.

### Save/load (game state)
- **Recommend: `bevy_save`** — serde-based snapshot/rollback, widely used.
- Distinct from `symtropy-physics::replay` (deterministic tick-level replay — too fine-grained for "save game" semantics).
- For cloud save with DID auth, layer on top via `mycelix-personal` vault.
- **Action item:** doc recipe + optional `symtropy-save-mycelix` bridge crate.

### Particle systems / VFX
- **Recommend: `bevy_hanabi`** — GPU-based, mature, widely adopted.
- CPU alternative: `bevy_particle_systems` (useful when determinism matters).
- **Action item:** doc recipe, no core work.

### Input abstraction / action mapping
- **Recommend: `leafwing-input-manager`** — action-based, remappable, most popular.
- Symthaea's HDC-perception approach is orthogonal (for cognitive-loop input encoding), not competing.
- **Action item:** doc recipe; `leafwing` integrates cleanly with Bevy.

### Profiler integration
- **Recommend: `tracy-client` + `bevy_dylib`** — Bevy has first-class Tracy support via the `trace_tracy` feature.
- Symthaea's `MetricsRegistry` is complementary (runtime telemetry, not frame-by-frame profiling).
- **Action item:** feature-flag `tracy` on `symtropy-bevy-core`; document in Book.

### Benchmarks (comparative)
- No crate — this is a work item. Target: one Criterion suite with head-to-head runs at 100/1k/10k bodies vs `bevy_xpbd` and Rapier3D on common workloads.
- **Action item:** belongs in Phase 0 (visibility), gated by the showcase demo.

### Starter template
- No crate — this is a `cargo generate` template (repo: `symtropy-starter`).
- **Action item:** part of the Phase 0.6 "Demo & Visibility" addition proposed in the ROADMAP gap analysis.

### Accessibility
- Bevy community has minimal coverage. `bevy_accessibility_kit` is nascent.
- **Action item:** add an "accessibility not yet covered" note to README + open issue for community contributions.

### Mobile
- **Action item:** either commit to Phase 4 mobile verification (iOS/Android Bevy launch) or move to "Will NOT Do" explicitly. Currently silent.

---

## Summary: what the adoption-focused diff to the ROADMAP should look like

1. **Add Phase 0.6 — "Demo & Visibility"** — showcase example, benchmarks, starter template. Gates Phase 1 on visibility, not just publication.
2. **Add a `docs/recipes/` directory** populated with short guides for the nine `Community` crate recommendations above. Each recipe is ~50-100 LOC of Rust + README, under 1 hour to write.
3. **Surface the Symthaea/Mycelix adoption story in the top-level README** — current README is engine-focused. A "Why Symtropy + Symthaea + Mycelix?" section converts the licensing split into an adoption argument.
4. **Add explicit rows for the gaps** in the ROADMAP's success-metrics table: example count, benchmark coverage, recipe count.
5. **Answer the mobile/accessibility question.** Either commit or explicitly not-commit — don't leave them absent.
6. **Audit the determinism contract** — "same-CPU float, Morton integer broadphase, BTreeMap iteration" is the working claim; write down the test that proves it and the failure mode if a user violates the assumptions.

Implementing just items (1), (2), (3) would materially change whether a new visitor to crates.io understands what Symtropy is *for*.

---

## Strategic decisions (resolved 2026-04-18)

1. **Mobile/XR narrative.** Mobile (iOS/Android) is **"Will NOT Do"** for the core engine — standard mobile gaming dilutes the focus on high-fidelity simulation and decentralized trust. WebXR and OpenXR stay in scope; the 4D-in-a-headset hero demo is the research payoff.
2. **Mycelix requirement.** **Opt-in, first-class.** `symtropy-bevy` remains standalone (AGPL); `symtropy-mycelix-bridge` is an official add-on plugin, not a transitive dep. The starter template includes a commented-out Mycelix section with identity + cloud-save enablement as a one-line opt-in.
3. **Symthaea performance budget.** `symthaea-bevy-brain`'s ~30 Hz cognitive loop is heavy by design. Reserve it for **Hero NPCs or complex procedural agents** where full consciousness-driven behavior earns its CPU cost. For generic NPCs (guards, orcs, mob spawns), recommend **`big-brain`** (utility-theory AI, lightweight). Two-tier AI strategy — Symthaea for the few, `big-brain` for the many — is the documented guidance.
4. **Bevy major-version cadence.** **N−1 cadence.** Always target current stable Bevy (today: 0.18), but do not rush to update on release day. Wait until core community crates (`leafwing-input-manager`, `oxidized_navigation`, `bevy_tnua`, `bevy_hanabi`) have stabilized their ports before Symtropy itself moves. Accepts ~8-week lag in exchange for reduced breakage.

---

*Last updated: 2026-04-18. When a row transitions from `Community`/`Missing` to `Symtropy`-covered, prune the row and log the close in the ROADMAP's "Completed" section.*
