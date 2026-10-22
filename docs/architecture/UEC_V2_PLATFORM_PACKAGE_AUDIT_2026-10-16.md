# Universal Embodiment v2: package, consumer and platform audit

**Audit baseline:** `main@5e83cea4986ba3845aa50e0d5dd03493c5adeddf`
**Date:** 2026-10-16  
**Scope:** static source/manifest inventory and first compatibility-first implementation slice; no directories moved.  
**Tracking:** [ROB-UEC2-001 #7673](https://github.com/Luminous-Dynamics/symthaea/issues/7673)  
**Canonical owner:** [UEC v2 contract issue #1274](https://github.com/Luminous-Dynamics/symthaea/issues/1274) and [qualification program #1288](https://github.com/Luminous-Dynamics/symthaea/issues/1288).

## Executive result

The architecture should consolidate **semantic contracts, verification and tooling**, not collapse all vehicles and robots into one physics crate.

Current UEC v1 in `symthaea-core::embodiment` is a useful compatibility façade but does too much conceptually in `step(thought_hv, dt, phi)`: output decoding, safety gating, controller behavior, physics progression, perception encoding and telemetry can occur inside a single opaque transition. The typed V2 proposal adds independently inspectable observation and action boundaries while keeping the old entry point.

The first migration in this branch is the simple multirotor simulator, whose state has a bounded, explicit 13-channel representation and deterministic tests. It exposes 13 typed observations and 4 action quantities. The simulator action is explicitly marked `Applied` only relative to the in-process `SimplePhysicsSimulator`; this is not HIL, an external flight controller, or actual flight evidence.

## 1. Workspace manifest inventory — no moves

Static recursive-tree audit:

- Tree at the named baseline: 11,080 entries, not truncated.
- All tracked Cargo manifests: 254. This includes the root/workspace packages, nested ROS message crates, nested ZK-proof and fuzz packages, separate crates outside the three workspace globs, path-patched libraries, stubs and vendored upstream crates. A Cargo manifest is not automatically an active workspace member.
- Direct manifests matching workspace glob roots: 23 under `crates/core/*`, 157 under `crates/domains/*`, 20 under `crates/bridges/*`; 200 direct package manifests in total.
- The root workspace explicitly excludes the direct core package `nixward-web` and direct domain packages `spark-engine`, `symthaea-lab`, and `symthaea-perception`. Nested exclusions such as ZK-proof host/method crates and the spore fuzz crate do not add direct glob members.
- Under the manifest patterns, this gives 196 non-excluded direct manifests plus the root package and explicit `xtask` member: 198 candidate workspace members.
- Nested integration packages are tracked separately: five ROS 2 message crates are path dependencies under the ROS bridge; ZK proof host/method subpackages, the `symthaea-muse-ui/src-tauri` app and the spore fuzz manifest are not extra direct glob members. Standalone crates outside the glob roots, `stubs/`, `patches/` and `vendor/` remain separately categorized.

This is a **static manifest census**, not a successful `cargo metadata --workspace` execution. Exact active workspace resolution must be rechecked locally with Cargo; it should be added to the qualification receipt once available. The package tree remains unchanged.

## 2. Known path-consumer edges

### `symthaea-hal`

Manifest consumers found by static source search:

| Consumer | Current dependency form | Audit decision |
|---|---|---|
| Root `symthaea` package | Optional path dependency | Keep existing feature compatibility; verify feature graph |
| `symthaea-ssm` | Optional; `default-features = false` | Generic sensor/instrument path |
| `symthaea-auv` | Path dependency; `default-features = false` | Must not require humanoid control types |
| `symthaea-vehicle` | Path dependency; `default-features = false` | Must not require humanoid control types |
| `symthaea-multirotor` | Path dependency; `default-features = false` | Must not require humanoid control types |
| `symthaea-helicopter` | Path dependency; `default-features = false` | Must not require humanoid control types |
| `symthaea-exoskeleton` | Optional dependency; `default-features = false` | Explicit human-coupled adapter; test feature wiring |

The original manifest had an unconditional `symthaea-humanoid` dependency. The branch makes this dependency optional under the explicit `humanoid` feature (enabled by default for compatibility). This is a first dependency-boundary change, not yet proof that every consumer is clean.

**Reusable without humanoid semantics:** `embedded-hal::i2c::I2c`, shared-bus wrappers, generic `SensorDecoder`/`EmbeddedSensor`/`HalSensorAdapter`, sensor decoding, I2C device primitives, recording/replay, mock buses, and generic GPIO e-stop polling.

**Humanoid-specific today:** `HumanoidCommand`, fixed 21-joint ordering, PCA9685 output bound to two board addresses and 21 pulse values, `CalibrationProfile::default_21`, per-joint angle/torque/current arrays, and runtime callback signatures returning `HumanoidCommand`. These remain behind the compatibility feature; do not relabel them as universal actuators.

### `symthaea-humanoid`

Direct manifest consumers found in the baseline:

- root `symthaea` package (optional platform feature);
- `symthaea-hal` (made optional behind `humanoid`);
- `symthaea-ros-bridge` (made optional behind `humanoid-adapter`);
- `symthaea-mujoco-bridge` (explicitly humanoid simulation bridge);
- `symthaea-exoskeleton` (optional adapter; human-coupled control requires a separately reviewed contract).

Examples/tests in other parts of the tree reference humanoid types, but that does not prove their enclosing package has a direct normal dependency. The exact Cargo dependency graph still requires `cargo metadata`; do not infer it from arbitrary text hits.

### `symthaea-ros-bridge`

The package is described as a ROS 2 bridge but the baseline had a direct normal dependency on `symthaea-humanoid` and public helpers taking `HumanoidCommand` and returning `HumanoidState`. The branch gates those converters behind `humanoid-adapter` and adds a typed action-frame converter using explicit `RosJointActionBindingV2` entries.

The generic converter:
- validates the V2 action frame first;
- permits only `Requested` and `Projected` stages as new command requests;
- requires explicit per-address joint, quantity and unit mappings;
- rejects unbound action addresses, duplicate mappings, duplicate joint fields, quantity/unit mismatch, or incomplete vectors;
- outputs a validated trajectory value only; it neither publishes nor claims execution.

The existing Humanoid adapter remains for source compatibility. A further audit must confirm all feature combinations and preserve the exact previous default API.

## 3. Production v1 embodiment inventory and physical model ownership

Source search for concrete `EmbodimentBridge` implementations identifies the following implementation families. This is a semantic ownership list, not a claim each is qualified for real hardware. All except the explicitly noted simulator pilot remain V1-only in this branch.

| Adapter | Physical/semantic family | Physics/meaning that must remain specific | V2 status |
|---|---|---|---|
| `symthaea-multirotor::FlightEmbodiment` | Multirotor aircraft | Rotor thrust/moments, rigid-body flight, flight-specific fallback | **Pilot adapter added** (simple simulator only) |
| `symthaea-helicopter::HelicopterEmbodiment` | Helicopter | Main/tail rotor dynamics, autorotation and rotorcraft fallback | V1; audit next |
| `symthaea-vehicle::VehicleEmbodiment` | Wheeled ground vehicle | Tires, steering, traction, braking and road constraints | V1 |
| `symthaea-humanoid::HumanoidEmbodiment` | Biped | Whole-body/contact dynamics, balance, fall protection and joint morphology | V1 |
| `symthaea-quadruped::QuadrupedEmbodiment` | Quadruped | Four-leg contact, gait and terrain | V1 |
| `symthaea-exoskeleton::ExoskeletonEmbodiment` | Human-coupled orthosis | Shared human/robot authority and user-safe fallback | V1 |
| `symthaea-manipulator::ManipulatorEmbodiment` | Fixed-base industrial manipulator | Kinematics, collision and force/torque semantics | V1 |
| `symthaea-surgical::SurgicalEmbodiment` | Surgical precision robot | Constrained precision, tool/tissue context and domain risk | V1 |
| `symthaea-auv::AuvEmbodiment` | Underwater vehicle | Hydrodynamics, buoyancy, currents, depth and communications loss | V1 |
| `symthaea-orbital::OrbitalEmbodiment` | Orbital servicing arm | Orbital mechanics, free-flying base and contact dynamics | V1 |
| `symthaea-agribot::AgribotEmbodiment` | Agricultural robot | Soil/terrain, implements and crop/field context | V1 |
| `symthaea-subterranean::SubterraneanEmbodiment` | Subterranean crawler/driller | Ground contact, digging resistance, spoil and thermal load | V1 |
| `symthaea-scavenger::ScavengerEmbodiment` | Recovery/recycling platform | Grasping, cutting/sorting and uncertain feedstock | V1 |
| `symthaea-biota::BiotaEmbodiment` | Ecological/interspecies platform | Animal sensing, right-of-way and distress response | V1 |
| `symthaea-clime::ClimeEmbodiment` | Habitat/environmental control | Atmosphere, HVAC, thermal inertia and life-support constraints | V1 |
| `symthaea-infrastructure::InfrastructureEmbodiment` | Stationary infrastructure | Hub, storage, microgrid and routing state | V1 |
| `symthaea-phone-embodiment::PhoneBridge` | Screen/ADB environment | UI state, screenshot freshness and bounded device actions—not physical robot dynamics | V1 |
| Root `HeliosEmbodiment` | Legacy metabolic/organism bridge | Domain-specific state; not ordinary mobile-robot physics | V1/legacy |
| Root `DetritivoreEmbodiment` | Legacy recycling/shredder embodiment | Material handling and machine-specific interlocks | V1/legacy |
| Root `MotorBridge` | Deprecated humanoid compatibility surface | Legacy delegation to humanoid embodiment | Compatibility only |

A source search also finds platform plugin files for the above families plus a `symthaea-process-discovery` plugin path whose status as a physical embodiment adapter is unresolved. `symthaea-gravcraft` explicitly says it does not implement `EmbodimentBridge`, does not provide a plugin or enum variant, and has no workspace consumer; do not count it as a platform until those artifacts exist.

**AuditUnresolved:** exact full list of every trait implementation/alias and every path consumer should be re-run using local `rg` and `cargo metadata`. The linked list records identified implementation files and makes uncertainty visible rather than quietly asserting an exhaustive AST-level proof from search snippets.

## 4. UEC v1 audit

Existing source of truth: `crates/core/symthaea-core/src/embodiment.rs` and [#1274](https://github.com/Luminous-Dynamics/symthaea/issues/1274).

Observed V1 limitations:
- `step(thought_hv, dt, phi)` fuses multiple physical/control stages in one opaque call.
- `EmbodimentResult` contains an unqualified `prediction_error` and a derived observation-confidence proxy; those metrics should not be used interchangeably.
- V1 has a central `EmbodimentPlatform` enum and a registry keyed on that enum; preserve this as a compatibility route but use stable, extensible profile IDs for V2.
- Default `actuator_health()` and `sensorimotor_accuracy()` implementations can return perfect-looking values when a platform does not override them. These are compatibility defaults, not measured evidence.
- A backend/simulator state transition must be distinguishable from real applied/measured actuator outcomes.
- Clock semantics, frame graph, units, calibration currentness, observation missingness and action lineage should not be inferred silently.
- Safety state, capability, readiness and authority are different concepts. High Phi or a command accepted by middleware must not mint actuation authority.

These are also scoped by existing qualification owners: [#1279 health/reliability](https://github.com/Luminous-Dynamics/symthaea/issues/1279), [#1281 metric semantics](https://github.com/Luminous-Dynamics/symthaea/issues/1281), [#1284 time semantics](https://github.com/Luminous-Dynamics/symthaea/issues/1284), [#1276 safety/readiness/authority](https://github.com/Luminous-Dynamics/symthaea/issues/1276), and [#1288 qualification order](https://github.com/Luminous-Dynamics/symthaea/issues/1288).

## 5. V2 first slice on this branch

Added `crates/core/symthaea-core/src/embodiment_v2.rs` with:
- `EmbodimentDescriptorV2`: profile vs instance vs morphology vs schema IDs, clock domain, capabilities, fallback/calibration references and backend provenance;
- `ObservationFrameV2` / `ObservationValueV2`: address, quantity, unit, source clock, missingness reason and optional uncertainty;
- `ActionFrameV2` / `ActionValueV2`: explicit stage `Requested / Projected / TransportAccepted / Applied / Measured`, typed quantities and source provenance;
- a read-only shared `validate_adapter_v2_conformance` gate to check descriptor/frame identity and source consistency;
- negative tests for non-finite values, duplicate addresses, bad digest, ambiguous missingness and invalid uncertainty.

The multirotor simulator now implements `EmbodimentV2Adapter`, using a deterministic seed-derived logical ID by default and offering a caller-provided ID for multi-instance runs. It exports 13 state scalar observations with explicit SI units and a four-quantity action record. Reset increments a clock-domain epoch so simulation timestamps do not rewind within the same domain. It retains the V1 `EmbodimentBridge` method and includes V1 API parity/conformance tests.

The ROS bridge now converts named joint arrays into UEC V2 observations with explicit profile-specific quantity/unit bindings and explicit NotReported missingness; it also accepts V2 action frames through explicit profile/joint/quantity/unit mappings. Conversion does not mean transport-published; transport success does not mean a physical actuator executed it. ROS 2/DDS must remain isolated and appropriately authenticated as configured by the system integrator; this Rust adapter is not a middleware security control.

The HAL now has an explicit `humanoid` feature gate around fixed-morphology command, calibration, servo-output, interlock and runtime pieces; `humanoid` remains enabled by default for compatibility. Generic sensor/embedded-hal/recording/mock/GPIO interfaces remain available without that feature. This change must be confirmed with `cargo check -p symthaea-hal --no-default-features` and the existing default feature suite.

## 6. Common conformance suite: required negative fixtures

The shared validator is the canonical V2 gate, with the multirotor simulator as the first consumer. Expansion is blocked until these cases are deterministic and independently asserted:

1. Unknown contract/profile version; missing descriptor fields; mismatched profile or bound instance.
2. Duplicate action/observation addresses; non-finite values; invalid or absent units.
3. Missing observations represented without a reason; unknown uncertainty accidentally synthesized as zero.
4. Mismatched clock domains, non-monotonic sequence, timestamp absent or stale beyond the platform profile.
5. Empty or duplicate capabilities; expired calibration/evidence; missing safe-fallback profile.
6. A requested action mislabeled Applied; transport receipt promoted to execution; Applied source not bound to backend evidence.
7. A platform returns perfect health/accuracy defaults without measurement provenance.
8. Safety override absent/unknown and fallback unavailable; Red/unknown incorrectly promotes authority.
9. ROS action with no explicit joint binding, unit mismatch, duplicate fields, incomplete vector, stale clock or replayed action receipt.
10. V1 call still behaves as before for all existing migrated-platform tests.

An unknown/stale/missing evidence state must return failure or indeterminate, not be repaired by a plausible-looking value. No aggregate score may mask a failed safety-critical fixture.

## 7. Paradigm shifts and benchmark targets

### Shift A — capability contracts instead of platform labels

A new adapter should declare a versioned profile and capability IDs without needing a central closed enum edit just to describe its semantic identity. Keep the v1 enum for compatibility until a registry migration is proven.

**Target:** 100% of identified active adapters have unique profile IDs, schema versions, clock/frame assumptions, actuator capability IDs and fallback-profile references before they claim V2 support.

### Shift B — event-sourced sensorimotor digital thread

Record requested → projected → transport accepted → applied → independently measured as different typed events. Bind configuration, frame/time/unit semantics, calibration and backend/model version.

**Target:** 100% of V2 action records preserve their causal stage; 0 known cases where transport publication is mistaken for physical execution.

### Shift C — physics-family reuse without false universal dynamics

Share the observation/action/clock/evidence contracts, deterministic conformance fixtures and adapter tooling. Keep multirotor, helicopter, fixed-wing, underwater, wheeled, legged, manipulator and human-coupled models separate where governing physics and safety differ.

**Target:** every package mapped to one contract owner and one physics/model owner or marked `AuditUnresolved`; no new duplicate generic interface crate; no blanket “same dynamics” assertion.

### Shift D — fail-closed evidence, not synthetic certainty

Unknown health, calibration, clock and actuator state remain unknown. Compatibility defaults cannot serve as measured evidence.

**Target:** every common critical negative fixture fails closed; zero synthetic-perfect health/accuracy receipts accepted as measured evidence.

### Shift E — portable contract conformance as the adoption unit

An integrator should be able to qualify a new hardware backend using a reusable conformance harness before depending on the cognitive stack. This lowers the cost of adding a robot without pretending all robots share control laws.

**Target:** a new simulator/provider adapter can pass the same contract suite without changing core schema semantics; V1 + V2 coexist during migration; regression rate and adapter implementation time are measured per platform.

### Timing and transfer metrics

For each platform profile, report p50/p95/p99 observation-to-projected-action latency, p99 jitter, deadline misses, actual vs requested step dt, transport delay/loss/reordering, safe-fallback activation latency, false-pass and false-fail rates, and out-of-distribution transfer to held-out scenarios. Benchmarks must bind the simulator/backend and exact commit; simulation and HIL data are separate populations.

## 8. Migration sequence

1. Freeze current package manifests, `Cargo.toml` path edges, platform implementations and tests; leave directories untouched.
2. Preserve UEC v1 as compatibility façade; verify V2 base types and common validator.
3. Qualify only the simple multirotor simulator as the first V2 adapter.
4. Verify HAL generic/default feature builds and ROS default/no-default feature matrices.
5. Add humanoid adapter as the second platform only after the common negative fixtures pass.
6. Continue with wheeled and AUV interfaces after time/health/authority owners clear their prerequisites.
7. Consider domain folder hierarchy only after package and consumer ownership stabilizes and Cargo path migration is separately planned.

## Validation statement

This document was built from GitHub source/manifest contents and code-search findings. This environment did not execute `cargo metadata`, `cargo check`, `cargo test`, HIL or simulator benchmarks. The branch's newly added tests are test sources, **not test results**. CI queued/skipped/missing checks and results on a different head do not count as passing validation.
