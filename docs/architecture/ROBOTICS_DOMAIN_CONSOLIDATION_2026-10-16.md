# Robotics Domain Consolidation: Ownership, Boundaries, and Migration

**Date:** 2026-10-16  
**Status:** architecture audit / migration proposal; no bulk package move is authorized by this document.  
**Tracking issue:** [ROB-DOMAIN-001 #7650](https://github.com/Luminous-Dynamics/symthaea/issues/7650)  
**Coordination owners:** [Universal Embodiment v2 #1274](https://github.com/Luminous-Dynamics/symthaea/issues/1274), [UEC qualification program #1288](https://github.com/Luminous-Dynamics/symthaea/issues/1288), [sensorimotor schema #974](https://github.com/Luminous-Dynamics/symthaea/issues/974), [transport owner/gap audit #6245](https://github.com/Luminous-Dynamics/symthaea/issues/6245), [robotics design bootstrap #4795](https://github.com/Luminous-Dynamics/symthaea/issues/4795).  
**Cross-system physical evidence:** [Mycelix AeroCommons #3638](https://github.com/Luminous-Dynamics/mycelix/issues/3638) and [reproducibility challenge #3643](https://github.com/Luminous-Dynamics/mycelix/issues/3643).

## Decision

**Yes: consolidate robotics into clear domain families and shared contracts. No: do not merge all robot platforms into one crate, one physics model, or one safety case.**

The correct unit of consolidation is a *stable responsibility and evidence contract*. Keep different platforms as independently testable implementations when their morphology, environment, actuator topology, dynamics, hazards, or qualification evidence differ.

The first increment is a dependency/ownership audit and compatibility-preserving migration plan. Do not rename or move all existing crates before the consumers, path references, feature flags, tests, and evidence owners are mapped.

## Why this change is warranted

The public workspace contains separate packages for at least these embodied platforms:

- \`symthaea-multirotor\`
- \`symthaea-helicopter\`
- \`symthaea-auv\`
- \`symthaea-vehicle\`
- \`symthaea-quadruped\`
- \`symthaea-humanoid\`
- \`symthaea-manipulator\`
- \`symthaea-agribot\`

The code exposes repeated *roles* across platform modules: controller, embodiment adapter, encoder, active-inference agent, simulator, perturbation model, plugin, and training. For example, multirotor, AUV, humanoid, quadruped, and manipulator each declare some combination of these roles. That is evidence to compare the contracts and dependency direction—not enough evidence to merge their implementations indiscriminately.

Several existing issues already define the correct shared-contract owners. In particular, #1274 says the current \`symthaea-core::embodiment\` and \`PlatformRegistry\` are the real common substrate and explicitly says **not** to create a parallel \`symthaea-robotics-interface\`. This proposal preserves that decision.

### Direct dependency-direction findings to audit

**1. The hardware abstraction layer is currently humanoid-specific.**

The current \`symthaea-hal/Cargo.toml\` directly depends on \`symthaea-humanoid\`. Its library documentation describes a 21-servo \`HumanoidCommand\` path, with an example importing \`symthaea_humanoid::types::HumanoidCommand\`. That may be legitimate for its current Raspberry Pi humanoid target, but a reusable HAL should not *implicitly* require a humanoid topology.

Proposed resolution after consumer audit: retain portable hardware drivers, I/O primitives, interlocks, recording/replay, and calibration where truly generic; place the 21-joint command mapping, joint calibration topology, and humanoid-specific adapter behind the humanoid platform layer. Only split APIs where a second real consumer proves a useful shared contract.

**2. The ROS 2 bridge is also tied to the humanoid crate.**

The current \`symthaea-ros-bridge/Cargo.toml\` depends on \`symthaea-humanoid\`, while its description presents it as a Symthaea ROS 2 bridge that consumes sensor/proprioception data. A generic transport/middleware adapter should consume typed shared observation/action contracts and leave morphology-specific command mapping to a platform plugin.

PX4's official ROS 2 architecture illustrates why this seam matters: it separates flight-controller firmware topics, versioned message definitions, middleware, and companion-computer applications. PX4 also warns that the uXRCE-DDS transport can be unauthenticated and expose uORB directly, so connecting adapters require an explicit isolation and security boundary. A generic bridge cannot silently become a trusted control path merely because the data types compile.

Sources: [PX4 ROS 2 User Guide](https://docs.px4.io/main/en/ros2/user_guide); [PX4 uXRCE-DDS architecture and security warning](https://docs.px4.io/main/en/middleware/uxrce_dds).

**3. The workspace is a flat package catalogue, not yet a nested domain tree.**

The root Cargo workspace includes \`crates/domains/*\`. A physical move to nested directories would change many path dependencies and workspace glob behavior. The safer first phase should classify ownership without touching paths; move directories only after a consumer/manifest inventory and reversible migration plan.

## Proposed architecture

~~~mermaid
flowchart TD
    A["Symthaea cognitive / planning / learning"] --> B["Shared embodiment contract: existing owner #1274"]
    B --> C["Typed observation and semantic encoder boundary: #974"]
    B --> D["Requested → projected → applied action stages"]
    B --> E["Clock, frame, units, calibration, configuration identity"]
    D --> F["Platform-specific authority / interlocks"]
    C --> G["Aerial domain family"]
    C --> H["Ground-mobile domain family"]
    C --> I["Legged domain family"]
    C --> J["Manipulation domain family"]
    C --> K["Marine domain family"]
    C --> L["Field / composite embodiment"]
    G --> M["Multirotor / helicopter / fixed-wing specializations"]
    H --> N["Wheeled-vehicle specialization"]
    I --> O["Quadruped / humanoid specializations"]
    J --> P["Manipulator / end-effector specializations"]
    K --> Q["AUV / future surface-vessel specializations"]
    L --> R["Agribot = base + tool + environment composition"]
    M --> S["Validated physics, simulation, hardware and test evidence"]
    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S
    S --> T["SE-VV / ETK / external qualification evidence"]
    T --> U["Mycelix AeroCommons / manufacturing / physical-evidence lineage"]
~~~

This diagram shows logical ownership. It does not authorize any direct actuator path or make physics interchangeable across domains.

### Layer 0 — canonical shared contracts

Reuse and graduate the existing owners rather than creating parallel types:

- \`symthaea-core::embodiment\`, \`EmbodimentBridge\`, and \`PlatformRegistry\`; migrate via UEC v2 in #1274/#1288.
- Typed observation/action stages, explicit state/action schemas, capability/profile identity, sequence/time, lifecycle, and safe-fallback references.
- Role-bound sensorimotor semantics via #974: meaning is not a channel index; missing is not zero; frames and units remain explicit.
- Existing physical quantity / unit / frame semantics, clock/time semantics, calibration and configuration lineage; map to canonical owners rather than copying definitions into platform crates.

Shared contracts should remain narrow, typed, versioned, and testable without a specific robot's simulator or physics dependency.

### Layer 1 — cross-cutting infrastructure and bridges

Potential shared responsibilities:
- hardware I/O traits and vendor/device drivers;
- interlocks, watchdogs, e-stop input semantics and actuator-command rejection;
- calibration/recording/replay/telemetry;
- simulator and hardware-in-loop bridge protocols;
- ROS 2/PX4/ArduPilot/vendor adapters with versioned message mapping;
- common benchmark manifests, experiment identity and results/evidence envelopes.

The adapter layer translates; it does not invent physical meaning or approval. Keep bus transport, platform command topology, simulator behavior, and real hardware identity separately recorded.

### Layer 2 — domain families with separate physics

| Logical domain | Current platform owners to audit | Shared only where valid | Must remain platform-specific |
|---|---|---|---|
| **Aerial** | \`symthaea-multirotor\`, \`symthaea-helicopter\`; fixed-wing engineering remains tracked in #6249 and energy screening in #7646 | Flight-domain frames, mission/configuration identity, telemetry, energy interfaces, evidence vocabulary, selected envelope semantics | Rotor dynamics, helicopter rotor/actuator models, multirotor allocation, fixed-wing aerodynamics/aeroelasticity/stall/load models, eVTOL transition dynamics, aircraft-specific qualification |
| **Ground / wheeled** | \`symthaea-vehicle\` | Base observation/action contract, localization and resource semantics, vehicle configuration identity | Tire/contact, steering, braking, powertrain, crash/roadworthiness, surface/friction assumptions |
| **Legged** | \`symthaea-quadruped\`, \`symthaea-humanoid\` | Joint/contact representations, timing, actuator limits, common balance/ground-contact vocabulary when mathematically compatible | Morphology/DOFs, contact configurations, gait, centroidal dynamics, fall protection, footstep generation, humanoid-specific solver/benchmarks |
| **Manipulation** | \`symthaea-manipulator\`; humanoid manipulation consumers separately identified | Joint/actuator units, tool frames, contact/force observations, workspace-reference semantics | Kinematics, end-effector/grasp/contact models, torque/force limits, collision and task-specific assurance |
| **Marine / underwater** | \`symthaea-auv\` | Generic observation, navigation identity, energy, fault/state and mission contracts | Buoyancy, added mass, hydrodynamic drag, depth pressure, underwater acoustics, chemical sensors, thruster layout and marine safety cases |
| **Field / agricultural composite robots** | \`symthaea-agribot\` | Composition/mission and tool-interface contracts; reuse existing mobile-base or manipulation owner where proven | Tooling/implement dynamics, soil/water/crop models, contamination, field conditions, agronomic requirements |
| **Space / planetary** | \`symthaea-orbital\` and relevant rover/planetary owners to be enumerated by #6245 | Typed embodiment and evidence contracts | Orbital/attitude dynamics, vacuum/thermal/radiation, planetary gravity/regolith, life support and mission-specific qualification |

**Aerial should be a family of related specializations, not a universal flight-physics crate.** Similarly, “robotics” should be the composition and common-contract layer, not the owner of all robot dynamics.

### Layer 3 — qualification, physical evidence, and authority

Keep distinct:
- proposal / optimization candidate;
- analytical result and simulation result;
- hardware-in-loop result;
- fabricated/as-built configuration;
- commissioned device and calibration;
- measured physical result;
- professional engineering review and external approval;
- permission to operate.

Existing engineering verification/validation, formal safety, runtime authority, observation lineage and external domain owners remain authoritative for their own claim classes. A shared robotics API or common training result must not mint an aircraft, road, marine, or manipulator safety claim.

Mycelix should preserve component/configuration provenance, maintenance/inspection/test records and external attestations through AeroCommons. Consensus or a signed provenance receipt is not physical truth.

## Consolidation rules

1. **One semantic owner per shared contract.** Do not reimplement embodiment IDs, frame, clock, calibration, units, or action lineage in every platform crate.
2. **One model owner per physical theorem.** Similar type names do not justify sharing aerodynamic, hydrodynamic, tire/contact, or legged dynamics equations.
3. **Platform crates own morphology and application.** Keep actuator allocation, geometry, controller mapping, sensor placement and platform-specific fault response close to the platform that qualifies them.
4. **Generic dependencies point inward only to shared contracts/primitives.** HAL and generic bridges should not need a concrete robot crate to represent an abstract device. Exceptions need explicit named reason and tests.
5. **Adapters are not authority.** A ROS transport, simulator bridge, model, HDC encoder, or plugin registry cannot override plant interlocks and physical operating authority.
6. **Compatibility first.** Keep current package names and public entry points while shared v2 contracts graduate; migrate one platform at a time with explicit compatibility and golden-trace evidence.
7. **No layout-only refactor before dependency freeze.** A path move may make the tree prettier while leaving ownership tangled. Prove dependency direction before changing folders.
8. **No shared benchmark victory by score aggregation.** Report physics/task performance separately across platforms; use a Pareto view and preserve each domain's hard constraints.

## Benchmarks and exit gates

### Architecture / ownership
- **100% inventory coverage:** every active platform crate, public API, major modules, consumers, test harness, backend, hardware target, and evidence owner is mapped or explicitly \`AuditUnresolved\`.
- **Zero unnamed duplicate authorities:** each common contract has one canonical owner; platform implementations either conform or document a justified specialized contract.
- **Dependency-direction gate:** after migration, generic HAL and generic transport/ROS bridge surfaces do not require a concrete platform crate unless an exception is reviewed and documented.
- **No accidental physics unification:** aerial, wheeled, legged, manipulation and marine domain equations keep separate model identities and validity envelopes.

### Contract conformance
- Every supported platform emits typed, identity-bound observations and traceable requested/projected/applied action transitions where that layer is implemented.
- 100% of required negative fixtures fail closed: missing/stale sensors, wrong units/frames, changed firmware/controller/calibration, incompatible actuator map, stale model, rejected safety command, transport loss, simulator/physical evidence confusion, unauthorized actuation.
- Compatibility fixtures show that a contract extraction has not silently changed the legacy timing, action scaling, actuator limits, serialization, or fallback behavior.
- There is no generic success label that upgrades simulation to HIL/physical qualification or authority.

### Runtime / control performance
Measure **per platform and declared control-rate contract**:
- end-to-end control-loop latency percentiles and worst observed latency;
- jitter, deadline misses, scheduler overrun, sensor freshness and transport loss;
- memory/CPU budget and thermal/resource pressure on target hardware;
- fault detection and safe-rejection / fallback latency;
- physical/simulation divergence residual and uncertainty where measured.

Do not invent one universal 500 Hz/200 Hz/etc. rate for all robots. Declare required and allowable timing for each loop and verify on the exact target/backend.

### Transfer and generalization
- Evaluate shared sensorimotor semantics and any transferred policy on held-out morphology, task and environment splits.
- Compare against from-scratch and simple non-transfer baselines.
- Report success, energy, sample efficiency, physical constraint violations, recovery, and uncertainty separately.
- A transfer win in one simulator does not imply safe transfer to real hardware or another physical domain.

### Consolidated development workflow
- Scoped build/test entry points per platform/family, and a separate common-contract conformance lane.
- Exact commit/toolchain/dependency/backend/fixture identity in evidence records.
- CI queue, skipped workflow, missing run, and unexecuted local test are not PASS.
- Physical and HIL gates require their own evidence, not just source-level or simulator tests.

## Recommended migration order

1. **Freeze the package and owner inventory.** Keep the current paths unchanged.
2. **Audit #1274/#1288 v2 contract completion** against each platform's encoder, embodiment, control, safety, and backend.
3. **Investigate the HAL seam first.** Identify every use of humanoid-specific types. Keep generic device I/O and interlocks reusable; move platform-specific command mapping only when its compatibility behavior is characterized.
4. **Investigate the ROS bridge seam.** Replace concrete humanoid assumptions with typed UEC observations/actions and explicit platform-specific adapters; isolate/authenticate middleware and validate message/schema versions.
5. **Add common conformance fixtures.** Run the same missingness, identity, stale calibration, command rejection, and evidence-promotion adversarial cases for the migrated platform set.
6. **Graduate one specialization at a time.** Aerial first may be practical for the current work, but do not merge multirotor and helicopter physics. Compose fixed-wing #6249 and aircraft energy #7646 rather than creating parallel aircraft owners.
7. **Evaluate logical and physical directory taxonomy separately.** Only after package ownership stabilizes, decide whether nested paths improve navigation enough to justify the workspace-glob and path-ref migration. Preserve package names and roll back on consumer breakage.
8. **Expand to ground, legged, manipulation, marine, field, and space/planetary families** according to the frozen coverage matrix and evidence maturity.

## External integration references

- [Universal Embodiment v2 #1274](https://github.com/Luminous-Dynamics/symthaea/issues/1274) — canonical shared robotics contract; do not create a parallel interface.
- [Universal Embodiment qualification #1288](https://github.com/Luminous-Dynamics/symthaea/issues/1288) — evidence/authority migration sequence.
- [Sensorimotor semantic schema #974](https://github.com/Luminous-Dynamics/symthaea/issues/974) — portable semantic identity and missing/frame/unit guarantees.
- [Transport owner/gap audit #6245](https://github.com/Luminous-Dynamics/symthaea/issues/6245) — distinguishes shared transport semantics from mode-specific physics.
- [External robotics/transport benchmarks #6266](https://github.com/Luminous-Dynamics/symthaea/issues/6266) — freeze task-specific external benchmark coverage and explicit gaps.
- [Mycelix AeroCommons #3638](https://github.com/Luminous-Dynamics/mycelix/issues/3638) / [physical evidence lineage #3643](https://github.com/Luminous-Dynamics/mycelix/issues/3643) — artifact, as-built configuration and test/inspection provenance.
- [PX4 ROS 2 User Guide](https://docs.px4.io/main/en/ros2/user_guide) and [uXRCE-DDS warning](https://docs.px4.io/main/en/middleware/uxrce_dds) — middleware is a versioned, security-relevant integration boundary, not robot physics or actuation authority.

## Explicit nonclaims

This proposal does not claim that all platform implementations are duplicative, that the code is already consolidated, or that all platforms can share the same controller/dynamics. It does not establish robot competence, flight/road/seaworthiness, physical safety, HIL qualification, or external authorization. The next valid result is a verified ownership/dependency map followed by narrow, evidence-backed migrations.
