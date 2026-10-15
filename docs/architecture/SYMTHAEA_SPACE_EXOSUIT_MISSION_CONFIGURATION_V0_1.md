# SX-031: Mission-Level Suit Configuration Verification v0.1

**Status:** experimental software contract checker; synthetic tests only  
**Implementation:** suit_mission_configuration  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/7520  
**Predecessor:** SX-030 pairwise contracts, PR #7518

## Paradigm shift

Pairwise compatibility is necessary but not sufficient. A suit may match a lander interface while the mission as a whole lacks rescue support, a required connection, a required component, or a connected path to a required worksite. SX-031 treats the **declared mission configuration as the unit of verification**.

The model uses four explicit distinctions:

1. **Components are not inferred.** A configuration pins component IDs, hardware revisions, interface manifests, and a suit variant/profile. It does not fill in missing vehicles, service stations, evidence, or adapters.
2. **Requirements are explicit.** Required component kinds and required connection IDs are caller-owned declarations. An absent required component or connection is emitted as a blocker rather than silently removed from the test set.
3. **Required failure and optional degradation differ.** An incompatible required connection blocks the declared policy result. An incompatible optional connection yields a degraded result. An empty required-interface set is never an implicit pass.
4. **Receipts bind exact inputs and evaluated outputs.** Canonically ordered components, connections, capability claims, interface manifests, policies, and result fields contribute to SHA-256 digests. These hashes provide deterministic change evidence, not signatures, provenance authentication, or qualification.

This remains deterministic conventional verification logic. Symthaea/HDC/Phi is not allowed to waive a missing requirement, alter contract revisions, infer a connector adapter, or authorize a physical operation.

## Result vocabulary

| Result | Meaning |
|---|---|
| Blocked | At least one required condition is missing, disconnected, or below its declared software/evidence policy. |
| Degraded | No blocker was found, but at least one optional connection does not meet its declared compatibility policy. |
| MeetsDeclaredMissionConfigurationPolicy | The explicitly declared graph and evidence claims satisfy this software policy only. |

Every report sets represents_physical_qualification = false. Evidence labels such as BenchTested or Qualified are caller-declared maturity labels in the existing model; this evaluator does not certify their authenticity or sufficiency.

## v0.1 input contract

A mission configuration pins:

- schema version, mission ID and configuration revision;
- one suit component ID, suit mission profile, exact variant/hardware revision, capability claims and minimum evidence floor;
- component IDs/kinds/revisions with a manifest whose identity must match its parent component;
- required component kinds and separately declared required connection IDs;
- connection endpoints and explicit pairwise interface policy, including a non-empty operating-environment set.

Structural ambiguity (duplicate IDs, duplicate component pairs, wrong manifest identity, invalid scopes, missing suit root) returns an error. Mission deficits that can be meaningfully diagnosed (missing required support role or connection, disconnected required node, insufficient profile evidence, incompatible required connection) return a deterministic blocked report.

A declared graph edge is topology only; it is not a physical wire/coupling, power budget, resource-flow proof, or safe rescue path. Pairwise matching continues to use SX-030's exact contract revision, schema digest, connector profile, environment and evidence floor.

## Synthetic fault/replay corpus

The crate currently includes unit cases and an integration replay for these properties:

| Scenario | Expected result |
|---|---|
| Suit + lander with synthetic matching lunar grounding declarations | Meets declared policy; physical-qualification flag remains false |
| Missing required suit–lander edge | Blocked: missing required connection and disconnected required component |
| Required component kind absent (for example, rescue capability) | Blocked |
| Required edge uses evidence below caller floor | Blocked |
| Required edge declares no required interfaces | Blocked; not an implicit pass |
| Failed optional edge while required graph remains connected | Degraded |
| Suit capability evidence bound to a different revision | Invalid configuration |
| Duplicate component pair or duplicate connection identity | Invalid configuration |
| Change configuration revision after report creation | Receipt verification fails |
| Reverse input vector ordering without changing meaning | Report and receipt are identical |
| Synthetic graph with 2, 3, 10, 34 and 130 components | Same declared-policy outcome; component/edge counts remain exact |

All scenario artifacts and identifiers are synthetic. No thermal, electrical, electrostatic discharge, pressure, ECLSS, mechanical, human-factors, or rescue hardware behaviour is simulated by these tests.

## Benchmark aims

These are **proposed acceptance targets, not measured results**. Freeze the runner CPU, OS, Rust toolchain, build profile, scenario generator seed and graph structure before converting timing targets into CI gates.

| Benchmark | Initial target |
|---|---|
| Required-condition negative controls | 100% blocked for every seeded missing/ambiguous/out-of-scope required condition |
| Optional-condition controls | 100% degraded (not blocked) when optional links fail and no blockers exist |
| Receipt tamper/mutation tests | 100% rejection for changed manifest, policy, graph, revision, requirement set or result |
| Determinism | Identical report and receipt across 1,000 permutations of semantically unordered input vectors |
| Fault-campaign coverage | Single faults + pairwise combinations, with every declared fault mapped to an expected blocker/degradation/result |
| Generated graph corpus | 2,000 seeded graphs including valid cases, missing nodes/edges, duplicate IDs, disconnected nodes, scope mismatches and optional failures; zero unexpected panics |
| 128 components / 256 connections | p95 evaluation under 25 ms on a documented reference runner |
| 1,024 components / 4,096 connections | p95 evaluation under 250 ms and peak RSS under 256 MiB on the same runner |

For correctness benchmarks, report counts and denominators—not just percentages. Any change to the generated corpus must preserve known-positive cases and negative controls. A test that never exercises a missing mandatory interface cannot support a zero-false-acceptance claim.

## Next paradigm shifts

1. **Capability-to-task proof graph:** map each EVA task to required capabilities, resources, location/environment limits, and evidence artifacts. A mission is blocked when any mandatory dependency has no admissible evidence path.
2. **Explicit adapter nodes:** interoperation between unlike versions or connectors is possible only through a declared adapter component with its own exact contract, version, tests and evidence. Never auto-insert adapters.
3. **Mission-wide fault injection:** remove or replace the rover, service node, relay, rescue option, suit module or evidence artifact; corrupt one contract digest; alter one environment; or degrade optional telemetry. Verify outcomes against independent expected dispositions.
4. **Minimal blocker explanations:** produce a deterministic, human-reviewable cut set of missing or blocked dependencies to help a reviewer see the smallest set of declarations that prevents the mission graph from meeting policy.
5. **Independent oracle:** keep a separate replay implementation for canonical digests and expected outcomes. Do not let the production evaluator generate its own gold expectations.
6. **Evidence provenance:** add signed/artifact-backed records only when signature policy, key ownership, revocation, timestamp semantics and artifact retention are designed. A digest alone is not authentic evidence.

The current implementation begins the graph, receipt, negative-control and scale-corpus work; it does not yet implement task planning, cut-set optimization, stochastic reliability, physical simulation, FMEA, fault-tree analysis, safety-case approval, or hardware qualification.

## Research basis

The project direction responds to public interface and human-factors work that emphasizes standards at subsystem boundaries rather than mandating a single design. NASA's April 2026 OIG report recommended interoperability standards between Artemis lunar vehicles and spacesuits; NASA's International Deep Space Standards describe a set of cross-program interface standards. NASA's Lunar Grounding Challenge, opened October 5, 2026, highlights lunar electrostatic charging and discharge risks, including the distinct conditions of shadowed regions. These sources motivate explicit interface/environment evidence, not a claim that this repository meets NASA requirements.

- NASA OIG, IG-26-006: https://oig.nasa.gov/news/spacesuit-development-delays-could-impact-artemis-timeline-and-iss-operations/
- NASA International Deep Space Standards: https://www.nasa.gov/international-deep-space-standards/
- NASA Lunar Grounding Challenge: https://www.nasa.gov/directorates/stmd/prizes-challenges-crowdsourcing-program/center-of-excellence-for-collaborative-innovation-coeci/lunar-grounding-challenge/
- NASA-STD-3001 Volume 2, Revision F: https://standards.nasa.gov/node/237
