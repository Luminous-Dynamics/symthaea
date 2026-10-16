# INTX-PROGRAM-001: Paradigm Shifts and Benchmark Roadmap

**Status:** engineering proposal; targets below are not test results  
**Parent:** [INTX-PROGRAM-001 / #7700](https://github.com/Luminous-Dynamics/symthaea/issues/7700)  
**Executable corpus scaffold:** [PR #7701](https://github.com/Luminous-Dynamics/symthaea/pull/7701)  
**Next test integration:** [INTX-PROGRAM-BENCH-001 / #7702](https://github.com/Luminous-Dynamics/symthaea/issues/7702)

## Executive thesis

Do not define the goal as “an AI that can connect to more equipment.” Define it as **an evidence-carrying engineering compiler for existing equipment**: a system that translates human or machine-readable requirements into a target-specific candidate change, verifies it with independent oracles, and preserves a proof-shaped record of what was proposed, approved, attempted, and observed.

The protocol adapter should be a narrow I/O boundary. The engineering model, change artifact, verification suite, authorization gate, and reconciliation record should be reusable across device vendors and domains without pretending that all devices have equivalent safety or programming semantics.

## Six paradigm shifts

### 1. From protocol drivers to an equipment compiler

A driver moves bytes. An engineering compiler needs to preserve meaning across:

```
requirements and constraints
 -> canonical typed change intent
 -> target/profile-specific engineering representation
 -> vendor/native artifact
 -> static and dynamic validation evidence
 -> separately authorized commissioning
 -> observed and reconciled result
```

Proposed versioned objects (reuse equivalent existing Symthaea/INTX types where available; do not duplicate ownership):

- `EquipmentProfile`: exact model/firmware applicability, native resource semantics, allowed operation classes, data quality and known limitations.
- `AsFoundSnapshot`: evidence-bound, timestamp/freshness-aware source state; an observation, not unquestionable truth.
- `ChangeIntent`: typed requested change, requirements, invariants, scope, and intended effect.
- `EngineeringArtifact`: source/configuration, dependencies/toolchain, canonical digest, target compatibility, and diff.
- `ValidationBundle`: independent static, schema, compiler, simulation, fault, and review evidence with explicit coverage/nonclaims.
- `CommissioningRequest`: exact artifact/target/preconditions/current approval/window/recovery constraints; not itself authority.
- `CommissioningReceipt`: native response, readback, separate process observations, reconciliation, and unresolved discrepancies.

These are conceptual names, not a request to invent parallel canonical types. Start with an ownership map and reuse INTX identity, authority, credential brokerage, preflight, and outcome reconciliation.

### 2. From model-written control code to a deterministic backend

Symthaea may explain, synthesize, compare, and propose. It must not transform model output directly into arbitrary writes or live control commands.

1. Translate the requested change into a typed intermediate representation.
2. Reject unsupported or underspecified operations before code generation.
3. Bind every changed parameter/logic block to a requirement, constraint, and test.
4. Use a pinned device-specific compiler/toolchain/backend to emit candidate artifacts.
5. Run independent validators and preserve the exact inputs/versions/digests.
6. Keep deployment credentials and physical dispatch authority outside the generator.

This reduces hidden coupling between a language model, a vendor protocol and physical output. A compile pass proves only the compiler's stated property, not process safety.

### 3. From “digital twin as truth” to a multi-oracle evidence graph

Do not let one simulator grade its own output. Admit different evidence sources with explicit claim ceilings:

- syntax/type/schema checks;
- vendor compiler or native static-analysis output;
- independent invariants and range/unit checks;
- simulation or digital-twin trajectories;
- fault-injected protocol/service tests;
- exact-target readback;
- independent sensor/metrology/process evidence;
- review and site qualification records.

Store disagreement as a first-class discrepancy. Do not average away disagreement into one confidence score. Native acknowledgement, readback, and observed physical/process state remain distinct.

### 4. From vendor lock-in to semantic profiles and companion standards

Build one shared lifecycle with small, versioned native profiles, not one universal register map. Preserve native identity, status/error, units/scaling, clocks, quality, authentication, and state-machine semantics.

OPC UA already provides companion information models and profiles; its UAFX series includes an OfflineEngineering part (10000-083), Device Onboarding (10000-021), Programs (10000-010), Safety (10000-015), and other relevant parts. Where a device actually supports these parts, map the profile to them rather than inventing a competing representation. They do not imply that every device implements every part or that conformance grants Symthaea execution authority. See the [OPC Foundation specification index](https://profiles.opcfoundation.org/document/).

IEC 61131-3:2025 defines syntax and semantics for Structured Text, Ladder Diagram and Function Block Diagram, alongside Sequential Function Chart structuring and configuration elements. Use its defined language surface plus vendor-specific capability manifests; don't assume a source program is portable merely because two controllers claim the same language. See [IEC 61131-3:2025](https://webstore.iec.ch/en/publication/68533).

### 5. From one-click deployment to a controlled engineering change protocol

Make commissioning resemble a controlled transaction with explicit states, not a generic API call:

```
prepare snapshot + candidate artifact
 -> validate target, exact digest, preconditions and recovery
 -> independent review and separate current approval
 -> recheck authority/currentness at point of use
 -> bounded dispatch
 -> native response
 -> readback and independent observation
 -> reconcile final outcome or preserve OutcomeUnknown
```

Rollback is not universally safe or equivalent to restoring bytes. It is a new effect with its own preconditions, authority, verification, and possibly unavailable recovery. For some device states, the correct outcome after ambiguity is to halt further changes and require trained intervention—not retry.

Safety-critical local controllers and certified protective functions remain local. Network loss must not make Symthaea a required safety loop; “fail closed” for authorization does not mean blindly forcing every physical process to stop.

### 6. From happy-path demos to negative-control, mutation, and lifecycle qualification

The benchmark should chiefly test whether the system reliably refuses invalid work, preserves ambiguity, and detects divergence—not just whether it can issue a successful request.

Each connector profile should carry a machine-readable conformance contract: supported observation/effect operations, native state semantics, identity vectors, error mapping, idempotency/reconciliation class, freshness, bounded resource use, test corpus version, and exact implementation identity. Requalification follows relevant profile, firmware, toolchain, dependency, or adapter changes.

## Benchmark roadmap

The 50-case table in `crates/intx-program-bench/fixtures/cases.tsv` is the seed corpus. The current Rust harness checks the fixture structure and a few stated relationships; it does **not** yet execute the production action runtime or test a real protocol adapter. Do not report it as runtime conformance.

| Stage | Benchmark | Acceptance target | Evidence boundary |
|---|---|---|---|
| B0 | Corpus/schema integrity | Exactly 50 stable unique seeds across 5 dimensions; every row has a defined expected state, dispatch class, and reconciliation requirement | Validates data only |
| B1 | Authority negative controls | Zero dispatch permits for frozen denied, stale, wrong-target, incompatible, malformed, or safety-invariant cases | Exact tested software APIs; still not physical safety |
| B2 | Outcome semantics | 100% of post-dispatch ambiguous cases remain `OutcomeUnknown` until reconciliation; no automatic unsafe retry | Synthetic tests against the existing runtime path |
| B3 | Semantic preservation | 100% preservation of safety-relevant native identity, type, unit/scaling, quality/status, timestamps, source provenance, and native error/finality distinctions in frozen vectors | Only declared profile fields |
| B4 | Artifact provenance | Deterministic builds yield identical digests for identical pinned inputs; nondeterminism is explicit and fully recorded rather than misreported as reproducible | Exact backend/toolchain |
| B5 | Change traceability | 100% of changed parameters/logic blocks link to requirement(s), constraint(s), validator(s), and the exact artifact digest; no orphaned safety-relevant changes | Artifact corpus and declared safety model |
| B6 | Differential validation | Any deliberately injected disagreement between static checker, simulator, expected invariant, or readback is surfaced as discrepancy; no majority vote may erase a safety violation | Synthetic and named-lab evidence, each labeled |
| B7 | Cross-implementation portability | Same golden semantic suite passes on two independent implementations of a declared protocol subset; test Modbus separately from OPC UA rather than claiming shared wire semantics | Protocol interoperability only |
| B8 | Mutation/fault campaign | Start with the 50 seeds; generate at least 250 deterministic mutations across identity, data typing, staleness, approval, artifact digest, replay, timeout, and reconciliation. Zero hidden dispatch bypasses | Exact code/corpus/tool versions; expand based on found defects |
| B9 | Staged commissioning | Every injected failure point leads to a documented safe state or unresolved/manual-recovery state; every live profile has its own independent approval and field qualification | Named target/site/process only |

### Metrics that must remain separate

Report separate counts/rates for:

- unsafe false authorization;
- false rejection of a valid, explicitly permitted synthetic request;
- semantic mapping loss and unit conversion errors;
- stale-data and quality-flag preservation;
- artifact reproducibility;
- ambiguous outcomes and reconciliation completion;
- duplicate/replay suppression;
- discrepancy detection latency;
- resource consumption and observation latency under a declared workload;
- effort required to add or update a device profile after a measured baseline;
- unresolved coverage, unsupported functions and recovery limitations.

Do not create a single “industrial readiness score” that lets throughput offset a dangerous false permit. Safety/security invariants are hard gates; latency and engineering effort are independent performance metrics. Any target requiring real measurements needs a baseline before claiming a percentage improvement.

## Immediate implementation order

1. Keep the first pilot offline and device-free.
2. Use the dependency-free corpus package for focused feedback, while explicitly acknowledging that it validates corpus structure only.
3. Implement [INTX-PROGRAM-BENCH-001 / #7702](https://github.com/Luminous-Dynamics/symthaea/issues/7702): bind each seed to the existing authority and reconciliation paths through synthetic adapters; do not create a second action runtime.
4. Confirm exact action-runtime/INTX API ownership at the tested branch head before writing the integration layer.
5. Next, add one read-only OPC UA profile against a reference/test server and preserve native status/time/type metadata. Keep it separate from any later effectful programming profile.
6. Introduce offline source/configuration generation and differential validation only after the target language/vendor toolchain and exact target profile are declared.
7. Require a documented independent review and target-specific qualification before any live commissioning pilot.

## Safety and scope boundary

The initial tranche provides offline engineering, read-only inventory, synthetic test evidence, and reviewable candidate artifacts. It does not authorize live PLC writes, patient-connected actuation, therapy/medication changes, protection-relay programming, breaker switching, or other critical physical effects. Medical and utility integrations need their own regulatory, domain, and site qualification; standard-compatible communication is not a substitute.

## References

- [IEC 61131-3:2025 — Programmable controllers: programming languages](https://webstore.iec.ch/en/publication/68533)
- [OPC Foundation — official specification index, including UAFX OfflineEngineering and device/profile parts](https://profiles.opcfoundation.org/document/)
- [OPC Foundation — UA Companion Specifications](https://opcfoundation.org/about/opc-technologies/opc-ua/ua-companion-specifications/)
- [NIST SP 800-82 Rev. 3 — Guide to Operational Technology Security](https://www.nist.gov/publications/guide-operational-technology-ot-security)
- [FDA — Medical Device Interoperability](https://www.fda.gov/medical-devices/digital-health-center-excellence/medical-device-interoperability)
- [CISA — Industrial Control Systems recommended practices](https://www.cisa.gov/resources-tools/resources/ics-recommended-practices)

Passing this plan's seed corpus is not equipment certification. Every claim must remain bound to exact tested source, corpus, adapter/profile, target, environment and evidence identity.
