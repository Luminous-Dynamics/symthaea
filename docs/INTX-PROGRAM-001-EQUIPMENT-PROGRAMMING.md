# INTX-PROGRAM-001: Equipment Programming and Commissioning Contract

**Status:** design proposal; not a qualification result  
**Owner:** Symthaea Interaction Fabric (INTX)  
**Parent:** [INTX-000 / #4397](https://github.com/Luminous-Dynamics/symthaea/issues/4397)  
**Tracking issue:** [INTX-PROGRAM-001 / #7700](https://github.com/Luminous-Dynamics/symthaea/issues/7700)

## 1. Problem statement

Symthaea needs to work with existing industrial, medical, laboratory, building, and utility equipment. A transport connector is necessary but insufficient: successfully communicating with a controller does not prove a proposed program or configuration is correct, applicable to that exact model/firmware, safe for the process, or correctly applied.

The missing layer is a governed engineering lifecycle that composes protocol adapters, domain semantics, offline validation, artifact provenance, site/operator approval, effect authority, and post-change verification.

This document proposes that layer. It does not implement a device protocol, authorize live equipment access, or establish that any equipment family is qualified.

## 2. Core model

Use one shared lifecycle, with explicitly different device-family profiles:

    asset identity + native observations + current configuration evidence
      -> change request with explicit requirements and constraints
      -> candidate source/configuration (never authority)
      -> pinned toolchain + dependencies + compatibility checks
      -> offline validation / simulation / regression corpus
      -> independent engineering and safety review
      -> content-addressed release package
      -> separately authorized commissioning request
      -> adapter dispatch under current authority (only for an approved profile)
      -> readback + independent operational observations
      -> reconciliation + as-commissioned evidence

The shared lifecycle must reuse INTX identity, authority, credential-broker, preflight, reconciliation, and connector-conformance work. It must not create a second action state machine or new route from model output directly to a device.

Keep these propositions separate:

- protocol reachable != engineering authority;
- a valid configuration schema != a correct configuration;
- generated source != a correct program;
- successful compilation != safe behavior;
- a simulator pass != field qualification;
- a signed artifact != compatible target;
- controller acknowledgement != correct physical effect;
- matching readback != independent verification of process behavior.

## 3. Effect/qualification ladder

These are evidence tiers, not marketing labels or universal safety levels.

| Tier | Capability | Permitted claim |
|---|---|---|
| E0 | Bounded read-only inventory and observation | Observations were acquired under the declared profile; freshness and provenance remain visible |
| E1 | Offline candidate generation and diff | A candidate artifact and its inputs are content-bound; no device effect occurred |
| E2 | Static checks, build, vendor/native simulation, synthetic regression | The exact artifact passed only the listed checks in the stated environment |
| E3 | Signed/content-addressed export package with target compatibility and recovery notes | A reviewable package exists; signature does not imply correctness or deployment authority |
| E4 | Isolated lab or vendor-supported staging commissioning, after separate admission | The exact tested device/profile and commissioning procedure passed the listed lab gate |
| E5 | Named site/process field qualification after independent review | Only the explicitly scoped site, device/version, process, and change class are covered |

No tier inherits automatically from a previous device, firmware revision, model, site, toolchain, library version, or protocol adapter update. Missing prerequisites remain NotQualified or Unknown; they are never silently treated as a pass.

The first implementation must stop at E0-E3 and synthetic qualification. E4 is a later, named-profile tranche. E5 requires domain-specific safety, regulatory, and site governance.

## 4. Equipment profile: minimum semantic contract

A profile is versioned, reviewable, and bound to exact applicability. It should reference, rather than duplicate, shared INTX types.

Required information:

- **Identity:** profile ID/version, manufacturer, model/family, compatible hardware revisions, firmware/software version constraints, native endpoint identity, and the evidence source for each claim.
- **Transport:** protocol/version, authenticated endpoint/session properties where supported, gateway/interface scope, network-zone assumptions, and bounded request/response limits.
- **Resources:** canonical native resource/configuration/program identities; mapping from native identity to domain meaning; source and native timestamps; quality/status, units, scaling, and freshness policy.
- **Operations:** named observe, preflight, export, configure, program-transfer, activate, stop, and reconcile semantics only when supported. Unsupported operations must be absent or rejected, not generalized from a protocol's ability to write bytes.
- **Artifact:** source/configuration formats, toolchain and dependency identity, target compatibility, deterministic/canonicalization rules, content digest, and reviewable diff.
- **Validation:** static checks, simulation availability, known-answer corpus, external tool versions, coverage boundary, and known unsupported behaviors.
- **Authority:** observation/effect classification, resource-operation binding, required human presence, required approvals, purpose, context, maintenance window, and point-of-use currentness. Connector metadata and model output do not grant permission.
- **Outcome/recovery:** native acknowledgement semantics, retry/idempotency class, unknown-outcome behavior, readback method, recovery/rollback preconditions and limits, and independent evidence needed to establish behavior.
- **Safety and privacy:** safety-critical functions that are explicitly out of scope, domain/impact class, protected data classifications, redaction requirements, and mandatory external review.
- **Qualification evidence:** exact source commit and executable/package digests, fixture/corpus digest, test environment, result matrix, reviewers/approvals where required, expiry/supersession rule, and explicit nonclaims.

The profile must preserve native semantics. A common envelope must not make different error codes, state transitions, freshness properties, or authentication strengths appear equivalent.

## 5. Initial domain matrix

Protocol coverage and programming coverage are separate. A transport being supported does not mean program transfer is supported.

| Family | Candidate standards / interfaces | Initial work | Explicitly excluded from the initial tranche |
|---|---|---|---|
| PLCs and process controllers | OPC UA; Modbus TCP/RTU; IEC 61131-3 source/toolchains | Inventory, typed observations, offline logic/config candidates, static checks and available simulator tests | Generic memory/register writes, unreviewed logic download, live safety-PLC or interlock changes |
| CNC and industrial machines | Vendor/controller-specific APIs, OPC UA or MTConnect observations where available | Identify exact machine, tool/firmware profile, program artifact and preconditions; offline simulation and diff | Unqualified live toolpaths, bypassing guarding, or assuming a generic protocol equals a qualified machining operation |
| Industrial robots | Vendor programming environment; ROS 2 and relevant fieldbus/device adapters where appropriate | Offline program/trajectory artifacts, geometry/calibration/fixture preconditions, digital-twin or vendor simulation | Direct cognition-to-motion, unattended deployment, or bypassing safety-rated controllers |
| Buildings and facilities | BACnet, Modbus, OPC UA, MQTT/Sparkplug | Bounded inventory, read-first telemetry, configuration-diff and offline/staging validation | Fire/life-safety overrides or unreviewed changes to critical ventilation, access, or other safety-related functions |
| Power, water, and utility operations | IEC 61850, DNP3, other site-qualified SCADA/IED interfaces | Observation, asset/configuration inventory, offline settings artifacts, synthetic fault cases | Live breaker switching, protection-relay deployment, grid dispatch, or production water-process writes |
| Medical equipment | IEEE 11073 where implemented; DICOM and FHIR for their respective data/record scopes; vendor-specific interfaces | Inventory, provenance-aware data exchange, offline/vendor-approved non-clinical testing | Patient-connected actuation, therapy/medication changes, clinical alarm or limit changes, diagnosis/treatment authority |
| Laboratory instruments | Vendor-supported APIs and standard-specific interfaces | Inventory, calibration/configuration identity, non-clinical test artifacts and readback checks | Assuming an instrument result is ground truth without calibration, method, sample, and provenance context |

Domain and jurisdiction details need review before any live implementation. In particular, DICOM, FHIR, and IEEE 11073 serve different purposes; they must not be collapsed into one generic medical-control interface.

## 6. Programming and commissioning package

A release bundle should bind at least:

1. Equipment profile ID/version and exact target/model/firmware applicability.
2. Change request, explicit requirements, preconditions, constraints, and source snapshot identity.
3. Program/configuration source, generated artifacts, complete dependency and toolchain identities, build parameters, and resulting digests.
4. Static-analysis, type/range/interface, schema, compatibility, and regression results.
5. Simulator/provider identity, simulator version, environment/input-state snapshot, covered behaviors, and unsupported behaviors.
6. Human-review and safety-impact assessment references, plus approval scope and expiry; the package cannot self-approve.
7. Recovery/rollback plan, its preconditions, and evidence that recovery is available for the target.
8. Authorized target, permitted operation, maintenance window and deployment conditions, held separately from the artifact itself.
9. Native transfer/acknowledgement evidence, final readback, independent operational observations, reconciliation result, and nonclaims.

No secret, private key, bearer token, credential payload, or protected health information belongs in the ordinary artifact or trace. Credential use remains behind the existing broker. An artifact signature establishes a bounded integrity/provenance claim, not safety or correctness.

## 7. Fault and qualification benchmark

Begin with at least **50 deterministic synthetic cases** before connecting to real equipment. Freeze the corpus and expected outcome labels; bind corpus/version/digests to results. Keep positive, negative, and ambiguous-outcome cases.

### C0 — identity and compatibility (10 cases)

Cover wrong target, look-alike model, stale firmware compatibility, changed endpoint identity, conflicting native identity, namespace collision, incompatible toolchain, unknown profile version, untrusted discovered metadata, and stale inventory snapshot.

### C1 — artifact correctness and provenance (10 cases)

Cover source mutation, dependency mutation, non-reproducible build, invalid type/unit/range, unsupported function, incorrect parameter mapping, missing required constraint, digest mismatch, unsigned/invalid release package, and target-binding mismatch.

### C2 — policy and safety invariants (10 cases)

Cover absent/revoked approval, expired maintenance window, wrong purpose/context, safety-function modification, unauthorized operation, protected-data egress, simulation treated as field proof, connector metadata claiming authority, stale precondition, and attempt to use an out-of-scope clinical or utility function.

### C3 — transport, failure, and reconciliation (10 cases)

Cover dropped request before dispatch, timeout after possible commit, remote reject, stale replay, duplicate response, partial transfer, reconnect to changed server, readback mismatch, controller acknowledgement with no independent process evidence, and recovery not available.

### C4 — audit and release lifecycle (10 cases)

Cover source-to-binary digest change, toolchain supersession, profile supersession, missing reviewer evidence, expired approval, telemetry containing secret material, rollback incompatible with current state, deployment receipt bound to the wrong artifact, incomplete as-commissioned record, and re-qualification required after hardware/firmware change.

### Hard gates and independent metrics

- **False authorization:** zero dispatches in every frozen denied/negative-control case.
- **Identity integrity:** 100% rejection of wrong-target, stale-profile, and mismatched-artifact fixtures.
- **Outcome uncertainty:** every timeout after possible remote commit remains OutcomeUnknown until profile-specific reconciliation; no blind retry of unsafe operations.
- **Readback truthfulness:** all intentional source/config/readback mismatches are detected and never reported as clean deployment.
- **Evidence provenance:** every result binds exact tested source head, corpus digest, profile, tools, target model, and toolchain/environment identities.
- **Claim discipline:** tests and reports distinguish simulation evidence, lab evidence, and field qualification; no CI queue is counted as a pass.
- **Operational performance:** measure observation latency/throughput, resource limits, and recovery time only for a declared profile and fixture. Do not combine safety, capability, and latency into one score.

The acceptance target is exact known-answer behavior at the tested revision. Passing synthetic fixtures does not certify a real controller, prove safety, or establish a vendor-supported deployment path.

## 8. Implementation boundaries and sequence

1. Freeze profile vocabulary and ownership; reuse INTX core/authority/reconciliation types when their exact qualified revisions are available.
2. Add pure profile/manifest parsing and deterministic artifact identity rules, with unknown-version fail-closed tests.
3. Add source/configuration diff and synthetic fixture support without network or credential dependencies.
4. Add a simulation-provider interface that records the provider's actual claim ceiling and does not promote output to approval.
5. Add conformance adapters one at a time, initially read-only and tested against local emulators/reference servers.
6. Admit any staged commissioning profile only after its required authority, credential, reconciliation, fault, and safety dependencies are independently qualified.

This should be Rust-first and use existing workspace/toolchain governance. New dependencies or crates must follow the repository's current dependency and workspace conventions. This document alone is not permission to write a production connector or touch hardware.

## 9. Versioned external references

These references inform profile boundaries; they do not replace the applicable licensed standard text, vendor manual, local procedures, or jurisdiction-specific review.

- [IEC 61131-3:2025 — Programmable controllers: programming languages](https://webstore.iec.ch/en/publication/68533)
- [OPC Foundation — OPC UA Companion Specifications](https://opcfoundation.org/about/opc-technologies/opc-ua/ua-companion-specifications/)
- [Modbus Organization — specifications and implementation guides](https://www.modbus.org/modbus-specifications)
- [BACnet Committee — building automation interoperability standard](https://bacnet.org/)
- [IEC 61850-1-1:2026 — power utility automation overview](https://webstore.iec.ch/en/publication/105222)
- [DNP Users Group — DNP3 overview](https://www.dnp.org/About/Overview-of-DNP3-Protocol)
- [NIST SP 800-82 Rev. 3 — Guide to Operational Technology Security](https://www.nist.gov/publications/guide-operational-technology-ot-security)
- [CISA — ICS recommended practices](https://www.cisa.gov/resources-tools/resources/ics-recommended-practices)
- [FDA — medical device interoperability](https://www.fda.gov/medical-devices/digital-health-center-excellence/medical-device-interoperability)
- [IEEE 11073-00000 — point-of-care medical device communication framework](https://standards.ieee.org/ieee/11073-00000/3516/)
- [DICOM standard overview](https://dicom.nema.org/medical/dicom/current/output/chtml/part01/chapter_1.html)
- [HL7 FHIR specification](https://hl7.org/fhir/)

## 10. Nonclaims

This proposal does not establish support for any real device, device safety, functional-safety certification, clinical efficacy, regulatory compliance, field readiness, vendor approval, universal interoperability, or authority to change a live process. Those claims require their own exact-profile evidence and independent domain approval.
