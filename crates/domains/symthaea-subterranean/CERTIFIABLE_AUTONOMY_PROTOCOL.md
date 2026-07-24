# Certifiable Autonomy Protocol

## Purpose

This protocol defines how `symthaea-subterranean` turns operational safety mechanisms into a bounded, reproducible release argument.

It does not claim regulatory certification, independent assurance, cryptographic identity verification, calibrated hardware performance, or suitability for deployment in a human-occupied mine. It provides the internal structures needed to assemble and reject a safety case honestly.

## Authority order

The physical command still follows the established authority ordering:

1. fused and integrity-checked observations;
2. physical hazard assessment and latching;
3. consciousness, moral, operator, degraded-operation, partition, and mission constraints;
4. learned nominal control;
5. verified recovery planning;
6. field power and thermal envelope;
7. maintenance derating;
8. actuator isolation;
9. independent final-command invariant monitor;
10. plant actuation and evidence recording.

The invariant monitor is the final internal authority. It may remove productive or motion authority but may not create it. Cooling and other survival actions are preserved only when the corresponding hardware remains available.

## Stable requirements

Release-blocking requirements have stable identifiers such as `SUB-SAF-001`. The canonical registry records:

- requirement title;
- criticality;
- expected verification method;
- whether the requirement blocks release.

Identifier stability matters more than prose stability. Scenario manifests, traceability links, safety claims, and release bundles reference identifiers rather than searching text.

## Runtime invariants

The final command is checked for:

- finite and bounded values;
- productive work under Red safety;
- motion during tunnel conflict;
- productive work with an infeasible return reserve;
- demand on isolated actuators;
- motion while capability disposition requires recovery hold;
- productive work under a sensor fault.

A violation modifies the same physical command, raises the reported safety tier to Red, selects `invariant_stop`, and records the exact invariant codes in bounded evidence.

## Fault-tree analysis

The canonical fault tree models five top events:

- uncontrolled productive motion;
- thermal runaway;
- entrapment;
- unauthorized recovery of motion;
- unsafe restart.

Minimal cut sets are bounded by event count and total set count. This protects the evaluator from unbounded combinatorial growth. The fault tree is a structured engineering argument, not a probabilistic risk calculation; event probabilities and hardware failure rates must come from external validated data.

## Reproducible scenarios

A scenario manifest contains:

- schema version;
- stable scenario identifier;
- deterministic seed phrase;
- fixed timestep and step count;
- consciousness value;
- physically validated initial-state overrides;
- requirement identifiers;
- acceptance criteria and tags.

Manifest fingerprints are order-independent for set-like fields. The included fingerprint is deterministic drift detection only and must not be treated as a cryptographic signature.

The scenario runner records final-state validity, battery, peak hazard, invariant breaches, and productive work under Red safety. Re-running an unchanged manifest must produce an identical report under the deterministic reference environment.

## Traceability

Every release-blocking requirement must link to at least one compatible verification artifact:

- deterministic test;
- runtime invariant;
- analysis report;
- evidence field.

The traceability gate rejects missing requirements, duplicate links, unknown scenario references, and verification-method mismatches.

## Safety case

The structured safety case contains one claim per requirement, a bounded argument, evidence references, and a disposition:

- supported;
- unsupported;
- rejected.

A release-eligible case requires all release-blocking claims to be uniquely supported and no modeled top event to remain active.

## Release signoff

Technical evidence alone does not authorize release. The signoff gate consumes externally verified approval assertions and enforces:

- three distinct hardware-backed signers;
- safety-engineer, verification-authority, and release-manager roles;
- no waiver for catastrophic requirements;
- two distinct hardware-backed roles for a non-catastrophic waiver;
- nonempty rationale and explicit waiver expiry.

Authentication, certificate validation, signature checking, and personnel authorization remain external responsibilities.

## Certification bundle

The deterministic bundle contains:

- build identity and source-tree identifier;
- requirement registry;
- scenario manifests and reports;
- traceability matrix and assessment;
- active basic faults and top events;
- minimal cut sets;
- safety case and assessment;
- release-gate result.

Bundle validation rejects missing reports, extra reports, duplicate scenario IDs, fingerprint drift, incomplete traceability, unsupported claims, and blocked release decisions.

The digest provider is injected. The built-in deterministic digest supports tests and accidental-change detection only. Production release bundles require an approved cryptographic digest and signature adapter outside this crate.

## Required external gates

Before any field claim, the complete workspace must additionally provide:

- Rust 1.94 formatting, Clippy, tests, and dependency review;
- real `symthaea-core` and `symthaea-fep` integration;
- authenticated operator, sensor, peer, and update transports;
- cryptographic bundle signing and protected key custody;
- calibrated sensor-independence evidence;
- hardware-in-the-loop actuator and watchdog testing;
- power-loss, thermal-soak, communications-partition, and restart testing;
- controlled 200 Hz latency measurements;
- independent safety review and applicable regulatory assessment.
