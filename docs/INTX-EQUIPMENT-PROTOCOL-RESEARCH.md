# INTX Equipment Interoperability: Protocol and Test-Stack Research

**Date checked:** 2026-10-16  
**Status:** source-based candidate assessment, not implementation or certification  
**Parent design:** [INTX-PROGRAM-001](./INTX-PROGRAM-001-EQUIPMENT-PROGRAMMING.md)  
**Paradigm roadmap:** [INTX-PROGRAM-001 paradigm shifts](./INTX-PROGRAM-001-PARADIGM-SHIFTS.md)

## Executive recommendation

First prove the equipment semantic and authority boundaries with a **read-only OPC UA profile** against an independent test server. Do not start with live write methods, PLC download, or a broad driver catalog.

Use the public standards and candidate implementations below as evidence for a tightly scoped adapter spike. No library should be added to the production workspace until exact version, Rust toolchain compatibility, enabled features, security configuration, maintenance posture, and license obligations have been reviewed.

## Standards-first: do not invent another engineering descriptor

The OPC Foundation's current UAFX OfflineEngineering specification is **OPC 10000-83 v1.00.03**, published June 8, 2026. It defines a data format/workflow for sharing AutomationComponent information during offline engineering and later commissioning; the defined Descriptor can bundle information models and other artifacts and has signature requirements. Use it as an interoperability profile when a target/toolchain supports it, not as a universal claim that any OPC UA server can import a program.

- [Official Part 83 specification and current version](https://reference.opcfoundation.org/specs/OPC-10000-83)
- [Part 83 full text / scope and workflow](https://reference.opcfoundation.org/specs/OPC-10000-83/full)
- [OPC UA specification index](https://profiles.opcfoundation.org/document/)

IEC 61131-3:2025 defines the syntax and semantics of Structured Text, Ladder Diagram and Function Block Diagram, plus Sequential Function Chart structuring and configuration elements. It is a language standard, not a universal vendor-neutral live deployment protocol. A matching language name alone must not be treated as binary compatibility, device compatibility, or verified process correctness.

- [IEC 61131-3:2025](https://webstore.iec.ch/en/publication/68533)

## Candidate implementation stack

### 1. OPC UA client/server library candidate: async-opcua 0.19.0

Sources:
- [FreeOpcUa/async-opcua](https://github.com/FreeOpcUa/async-opcua)
- [async-opcua 0.19.0 package documentation](https://docs.rs/crate/async-opcua/0.19.0)

The project documents a Rust client/server implementation. The 0.19.0 package was published in July 2026, and the upstream repository was updated in October 2026. The upstream project states it targets current stable Rust and does not promise support for older compiler versions; Symthaea currently pins Rust 1.96.0. This is a **compatibility gate, not an automatic dependency choice**.

Before adding it to Symthaea:
- prove it builds under the exact pinned 1.96.0 toolchain and required target(s);
- pin exact versions and review the full enabled-feature dependency graph;
- test certificate trust, endpoint/application identity, security policy and mode, user/session permissions, timeouts, reconnect identity, bounded message sizes, and stale-subscription handling;
- verify required native status, source/server timestamps, quality, NodeId/NamespaceUri, data type, and units survive the adapter without lossy normalization;
- inspect upstream compatibility notes and licensing (MPL-2.0) for the intended distribution model;
- keep effectful services/writes outside the first adapter; do not assume support for an OPC UA method means Symthaea is authorized to invoke it.

If Rust 1.96.0 compatibility fails, do not silently upgrade the repository toolchain or weaken workspace pins. Record the incompatibility and evaluate a compatible exact version or a separately isolated bridge.

### 2. Independent OPC UA conformance/test target: open62541 v1.5.9

Sources:
- [open62541 project](https://github.com/open62541/open62541)
- [v1.5.9 release notes](https://github.com/open62541/open62541/releases)

The project is an OPC UA client/server implementation in C under MPL-2.0. Its v1.5.9 maintenance release was published October 4, 2026. That makes it a strong **independent server/test implementation** to evaluate against a Rust client profile. It should be run as an isolated test service or container with synthetic data, not treated as a dependency that Symthaea needs to call through C in production.

Test objective: the same frozen Rust-side golden vectors pass against this separately implemented server, including namespace identity, typed values, quality/status, timestamp behavior, subscriptions, certificate failures, timeouts/reconnects and resource bounds. This demonstrates interoperability for only the exercised profile, not full OPC UA conformity or device safety.

### 3. Modbus candidate: tokio-modbus 0.17.0

Sources:
- [tokio-modbus upstream](https://github.com/slowtec/tokio-modbus)
- [tokio-modbus 0.17.0 documentation](https://docs.rs/tokio-modbus/0.17.0/tokio_modbus/)

The library documents a pure Rust implementation supporting asynchronous and synchronous Modbus TCP/RTU client usage and server functionality. Consider it a separate, subsequent adapter, not a fallback with equivalent identity guarantees.

Modbus addresses, unit IDs, and successful function responses do not intrinsically authenticate a device or establish a physical effect. Any configured mapping must therefore be versioned/content-bound, and the initial profile must use bounded reads with source/gateway provenance. Generic raw register/coil writes are not the semantic API.

## Recommended first pilot

Scope:
- one read-only OPC UA client profile;
- synthetic address space with stable, versioned expected values;
- an independent open62541 test server;
- no production plant, patient-connected equipment, utility switchgear, or writes;
- exact Rust 1.96.0 toolchain and lockfile review before selecting the Rust library;
- evidence receipt binding source head, library versions, server version/config, golden corpus digest, test results, certificate profile, and nonclaims.

Minimum protocol-level cases before any device vendor trial:
1. namespace URI plus NodeId identity survives NamespaceArray remapping;
2. scalar/array/structured types do not silently coerce;
3. native status/quality and source/server timestamps survive intact;
4. stale or expired subscription updates cannot be promoted to current observations;
5. endpoint/application certificate mismatch fails closed;
6. unexpected endpoint after reconnect cannot inherit the prior trust/lease;
7. size/time/rate bounds prevent resource exhaustion in the declared profile;
8. browse/discovery results cannot mint effect authority;
9. duplicate/reordered/replayed events are retained or deduplicated according to an explicit profile;
10. a transport timeout is distinguished from an explicit native rejection and does not imply no effect.

Only after those fixtures pass should we consider an offline descriptor/configuration workflow. OPC UA UAFX OfflineEngineering, IEC 61131-3-compatible tooling, and device-vendor toolchains remain distinct profile layers.

## Licensing / architectural caution

A library's open-source license, protocol support list, recent release, or local demo does not establish production suitability. Review exact dependency licensing, vulnerability-response policy, native cryptographic behavior, feature flags, supported profiles and target constraints before adoption. Do not replace Symthaea's Rust-first production stack with Python tooling merely because a test simulator has a Python interface.

## Explicit nonclaims

This note does not establish that Symthaea currently implements OPC UA or Modbus, that the candidate crates pass Symthaea's toolchain or security gates, that either server/client is fully certified, or that any real equipment is compatible. The proposed interoperability tests are not functional-safety certification or field qualification.


## Existing Symthaea equipment-control baseline

A separate source audit on the default-branch commit `5e83cea4986ba3845aa50e0d5dd03493c5adeddf` found an existing domain-specific 3D-printing/machine-control path:

- `crates/domains/symthaea-fabrication-kernel/src/printer_control.rs` contains `PrinterApi`, `MockPrinter`, typed status/temperature results, and OctoPrint/Moonraker client stubs that fail closed with `BackendUnavailable`.
- `src/machine.rs` includes typed machine profiles/capabilities, bounded timed sessions, negotiation and `ValidatedGCode`.
- `src/submission.rs` adds a single-use print authorization path bound to G-code and machine-profile fingerprints, attestation/release evidence, and session lease.
- `src/session.rs` adds sequence/nonce/expiry anti-replay tracking; `src/operator_command.rs` includes signed bounded operator commands; `src/execution_guard.rs` ties containment decisions to verified machine telemetry.

This is a better immediate bridge target than building another machine-control stack from scratch. However, it is **not** a generic industrial adapter, and the live OctoPrint/Moonraker clients are still stubs at that baseline. The next focused issue is [INTX-FAB-ADAPTER-001 / #7706](https://github.com/Luminous-Dynamics/symthaea/issues/7706): expose read-only observations and artifact evidence while preserving the current machine/session/authorization owners, and only later evaluate real transport. It must not expose raw G-code submission directly to cognition.

This domain adapter is a complementary first implementation slice; the independently implemented OPC UA test-server pilot in [#7704](https://github.com/Luminous-Dynamics/symthaea/issues/7704) remains the standards-based industrial interoperability tranche. Neither should be advertised as generalized support for existing medical, utility, or factory devices.
