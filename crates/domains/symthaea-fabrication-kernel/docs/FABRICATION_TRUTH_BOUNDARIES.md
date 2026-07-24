# Fabrication Kernel Truth Boundaries

This crate distinguishes implemented evidence from declared direction.

The durable gateway, signed telemetry, submission reconciliation, and governed
trust-rotation boundary is described in `FABRICATION_DURABLE_GATEWAY.md`.

## Closed-solid geometry

The BSP backend now performs solid-classifying union, subtraction, and
intersection for closed, consistently oriented triangle meshes. Oracle tests
cover overlapping, contained, disjoint, and empty cube cases by expected volume.
The implementation is not yet a proof-grade CAD kernel. A bounded triangle
self-intersection scan now fails closed for interior crossings and incomplete
work budgets, but exact coincident-surface classification, minimum feature
sizes, and adversarial numerical conditioning remain outside the proof boundary.

## Mesh validation

`ValidationReport::is_valid` requires a non-empty mesh, valid indices, finite
vertices and normals, matching vertex/normal counts, no degenerate or duplicate
triangles, and no non-manifold edges. The report separately records boundary
edges, non-manifold edges, and connected components. `is_printable`
additionally requires a closed edge incidence map, consistent normals, and
positive signed volume. `FabricationReadyMesh` grants an owned capability only
when that baseline gate passes. This remains closed-solid evidence rather than
complete process qualification.

## File formats

`export_3mf_model_xml` emits only the `/3D/3dmodel.model` XML payload.
`export_3mf_package` emits a complete core stored ZIP/OPC package with content
types and root relationships. Materials, production extensions, thumbnails,
and package signatures are not yet emitted.

`parse_step_subset` is a bounded, line-oriented parser for direct numeric point
and B-spline payloads. Explicit input, line, entity, and nesting limits protect
untrusted ingestion. It does not claim general STEP entity-reference, topology,
assembly, or B-Rep support. STL parsing likewise exposes bounded profiles.

## Semantic geometry

HDC intent encodes scale, rotation, and translation with deterministic scalar
identity. `GeometricThought` reconstructs its skipped HDC field during
serde deserialization. This is deterministic semantic encoding, not yet a
validated HDC-to-CSG decoder or geometric round-trip proof.

## Printer control

Only `MockPrinter` is operational. OctoPrint, Moonraker, and custom HTTP
backends fail closed at configuration and connection time until authenticated
live transports exist. A declared backend must never report a successful
connection without an implementation behind it.

## Slicing, units, and machine policy

Geometry is canonically millimetres; analytical mechanics is SI. Checked unit
wrappers and generative conversion prevent mesh extents from being interpreted
as metres. Rectilinear, grid, and honeycomb infill share an even-odd clipping
path that excludes hole interiors. Strict slicing and G-code APIs reject invalid
inputs without clamping and rejects jobs with no printable geometry.
`MachineProfile` validation checks build bounds, homing, feed rates,
temperatures, finite values, retraction, and positive extrusion motion before
creating `ValidatedGCode`.

## Required next gates before live hardware authority

1. Exact coincident-surface classification and minimum wall/hole/clearance checks.
2. Process-aware orientation and optimized, removable support qualification.
3. Firmware-dialect parsing and independent G-code simulation.
4. Collision-aware kinematics and machine-state interlocks.
5. Material/process certification and thermal-distortion evidence.
6. Explicit simulation-to-live authorization and immutable evidence capture.

## Repair, feature screening, and provenance

Conservative sanitation is not topology healing. The minimum-feature ray oracle is bounded screening rather than exact wall-thickness proof. Provenance fingerprints detect deterministic mismatch but are neither signatures nor tamper-evident attestations.

## Cryptographic and runtime boundary (v0.10)

Manifest SHA-256 digests are cryptographic byte identities, but trust requires
an external validated signature provider. Signature algorithm labels are not
implementations. Session nonces provide freshness only when issued and consumed
by an authenticated gateway. Execution-guard decisions do not actuate physical
safety systems without explicit backend integration.


## Governance and recovery boundary (v0.11)

The governed path evaluates signature validity together with a fresh,
sequence-numbered key-lifecycle snapshot. Snapshot hashes and audit-chain hashes
are cryptographic identities, but production trust still depends on authentic
snapshot distribution, persistent rollback protection, and durable external
audit anchoring. Execution checkpoints preserve deterministic guard state. The
recovery API authorizes only a complete restart in a fresh session; it makes no
claim of safe arbitrary mid-G-code continuation.

## Federated certification boundary

Version 0.14 can prove that multiple lifecycle-eligible gateways endorsed one exact durable state generation, retain anti-equivocation history, verify a contiguous disaster-recovery chain, bind signed operator commands to one execution, and block release certification on unresolved incident evidence. This is evidence-level federation; it is not a Byzantine consensus transport, a clock-synchronization system, or a substitute for supervised physical qualification.
