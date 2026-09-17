# EKM-040 — Typed Belief-Revision Schema Wire

## Purpose

EKM-039 gives restart V2 an explicit typed policy/decision integrity layer. EKM-040 gives that typed sidecar its own bounded canonical wire representation without changing the legacy EKM-034 V1 restart envelope.

## Central invariant

**Typed policy and decision semantics may be serialized only through explicit stable fields and tags; parsing those bytes still yields untrusted data, not operational epistemic state.**

## Wire framing

The envelope uses:

- fixed `SYMTHAEA-EKM-SCHEMA-WIRE` magic
- explicit wire version
- explicit payload length
- maximum 64 MiB payload
- maximum 1,000,000 records per bounded collection
- domain-separated BLAKE3 corruption checksum
- rejection of truncation, trailing bytes, unknown tags, invalid booleans, integer conversion overflow, non-finite/out-of-range numeric values, and noncanonical policy ordering

## Explicit typed payload

Each schema-bound revision record carries:

- receipt ID
- claim ID
- bit-preserved proposed delta
- evaluation cycle
- every canonical policy field
- ordered uncertainty caps with stable dimension tags
- decision eligibility
- declared provenance-root count
- complete typed decision failures
- complete nested knowledge-weight routing failures

There are no `Debug` strings and no Rust enum discriminants in the format.

## Canonical policy enforcement

The encoder independently validates `BeliefRevisionPolicySchemaV1` before writing bytes. This is necessary because the schema is intentionally inspectable and its fields are public: wire canonicality must not depend on trusting how an in-memory DTO was assembled.

Uncertainty caps must use strictly increasing stable dimension tags. Duplicate or reordered caps are rejected rather than normalized silently at serialization time.

The decoder enforces the same ordering and rebuilds the policy schema through the existing validated constructors/builders.

## Decode boundary

`BeliefRevisionSchemaWireV1::decode` returns only `BeliefRevisionSchemaWireSnapshotV1`.

It does not create:

- `BeliefRevisionSchemaHistoryCapsuleV1`
- `BeliefRevisionHistory`
- `EpistemicRestartCapsuleV2`
- a quarantine object
- an authorization object
- any writable belief state

Checksum validity proves only byte integrity, not semantic validity, authenticity, provenance, or authorization.

## Compatibility

EKM-034 V1 remains unchanged. EKM-040 is a sidecar wire format for the new typed semantics. A later top-level restart-v2 bundle can bind the legacy base-state wire and this typed schema wire under the EKM-039 V2 digest without reinterpreting old artifacts.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
