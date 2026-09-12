# ADR-015: RSK schema-bound Rust types

**Status**: Accepted for reference implementation

**Date**: 2026-09-12

**Change Class**: A

## Context

The Replicator Safety Kernel (RSK) reference evaluator currently represents capability authority as an opaque `u64` bitset and resource authority as scalar `u64` counters. Those forms are intentionally simple reference semantics, but they do not bind the *meaning* of those values to a frozen schema.

The normative contracts in `RSK_CAPABILITY_SCHEMA_V0_1.md`, `RSK_RESOURCE_ACCOUNTING_V0_1.md`, and PR #1856 require production-target authority values to carry exact semantic identity:

```text
CapabilitySet = (CapabilitySchemaId, Bits)
ResourceAmount = (ResourceAccountingSchemeId, DimensionVector)
```

PR #1867 adds a Rust-independent canonicalization/golden-vector profile. The next step is to establish matching Rust types without prematurely rewriting the existing authority/ledger state machine while its CI remains queued.

## Decision

Add a new public `semantic` module to `symthaea-replicator-safety` containing dependency-free production-target reference types:

- `CapabilitySchemaId`;
- `BoundCapabilitySet`;
- `ResourceAccountingSchemeId`;
- `ResourceDimensionId`;
- `ResourceQuantity`;
- `ResourceVector`;
- `SemanticBindingError`.

The legacy `CapabilitySet` and scalar resource fields remain in place for the existing reference evaluator during this tranche.

### Capability rules

`BoundCapabilitySet` always carries an exact `CapabilitySchemaId`.

Subset/intersection operations across different schema IDs return an explicit `CapabilitySchemaMismatch` error. There is no implicit translation and no name/description-based equivalence.

### Resource rules

`ResourceVector` always carries one exact `ResourceAccountingSchemeId` and a canonical strictly increasing sequence of numeric dimension IDs.

Arithmetic requires:

1. identical accounting-scheme IDs;
2. identical dimension sets in identical canonical order;
3. checked arithmetic with explicit underflow/overflow failure.

A conservative transition check may verify only that a target remaining vector does not exceed a caller-supplied conservative envelope. It must not infer or manufacture the semantic mapping that produced that envelope.

## Golden-vector convergence

The Rust crate adds dev-only `sha2` and `serde_json` dependencies and tests the committed cross-language corpus:

`docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_1.json`

The Rust tests independently canonicalize the JSON schema objects and recompute their SHA-256 identities. This tests agreement with the Rust-independent Python reference without adding serialization/hash dependencies to the runtime RSK crate.

## Non-goals

This ADR does not:

- replace all existing `CapabilitySet` or scalar resource fields;
- modify grant/authorization/ledger semantics;
- implement schema registries;
- approve a production capability vocabulary;
- define physical resources or measurements;
- verify semantic migration proofs;
- grant positive authority.

## Safety rationale

The change makes semantic mismatch representable in the Rust type system before the types are threaded through positive authority paths. That is preferable to a broad migration because it allows the schema identity and arithmetic contracts to compile and stabilize independently.

The key invariant is:

```text
same numeric value under a different schema != same authority value
```

## Follow-up

After this tranche has compiler/test evidence, a separate Class A change should bind the new schema-aware values through:

- lineage policy;
- replication grants;
- requests;
- bounded authorization;
- descendant commit evidence;
- checkpoints/replay;
- admitted-release/runtime identity.

Cross-schema migration must remain unavailable without an opaque separately verified translation capability.

## Validation requirements

Before this tranche is considered evidenced:

- exact-head `cargo check --all-targets --locked` passes for the RSK authority crate;
- exact-head tests pass;
- exact-head Clippy with `-D warnings` passes;
- Rust recomputation matches the committed capability/resource golden digests;
- mismatch, underflow, overflow, dimension-set and conservative-envelope negative tests pass.

Production admission remains **DENIED / NOT YET ELIGIBLE**.