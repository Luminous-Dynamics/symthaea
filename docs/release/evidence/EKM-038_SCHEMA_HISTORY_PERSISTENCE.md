# EKM-038 — Schema-Bound Revision Persistence

## Purpose

EKM-037 binds canonical policy schemas to revision receipts at evaluation time. EKM-038 makes that binding restart-capturable and explicitly rejects mixed histories where some receipts still lack typed policy provenance.

## Added contract

`BeliefRevisionSchemaHistoryCapsuleV1` captures:

- capture cycle
- linked EKM-031 capture cycle
- linked revision count
- linked next revision receipt ID
- complete `SchemaBoundBeliefRevisionRecordV1` sequence

## Central invariant

**A restart image that claims typed belief-revision semantics must carry canonical policy/decision schema state for every persisted revision receipt, not only for a subset.**

## Cross-checks

Capture verifies:

- EKM-031 version and exact capture epoch
- live receipt history ↔ schema sidecar alignment
- complete receipt-count equality
- receipt ID equality
- claim identity
- bit-exact proposed delta
- evaluation cycle
- receipt does not postdate capture
- typed decision snapshot equality
- canonical policy schema rebuild equality against the immutable receipt policy
- next receipt ID lineage inherited from EKM-031

A raw/legacy receipt inserted without an EKM-037 schema record makes capture fail closed.

## Authority boundary

The capsule is export/validation only. It does not hydrate `BeliefRevisionHistory`, create executable authorization, mutate epistemic support, ingest evidence, alter causal/world-model state, activate restored state, or perform file/network I/O.

## Migration implication

The existing EKM-034 V1 wire format remains a legacy envelope with opaque policy/decision substructures. The next restart version should carry EKM-038 alongside EKM-031 and encode the canonical policy schema plus typed decision failures explicitly rather than modifying V1 semantics in place.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
