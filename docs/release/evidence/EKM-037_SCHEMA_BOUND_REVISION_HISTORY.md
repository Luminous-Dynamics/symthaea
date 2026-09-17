# EKM-037 — Schema-Bound Belief Revision History

## Purpose

EKM-036 defines explicit policy and decision snapshot schemas, but a schema is only useful for persistence if it is bound to the decision at the time that decision is created. EKM-037 adds that binding without modifying belief support or authority.

## Added contract

`BeliefRevisionSchemaHistoryV1` wraps the existing `BeliefRevisionHistory` evaluation path and records a `SchemaBoundBeliefRevisionRecordV1` under the same `BeliefRevisionReceiptId`.

Each sidecar record binds:

- receipt ID
- claim ID
- proposed delta
- evaluation cycle
- exact `BeliefRevisionPolicySchemaV1`
- exact typed `BeliefRevisionDecisionSnapshotV1`

## Central invariant

**A persisted belief-revision decision must retain the explicit policy semantics that generated it, not attempt to infer those semantics after the fact.**

## Fail-closed alignment

Before recording a new decision, the wrapper requires the underlying receipt history and schema sidecar history to have exactly equal lengths. If a caller inserted a raw receipt through the legacy API, the next schema-bound evaluation fails before adding another sidecar record.

`validate_alignment()` independently checks:

- contiguous sidecar receipt IDs
- receipt existence
- claim identity
- bit-exact proposed delta
- evaluation cycle
- typed decision snapshot equality
- policy schema rebuild equality against the immutable receipt policy

## Legacy bypass boundary

The underlying `BeliefRevisionHistory::evaluate_and_record` remains public in this tranche. Therefore EKM-037 does not yet make schema-bound recording structurally mandatory. It makes bypass detectable and gives a migration target.

A later authority-surface tranche should route new persisted/restart-capable decisions through the schema-bound path and treat raw receipt insertion as legacy-only.

## Authority boundary

This PR does not mutate epistemic support, ingest evidence, authorize a mutation, change causal/world-model state, perform restart hydration, activate restored state, or perform file/network I/O.

The wrapper delegates decision evaluation and receipt creation to the existing gate/history implementation and only records additional immutable audit semantics.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
