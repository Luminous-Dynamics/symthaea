# EKM-075 — Epoch-Bound Revision Receipt Schema Contract

## Purpose

EKM-074 defines the identity requirements for a post-restart authority epoch. EKM-075 defines the versioned belief-revision receipt migration required before that epoch can safely authorize new mutations.

The central invariant is:

**legacy V1 receipts remain readable audit history, but no V1 receipt can be inferred or promoted into operational post-restart authority. Operational V2 receipts require explicit epoch binding at evaluation time.**

## V2 semantic field inventory

A future operational V2 receipt preserves every existing audit semantic and adds explicit authority-epoch binding:

- receipt ID
- claim ID
- proposed delta
- rationale
- evidence-basis snapshots
- duplicate basis IDs
- typed policy schema
- calibration snapshot
- uncertainty assessment
- typed decision snapshot
- evaluation cycle
- authority-epoch digest
- authority-epoch sequence

## Migration rules

The canonical V2 contract requires:

- global receipt IDs remain monotonic across restart epochs
- all V1 audit semantics remain preserved
- legacy V1 receipts stay archival-only
- operational V2 receipts require both epoch digest and epoch sequence
- evaluation may not predate epoch activation
- epoch identity may not be inferred from capture cycle
- epoch identity may not be inferred from restart digest
- epoch identity may not be inferred from preflight or sandbox state
- legacy V1 receipts may never be silently upgraded to operational V2 by inference
- persistence must carry the epoch binding
- canonical wire encoding/hashing must carry the epoch binding
- prepared mutations must retain the receipt epoch
- authorizations must match the prepared receipt epoch

## Current runtime status

EKM-075 deliberately records that the migration is not implemented yet:

- `current_receipt_type_is_v2 = false`
- `current_persistence_carries_v2_epoch_binding = false`
- `current_wire_carries_v2_epoch_binding = false`
- `prepared_mutation_carries_epoch_binding = false`
- `authorization_enforces_epoch_binding = false`
- `mutation_authority_exported = false`
- `activation_authorized = false`

## Why no optional epoch field

Making epoch binding optional on the existing receipt would create an ambiguous authority state: a caller could not tell whether `None` meant pre-restart archival history, an uninitialized post-restart receipt, or a bypassed enforcement path.

V2 therefore treats epoch binding as mandatory for operational receipts, while V1 remains a distinct archival representation.

## Why no inferred migration

A capture cycle, V2 restart digest, preflight receipt, or sandbox digest can all exist before a candidate actually becomes live. Using any of them to infer an authority epoch could mint mutation authority for a state that was never atomically activated.

The epoch must instead come from the successful activation-commit boundary defined by EKM-072/074.

## Authority boundary

EKM-075 does not:

- alter `BeliefRevisionReceipt`
- create a V2 receipt runtime type
- alter persistence or wire formats
- alter `PreparedBeliefMutation`
- alter `BeliefMutationAuthorization`
- alter `BeliefMutationAuthority`
- mint an authority epoch
- restore operational history
- authorize mutation or activation

## Qualification status

This tranche is stacked on EKM-074 / PR #4065. EKM-074 exact-head CI #7513 was fully queued when this evidence contract was prepared.

GitHub Actions remains the executable authority. Static/API review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.

## Next boundary

After executable qualification, the next change should introduce the actual immutable V2 receipt data type and canonical snapshot/persistence representation **without yet wiring it into mutation authority**. Only a later separately qualified tranche should change `BeliefMutationAuthority::prepare` to require the active epoch and produce V2 operational receipts.
