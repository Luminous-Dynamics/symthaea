# EKM-076 — Epoch-Bound Revision Receipt V2 Data

## Purpose

EKM-075 defines the migration contract for future post-restart operational revision receipts. EKM-076 defines the exact immutable typed data shape and canonical digest for that future V2 receipt **without adding a production constructor or mutation-authority path**.

The central invariant is:

**a V2 receipt data object may have a valid canonical shape and digest without proving that its authority epoch was actually issued, that its evaluation occurred after epoch activation, or that it is accepted by the mutation facade.**

## Exact V2 semantic data

`EpochBoundRevisionReceiptDataV2` carries:

- global belief-revision receipt ID
- claim ID
- proposed epistemic-support delta
- rationale
- unique frozen basis evidence snapshots
- duplicate-basis diagnostics
- canonical typed policy schema
- optional calibration snapshot
- optional typed uncertainty assessment
- canonical typed decision snapshot
- evaluation cycle
- exact authority-epoch digest
- authority-epoch sequence
- canonical receipt digest

The representation uses the existing EKM typed policy/decision snapshots and V1 basis/uncertainty DTOs rather than creating parallel semantic structures.

## Canonical digest

The receipt digest is domain-separated with:

`symthaea-ekm-epoch-bound-revision-receipt-data-v2`

It binds the canonical EKM-075 schema-contract digest and every V2 receipt field.

Policy and decision encoding mirrors the stable explicit tags already used by EKM-040. Evidence kinds, evidence polarities, uncertainty dimensions, knowledge-weight sources and knowledge-weight dimensions use explicit tags rather than Rust enum discriminants or `Debug` output.

Floating-point values are hashed by exact IEEE bit representation after bounded/finite validation.

## Structural validation

The passive DTO validates, among other things:

- non-zero receipt ID
- non-zero epoch sequence
- finite bounded proposed delta
- bounded rationale and optional evidence text
- unique frozen basis IDs
- exact requested-ID/snapshot-ID equality
- basis evidence not observed after evaluation
- unique duplicate-basis diagnostics referring to the unique basis
- valid canonical policy schema and cap ordering
- bounded calibration/uncertainty values
- uncertainty assessment not later than evaluation
- typed decision snapshot version
- `eligible == failures.is_empty()`
- bounded decision/routing failure collections
- exact canonical receipt digest

These are representation invariants, not a replacement for belief-gate re-execution.

## No production constructor

EKM-076 intentionally provides **no public or crate-private production constructor** for `EpochBoundRevisionReceiptDataV2`.

This prevents archival V1 receipts from being converted into apparently operational V2 receipts by attaching an inferred or caller-supplied epoch.

A later tranche may add a constructor only at the fresh evaluation boundary after an operational authority epoch exists.

## Machine-readable authority status

`EpochBoundRevisionReceiptAuthorityStatusV2` separates passive verification from authority verification.

After shape/digest verification it still reports:

- `epoch_issuance_verified = false`
- `evaluation_after_epoch_activation_proven = false`
- `accepted_by_current_mutation_facade = false`
- `mutation_authority = false`
- `activation_authorized = false`

Therefore:

**shape/digest valid != epoch valid != fresh evaluation valid != mutation authority**

## Authority boundary

EKM-076 does not modify:

- `BeliefRevisionReceipt`
- `BeliefRevisionHistory`
- persistence capsules
- restart wire formats
- `PreparedBeliefMutation`
- `BeliefMutationAuthorization`
- `BeliefMutationAuthority`
- authority-epoch issuance
- operational-history continuation
- live activation

Historical V1 receipts remain archival-only under EKM-073.

## Qualification status

This tranche is stacked on EKM-075. At preparation time EKM-075 exact-head CI #7515 remained queued.

GitHub Actions remains the executable qualification authority. Static/API review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.
