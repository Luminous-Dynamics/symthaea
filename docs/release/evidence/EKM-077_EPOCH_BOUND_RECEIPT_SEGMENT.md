# EKM-077 — Epoch-Bound Revision Receipt Segment

## Purpose

EKM-076 defines one immutable epoch-bound revision receipt V2. EKM-077 defines the passive persistence grouping for receipts produced within one authority epoch.

The central invariant is:

**receipt IDs remain globally monotonic across restart epochs, while every operational V2 segment belongs to exactly one authority epoch.**

## Segment shape

`EpochBoundRevisionReceiptSegmentV2` carries:

- exact authority-epoch digest
- authority-epoch sequence
- first global receipt ID in the segment
- next global receipt ID after the segment
- capture cycle
- ordered epoch-bound V2 receipt records
- canonical segment digest

There is intentionally no production constructor.

## Validation

A segment must satisfy:

- V2 segment version
- non-zero authority-epoch sequence
- non-zero receipt-ID boundaries
- at most one million records
- `next_receipt_id == first_receipt_id + record_count`
- every record independently passes EKM-076 shape/digest verification
- every record carries the exact segment epoch digest and sequence
- record IDs are contiguous from `first_receipt_id`
- evaluation cycles do not regress within the append-only segment
- no receipt evaluation occurs after the segment capture cycle
- exact canonical segment digest

## Empty epoch segments

An empty segment is valid when an authority epoch has been activated but no fresh revision receipt has yet been evaluated.

For an empty segment:

`first_receipt_id == next_receipt_id`

This preserves the global receipt cursor without fabricating a receipt.

## Canonical segment identity

The segment digest is domain-separated with:

`symthaea-ekm-epoch-bound-revision-receipt-segment-v2`

It binds:

- epoch digest and sequence
- first/next receipt IDs
- capture cycle
- record count
- each canonical EKM-076 receipt digest in order

Because each receipt digest already binds its full typed semantic content, the segment does not duplicate a second semantic encoding of each record.

## Multi-epoch model

EKM-077 models **one epoch per segment**.

Future operational history can therefore preserve a sequence such as:

- archival legacy V1 history
- epoch E1 receipt segment
- epoch E2 receipt segment
- ...

while receipt IDs continue globally across every segment.

A later history-chain tranche may bind multiple segments and epoch transitions. EKM-077 does not do that yet.

## Authority boundary

A valid segment still reports:

- `operational_history_constructed = false`
- `authority_epoch_issuance_verified = false`
- `mutation_authority = false`
- `activation_authorized = false`

EKM-077 does not modify `BeliefRevisionHistory`, persistence capsules, restart wire formats, prepared mutations, authorizations, the mutation facade, epoch issuance, or activation.

## Qualification status

This tranche is stacked on EKM-076 / PR #4091. EKM-076 exact-head CI #7547 remained queued when this contract was prepared.

GitHub Actions remains the executable qualification authority. Static/API review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.
