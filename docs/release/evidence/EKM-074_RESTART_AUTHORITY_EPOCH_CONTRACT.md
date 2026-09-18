# EKM-074 — Restart Authority Epoch Contract

## Purpose

EKM-073 makes every pre-restart belief-revision receipt archival-only. EKM-074 defines the canonical identity and receipt-binding rules required before post-restart mutation authority can exist again.

The central invariant is:

**a restart authority epoch may become operational only after a successful atomic activation commit; preflight, review eligibility, sandbox construction, or an integrity digest cannot mint authority.**

## Epoch identity schema

A future operational epoch must bind all of the following:

- deployment identity
- trust-domain identity
- monotonic epoch sequence
- previous epoch digest
- exact activated restart V2 digest
- exact successful activation-commit receipt digest
- committed live generation / CAS generation
- activation cycle

The activation-commit receipt does not exist yet, so EKM-074 defines this schema without issuing an epoch.

## Receipt binding schema

Future post-restart belief-revision receipts must:

- preserve globally monotonic receipt IDs across epoch boundaries
- bind the exact authority-epoch digest
- bind the epoch sequence
- never evaluate before the epoch activation cycle
- be prepared only under the currently active epoch
- require authorization to match the prepared receipt's epoch
- persist the epoch binding
- serialize/hash the epoch binding canonically
- leave all earlier-epoch receipts archival-only

This avoids resetting audit identity at restart while preventing stale receipts from becoming fresh authority.

## Enforcement point

The eventual enforcement point should be the existing `BeliefMutationAuthority::prepare` / `apply` public facade, not the lower EKM-026 firewall. EKM-074 does **not** modify that facade yet.

## Current implementation status

The canonical contract explicitly records:

- `epoch_issuance_requires_successful_activation_commit = true`
- `preflight_may_issue_epoch = false`
- `review_eligibility_may_issue_epoch = false`
- `sandbox_may_issue_epoch = false`
- `receipt_ids_global_across_epochs = true`
- `historical_receipts_archival_only = true`
- `current_receipt_schema_has_epoch_binding = false`
- `current_mutation_facade_enforces_active_epoch = false`
- `operational_epoch_issuance_implemented = false`
- `operational_history_continuation_implemented = false`
- `mutation_authority_exported = false`
- `activation_authorized = false`

## Why the epoch is not minted from EKM-071

EKM-071 proves only that an atomic activation transaction design is eligible for review. It explicitly has no live-state lock, no compare-and-swap, no closed TOCTOU window, no installed-state verification, and no trusted-checkpoint commit.

Minting an authority epoch before those actions succeed would create post-restart mutation authority for a state that might never become live.

## Relationship to EKM-072

EKM-072 requires trusted checkpoints to commit only after the newly installed state has been verified under the live guard. The future epoch identity should be issued at that same successful commit boundary and include the activation-commit receipt digest plus committed generation.

## Authority boundary

EKM-074 does not:

- mint an authority epoch
- modify `BeliefRevisionReceipt`
- modify `BeliefMutationAuthority`
- restore operational revision history
- issue a mutation authorization
- swap live state
- advance trusted checkpoints
- authorize activation

## Qualification status

This tranche is stacked on EKM-073. Parent exact-head CI #7474 remained queued when this evidence contract was prepared. GitHub Actions remains the executable authority; authored code and static review are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.

## Next boundary

After executable qualification, the next schema tranche can introduce a versioned epoch-bound receipt representation and migrate `BeliefMutationAuthority::prepare` to require the active epoch for newly evaluated receipts. Historical V1 receipts should remain readable and archival, never silently upgraded to fresh authority.
