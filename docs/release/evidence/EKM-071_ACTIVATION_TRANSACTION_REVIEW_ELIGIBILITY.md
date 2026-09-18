# EKM-071 — Activation Transaction Review Eligibility

## Purpose

EKM-069 composes a fresh activation-preflight review but correctly leaves transaction review ineligible because the EKM-054 protected joint trust-context checkpoint is not independently proven to be the current head. EKM-070 supplies that missing current-head attestation.

EKM-071 combines both at one exact logical review cycle and may report only that a later atomic activation transaction is eligible for design/review.

The central invariant is:

**review eligibility requires the candidate sandbox, actual live EKM epoch, mutation-seal current-head evidence, protected trust checkpoint, and trust-context current-head evidence to all remain mutually bound and fresh at the same review cycle.**

## Composition

`ActivationTransactionReviewEligibilityReceiptV1::evaluate`:

1. Re-runs EKM-069 at the exact EKM-071 `reviewed_at_cycle` instead of trusting an older detached preflight receipt.
2. Therefore re-verifies the exact EKM-067 sandbox source chain and sandbox digest.
3. Rechecks the EKM-068 live epoch against the actual live ledger, support store, revision history, and schema history at the same review cycle.
4. Requires EKM-061 candidate mutation-seal current-head evidence to remain unexpired.
5. Requires the EKM-054 protected joint trust-context checkpoint to remain unexpired.
6. Verifies the EKM-070 current-head receipt internally and requires it to remain fresh.
7. Requires EKM-070 to bind the exact same EKM-054 checkpoint statement digest, sequence, joint-context digest, deployment ID, and trust-domain ID.
8. Emits one domain-separated receipt digest over the fresh preflight identity, current-head statement/proof identities, review cycle, and all authority flags.

`verify_against(...)` re-runs the complete composition and requires exact receipt and digest equality.

## Positive claim

A successful receipt may report:

- `live_epoch_unchanged_at_review = true`
- `candidate_current_head_proven_at_review = true`
- `trust_context_current_head_proven_at_review = true`
- `exact_trust_checkpoint_binding_proven = true`
- `activation_transaction_review_eligible = true`

This means only that the evidence is coherent enough to review a later atomic transaction design.

## Authority boundary

The receipt simultaneously requires:

- `live_state_lock_acquired = false`
- `compare_and_swap_implemented = false`
- `time_of_check_use_window_closed = false`
- `live_state_swap_authorized = false`
- `rollback_authorized = false`
- `activation_authorized = false`
- `trusted_state_mutated = false`
- `trusted_checkpoint_commit_authorized = false`

No mutable sandbox object is exported. No live state is mutated. No checkpoint is advanced.

## Remaining TOCTOU boundary

EKM-071 still performs checks before a hypothetical live-state swap. Another writer could change live state after the final EKM-068 recheck but before a future swap.

Therefore:

**activation transaction review eligible ≠ activation transaction safe to execute.**

Closing that window requires a later live-state lock, generation/CAS token, or equivalent atomic compare-and-swap protocol whose success condition includes the exact EKM-068 live baseline immediately at commit time.

## Qualification status

This tranche is stacked on EKM-070. GitHub Actions remains the executable authority. Static/API review and authored tests are not rustfmt, compile, Clippy, unit-test, integration-test, runtime, or deployment evidence.

No activation transaction should be implemented or enabled merely because this review receipt exists.
