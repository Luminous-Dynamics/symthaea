# EKM-069 — Activation Preflight Composition

## Purpose

EKM-067 creates a sealed split-state sandbox and EKM-068 can prove that the actual live EKM state remains unchanged from a verified baseline. EKM-061 separately proves that the protected mutation-seal checkpoint is the current deployment head.

A tempting next step would be to call those conditions sufficient for activation review. EKM-069 deliberately refuses to do that because EKM-054's protected joint trust-context checkpoint does not independently prove *current-head* status. Its evidence taxonomy includes generic signed checkpoints, which can prove authenticity without proving latest-state freshness.

The central invariant is:

**preflight composition may prove sandbox identity, live-epoch continuity, candidate mutation-seal currentness, and protected trust-checkpoint validity while still refusing activation-transaction eligibility until the joint trust-context checkpoint receives its own current-head attestation.**

## Exact composition

`ActivationPreflightReceiptV1::evaluate`:

1. verifies the exact EKM-067 sandbox;
2. verifies EKM-061 current-head evidence and its validity window;
3. verifies the EKM-054 protected trust checkpoint and its validity window;
4. checks all source authority flags remain false;
5. requires the EKM-061 restart capture cycle to equal the EKM-067 sandbox capture cycle;
6. independently re-derives EKM-067 from the exact EKM-057/059/060/061/062/063/064/065/066 source chain and requires the sandbox digest to match;
7. rechecks EKM-068 against the actual live ledger, inventory, support store, revision history and schema history at the exact preflight cycle;
8. binds all identities and proof digests into one domain-separated receipt.

This prevents a valid freshness proof, trust checkpoint, or live-state receipt from being detached and attached to a different restart candidate.

## Machine-readable distinction

A successful EKM-069 composition reports:

- `source_sandbox_rederived = true`
- `live_epoch_unchanged = true`
- `candidate_current_head_proven = true`
- `protected_trust_checkpoint_valid = true`
- `protected_trust_checkpoint_currentness_independently_proven = false`
- `trust_context_current_head_attestation_required = true`

Therefore it also reports:

- `activation_transaction_review_eligible = false`
- `live_state_swap_authorized = false`
- `rollback_authorized = false`
- `activation_authorized = false`
- `trusted_checkpoint_commit_authorized = false`

## Why EKM-054 validity is not currentness

A valid signature over checkpoint N proves that an accepted authority signed checkpoint N. It does not prove checkpoint N+1 does not exist.

Likewise, an unexpired protected checkpoint can be authentic but stale relative to a newer protected state. EKM-061 already enforced this distinction for mutation-seal checkpoints by introducing a provider contract whose evidence semantics explicitly mean current/monotonic head. The joint trust-context checkpoint needs the same separation.

## Live-state boundary

EKM-069 rechecks EKM-068 at the exact preflight cycle, but it still does not lock or atomically compare-and-swap the live objects. The final TOCTOU window remains intentionally unresolved until an eventual activation transaction protocol.

## Authority boundary

EKM-069 does not:

- export the sandbox's writable support store;
- construct a restored operational revision history;
- mutate the live EKM state;
- swap live state;
- authorize rollback;
- advance restart anchors, verifier checkpoints, trust-context checkpoints, or mutation-seal checkpoints;
- mutate legacy confidence, causal/world-model state, or action state.

## Qualification status

This tranche is stacked on EKM-068 / PR #4010. Parent EKM-068 exact-head CI #7449 was queued when EKM-069 was prepared.

GitHub Actions remains the executable authority. Static/API review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime or deployment evidence.

## Next boundary

The next safe tranche is a **joint trust-context checkpoint current-head attestation** with an evidence taxonomy limited to mechanisms whose contract explicitly means current monotonic state (for example monotonic protected storage, a hardware monotonic counter, a transparency-log head, or a current-head witness quorum).

Only after that evidence is bound to the exact EKM-054 checkpoint should a later composition receipt set `activation_transaction_review_eligible = true`. Even then, swap and checkpoint commit remain unauthorized until a separate atomic activation transaction exists.