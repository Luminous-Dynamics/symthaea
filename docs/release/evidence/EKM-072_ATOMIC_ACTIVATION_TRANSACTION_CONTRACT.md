# EKM-072 — Atomic Activation Transaction Contract

## Purpose

EKM-071 may establish that a future activation transaction is eligible for review, but it intentionally proves that no lock, compare-and-swap mechanism, rollback implementation, or activation capability exists yet.

EKM-072 defines the candidate-independent contract that any future executor must satisfy. It does not implement that executor.

## Atomic activation bundle

V1 requires these state components to move coherently as one bundle:

1. epistemic ledger
2. epistemic support store
3. operational revision history
4. revision-schema history
5. mutation-authorization-consumption state

This is deliberately stronger than EKM-067. The EKM-067 sandbox contains a real private support store and immutable revision audit, but not an operational revision history. Therefore EKM-067 is not, by itself, an activation bundle.

## Required phase order

The canonical V1 transaction order is:

1. `AcquireExclusiveLiveEpochGuard`
2. `ReverifyActivationReviewUnderGuard`
3. `CompareExpectedLiveEpoch`
4. `CaptureRollbackBundle`
5. `StageCompleteCandidateBundle`
6. `AtomicLiveStateSwap`
7. `VerifyInstalledStateUnderGuard`
8. `CommitTrustedCheckpoints`
9. `ReleaseLiveEpochGuard`

The order is part of the domain-separated contract digest.

## Required invariants

- exclusive live-epoch guard required
- activation review must be reverified while the guard is held
- compare-and-swap / expected-generation check required
- rollback material must exist before the live swap
- the complete activation bundle must swap atomically
- post-swap equivalence must succeed before trusted checkpoints advance
- trusted checkpoint commit must follow installed-state verification
- partial bundle swap forbidden
- partial checkpoint commit forbidden

## Authority boundary

This tranche contains no executor and records:

- `executor_implemented = false`
- `live_state_lock_implemented = false`
- `compare_and_swap_implemented = false`
- `rollback_implemented = false`
- `live_state_swap_authorized = false`
- `activation_authorized = false`
- `trusted_checkpoint_commit_authorized = false`

It does not accept a candidate, touch live state, expose sandbox internals, or mutate trust state.

## Why this comes before an executor

The transaction boundary should be reviewable before implementation. In particular, the contract makes two dangerous shortcuts invalid by construction:

1. a writable support store cannot be mistaken for a complete swappable epistemic state;
2. trusted checkpoints cannot advance immediately after a swap without first verifying the installed state.

## Next blockers

A future executor remains blocked on at least:

- an explicit live EKM epoch/generation guard with exclusive mutation exclusion;
- a compare-and-swap or equivalent commit primitive;
- construction of a complete candidate activation bundle, including operational revision history and authorization-consumption state;
- a complete rollback bundle with exact pre-swap identity;
- post-swap equivalence verification under the same guard;
- atomic or recoverable trusted-checkpoint commit semantics.

## Qualification status

This tranche is stacked on EKM-071. GitHub Actions remains the executable authority. Static/API review and authored unit tests are not rustfmt, compile, Clippy, unit-test, integration-test, runtime, or deployment evidence.

No future implementation should infer execution authority merely from this contract being present or valid.
