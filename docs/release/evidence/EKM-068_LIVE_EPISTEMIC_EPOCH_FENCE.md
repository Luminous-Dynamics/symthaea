# EKM-068 — Live Epistemic Epoch Fence

## Purpose

EKM-067 can retain a real `EpistemicSupportStore` inside a sealed restart sandbox, but a later activation review must also prove that the currently running epistemic state did not change while the candidate was being reviewed.

A restart digest cannot be compared naively across later observation cycles because restart capsules bind their capture epoch. EKM-068 therefore reuses the existing source-live validators instead of inventing a second state model.

The central invariant is:

**a live epoch fence is valid only when a complete typed EKM V2 capsule is proven to match the actual live ledger, support store, revision history, and schema history at one atomic capture cycle; later rechecks must prove those live objects remain exactly represented by that baseline.**

## Added contract

### `LiveEpistemicEpochObservationV1`

A baseline observation:

- accepts a complete `EpistemicRestartCapsuleV2`
- requires `source_capsule.captured_at_cycle == observed_at_cycle`
- calls `EpistemicRestartCapsuleV1::verify_source_live` against the actual live ledger/inventory/support store/revision history
- independently calls `BeliefRevisionSchemaHistoryCapsuleV1::validate_live` against the actual live schema history
- binds V2, schema and base-manifest digests plus ledger/support/revision counts
- retains only the immutable source capsule, never references to the live objects

### `LiveEpistemicEpochFenceV1`

The fence:

- requires the exact EKM-067 sandbox to pass `verify()`
- binds the sandbox digest to one verified live observation
- requires the fence cycle to equal the live observation cycle
- requires the fence not to predate sandbox construction
- contains no swap, activation or checkpoint-commit capability

### `LiveEpistemicEpochContinuityReceiptV1`

A later recheck:

- requires the exact same sandbox digest
- re-runs the source-live validation against the actual current ledger/store/history/schema objects
- allows the wall-clock/observation cycle to advance
- requires the epistemic state/history semantics to remain identical to the baseline
- returns only a read-only continuity receipt

## Why this is stronger than digest equality

The V2 restart digest commits the capture epoch. Re-capturing identical state at a later cycle can therefore produce a different capsule identity even when no semantic state changed.

EKM-068 instead retains the original verified capsule privately and asks the existing persistence contracts whether the actual live objects have diverged from that exact baseline. This distinguishes:

- time advanced; state unchanged
- state/history actually changed

without normalizing or weakening restart digests.

## Concurrency boundary

EKM-068 does not lock the live state and does not make a compare-and-swap atomic.

It establishes and rechecks a baseline only. A later activation-preflight tranche must still recheck this fence immediately before any proposed swap and combine that result with fresh candidate-currentness and trust-checkpoint evidence.

If the live state changes after the last EKM-068 recheck and before a later swap, EKM-068 alone cannot prevent that race. Closing that final time-of-check/time-of-use window requires an eventual atomic activation transaction or equivalent live-state lock/CAS protocol.

## Trust boundary

The caller must supply the actual current EKM live objects when capturing and rechecking the fence. Passing a detached or non-authoritative object graph is outside this module's ability to detect.

This tranche does not bind to legacy `KnowledgeManager`; that manager still coordinates the older graph/causal/ontology engine rather than the EKM authority state. Activation should not silently collapse those two state models.

## Authority boundary

EKM-068 reports only continuity evidence. It does **not**:

- expose a mutable live object
- expose the retained source capsule
- expose the EKM-067 support store
- construct an operational revision history
- authorize belief/evidence mutation
- authorize activation
- authorize a live-state swap
- authorize rollback
- authorize trusted checkpoint advancement
- mutate legacy confidence, causal/world-model state, or action state

The observation/fence/continuity objects retain:

- `mutation_authority = false`
- `activation_preflight_authorized = false`
- `activation_authorized = false`
- `trusted_checkpoint_commit_authorized = false`

## Qualification status

This tranche is stacked on EKM-067 / PR #3990.

GitHub Actions remains the executable authority. Parent EKM-067 exact-head CI #7432 was still queued when EKM-068 was prepared. Static/API review and unit-test authorship are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime or deployment evidence.

## Next boundary

After executable qualification, the next safe tranche is an **activation-preflight receipt** that composes:

1. exact EKM-067 sandbox verification;
2. EKM-068 live-epoch continuity recheck;
3. fresh EKM-061 mutation-seal current-head evidence;
4. fresh EKM-054 protected joint trust-context checkpoint evidence;
5. explicit proof that no trusted checkpoint is advanced during preflight.

Even that preflight receipt should keep live swap, activation and checkpoint commit unauthorized. The actual swap must remain a later atomic transaction.