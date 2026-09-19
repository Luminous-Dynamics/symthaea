# EKM-079 — Passive Authority-Epoch Issuance Record

## Purpose

EKM-072 defines the required atomic activation transaction. EKM-074 requires a successful activation commit to be the only boundary that may issue a new restart-authority epoch. EKM-079 defines the passive, non-circular data representation that a future successful executor would need to emit.

The central invariant is:

**activation review, transaction-contract identity, CAS generation, rollback/candidate/installed-state identities, every required transaction phase, and the resulting authority epoch must all be bound into one deterministic lineage — but shape/digest validity is not proof that the transaction actually executed.**

## Non-circular digest structure

EKM-079 separates three identities:

1. `ActivationCommitReceiptDigestV1`
   - deployment/trust-domain identity
   - activated restart V2 digest
   - exact EKM-071 review-receipt digest identity
   - exact canonical EKM-072 transaction-contract digest
   - expected and committed live generation
   - rollback bundle digest
   - candidate bundle digest
   - installed-state digest
   - activation cycle
   - one evidence digest for each required EKM-072 phase in exact order

2. `RestartAuthorityEpochDigestV1`
   - deployment/trust-domain identity
   - epoch sequence
   - explicit predecessor epoch identity
   - activated restart V2 digest
   - activation-commit receipt digest
   - committed live generation
   - activation cycle

3. `RestartAuthorityEpochIssuanceRecordDigestV1`
   - activation-commit receipt digest
   - authority-epoch digest
   - activated V2 digest
   - committed generation
   - activation cycle

The commit digest is therefore computed before the epoch digest, avoiding a circular dependency.

## Genesis predecessor semantics

EKM-074 required a predecessor field but did not define genesis encoding. EKM-079 makes it explicit:

- epoch sequence `1` requires `previous_epoch_digest = None`
- epoch sequence `> 1` requires a non-zero predecessor epoch digest

No magic all-zero predecessor hash is used for genesis.

## Exact activation phase inventory

The record carries one digest for every EKM-072 phase in the canonical order:

1. acquire exclusive live-epoch guard
2. reverify activation review under guard
3. compare expected live epoch/generation
4. capture rollback bundle
5. stage complete candidate bundle
6. atomic live-state swap
7. verify installed state under guard
8. commit trusted checkpoints
9. release live-epoch guard

Phase evidence must be non-zero, ordered exactly like the canonical EKM-072 contract, and individually identified.

## Generation rule

V1 represents the live epoch/CAS token as a monotonic integer generation. A valid passive record requires the committed generation to be exactly the checked successor of the expected generation.

This is a schema rule only; EKM-079 contains no lock, CAS implementation, or live-state mutation.

## Constructor boundary

`RestartAuthorityEpochIssuanceRecordV1` deliberately has no production constructor.

A later executor tranche may create it only from an actually executed atomic activation transaction. EKM-079 does not allow preflight state, a sandbox, detached digests, or archival history to mint an epoch issuance record.

## Claim boundary

`verify()` proves only:

- passive field invariants
- canonical EKM-072 contract binding
- exact phase inventory/order
- predecessor-shape rules
- canonical activation-commit digest
- canonical authority-epoch digest
- canonical issuance-record digest

It does **not** prove that:

- the activation executor actually ran
- the exclusive guard was genuinely acquired
- the CAS operation really occurred
- the rollback bundle was durably captured
- the installed state was independently equivalent to the candidate
- trusted checkpoints were really committed
- the phase-evidence digests correspond to genuine external/runtime evidence
- the authority epoch may become operational

The record therefore reports:

- `successful_activation_event_independently_verified = false`
- `trusted_checkpoint_commit_independently_verified = false`
- `epoch_issuance_authorized = false`
- `mutation_authority = false`
- `activation_authorized = false`

## Authority boundary

EKM-079 does not implement:

- an activation executor
- live locking
- compare-and-swap
- rollback
- installed-state verification
- trusted-checkpoint commit
- epoch issuance
- operational V2 receipt construction
- operational history continuation
- mutation-facade integration
- mutation authority

## Qualification status

This tranche is stacked on EKM-078 / PR #4100. EKM-078 exact-head CI #7554 remained queued when EKM-079 was prepared.

GitHub Actions remains the executable qualification authority. Static/API review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.

## Next boundary

The next safe tranche should independently validate an EKM-079 record against real runtime/external activation evidence and then bind those verified issuance records to EKM-078 epoch segments. Only that later layer may consider upgrading `epoch_issuance_chain_verified` from false, and it must still remain separate from mutation-facade authority.
