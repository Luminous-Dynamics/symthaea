# EKM-066 — Immutable Revision Audit Restoration

## Purpose

EKM-064 historically re-executes the EKM-025 receipts that actually sourced persisted belief mutations, because those receipts are needed by EKM-026. Rejected and unapplied revision receipts are intentionally represented only by private ID-cursor placeholders during that replay.

EKM-065 therefore separates support-store hydration review from complete revision-history hydration review.

EKM-066 restores the complete persisted EKM-025 audit semantics as a **read-only typed audit image** without constructing an operational `BeliefRevisionHistory` and without pretending rejected/unapplied decisions were historically re-executed.

The central invariant is:

**restoring immutable audit semantics is not the same operation as re-executing a historical decision or restoring mutation authority.**

## Source contract

EKM-066 re-verifies the complete EKM-065 source chain and then consumes the already validated restart representations:

- EKM-034 base restart wire for receipt inputs and immutable evidence-basis snapshots
- EKM-040 typed revision-schema wire for canonical policy and decision semantics
- EKM-064 replay report for the exact set of mutation-source revisions that were historically re-executed
- EKM-065 support-hydration eligibility receipt for quarantine/trust/replay binding

No V1 `policy_debug` or `decision_debug` string is used as the restored policy/decision semantic source.

## Restored record

Each `ImmutableRevisionAuditRecordV1` preserves:

- receipt ID
- claim ID
- proposed delta
- rationale
- immutable basis snapshots
- duplicate-basis diagnostics
- complete typed `BeliefRevisionPolicySchemaV1`
- calibration snapshot
- uncertainty assessment
- complete typed `BeliefRevisionDecisionSnapshotV1`
- evaluation cycle
- optional persisted mutation ID
- whether that receipt was actually historically re-executed by EKM-064

Every record reports `mutation_authority = false`.

## Historical re-execution classification

EKM-066 requires a strict one-to-one mapping between:

- persisted mutation source-revision IDs, and
- EKM-064 historically replayed source-revision IDs.

A mutation-source receipt missing from EKM-064 is rejected. A non-mutation receipt unexpectedly marked historically replayed is also rejected.

The resulting counts distinguish:

- mutation-source receipts historically re-executed through EKM-026
- rejected/unapplied receipts restored from typed persistence without historical decision re-execution

The latter are audit restoration, not experimental replay.

## Digest semantics

The restoration digest binds:

- exact restart capture cycle
- exact EKM-042 outer checksum
- exact EKM-034 base-wire checksum
- exact EKM-040 typed-schema-wire checksum
- exact EKM-065 eligibility receipt digest
- receipt counts and classifications
- ordered receipt IDs, claim IDs, delta bit patterns, mutation linkage, eligibility, evaluation cycles and authority flags

The canonical base/schema wire checksums already commit the complete persisted receipt and typed policy/decision payloads. EKM-066 does not fall back to Rust `Debug` hashing.

## Explicit claims

A successful restoration may report:

- `complete_immutable_revision_audit_restored = true`

It simultaneously reports:

- `operational_revision_history_constructed = false`
- `nonmutation_decisions_historically_reexecuted = false`
- `mutation_authority = false`
- `writable_hydration_authorized = false`
- `writable_state_export_authorized = false`
- `activation_authorized = false`

This prevents a passive audit image from becoming a disguised belief-mutation capability.

## Why this is preferable to arbitrary receipt injection

A tempting shortcut would be to add a private constructor that inserts arbitrary persisted receipt IDs directly into `BeliefRevisionHistory`. EKM-066 deliberately does not do that.

The operational history type retains its existing semantics. The immutable restoration is a separate read-only representation whose role is audit continuity, not authority restoration.

A future operational restore protocol can decide how to combine this immutable audit overlay with a sealed writable support store without weakening EKM-025/026 authority boundaries.

## Authority boundary

EKM-066 does not:

- construct `BeliefRevisionHistory`
- construct or mutate `EpistemicSupportStore`
- create a belief mutation authorization
- call the mutation firewall
- grant support or revision hydration authority
- advance any trusted checkpoint
- mutate legacy confidence
- mutate causal/world-model/action state
- perform file/network I/O or key custody

## Qualification status

This tranche is stacked on EKM-065. GitHub Actions remains the executable authority. Parent EKM-065 exact-head CI #7369 was queued when this evidence note was prepared.

Static/API review is not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.

## Next boundary

The next safe composition layer can bind:

1. EKM-064 exact support-mutation replay,
2. EKM-065 support hydration review eligibility, and
3. EKM-066 complete immutable revision audit restoration.

That composition may establish that a future sealed hydration sandbox has both reproducible writable support state and complete immutable audit continuity. It should still not grant live-state activation or trusted-checkpoint commit authority.
