# Fabrication Durable Gateway Boundary

Version 0.13 introduces a durable gateway authority layer around the existing
release, machine-session, submission, containment, and audit capabilities.

## Evidence flow

1. A machine emits an integer-valued telemetry payload bound to one manifest,
   machine, timed session, printer job, and monotonically increasing frame.
2. An external provider signs the canonical payload. The kernel verifies the
   signature, trust-snapshot freshness, key lifecycle, telemetry usage, payload
   freshness, and expected identities.
3. A persistent telemetry tracker rejects rollback, same-sequence substitution,
   job substitution, and observation-time regression.
4. A governed submission is written as `Prepared` and durably persisted before
   any printer call. A printer-layer error becomes `Uncertain`; it is never
   treated as evidence that no physical job was accepted.
5. Submission-ledger transitions are mirrored into the audit journal and must
   reconcile exactly. Missing, duplicate, altered, or orphaned evidence blocks
   gateway replay authority.
6. Trust, audit, session anti-replay, telemetry anti-replay, and submission
   evidence are sealed into one hash-chained gateway generation.
7. The state store uses a same-directory lock, bounded envelope, synced pending
   file, atomic rename, directory sync, and optimistic current-digest guard.
8. Gateway replay binds the operational replay digest, exact state generation,
   all retained evidence digests, reconciliation result, and rotation policy.

## Trust rotation

Trust snapshots are rotated through a separately signed proposal. Signers must
be currently eligible for `TrustRotation`. The policy controls signer quorum,
algorithm diversity, activation delay, emergency behavior, key usage coverage,
and scheduled overlap. Existing keys cannot silently disappear; removal must be
represented as an explicit lifecycle record.

Governance authorization and activation are distinct. Activation stages both
the monotonic trust tracker and the audit journal before committing the new
in-memory snapshot, preventing a partial authority transition.

## Non-claims

This layer does not make a generic printer transport exactly-once. Most printer
APIs cannot prove whether a timed-out request was accepted. The correct state is
therefore `Uncertain`, requiring external reconciliation. The atomic state store
also depends on the host filesystem honoring same-directory rename and sync
semantics; deployment-specific fault injection remains required.
