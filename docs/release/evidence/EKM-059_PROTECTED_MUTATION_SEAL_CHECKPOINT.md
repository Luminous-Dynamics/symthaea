# EKM-059 — Protected mutation-evidence-seal checkpoint

## Status

Draft implementation. GitHub Actions remains the executable authority. Static review, queued jobs, deterministic hashing, and successful provider-interface construction are not format, compile, Clippy, test, runtime, or deployment proof.

## Purpose

EKM-058 can prove that an untrusted EKM-057 mutation-evidence-seal sidecar is cross-component consistent with one restart-v2 history, but it deliberately cannot prove that an attacker did not omit historical non-basis evidence and recompute sidecar-local digests.

EKM-059 closes the missing **protected source commitment** boundary without widening EKM-054 V1 or granting restore authority.

The central invariant is:

> The original EKM-056 mutation-evidence-seal capsule digest becomes protected restart evidence only when an external deployment verifier accepts one canonical statement that also binds the exact protected trust checkpoint, restart anchor, restart validation receipt, restart V2 digest, capture epoch, and mutation count.

## Why EKM-054 V1 is not modified

EKM-054 already defines an externally verified, rollback-aware checkpoint over the joint deployment trust context. Changing that V1 statement would retroactively widen its semantics and complicate continuity.

EKM-059 therefore creates a separate child commitment whose parent is the exact verified EKM-054 checkpoint.

## Protected statement

`RestartMutationSealCheckpointStatementV1` binds:

- checkpoint sequence and optional predecessor digest;
- deployment ID and trust-domain ID;
- exact parent EKM-054 checkpoint sequence and statement digest;
- exact joint trust-context digest;
- exact externally verified restart-anchor sequence and statement digest;
- exact restart capture cycle;
- exact EKM-044 restart-validation-receipt digest;
- exact claimed restart-v2 digest;
- EKM-056 seal-capsule capture cycle;
- linked EKM-030 mutation capture cycle;
- exact persisted mutation count;
- exact original EKM-056 seal-capsule digest;
- issue/expiry window;
- external authority identifier;
- protected-storage/signature/hardware/transparency/witness evidence kind.

The statement is domain-separated and canonically BLAKE3 hashed. The digest is then passed to `RestartMutationSealCheckpointVerifierV1`; the hash is not treated as trust by itself.

## Source binding

Before a statement can be created or verified, EKM-059 requires:

- the EKM-054 parent receipt to pass its own internal verification;
- the restart-anchor statement digest to re-derive exactly;
- all source authority flags to remain false;
- the parent checkpoint's anchor sequence/capture cycle to match the exact verified anchor;
- the verified anchor's receipt digest/capture cycle to match the exact restart validation receipt;
- the EKM-056 linked mutation epoch to equal the restart receipt capture epoch;
- the EKM-056 seal capture to be no earlier than its mutation epoch;
- issue time to postdate parent verification, anchor verification, and seal capture;
- parent checkpoint and anchor evidence to remain unexpired at issue time.

Verification rechecks every statement/source equality before invoking the external provider.

## Verified receipt

`VerifiedRestartMutationSealCheckpointV1` records:

- the canonical statement;
- its externally verified statement digest;
- a domain-separated digest of the opaque proof;
- verification cycle;
- `seal_capsule_digest_protected = true`;
- `trusted_state_mutated = false`;
- `capsule_construction_authorized = false`;
- `writable_hydration_authorized = false`;
- `activation_authorized = false`.

`verify_internal()` re-derives the statement digest and rejects any unexpected authority bit.

## Continuity

The child checkpoint has its own monotonic continuity gate. Forward progress requires:

- unchanged deployment and trust domain;
- exact `+1` checkpoint sequence;
- exact predecessor digest;
- no parent EKM-054 checkpoint sequence rollback;
- no restart-anchor sequence rollback;
- a strictly newer restart capture epoch;
- a different mutation-seal capsule digest;
- no verification-cycle rollback.

The continuity decision is review-only and still grants no hydration or activation authority.

## What EKM-059 proves

After a deployment verifier accepts the canonical statement, EKM-059 can establish that the configured external trust mechanism protected **this exact seal-capsule digest in this exact restart lineage**.

It does not, by itself, decode or reconstruct the seal capsule and it does not independently establish that a supplied EKM-057 sidecar matches that protected digest.

That is the next boundary.

## Next boundary

EKM-060 should combine:

1. a successful EKM-058 cross-component validation report;
2. the exact EKM-057 decoded sidecar;
3. the exact restart validation receipt;
4. an internally valid `VerifiedRestartMutationSealCheckpointV1`.

It should require the EKM-058 recomputed seal-capsule digest to equal the protected EKM-059 digest and recheck all restart lineage fields. Only then may a read-only admission receipt state that the supplied historical census is equivalent to the externally protected source commitment.

Even EKM-060 should keep capsule construction, writable hydration, and activation false.
