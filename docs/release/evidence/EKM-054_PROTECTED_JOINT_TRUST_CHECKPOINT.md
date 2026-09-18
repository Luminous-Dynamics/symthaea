# EKM-054 — Protected Joint Trust-Context Checkpoint Evidence

## Purpose

EKM-052 binds restart-anchor state and verifier-trust state into one immutable deployment trust epoch. EKM-053 seals the external-byte quarantine capability surface behind that joint context. Neither tranche by itself proves that the context digest is protected outside process memory against rollback, substitution, or storage tampering.

EKM-054 adds a provider-neutral checkpoint-evidence contract for protecting the exact EKM-053 `RestartTrustContextHandleV1` digest.

The central invariant is:

**external checkpoint evidence must bind one exact deployment/trust-domain context digest and its embedded anchor/verifier chronology before it can be considered verified audit evidence.**

## Canonical checkpoint statement

`RestartTrustContextCheckpointStatementV1` binds:

- positive checkpoint sequence
- deployment ID
- trust-domain ID
- exact joint trust-context digest
- context commit cycle
- embedded restart-anchor sequence
- embedded anchor capture cycle
- embedded verifier trust-snapshot sequence
- exact predecessor checkpoint digest
- issue/expiry cycle
- bounded authority ID
- typed upstream evidence mechanism

Genesis is sequence `1` with no predecessor. Non-genesis statements require a predecessor digest.

The issue cycle may not predate the joint context commit, and the context commit may not predate its embedded anchor capture cycle.

## Provider boundary

`RestartTrustContextCheckpointVerifierV1` is an integration seam only. It receives:

- evidence kind
- authority ID
- canonical statement digest
- opaque proof bytes

V1 supports descriptive mechanism classes for:

- protected storage
- signed checkpoint
- hardware attestation
- transparency checkpoint
- witness quorum

The knowledge layer does not implement the cryptographic or storage mechanism itself. A deployment provider remains responsible for actual signature verification, TPM/secure-element attestation, transparency-log inclusion/consistency, witness-quorum validation, or protected-storage guarantees.

A malicious or misconfigured provider can still accept invalid evidence. Provider acceptance is therefore **not** equivalent to universal trust.

## Verified receipt

A successful provider verification produces `VerifiedRestartTrustContextCheckpointV1`, binding:

- exact checkpoint statement
- canonical statement digest
- domain-separated proof digest
- verification cycle

The receipt independently rechecks its statement digest and hard-codes:

- `trusted_state_mutated = false`
- `quarantine_construction_authorized = false`
- `writable_hydration_authorized = false`
- `activation_authorized = false`

The receipt is audit evidence. It is not a capability token.

## Continuity review

`RestartTrustContextCheckpointContinuityGateV1` compares two verified receipts without mutating either one.

Forward progress requires:

- stable deployment ID
- stable trust-domain ID
- exactly contiguous checkpoint sequence
- exact predecessor checkpoint digest
- a different joint-context digest
- strictly newer context commit cycle
- no restart-anchor sequence rollback
- no anchor capture-cycle rollback
- no verifier trust-snapshot rollback
- no verification-cycle rollback

Failures remain typed and diagnostic.

This makes four chronologies independently reviewable:

1. epistemic restart capture chronology
2. restart-anchor chronology
3. verifier trust-snapshot chronology
4. externally protected joint-context checkpoint chronology

Progress in one chronology cannot silently excuse rollback in another.

## Persistence boundary

EKM-054 does **not** itself write a checkpoint anywhere and does not advance trusted checkpoint state.

Actual rollback resistance still depends on a deployment binding the provider interface and checkpoint digest to a mechanism whose own state cannot be silently rolled back. That may later be a signed append-only log, independent witnesses, hardware monotonic state, protected remote storage, or another qualified mechanism.

## Authority boundary

EKM-054 does not:

- construct quarantine state
- hydrate writable support state
- rebuild writable revision history
- activate restart state
- advance restart anchors
- advance verifier checkpoints
- persist checkpoint state
- perform file/network I/O
- hold signing keys
- mutate evidence, belief, causal, world-model, or action state

## Qualification status

This tranche is stacked on EKM-053. GitHub Actions remains the executable authority. Queued/cancelled jobs are not format, compile, Clippy, test, or runtime evidence.
