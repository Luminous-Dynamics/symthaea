# EKM-061 — Mutation-seal checkpoint currentness

## Status

Draft implementation. GitHub Actions remains the executable authority. Static review and provider-interface design are not compile, test, cryptographic, monotonic-storage, transparency-log, witness-quorum, or deployment evidence.

## Purpose

EKM-059 protects an exact mutation-seal checkpoint, and EKM-060 can prove that a supplied sidecar reproduces that protected source commitment. Neither fact alone proves the checkpoint is still the deployment's current head.

A historical signature can remain valid. A historical checkpoint can remain authentic. **Authenticity is not freshness.**

EKM-061 therefore introduces a separate, time-bounded current-head attestation contract.

The central invariant is:

> Currentness may be asserted only when a deployment provider whose evidence semantics explicitly mean monotonic/current-head state accepts a canonical statement bound to the exact EKM-059 checkpoint.

## Evidence taxonomy

`RestartMutationSealCurrentnessEvidenceKindV1` deliberately excludes a generic detached-signature mode.

Allowed classes are:

- `MonotonicProtectedState`
- `HardwareMonotonicCounter`
- `TransparencyLogHead`
- `CurrentHeadWitnessQuorum`

A deployment implementation is responsible for making those semantics real. For example, a transparency provider must actually verify the referenced checkpoint is at the accepted current log head, not merely that it once appeared in the log.

## Currentness statement

`RestartMutationSealCurrentnessStatementV1` binds:

- deployment ID and trust-domain ID;
- exact EKM-059 checkpoint sequence;
- exact EKM-059 checkpoint statement digest;
- restart capture cycle;
- protected EKM-056 seal-capsule digest;
- EKM-059 verification cycle;
- EKM-059 expiry cycle;
- currentness attestation cycle;
- a validity window that may not extend past the source EKM-059 checkpoint;
- currentness authority ID;
- current-head evidence kind.

The statement is domain-separated and canonically BLAKE3 hashed before being passed to the external currentness verifier.

## Verification rules

Before invoking the provider, EKM-061:

- re-verifies the EKM-059 checkpoint internally;
- requires its protected seal-capsule flag to remain true;
- requires all mutation/hydration/activation authority flags to remain false;
- binds every statement field back to the exact EKM-059 checkpoint;
- requires attestation at or after EKM-059 verification;
- rejects an attestation after EKM-059 expiry;
- requires the currentness validity window to be non-empty and no longer than the source checkpoint validity window.

At observation time it additionally requires:

- observation not before attestation;
- observation before currentness evidence expiry.

Only then is the canonical statement digest passed to `RestartMutationSealCurrentnessVerifierV1`.

## Verified currentness receipt

A successful external provider response produces `VerifiedRestartMutationSealCurrentnessV1` with:

- `current_head_proven = true`;
- source checkpoint statement digest;
- proof digest;
- verification cycle;
- `trusted_state_mutated = false`;
- `historical_replay_authorized = false`;
- `writable_hydration_authorized = false`;
- `activation_authorized = false`.

`verify_internal()` re-derives the statement digest and rejects any unexpected authority bit.

## Claim boundary

`current_head_proven = true` means only that the configured provider accepted the exact checkpoint as current under the selected explicit current-head evidence contract and within the receipt's validity window.

It does not mean:

- the provider implementation is infallible;
- the underlying evidence is globally authoritative;
- the checkpoint remains current after the receipt expires;
- the restart should be hydrated or activated;
- the historical evidence itself is true.

## Next boundary

A later read-only gate can combine:

1. EKM-060 protected-source sidecar equivalence; and
2. an unexpired EKM-061 current-head receipt for the exact same EKM-059 checkpoint.

Only that combined gate should make the historical replay path *eligible for isolated reconstruction review*. It should still not directly authorize writable hydration or activation.
