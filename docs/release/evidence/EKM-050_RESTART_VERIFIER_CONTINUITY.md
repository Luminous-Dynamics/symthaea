# EKM-050 — Restart Verifier Trust Continuity

## Purpose

EKM-049 records which verifier profile and trust snapshot were stable across one restart-admission review. A valid candidate can still present an older verifier trust snapshot or silently replace the verifier implementation/configuration.

EKM-050 adds a read-only continuity gate against a caller-held trusted verifier checkpoint.

The central invariant is:

**restart-state forward progress may not conceal verifier-trust rollback or verifier replacement.**

## Strict V1 continuity rules

The candidate must preserve exactly:

- verifier identity
- implementation identity
- implementation version
- configuration digest

Verifier/software/configuration replacement is intentionally rejected in V1. A future explicit migration protocol can authorize reviewed transitions rather than making replacement an implicit side effect of restart.

Trust-snapshot continuity allows:

- same sequence + same digest: stable trust-state reuse
- greater sequence: trust-snapshot advance

It rejects:

- lower sequence: trust-snapshot rollback
- same sequence + different digest: same-sequence substitution

## Trust boundary

`TrustedRestartVerifierStateV1` is derived from a previously accepted EKM-049 provenance receipt. The gate cannot make the caller-held checkpoint rollback-resistant by itself; durable monotonic storage remains an external deployment responsibility.

The gate performs no cryptographic verification. EKM-049 still owns verifier-profile binding and EKM-046/047 still own proof-provider and local-policy checks.

## Authority boundary

EKM-050 does not:

- mutate the trusted verifier checkpoint
- mutate the trusted anchor tracker
- construct quarantine state
- hydrate writable state
- activate a restart
- authorize verifier migration
- perform file/network I/O
- mutate evidence, belief, causal, world-model or action state

Every decision reports `trusted_state_mutated = false`, `quarantine_construction_authorized = false`, and `activation_authorized = false`.

## Qualification status

This tranche is stacked on EKM-049. The EKM-049 exact-head CI and forced EKM-046 Format Check remain queued at authoring time; queue/cancellation state is not executable qualification evidence.
