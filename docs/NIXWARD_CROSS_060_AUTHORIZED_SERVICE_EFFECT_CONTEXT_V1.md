# Nixward CROSS-060 — Authority-Bound Service Effect Context V1

## Purpose

CROSS-060 moves the service-definition commitment from a post-state-only concept into the authority chain.

A service effect can no longer be considered authorization-ready merely because a later post-state expectation names an authorized definition digest.

## Typed context

`NixServiceEffectContextV1` is a neutral, transport-independent value containing:

- typed lifecycle operation;
- canonical service unit;
- authorized NixOS generation;
- exact pre-state digest;
- authorized systemd definition commitment digest;
- optional pre-invocation identity;
- required stability window.

The context has its own deterministic domain-separated digest.

The type is deliberately independent of the authorization, executor, observer, and receipt modules to prevent a provenance cycle.

## Intent binding

`NixActionIntentV1` now carries an optional service-effect context.

The field is optional only for serializable/unqualified intent data. For a typed Service action, the context must agree exactly with:

- operation;
- unit;
- generation;
- the exact rich service pre-state identity;
- authorized definition digest;
- pre-invocation identity;
- stability contract.

The pre-state identity is required to use the canonical:

```
nixward-service-pre-state-v1|generation=<generation>|unit=<unit>|state=<digest>
```

form when a service effect context is present. Therefore the context's pre-state digest cannot float independently of the intent's actual pre-state commitment.

The context digest is included in the intent digest.

## Authorization binding

`NixExecutionAuthorizationRecordV1` now carries the digest of the service-effect context.

For typed Service actions, local explicit confirmation will not mint an authorization record without that context.

The authorization record can independently validate itself against the exact intent:

```
intent digest
    +
service-effect-context digest
    +
approved decision
    =
authorization-compatible
```

The live execution authority promotion path invokes the same context gate. This is important because otherwise a valid consumed approval could become live authority merely from matching an intent digest.

For non-Service actions, a service-effect context is rejected rather than silently ignored.

## Post-state transitive binding

Post-state expectation validation now requires the same context and binds:

```
intent
  -> service effect context
  -> authorization record
  -> post-state expectation
  -> observed evidence
  -> receipt
```

The expectation cannot substitute a different generation, definition commitment, invocation identity, stability contract, or pre-state digest.

## Source identity vs content identity

The authorized definition field is intentionally named a **commitment digest** rather than being presented as a file-content hash.

The current definition identity model separately represents systemd's FragmentPath/DropInPaths source identity. CROSS-067 makes that source identity self-verifying in the durable receipt.

A future content-observation tranche may add exact bytes/content hashes. CROSS-060 does not collapse these distinct provenance layers.

## Qualification vectors

The boundary must reject:

- Service intent without effect context entering local authorization;
- Service intent without effect context entering live execution authority;
- context operation/unit mismatch;
- context generation mismatch;
- context pre-state mismatch;
- context definition commitment mismatch;
- context invocation mismatch;
- context stability-contract mismatch;
- context present on a non-Service intent;
- authorization record with missing context digest;
- authorization record with a mismatched context digest;
- replay of an authorization context against a different service intent.

## Claim ceiling

This tranche does not provide cryptographic proof that the authorized definition bytes existed at approval time. It binds the service-definition commitment into the authorization identity.

It also does not itself eliminate execution TOCTOU, systemd-manager rollover, or post-state observation races; those are handled by the adjacent CROSS-061/062/063/065/066/067 hardening layers.

## Qualification standard

No state is called PASS from branch mergeability or static inspection alone.

The exact checked-in HEAD must complete the relevant GitHub Actions successfully. Queued, pending, cancelled, or merely mergeable states are not qualification evidence.
