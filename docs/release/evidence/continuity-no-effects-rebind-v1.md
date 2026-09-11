# Continuity No-Effects Restart Rebind V1 — Frozen Subject

## Stack base

Stacked directly on draft PR #1499 exact head:

`0f00e4cdf80d619adacfaa37e9d3b99e9255e00b`

## Exact code subject

`090a7e01fbd7b44b63b7e0f82c14f6cbb23896d4`

## Theorem

```text
persisted NoEffectsExecutionEnvelopeV1
+ exact durable A -> B intent
+ exact NoExternalEffectsDeclarationV1
+ exact reconstructed QualifiedNoExternalEffectsAuthorizationV1
+ exact QualifiedNoExternalEffectsCoverageV1
+ exact ExecutionBackendProfileV1
    -> ReboundNoEffectsExecutionV1
```

## Canonical reconstruction

Rebind does not trust the persisted envelope's self-hash alone. It recomputes the exact V1 envelope digest from the live intent/declaration/authorization/coverage/backend parents using the original envelope hash domain and field order, then requires that digest to equal the persisted envelope ID.

This catches a persisted envelope whose duplicated backend/material fields were altered and whose self-hash was recomputed afterward.

## Exact-time rule retained

The live no-effects coverage proof must still have `analyzed_at_unix_ms == exact commit time` from the exact A -> B lineage.

## Non-claims

`ReboundNoEffectsExecutionV1` is non-Serde proof of semantic reconstruction only. It grants no execution, retry, recovery, promotion, or bootstrap authority and does not replace the exact rollback-resistant pre-mutation `QualifiedNoEffectsExecutionCommitmentV1` required downstream.

CI/compiler/test qualification is not established until exact-head executable evidence exists.