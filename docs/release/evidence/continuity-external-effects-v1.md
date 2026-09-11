# External effects v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact parent:

`architecture/continuity-current-crash-recovery-v1`

Parent exact head at branch creation:

`777509ecb220303f1772d1bfaf5357a2fb327aba`

Exact code subject before this evidence-only commit:

`0ab51d53f67fdc5c6c078743a0b44515a0f68f31`

## Theorem under qualification

```text
exact A -> B transition lineage
+ exact predeclared external-effect coverage manifest
+ canonical exact external-effect obligations
+ fresh authenticated one-per-obligation observations
+ every observation satisfies its exact recovery predicate
    -> QualifiedExternalEffectReconciliationV1
```

Supported first-class effect classes include database mutation, emitted message/event, external API mutation, network control-plane mutation, storage mutation, identity/credential mutation, device actuation, and canonical custom effects.

Recovery predicates are explicit and effect-specific: effect absent, source-equivalent state restored, compensation established under the exact compensation contract, or persistence explicitly accepted under the exact policy.

The implementation must fail closed on incomplete/duplicate observation sets, UNKNOWN or present-unreconciled states, cross-contract evidence, stale/future evidence outside policy, verifier/profile substitution, policy substitution, non-canonical obligation ordering, zero/missing evidence material, wrong source-state digest, and wrong compensation/acceptance policy digest.

An empty obligation set must not establish reconciliation. A future separate proof is required to establish by construction/analysis that a transition has no external effects within an exact declared coverage scope.

## Scope honesty

This proof establishes reconciliation only for obligations contained in the exact coverage manifest. It does **not** establish that no other external effect exists outside the declared scope.

## Authority boundary

The contract, observations, and reconciliation proof grant no physical execution, retry, LKG promotion, bootstrap, or external compensation authority. The contract must be bound into the durable pre-mutation execution lineage before it may strengthen any recovery proof.

No test or CI pass is claimed here. Exact-head CI and all parent qualification remain required.
