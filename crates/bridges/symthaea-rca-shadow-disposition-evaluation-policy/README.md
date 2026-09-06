# Symthaea RCA Shadow Disposition Evaluation Policy

RCA-003b.3d freezes the exact **evaluation surface** before any result-bearing shadow-disposition engine exists.

## Why another policy binding is necessary

The preregistered disposition policy predates both preflight contracts:

```text
RCA-003b.3b ShadowDispositionPreflightV1
RCA-003b.3c LineageBoundShadowDispositionPreflightV1
```

If those profiles were not policy-bearing, an engine could choose a different preflight semantics under the same preregistered thresholds.

This layer closes that gap without rewriting earlier policy artifacts:

```text
RegisteredEffectiveShadowDispositionPolicyV1
        +
exact raw-preflight profile
        +
exact lineage-bound-preflight profile
        ↓
RegisteredShadowDispositionEvaluationPolicyV1
```

## Identity

The domain-separated BLAKE3 `evaluation_policy_id` binds:

- evaluation-policy profile/schema;
- exact effective-policy ID;
- exact effective-policy profile-contract digest;
- exact raw-preflight profile digest;
- exact canonical-lineage-bound preflight profile digest.

Any preflight contract drift therefore requires a new evaluation-policy identity before result-bearing evaluation.

## Registration timing

This crate accepts only:

```text
RegisteredEffectiveShadowDispositionPolicyV1
```

It obtains the two current preflight profile digests from their contract modules.

It accepts **no** case, raw preflight, lineage-bound preflight, witness, or interpretation-lineage instance. Registration can therefore occur before result-bearing evaluation.

## Persistence

The registered artifact derives `Serialize`; deserialization:

1. revalidates the nested effective policy;
2. recomputes the current raw-preflight profile;
3. recomputes the current lineage-bound-preflight profile;
4. recomputes the evaluation-policy profile;
5. recomputes the complete evaluation-policy ID;
6. rejects any mismatch.

## Authority separation

```text
RegisteredShadowDispositionEvaluationPolicyV1
        !=
lineage-bound preflight
        !=
shadow disposition
        !=
canonical epistemic state
        !=
workspace/GWT authority
        !=
action authority
        !=
self-improvement promotion
```

This layer freezes **which evaluation contracts are permitted**. It does not evaluate an instance under them.

## Future engine rule

A later pure shadow-disposition engine should require:

```text
RegisteredShadowDispositionEvaluationPolicyV1
+
LineageBoundShadowDispositionPreflightV1
+
exact witnessed artifacts needed for reason trace
```

and verify that the lineage-bound preflight's embedded raw preflight carries the same effective-policy ID contained by the registered evaluation policy.

The engine should not accept raw preflight directly and should not invent policy/profile bindings itself.
