# Symthaea RCA Shadow Disposition Evaluation Policy

RCA-003b.3d freezes the exact **evaluation surface and decision semantics** before any result-bearing shadow-disposition engine exists.

## Why this binding is necessary

The preregistered disposition policy predates both preflight contracts, and the actual result-bearing engine does not exist yet:

```text
RCA-003b.3b ShadowDispositionPreflightV1
RCA-003b.3c LineageBoundShadowDispositionPreflightV1
RCA-003b.4 future pure disposition engine
```

If only the input profiles were policy-bearing, two different engines could interpret the same qualified artifacts using different precedence or outcome semantics.

This layer closes both gaps before result-bearing code exists:

```text
RegisteredEffectiveShadowDispositionPolicyV1
        +
exact raw-preflight profile
        +
exact lineage-bound-preflight profile
        +
exact pure-engine decision-contract profile
        ↓
RegisteredShadowDispositionEvaluationPolicyV1
```

## Non-result-bearing engine contract

`src/engine_contract.rs` contains **no classifier** and accepts no evaluation artifacts. It freezes only the semantic contract a future engine must implement:

- future input classes;
- exact identity-join requirements;
- evidence-item versus interpretation-root cardinality semantics;
- primary outcome taxonomy;
- predicate taxonomy;
- exact decision-rule precedence;
- qualified-defeater blocker semantics;
- conservative bilateral-disagreement behavior;
- scoped unknown-independence semantics;
- typed reason-trace requirements;
- serializer-independent result identity requirement;
- Serialize-only current-result boundary;
- purity and downstream-authority exclusions.

The stable V1 primary classes are:

```text
BlockedByQualifiedDefeater
Contested
Supported
TentativelySupported
Opposed
TentativelyOpposed
Underdetermined
```

The stable V1 rule precedence is:

```text
qualified defeater blocker
    >
qualified contestation
    >
bilateral qualified disagreement below contestation
    >
full support / full opposition
    >
tentative support / tentative opposition
    >
insufficient topology / underdetermined
```

There is no count-margin, vote, relation-strength, posterior, or winner-take-all tie-breaker.

## Engine-contract identity

`shadow_disposition_engine_contract_profile_digest_v1()` is a domain-separated BLAKE3 identity over:

- the normative contract text;
- engine-contract schema/profile;
- all primary class tags;
- all predicate tags;
- the exact ordered decision-rule precedence table.

Changing the taxonomy, predicate surface, or precedence therefore changes the engine-contract profile.

## Evaluation-policy identity

The domain-separated BLAKE3 `evaluation_policy_id` binds:

- evaluation-policy profile/schema;
- exact effective-policy ID;
- exact effective-policy profile-contract digest;
- exact raw-preflight profile digest;
- exact canonical-lineage-bound preflight profile digest;
- exact pure-engine decision-contract profile digest.

Therefore:

```text
preflight contract drift
or
decision-semantic drift
        ↓
new evaluation-policy identity required
```

The engine cannot decide what its decision rules mean after observing the case.

## Registration timing

This crate accepts only:

```text
RegisteredEffectiveShadowDispositionPolicyV1
```

It obtains the current preflight and engine-contract profile digests from their contract modules.

It accepts **no** case, raw preflight, lineage-bound preflight, witness, interpretation-lineage, or result instance. Registration can therefore occur before result-bearing evaluation.

## Persistence

The registered artifact derives `Serialize`; deserialization:

1. revalidates the nested effective policy;
2. recomputes the current raw-preflight profile;
3. recomputes the current lineage-bound-preflight profile;
4. recomputes the current pure-engine decision-contract profile;
5. recomputes the evaluation-policy profile;
6. recomputes the complete evaluation-policy ID;
7. rejects any mismatch.

## Authority separation

```text
pure engine contract
        !=
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

This layer freezes **which evaluation contracts and decision semantics are permitted**. It does not evaluate an instance under them.

## Future engine rule

A later pure shadow-disposition engine should require:

```text
RegisteredShadowDispositionEvaluationPolicyV1
+
LineageBoundShadowDispositionPreflightV1
+
exact witnessed artifacts needed for the typed reason trace
```

and verify that the lineage-bound preflight's embedded raw preflight carries the same effective-policy ID contained by the registered evaluation policy.

The engine should not accept raw preflight directly, should not rediscover independence/currentness/provenance, and should not invent precedence or policy/profile bindings itself.
