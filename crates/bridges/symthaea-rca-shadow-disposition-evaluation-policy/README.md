# Symthaea RCA Shadow Disposition Evaluation Policy

RCA-003b.3d freezes the exact **evaluation surface and decision semantics** before any result-bearing shadow-disposition engine exists.

## Why this binding is necessary

The preregistered disposition policy predates both preflight contracts, and the actual result-bearing engine does not exist yet:

```text
RCA-003b.3b ShadowDispositionPreflightV1
RCA-003b.3c LineageBoundShadowDispositionPreflightV1
RCA-003b.4 future pure disposition engine
```

If only the input profiles were policy-bearing, two different engines could interpret the same qualified artifacts using different precedence, rule meanings, outcome classes, or audit traces.

This layer closes those gaps before result-bearing code exists:

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

`src/engine_contract.rs` contains **no production classifier** and accepts no evaluation artifacts. It freezes only the semantic contract a future engine must implement:

- future input classes;
- exact identity-join requirements;
- evidence-item versus interpretation-root cardinality semantics;
- primary outcome taxonomy;
- predicate taxonomy;
- exact decision-rule precedence;
- exact decision-rule → primary-class mapping;
- qualified-defeater blocker semantics;
- qualified-contestation semantics;
- conservative bilateral-disagreement behavior;
- scoped unknown-independence semantics;
- exact reason-trace field schema;
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
QualifiedDefeaterBlocker
QualifiedContestation
BilateralQualifiedDisagreementBelowContestation
FullSupport
FullOpposition
TentativeSupport
TentativeOpposition
InsufficientTopology
```

The exact rule → class mapping is:

```text
QualifiedDefeaterBlocker                         -> BlockedByQualifiedDefeater
QualifiedContestation                            -> Contested
BilateralQualifiedDisagreementBelowContestation  -> Underdetermined
FullSupport                                      -> Supported
FullOpposition                                   -> Opposed
TentativeSupport                                 -> TentativelySupported
TentativeOpposition                              -> TentativelyOpposed
InsufficientTopology                             -> Underdetermined
```

There is no count-margin, vote, relation-strength, posterior, or winner-take-all tie-breaker.

## Qualified contestation

The frozen predicates are:

```text
D   defeater qualified
CS  support meets contested-side topology
CO  opposition meets contested-side topology
S   full support qualified
TS  tentative support qualified
O   full opposition qualified
TO  tentative opposition qualified
```

Policy registration already guarantees:

```text
S => TS
O => TO
```

V1 additionally freezes the classifier condition:

```text
QualifiedContestation = CS && CO && TS && TO
```

`CS && CO` alone is not enough. `contested_side_requirements` are separately preregistered and could be lower than a side's tentative threshold. Requiring `TS && TO` ensures `Contested` always means both sides independently qualify at least tentatively before the stronger contested topology can select the primary class.

If both tentative sides survive but qualified contestation is not established, V1 selects:

```text
BilateralQualifiedDisagreementBelowContestation -> Underdetermined
```

No count margin may erase either side.

## Exhaustive decision-lattice theorem

Under only the admissibility invariants `S => TS` and `O => TO`, there are exactly **72 admissible boolean states**. A test-only classifier in `engine_contract.rs` enumerates all 72 and proves every state selects exactly one frozen rule whose primary class exists in the frozen taxonomy.

The contestation requirement above is decision logic, not an additional admissibility constraint, so the 72-state theorem remains stable.

This proof is test-only. Production code in this tranche still exposes no classifier.

## Exact reason-trace schema

The future result must retain exact machine-readable audit fields. These tag tables are part of the engine-contract digest.

### Identity lineage

```text
engine_implementation_profile_digest
engine_contract_profile_digest
evaluation_policy_id
effective_policy_id
base_policy_id
lineage_bound_preflight_binding_id
raw_preflight_id
proposition_id
case_id
canonical_evidence_lineage_graph_id
registered_experiment_contract_digest
```

### Support/opposition/defeater slot facts

For each of the three slots:

```text
evidence_witness_id
evidence_item_count
interpretation_witness_id
interpretation_root_count
```

That produces 12 exact slot-fact fields.

### Predicate and decision facts

The seven frozen predicates are retained independently of the selected primary outcome. Final decision fields are exactly:

```text
decision_rule_id
primary_class
```

A higher-precedence result therefore cannot erase simultaneously true lower-level evidence facts. For example, `BlockedByQualifiedDefeater` must still preserve support/opposition/contestation predicates that were also satisfied.

## Engine-contract identity

`shadow_disposition_engine_contract_profile_digest_v1()` is a domain-separated BLAKE3 identity over:

- normative contract text;
- engine-contract schema/profile;
- all primary class tags;
- all predicate tags;
- exact ordered decision-rule precedence;
- exact rule → primary-class mapping;
- exact identity-trace field tags;
- exact slot-trace field tags;
- exact decision-trace field tags.

Changing any of those requires a new evaluation-policy identity before result-bearing evaluation.

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
decision-semantic/audit-schema drift
        ↓
new evaluation-policy identity required
```

The engine cannot decide what “winning,” “blocked,” “contested,” or “auditable” means after observing the case.

## Registration timing

This crate accepts only:

```text
RegisteredEffectiveShadowDispositionPolicyV1
```

It obtains current preflight and engine-contract profile digests from their contract modules.

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

This layer freezes **which evaluation contracts, decision semantics, and audit schema are permitted**. It does not evaluate an instance under them.

## Queue governance

The focused workflow uses the repository's mature evidence-run pattern:

```text
automatic PR/ref run -> stale automatic copy superseded
manual workflow_dispatch -> unique run_id, preserved independently
```

The focused job also has a 20-minute timeout. Superseded/cancelled runs remain infrastructure state, never PASS/FAIL epistemic evidence.

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

The engine should not accept raw preflight directly, should not rediscover independence/currentness/provenance, and should not invent precedence, rule mappings, trace fields, or policy/profile bindings itself.
