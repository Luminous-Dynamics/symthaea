# RCA-003b.3d Qualification Contract

Qualification target: `symthaea-rca-shadow-disposition-evaluation-policy`

Status until hosted Actions pass: **IMPLEMENTED, NOT QUALIFIED**.

## Required theorem

```text
non-result-bearing engine contract
        !=
registered effective policy
        !=
registered evaluation-surface policy
        !=
result-bearing evaluation
        !=
shadow disposition
        !=
canonical belief/workspace state
        !=
action authority
        !=
self-improvement promotion
```

## Required positive checks

Hosted qualification must establish:

1. exact effective-policy ID is evaluation-policy-bearing;
2. exact effective-policy profile digest is evaluation-policy-bearing;
3. exact raw-preflight profile digest is evaluation-policy-bearing;
4. exact canonical-lineage-bound preflight profile digest is evaluation-policy-bearing;
5. exact pure-engine decision-contract profile digest is evaluation-policy-bearing;
6. engine-contract digest binds normative contract, class tags, predicate tags, and ordered rule precedence;
7. qualified defeater precedes contestation and unilateral outcomes in the frozen rule table;
8. bilateral qualifying disagreement precedes unilateral support/opposition outcomes;
9. no vote/count-margin/relation-strength/posterior tie-breaker enters the engine contract;
10. drift in preflight or engine decision semantics requires a new evaluation-policy identity;
11. domain-separated serializer-independent BLAKE3 identities;
12. persistence revalidates nested effective policy and all current profile contracts;
13. tampered preflight/engine profile or identity state fails closed;
14. no result-bearing artifact instance input exists;
15. no classifier/disposition or downstream authority exists;
16. rustfmt, tests, and strict Clippy pass.

## Required negative checks

Qualification must fail if this crate gains:

- `BoundShadowEvidenceCaseV1` instance input;
- `ShadowDispositionPreflightV1` instance input;
- `LineageBoundShadowDispositionPreflightV1` instance input;
- evidence/interpretation witness instance input;
- interpretation-lineage instance input;
- an evaluate/decide/dispose/issue-disposition API;
- a `ShadowDispositionV1` or equivalent result-bearing output;
- executable threshold comparison/classification logic;
- count-margin, majority-vote, relation-strength, Bayesian/posterior, or winner-take-all arithmetic;
- canonical belief, workspace/GWT, action, or self-improvement promotion authority;
- deserialization that skips full re-registration and current engine-contract recomputation.

## Engine contract boundary

`src/engine_contract.rs` may expose only static/versioned contract information and `shadow_disposition_engine_contract_profile_digest_v1()`.

It must not accept a policy, preflight, witness, case, current time, RNG, callback, or mutable state. It must not emit a result-bearing disposition.

The contract profile must change if any of the following changes:

- primary outcome taxonomy;
- predicate taxonomy;
- decision-rule IDs;
- decision-rule precedence;
- cardinality semantics;
- blocker/contestation/underdetermination semantics;
- reason-trace requirements;
- purity/authority boundary.

## Evidence tier

A green focused workflow qualifies the exact preregistered evaluation-surface **and decision-contract identity** at one commit. It does not qualify a disposition engine; no result-bearing disposition engine belongs in this tranche.
