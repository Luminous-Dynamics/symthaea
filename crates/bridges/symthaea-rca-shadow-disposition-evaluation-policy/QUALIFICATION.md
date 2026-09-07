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
6. engine-contract digest binds normative contract, class tags, predicate tags, rule precedence, exact rule → class mapping, and exact reason-trace schema;
7. every frozen rule maps to exactly one frozen primary class;
8. the frozen decision lattice is total over all 72 predicate states admissible under `S => TS` and `O => TO`;
9. qualified defeater precedes contestation and unilateral outcomes in the frozen rule table;
10. `QualifiedContestation` requires `CS && CO && TS && TO`, so contested-side topology cannot bypass tentative qualification on either side;
11. bilateral qualifying disagreement precedes unilateral support/opposition outcomes;
12. no vote/count-margin/relation-strength/posterior tie-breaker enters the engine contract;
13. exact identity, slot, predicate, rule, and primary-class trace fields are profile-bearing;
14. drift in preflight, decision semantics, rule meaning, or trace schema requires a new evaluation-policy identity;
15. domain-separated serializer-independent BLAKE3 identities;
16. persistence revalidates nested effective policy and all current profile contracts;
17. tampered preflight/engine profile or identity state fails closed;
18. no result-bearing artifact instance input exists;
19. no production classifier/disposition or downstream authority exists;
20. automatic workflow runs supersede stale copies while manual historical `workflow_dispatch` runs remain independent;
21. rustfmt, tests, and strict Clippy pass.

## Required negative checks

Qualification must fail if this crate gains:

- `BoundShadowEvidenceCaseV1` instance input;
- `ShadowDispositionPreflightV1` instance input;
- `LineageBoundShadowDispositionPreflightV1` instance input;
- evidence/interpretation witness instance input;
- interpretation-lineage instance input;
- a production evaluate/decide/dispose/classify/issue-disposition API;
- a `ShadowDispositionV1` or equivalent result-bearing output;
- executable threshold comparison/classification logic outside `#[cfg(test)]` contract proofs;
- a `Contested` rule that can succeed without both tentative-side predicates;
- count-margin, majority-vote, relation-strength, Bayesian/posterior, or winner-take-all arithmetic;
- canonical belief, workspace/GWT, action, or self-improvement promotion authority;
- deserialization that skips full re-registration and current engine-contract recomputation.

## Engine contract boundary

`src/engine_contract.rs` may expose only static/versioned contract information and `shadow_disposition_engine_contract_profile_digest_v1()`.

It must not accept a policy, preflight, witness, case, current time, RNG, callback, or mutable state. It must not emit a result-bearing disposition.

A `#[cfg(test)]` classifier may exist solely to prove the frozen lattice is exhaustive/total. No equivalent classifier may exist in production code in this tranche.

The contract profile must change if any of the following changes:

- primary outcome taxonomy;
- predicate taxonomy;
- decision-rule IDs;
- decision-rule precedence;
- decision-rule → primary-class mapping;
- cardinality semantics;
- blocker/contestation/underdetermination semantics;
- identity-trace fields;
- support/opposition/defeater slot-trace fields;
- decision-trace fields;
- purity/authority boundary.

## Evidence tier

A green focused workflow qualifies the exact preregistered evaluation-surface **and decision/audit-contract identity** at one commit. It does not qualify a disposition engine; no result-bearing disposition engine belongs in this tranche.

A queued, pending, superseded, or cancelled run is infrastructure state only and must never be interpreted as PASS/FAIL evidence.
