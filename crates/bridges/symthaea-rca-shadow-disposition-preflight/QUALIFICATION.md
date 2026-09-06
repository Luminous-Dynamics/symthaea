# RCA-003b.3b Qualification Contract

Qualification target: `symthaea-rca-shadow-disposition-preflight`

Status until hosted Actions pass: **IMPLEMENTED, NOT QUALIFIED**.

## Required theorem

```text
individually valid artifacts
        !=
valid joint evaluation input
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

1. exact proposition equality across case, policy, eligibility, lineage, and supplied interpretation witnesses;
2. exact case-declaration set equals the supplied current eligible-declaration set;
3. exact interpretation-lineage declaration/eligibility mapping equals the supplied eligibility set;
4. all eligible declarations share the exact lineage eligibility-context commitment;
5. each supplied evidence witness uses the exact policy-bound profile;
6. every selected evidence item is an exact current case item on the correct semantic side;
7. V1 evidence-witness root set is exactly the one case observation root;
8. each supplied interpretation witness has a corresponding same-slot evidence witness;
9. each interpretation witness root set exactly equals the roots interpreting that same slot's evidence items;
10. every interpretation witness binds the exact proposition, profile, and interpretation-lineage id;
11. actual case item count and interpretation root-pair count obey policy ceilings;
12. actual registered experiment contract verifies integrity and exactly matches the policy preregistration digest;
13. complete serializer-independent BLAKE3 preflight identity;
14. issued preflight is private-field and non-deserializable;
15. zero witness slots remain a valid coherent preflight input;
16. rustfmt, tests, and strict Clippy pass.

## Required negative checks

Qualification must fail if the crate gains:

- threshold comparison against `min_pairwise_independent_*` fields;
- a `ShadowDispositionV1` or equivalent result-bearing output;
- evaluate/decide/dispose/issue-disposition logic;
- relation-strength arithmetic, voting, posterior, or Bayesian semantics;
- acceptance of an interpretation witness without its same-slot evidence witness;
- acceptance of interpretation roots from evidence outside the same slot witness;
- silent multi-root evidence generalization in the V1 preflight profile;
- oversized actual case/interpretation topology relative to registered policy ceilings;
- experiment digest matching without requiring the actual registered experiment artifact and integrity check;
- a `Deserialize` path for the issued preflight;
- live cognition, workspace/GWT, external-action, or self-improvement promotion dependencies.

## Evidence tier

A green focused workflow proves this exact preflight implementation compiles/tests/lints and that its static authority/join contract remains present. It does **not** prove a disposition algorithm, because no disposition algorithm belongs to this tranche.
