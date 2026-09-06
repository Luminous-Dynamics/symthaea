# RCA-003b.3d Qualification Contract

Qualification target: `symthaea-rca-shadow-disposition-evaluation-policy`

Status until hosted Actions pass: **IMPLEMENTED, NOT QUALIFIED**.

## Required theorem

```text
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
5. drift in either preflight contract requires a new evaluation-policy identity;
6. domain-separated serializer-independent BLAKE3 identity;
7. persistence revalidates nested effective policy and current preflight profiles;
8. tampered profile/identity state fails closed;
9. no result-bearing artifact instance input exists;
10. no disposition or downstream authority exists;
11. rustfmt, tests, and strict Clippy pass.

## Required negative checks

Qualification must fail if this crate gains:

- `BoundShadowEvidenceCaseV1` instance input;
- `ShadowDispositionPreflightV1` instance input;
- `LineageBoundShadowDispositionPreflightV1` instance input;
- evidence/interpretation witness instance input;
- interpretation-lineage instance input;
- an evaluate/decide/dispose/issue-disposition API;
- a `ShadowDispositionV1` or equivalent result-bearing output;
- threshold comparison or relation-strength arithmetic;
- canonical belief, workspace/GWT, action, or self-improvement promotion authority;
- deserialization that skips full re-registration.

## Evidence tier

A green focused workflow qualifies the exact preregistered evaluation-surface identity and authority boundary at one commit. It does not qualify a disposition engine; no disposition engine belongs in this tranche.
