# RCA-003b.3a Qualification Contract

Qualification target: `symthaea-rca-effective-disposition-policy`

Status until hosted Actions pass: **IMPLEMENTED, NOT QUALIFIED**.

## Required theorem

```text
registered base policy
        !=
effective profile-bound policy
        !=
case evaluation
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

1. exact wrapped base-policy identity is effective-policy-bearing;
2. exact wrapped base-policy profile digest is effective-policy-bearing;
3. exact current independent interpretation-root-set witness profile is effective-policy-bearing;
4. profile drift requires a new effective policy identity;
5. serializer-independent BLAKE3 effective-policy identity;
6. persistence revalidates the wrapped base policy and witness profile;
7. tampered profile/identity state fails closed;
8. no result-bearing evaluation API exists;
9. rustfmt, tests, and strict Clippy pass.

## Required negative checks

Qualification must fail if this crate gains:

- a bound-case instance input;
- an evidence-set-witness instance input;
- an interpretation-lineage instance input;
- an interpretation-set-witness instance input;
- a `ShadowDispositionV1` or equivalent result-bearing output;
- an evaluate/decide/dispose/issue-disposition function;
- a path to canonical belief, workspace/GWT, external action, or self-improvement promotion;
- deserialization that skips revalidation.

## Evidence tier

A green focused workflow proves this exact commit compiles/tests/lints and freezes the profile-binding/authority boundary. It does not qualify a disposition algorithm; no disposition algorithm belongs in this tranche.
