# Symthaea RCA Provenance-Bound Shadow Case

This crate is RCA-003a.2. It promotes a lower-level structural `ShadowEvidenceCaseV1` into a **provenance-bound, content-addressed shadow case** suitable as the input class for a future RCA-003b disposition experiment.

It still does not decide whether any proposition is true, supported enough to believe, actionable, or eligible for self-improvement promotion.

## Boundary

```text
instrumented runtime candidates
        +
one validated relevance context
        +
BoundEvidenceRelationDeclarationV1 per candidate
        ↓
recompute structural ShadowEvidenceCaseV1
        ↓
verify exact candidate ↔ bound declaration joins
        ↓
canonical complete-case BLAKE3 identity
        ↓
BoundShadowEvidenceCaseV1

        ✕ no truth disposition
        ✕ no confidence/posterior
        ✕ no canonical evidence admission
        ✕ no belief/workspace authority
        ✕ no action authority
        ✕ no self-improvement promotion
```

## Why this is a separate crate

`symthaea-rca-shadow-case` remains a deliberately lower-level structural diagnostic layer. It accepts structurally validated declared relations and reconstructs relevance/lineage/independence.

A future disposition policy should not be able to bypass declarer provenance accidentally. Therefore this crate is a separate dependency boundary. Future RCA-003b code can depend on `symthaea-rca-bound-shadow-case` rather than consuming raw structural cases directly.

## Relation provenance

Every candidate requires exactly one `BoundEvidenceRelationDeclarationV1`.

The bound declaration carries a derived identity over:

- declarer identity/version;
- declaration method;
- immutable provenance artifact;
- complete relation body.

The producer-supplied relation reference remains audit metadata. It does not define case provenance identity.

## Complete case identity

`case_id` is a serializer-independent, domain-separated BLAKE3 commitment over the exact issued case semantics, including:

- binding profile/schema;
- structural-case profile/contract;
- proposition id;
- exact relevance-context commitment;
- case-scope digest;
- lineage graph id;
- ordered candidate/observation/claim identities;
- producer relation references and derived declaration identities;
- declarer provenance;
- relation kinds and strengths;
- current-runtime relevance plus every defect;
- pairwise lineage-independence topology;
- declared relation topology;
- declared current-runtime defeater flag.

The encoding uses explicit semantic tags and length prefixes. JSON serialization, debug formatting, Rust `Hash`, and enum discriminants do not define identity.

## Persistence

`BoundShadowEvidenceCaseV1` has private fields and intentionally no `Deserialize` implementation.

It may serialize for audit. Archived bytes cannot be rehydrated as a trusted bound case. Trusted binding must be recomputed from revalidated candidates, relevance context, and bound declarations.

## Core theorem

```text
raw structural evidence case
        !=
provenance-bound content-addressed case
```

and:

```text
provenance-bound content-addressed case
        != truth
        != belief
        != evidence admission
        != action authority
        != self-improvement authority
```
