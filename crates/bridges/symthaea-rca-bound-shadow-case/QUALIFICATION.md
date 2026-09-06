# RCA Bound Shadow Case Qualification Theorem

RCA-003a.2 is qualified only if all of the following remain true:

1. Every candidate has exactly one `BoundEvidenceRelationDeclarationV1`.
2. Declaration joins are by exact candidate id; declarations outside the case fail closed.
3. The structural `ShadowEvidenceCaseV1` is recomputed internally from candidates + one validated relevance context + declaration-backed relations.
4. Caller ordering of candidates/declarations does not change canonical case identity.
5. `case_id` binds the binding profile/schema, structural profile, proposition, relevance context, case scope, lineage graph, candidate/claim/observation identities, declaration identities/provenance, relation kind/strength, relevance defects, independence topology, declared relation topology, and defeater flag.
6. Semantic variants use explicit stable tags; serializer output, debug formatting, Rust `Hash`, and enum discriminants do not define identity.
7. `BoundShadowEvidenceCaseV1` has private fields and no `Deserialize` implementation.
8. Archive bytes are audit material only; trusted binding must be recomputed.
9. This layer exposes no supported/true/confidence/posterior/admit/authorize/promote API.
10. It has no live cognitive-loop, workspace, action, or recursive-improvement capability.

The intended theorem is:

```text
coherent structural case
        +
relation declaration provenance
        +
canonical complete-case identity
        ->
BoundShadowEvidenceCaseV1

BoundShadowEvidenceCaseV1
        != truth disposition
        != canonical evidence admission
        != belief/workspace authority
        != action authority
        != self-improvement promotion
```

A future RCA-003b disposition experiment should accept this provenance-bound artifact class, not a raw structural `ShadowEvidenceCaseV1`.
