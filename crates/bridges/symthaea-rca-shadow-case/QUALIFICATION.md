# RCA-003a Shadow Evidence Case Qualification Theorem

RCA-003a is qualified only if all of these remain true:

1. One case binds one exact proposition digest and one exact validated runtime-relevance context.
2. Runtime relevance is recomputed internally for every candidate; callers cannot splice precomputed per-item relevance reports into a case.
3. Every declared relation joins to exactly one case candidate and targets the exact case proposition.
4. Relation labels and strengths remain declared inputs. They are not interpreted as verified truth, posterior probability, or evidence-admission authority.
5. Candidate lineage is reconstructed only from each candidate's committed `lineage_fragment()`; callers cannot supply an arbitrary lineage graph.
6. Multiple fields from one frozen observation retain one shared root and therefore cannot count as independent observations.
7. Pairwise independence comes only from the validated closed lineage graph. `Corroborates` never implies independence by itself.
8. Relation strengths are preserved verbatim and are not summed, averaged, normalized, or converted to probabilities.
9. `ShadowEvidenceCaseV1` has private fields, no public constructor, and no `Deserialize` implementation. Archived JSON is audit material only.
10. RCA-003a exposes no supported/tentative/true/false disposition, posterior, confidence, canonical evidence admission, belief/workspace authority, action authority, or self-improvement promotion.
11. The implementation has no live cognitive-loop or RCA observer-wrapper dependency and no ambient side-effect capability.

The intended theorem is:

```text
candidate evidence
        +
one exact runtime-relevance context
        +
declared proposition relations
        +
reconstructed provenance/independence
        ->
issued shadow evidence case

issued shadow evidence case
        !=
truth disposition
        !=
canonical evidence admission
        !=
belief/workspace authority
        !=
action authority
        !=
self-improvement promotion
```
