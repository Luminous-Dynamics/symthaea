# RCA Current-Runtime Relevance Qualification Theorem

RCA runtime relevance is qualified only if all of the following remain true:

1. Candidate source-generation, execution-lineage, adapter-profile, and adapter-contract identities are compared exactly.
2. No cryptographic identity is truncated or reinterpreted as a numeric generation.
3. Cycle indices are compared only when execution lineage matches exactly.
4. `max_cycle_lag` is supplied by an explicit validated use context and participates in its commitment.
5. The exact relevance-policy contract participates in the context commitment and every issued assessment.
6. `RuntimeRelevanceAssessmentV1` has private fields, no public constructor, and no `Deserialize` implementation.
7. Archived assessment bytes are audit material only; trusted relevance must be recomputed from candidate + validated context.
8. Relevance does not imply canonical evidence authority, proposition support, belief/workspace admission, action authority, or self-improvement promotion.
9. The crate has no live cognitive-loop or observer-wrapper dependency and no ambient clock, RNG, filesystem, networking, process, channel, or lock capability.

The intended theorem is:

```text
historically authentic runtime observation
        +
exact current execution identity
        +
explicit bounded lag policy
        ->
issued current-runtime relevance assessment

issued current-runtime relevance assessment
        !=
canonical evidence admission
        !=
proposition support
        !=
belief or workspace authority
        !=
action authority
        !=
self-improvement promotion
```
