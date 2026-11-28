# MEL-HEAR-005 — Crossed performer × source-work sampling design v1

**Status:** sampling-design classifier and source-work-bootstrap admission wrapper; crossed inference not implemented  
**Date:** 28 November 2026  
**Parent:** #14026  
**Consumes:** #13947 property-relative reducer and #14019 paired source-work estimator

## 1. Why this slice exists

The paired inference layer in #14019 gives every source work equal weight and resamples source-work clusters. That is a defensible v1 only when source-work clusters are independent for the declared estimand. If the same performer appears across multiple works, the nominal work clusters are not necessarily independent. Repeated recording sessions can create the same problem.

A threshold of 30 works does not repair a crossed sampling design. The design must be classified before source-work-only intervals are interpreted.

## 2. Implemented contract

The sampling manifest binds each exact comparison item and pair to:
- one source-work lineage digest;
- source performance-realization identity for each arm, so a reused performance cannot be misclassified as independent merely because performer/session/take metadata differ;
- CLEAN, Performance-STAY and Musical-FLIP arm membership;
- the reducer-bound performer identity in each arm;
- all declared participant identity commitments for multi-performer arms;
- session, take and performance-realization identity commitments;
- sampling frame, target population, inclusion/exclusion policy and purpose-limited pseudonymization scope;
- the explicit source-work-independence declaration and its written basis.

The comparison wrapper requires exactly one sampling observation for every baseline/candidate item. It binds work, performer, take and performance identity to #13947. Session identity remains a supplied sidecar commitment because the current #13947 reducer has no session field; it cannot be described as a verified join to the audio or comparison case. It joins item, pair and source-work identity, then checks the reducer-bound performer, take and performance identity for all three arms against both model runs. A mismatch rejects before the #14019 comparator is invoked.

## 3. Design classification

- **Work-disjoint clusters:** every required identity is present and no performer, session, take or performance realization spans source works.
- **Nested within work:** multiple performers, sessions, takes or performance realizations occur within a work, but none is reused across works.
- **Crossed performer, session, take or performance:** at least one declared identity on these axes occurs across multiple works.
- **Insufficiently identified:** any required sampling-frame, work, performer, session, take or performance identity is missing.

A multi-performer ensemble recorded within one work is nested, not crossed. Every declared ensemble participant is retained in the membership set; it is not silently assigned to one invented musician or counted as an independent work.

The first two shapes receive only a **conditional** source-work-bootstrap admission when the manifest also declares independent work sampling and supplies a non-empty design basis. Crossed panels are refused by the #14019 source-work-only wrapper. Incomplete panels and panels without an independence declaration remain descriptive-only.

## 4. What the admission flag does not mean

The independence declaration and pseudonymous IDs are supplied study metadata. The session axis is particularly limited: unless an independently qualified provenance record binds session IDs to source recordings, it is a declared grouping variable only. They do not authenticate a person, establish that two pseudonyms identify different people, establish that works are independently sampled, prove representativeness, or establish exchangeability. A hash is a locator/commitment, not identity proof.

This v1 therefore always emits false for performer identity authentication, performer-membership authentication, session identity authentication, take identity authentication, performance identity authentication, source-work identity authentication, sampling-independence authentication and population generalization. It does not implement crossed-cluster intervals. A crossed design is explicitly marked as requiring crossed inference, and its source-work-only comparison is rejected.

## 5. Adversarial controls

The Rust tests exercise:
- performer reused across source works;
- performance realization reused across source works;
- session reused across source works with disjoint performers;
- multiple performers and sessions nested within a work;
- explicit multi-performer arm membership;
- missing performer/session identity;
- missing independence basis;
- item-order invariance;
- duplicate item and duplicate participant rejection;
- source-work and performer substitution at the comparison join;
- rejection of a crossed panel before source-work-only paired inference;
- conditional admission of a complete work-disjoint panel after exact joins.

The checked-in crossed fixture is synthetic. It validates classification plumbing, not the existence of independent real-world performers or an actual crossed-cohort result.

## 6. Not implemented in this phase

No crossed bootstrap, multiway wild bootstrap, CR2 correction, or independent statistical oracle is selected. The next phase must compare candidate estimators against seeded data-generating processes with known dependence, including:
- independent work and performer axes;
- performer crossed across works;
- multiple performers per work;
- repeated sessions nested within performer × work;
- shared sessions across works;
- unbalanced cluster sizes and high-leverage performers;
- sparse property/factor strata and a small cluster axis;
- missing/rekeyed performer and session links.

Before adopting an estimator, freeze its resampling units, paired baseline/candidate indices, multiplicity treatment, PRNG, small-sample behavior and failure semantics. Use a known-answer reference or independently implemented oracle. Report bias, interval coverage, type-I error and interval width over preregistered simulation replications. Simulation does not prove real-cohort exchangeability.

## 7. Qualification boundary

The implementation is a sampling-design classifier and wrapper only. It does not change #13947's musical reducer or #14019's historical v1 estimator. It does not prove target-property labels, audio provenance, consent, rights, performer identity, model execution or population generalization. Exact-head Rust formatting, locked tests, warnings-denied Clippy and the focused fixture replay remain required.
