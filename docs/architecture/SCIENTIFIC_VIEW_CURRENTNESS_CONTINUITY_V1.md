# Scientific View Currentness Continuity v1

Status: architecture candidate. This document refines the SCI-014 current-within-view boundary defined by #808.

## 1. Purpose

#808 correctly separates historical replay correctness from present-state currentness and requires currentness to bind to an exact namespace-scoped scientific-view state rather than caller-supplied matching generations.

The next problem is scale.

If every disposition witness is considered stale whenever the entire scientific-view head changes for any reason, then an unrelated update to another proposition, source, registry partition, or policy domain can invalidate every live disposition in the Atlas.

That is safe but unnecessarily coarse.

This contract therefore freezes two distinct currentness modes:

```text
ExactHeadCurrentnessV1
    = simple, conservative, first implementation

QualifiedMaterialContinuityV1
    = optional later optimization requiring its own proof
```

The optimization must never weaken the simple theorem by becoming an implicit fallback.

## 2. Core theorem

Preserve:

```text
replay-valid disposition
    != exact-head-current disposition
    != material-continuity-current disposition
    != truth
    != global consensus
```

and:

```text
view head advanced
    != disposition scientifically changed
```

but also:

```text
caller asserts nothing relevant changed
    != qualified continuity
```

and:

```text
same primary disposition after two heads
    != proof that the intervening state was scientifically irrelevant
```

## 3. V1 default: exact view-head equality

The first executable currentness theorem should remain deliberately strict.

Conceptually:

```text
ReplayVerifiedScientificDispositionV1
    + QualifiedScientificViewStateV1 { head = H17 }
        -> CurrentWithinScientificViewDispositionV1 { head = H17 }
```

A live point-of-use check against a view currently at `H18` fails currentness unless the consumer explicitly operates against retained immutable snapshot `H17`.

Freeze:

```text
witness bound to H17
+ live view at H18
    -> StaleForLiveView
```

This does not invalidate the historical/replay theorem at H17.

## 4. Why exact-head equality is not the final scalability model

A large scientific view may contain millions of unrelated propositions and many independent registries.

For example:

```text
H17
    -> update astronomy proposition Z
    -> H18
```

should not necessarily force a full replay of a disposition about an unrelated economic proposition P if none of P's scientifically material dependencies changed.

However, deciding that an update was irrelevant is itself a scientific/currentness assertion and therefore requires qualification.

The system must not infer irrelevance from:

```text
different proposition label
no obvious matching ID
same summary disposition
caller-supplied changed-path list
cache metadata
wall-clock freshness
```

## 5. Material currentness dependency projection

A future qualified scientific reconstruction should emit or allow the currentness owner to derive an exact immutable dependency projection for the requested proposition/use.

Conceptually:

```text
ScientificCurrentnessDependencyProjectionV1 {
    view_namespace_id,
    proposition_id,
    requested_scientific_use,
    reconstruction_id,

    proposition_registry_dependencies,
    evidence_view / discovery dependencies,
    lifecycle dependencies,
    argument / adjudication dependencies,
    dependency-graph dependencies,
    compatibility / triangulation dependencies,
    disposition-policy dependencies,
    predicate-derivation dependencies,
    evaluator dependencies,
    information-cutoff semantics,
    negative-by-absence closure dependencies,
}
```

The exact shape is deferred.

The projection must be owner-derived from qualified reconstruction material, not caller-selected after the fact.

## 6. Projection completeness is an authority boundary

A projection can only safely support material continuity if it contains every state component whose change could alter:

- evidence admission;
- candidate accounting;
- reason topology;
- a policy predicate;
- policy selection;
- evaluator semantics;
- primary disposition;
- requested-use eligibility.

Freeze:

```text
incomplete dependency projection
    -> no continuity credit
```

Unknown dependency completeness must not be interpreted as unchanged.

The first implementation should therefore prefer exact-head equality until dependency projection itself has an independently qualified construction theorem.

## 7. Registry-owned ancestry and delta proof

A future continuity path may conceptually consume:

```text
old qualified view head H17
new qualified live view head H18
owner-derived material dependency projection D(P,use)
qualified H17 -> H18 ancestry/delta evidence
```

and produce:

```text
QualifiedMaterialContinuityV1
```

only if the view-state owner proves that every change between H17 and H18 is either:

1. outside the complete currentness dependency projection; or
2. covered by an explicitly qualified compatibility theorem proving no scientifically material change for the requested proposition/use.

The caller must not supply the authoritative delta.

## 8. No path-list shortcut

A source-control-style changed-file list is not scientific continuity evidence.

For example:

```text
changed paths do not mention proposition P
```

does not prove that:

- a shared dataset used by P did not change;
- a lifecycle event did not affect P's evidence;
- a dependency graph update did not connect P to another lineage;
- a policy profile did not change globally;
- a source registry generation did not reveal a new relevant candidate;
- a learned prior or evaluator artifact did not change.

Continuity must operate over semantic/material dependency identities, not display paths.

## 9. Negative-by-absence predicates are highly currentness-sensitive

Predicates such as:

```text
no-qualified-opposition-exists
no-active-defeater-exists
all-candidates-accounted-for
```

are especially vulnerable to stale closure.

Any advancement of a source/discovery registry that could introduce a candidate inside the exact closed-world scope invalidates the previous negative-by-absence receipt unless a qualified incremental-discovery theorem proves no newly admitted candidate matches the predicate scope.

Freeze:

```text
old closed-world absence receipt
+ source/discovery universe advanced
    != current absence proof
```

by default.

The safe V1 behavior is to stale the disposition and require reconstruction/replay.

## 10. Positive existential evidence is not automatically append-safe

It may appear safe to preserve a predicate such as:

```text
qualified-support-exists = Satisfied
```

when new evidence is appended, because the old support still exists.

But a new contribution may simultaneously introduce:

- stronger opposition;
- an undercutting defeater;
- a retraction;
- a dependency contamination finding;
- a target-compatibility change;
- a policy predicate that outranks support.

Therefore append-only source growth is not automatically disposition-preserving.

Continuity must evaluate the complete currentness dependency projection and policy-visible consequences, not individual predicate monotonicity in isolation.

## 11. Lifecycle and adjudication changes are material by default

If an admitted contribution's lifecycle or external adjudication state changes, the old currentness witness becomes stale by default.

Examples:

```text
active -> corrected
active -> superseded
active -> source-retracted
no external finding -> provenance failure finding
no defeater -> qualified undercutting defeater
```

A future compatibility theorem may establish that a specific change is immaterial to a requested use, but no generic lifecycle event should be assumed harmless.

## 12. Policy and evaluator continuity

A newer policy/evaluator version does not automatically invalidate historical replay, but it does affect live-view currentness when that view declares the new profile authoritative for the requested use.

Freeze:

```text
old policy still replayable
    != old policy current for live view
```

and:

```text
new evaluator semantically equivalent by claim
    != qualified evaluator continuity
```

Any cross-profile continuity requires an explicit qualified equivalence/migration receipt.

## 13. Forks, competing heads, and federation

A scientific view may fork or multiple qualified views may legitimately disagree.

Do not define currentness as:

```text
highest generation number wins
latest timestamp wins
majority of registries wins
```

A currentness witness is always scoped to one exact view namespace and one exact view lineage/profile.

If the registered view lineage forks and no owner policy resolves the fork, live currentness is unavailable/contested rather than arbitrarily selecting a branch.

Cross-view comparison remains a separate Theory Atlas operation.

## 14. Reorganizations and rollback

If a view owner permits rollback/reorganization, generation monotonicity alone is insufficient.

Currentness should bind to immutable head identity and verified ancestry, not only integer generation.

A witness from H17 does not become current again merely because a later branch reuses generation 17 or equivalent metadata.

Head identities must be non-ambiguous within the exact view profile.

## 15. TOCTOU and point-of-use fencing

Even a successful continuity proof from H17 to H18 can become stale when H19 appears.

Consequential scientific consumers should use one of:

```text
operate against immutable H18 snapshot
atomic compare-and-use on live H18
recheck exact live head immediately before use
```

The continuity witness should retain both source and destination heads:

```text
H17 -> H18
```

and must not be interpreted as continuity to arbitrary future descendants.

## 16. Continuity proof composition

Continuity proofs should compose only under explicit ancestry and profile rules.

Conceptually:

```text
QualifiedContinuity(H17, H18)
+ QualifiedContinuity(H18, H19)
    -> maybe QualifiedContinuity(H17, H19)
```

only if:

- the same view/profile semantics apply;
- the material dependency projection identity remains valid;
- no intermediate unresolved/fork state is hidden;
- composition is allowed by the qualified continuity profile.

Do not infer transitivity merely because head identifiers form a chain.

## 17. Continuity result space

Prefer an explicit result rather than a Boolean:

```text
ExactHeadMatch
QualifiedMaterialContinuity
StaleMaterialDependencyChanged
ProjectionIncomplete
ContinuityEvidenceUnavailable
ViewForkUnresolved
ProfileMismatch
AncestryUnverified
```

The exact enum is deferred.

Unknown/unavailable states must not become currentness.

## 18. Positive capability separation

A future material-continuity object should itself be a verifier-owned capability with private construction and no serde reconstruction path.

Conceptually:

```text
QualifiedMaterialContinuityV1
```

It should bind:

- exact view namespace/profile;
- source head;
- destination head;
- currentness dependency projection identity;
- ancestry/delta proof identity;
- continuity profile identity;
- verification execution lineage.

It does not itself grant scientific disposition currentness until composed with the exact replay-positive scientific disposition whose reconstruction generated the dependency projection.

## 19. Safe implementation order

Do not implement material continuity first.

Preferred order:

```text
1. #825 lower evaluation replay qualifies
2. #839/#848 full scientific reconstruction + replay qualifies
3. exact-head CurrentWithinScientificViewDispositionV1
4. exact-head TOCTOU / point-of-use tests
5. owner-derived currentness dependency projection
6. qualified view ancestry/delta evidence
7. QualifiedMaterialContinuityV1
8. continuity-aware currentness optimization
```

This prevents a performance optimization from defining scientific authority semantics.

## 20. Qualification vectors

A future continuity qualification should prove at minimum:

1. unrelated whole-view head advancement fails simple exact-head currentness;
2. caller assertion of irrelevance cannot mint continuity;
3. caller-supplied changed-ID/path list cannot mint continuity;
4. incomplete dependency projection fails closed;
5. omitted shared dataset dependency invalidates continuity;
6. omitted policy/evaluator dependency invalidates continuity;
7. source/discovery advancement invalidates old absence receipts by default;
8. new opposing evidence prevents continuity even if old support still exists;
9. lifecycle/adjudication change invalidates old currentness by default;
10. view A continuity cannot satisfy view B;
11. forked ancestry without a qualified branch decision fails currentness;
12. generation reuse/rollback cannot resurrect an old witness by metadata equality;
13. H17->H18 continuity does not imply H19 currentness;
14. stale/unknown continuity never becomes scientific opposition;
15. continuity never grants truth, recommendation, governance, resource, medical, or effect authority.

## 21. Exit gate

This architecture is complete when the repository can truthfully state:

> Currentness is exact-head-scoped by default. A replay-verified scientific disposition may remain current across a newer view head only when an owner-qualified continuity verifier proves that the head transition preserved every scientifically material dependency for that exact proposition and requested use. Unproven irrelevance is treated as staleness, not as continuity.

This is an optimization theorem over currentness, not a truth theorem.
