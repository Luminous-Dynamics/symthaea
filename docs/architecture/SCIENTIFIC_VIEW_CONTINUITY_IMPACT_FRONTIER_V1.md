# Scientific View Continuity Impact Frontier v1

Status: hardening companion to `SCIENTIFIC_VIEW_CURRENTNESS_CONTINUITY_V1.md`.

## 1. Purpose

A currentness dependency projection derived at source head `H17` contains the dependencies known to the replayed scientific disposition at `H17`.

That is not, by itself, enough to prove continuity to a later head `H18`.

A later state may introduce a **new dependency that did not exist at H17**:

```text
H17: proposition P has no admitted evidence from source S2
H18: source/discovery update introduces candidate E9 from S2
```

If continuity checks only whether existing H17 dependencies changed, E9 can be missed because it was not yet part of the old direct dependency set.

Freeze:

```text
old direct dependency set unchanged
    != no new scientifically material dependency was introduced
```

## 2. Dependency projection must include dependency-generating surfaces

The currentness projection therefore needs two classes of material dependency:

```text
A. instantiated dependencies
   objects already present in the reconstruction

B. dependency-generating surfaces
   qualified registries/rules/scopes whose advancement can introduce,
   remove, reinterpret, or reconnect scientifically material objects
```

Examples of dependency-generating surfaces include:

- evidence discovery/source registries;
- query/discovery profiles;
- proposition equivalence/namespace registries;
- lifecycle/adjudication registries;
- dependency-graph construction profiles;
- compatibility/triangulation registries;
- policy registries;
- evaluator registries;
- learned-prior/model registries when scientifically material;
- measurement/admission schemas;
- any registry capable of introducing new candidate evidence or defeaters.

The projection is incomplete if it binds existing objects but omits a surface that can create a new relevant object.

## 3. Impact frontier

Define conceptually an owner-derived:

```text
ScientificCurrentnessImpactFrontierV1
```

for the exact proposition/use/reconstruction.

The frontier identifies the semantic scopes in the destination-head delta that must be inspected because they can affect the scientific disposition even when no old direct dependency identity changed.

Conceptually it may bind:

```text
proposition / ontology scope
scientific-use scope
evidence discovery/query scope
source namespaces
measurement / admission scope
lifecycle/adjudication scope
dependency-construction scope
compatibility/triangulation scope
policy/evaluator scope
closed-world absence scopes
```

This is not a search hint. It is scientific currentness material.

## 4. Source-head projection cannot self-certify completeness for destination state

The H17 reconstruction cannot know every object that may appear at H18.

Therefore continuity requires both:

```text
H17 owner-derived dependency projection
+ H17->H18 owner-qualified delta evaluated against the impact frontier
```

The destination-side delta classifier must be owned/qualified by the scientific-view state boundary.

A caller may not submit:

```text
new objects are irrelevant = true
```

or a filtered delta that omits newly introduced candidates.

## 5. Newly introduced evidence

Any new candidate evidence inside the exact discovery/admission scope is material by default until classified through the normal owner admission path.

Freeze:

```text
new candidate exists
    -> continuity cannot be granted merely because it is not yet admitted
```

The candidate may ultimately be:

```text
Excluded(reason)
DuplicateAliasOf(...)
OutOfScope under qualified rule
```

and therefore have no disposition effect, but that conclusion must come from qualified destination-state processing.

## 6. New defeaters and lifecycle events

The same rule applies to new scientific events that target old evidence or inference paths.

A new:

```text
retraction
correction
provenance failure
undercutting defeater
scope defeater
dependency contamination edge
```

may be scientifically material even though it is a brand-new object absent from H17's direct object set.

The impact frontier must therefore include registries/schemas capable of creating such edges/events.

## 7. Ontology and proposition-equivalence drift

New semantic relationships can also introduce material change without editing the original proposition object.

For example:

```text
H17: proposition Q considered unrelated to P
H18: qualified semantic-equivalence/transform receipt relates Q to P
```

Evidence already attached to Q may now become relevant to P's current scientific view.

Therefore proposition/ontology/equivalence registries are dependency-generating surfaces whenever the requested view permits such transfer.

## 8. Dependency graph expansion

SCI-006 dependency topology may discover previously hidden shared lineage.

Example:

```text
H17: studies A and B appear declared-disjoint
H18: new provenance receipt shows shared transformation artifact T
```

Neither A nor B changed, yet the replication/triangulation interpretation did.

Currentness must therefore treat dependency-graph generation and dependency-edge discovery as material surfaces, not only direct evidence-object versions.

## 9. Policy reachability expansion

A policy update may introduce a new predicate or make previously auxiliary scientific state disposition-relevant.

Even if the old evaluated predicates remain unchanged, the new current policy may reach additional state.

Therefore material continuity across policy versions requires more than rule-output equivalence on the old predicate vector.

It requires a qualified policy migration/equivalence theorem covering the new policy's dependency reachability.

## 10. Closed-world absence receives the strictest default

For any predicate whose truth depends on absence, the impact frontier must include every surface capable of introducing a witness that would falsify that absence within the qualified scope.

Examples:

```text
no-qualified-opposition-exists
no-active-defeater-exists
no-relevant-new-evidence
all-candidates-accounted-for
```

If any such surface advances and destination-side incremental closure cannot be qualified, continuity fails closed.

## 11. Incremental qualification must equal full qualification on the material slice

A future incremental continuity algorithm is acceptable only if its theorem is equivalent to a full currentness reconstruction for the exact material slice it claims to preserve.

Freeze:

```text
faster delta algorithm
    != weaker scientific semantics
```

The optimization may inspect fewer objects because a qualified impact frontier proves other objects unreachable, not because it skips ordinary admission/reason/predicate rules.

## 12. No Bloom-filter/probabilistic omission authority by default

Probabilistic membership/index structures may accelerate discovery, but false negatives are unacceptable for currentness unless an explicit profile accounts for them without granting false continuity.

A Bloom filter or learned relevance classifier may help identify candidates, but cannot by itself prove:

```text
no material new dependency exists
```

Any probabilistic pruning used in the privileged continuity path needs explicit no-false-negative semantics or a fail-closed uncertainty result.

## 13. Incremental result identity

A qualified destination-delta assessment should retain an identity covering at least:

```text
source head
destination head
impact-frontier identity
source dependency-projection identity
raw owner delta identity
destination candidate classification receipts
new/changed dependency identities
incremental admission/reason/predicate receipts
continuity profile + implementation + execution lineage
```

The continuity witness must not retain only a Boolean `unchanged`.

## 14. Qualification vectors

A future impact-frontier qualification should prove at minimum:

1. new candidate evidence absent from H17 direct dependencies is still detected;
2. caller-filtered destination delta cannot hide that candidate;
3. candidate exclusion requires the normal qualified admission path;
4. new undercutting defeater invalidates continuity even when old evidence objects are byte-identical;
5. new proposition-equivalence receipt can make previously unrelated evidence material;
6. new dependency edge can invalidate replication diversity without modifying either study;
7. policy reachability expansion cannot be certified from old predicate equality alone;
8. omission of a dependency-generating registry makes projection incomplete;
9. probabilistic relevance pruning cannot mint negative currentness without a no-false-negative theorem;
10. incremental currentness produces the same scientific result as full reconstruction on the qualified material slice.

## 15. Exit gate

The continuity architecture is safe only when it can state:

> Material continuity is proven not merely by showing that the old dependencies did not change, but also by proving that the destination-head transition introduced no new scientifically material dependency reachable through any qualified dependency-generating surface for the exact proposition and requested use.

This remains a currentness optimization theorem, not a truth theorem.
