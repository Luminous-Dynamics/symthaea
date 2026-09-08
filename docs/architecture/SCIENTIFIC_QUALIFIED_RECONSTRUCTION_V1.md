# Scientific Qualified Reconstruction v1

Status: architecture candidate. This document specifies the missing SCI-014A seam between deterministic disposition-evaluation replay and a full replay-verified scientific disposition.

## 1. Purpose

The lower replay seam may prove that a persisted disposition evaluation is internally reproducible from explicit inputs and an explicit policy. That is valuable, but insufficient for scientific replay when the supplied predicate facts and reason identifiers were not themselves reconstructed from qualified scientific state.

Freeze the core non-equivalence:

```text
replay-verified disposition evaluation
    != replay-verified scientific disposition
```

and:

```text
caller supplies matching predicates/reasons
    != predicates/reasons were scientifically qualified
```

The full SCI-014 replay witness may exist only after an owner verifier reconstructs the exact scientifically material inputs from qualified upstream state and then successfully passes the independent disposition-evaluation replay boundary.

## 2. Authority ladder

```text
ordinary persisted / caller-shaped data
        ↓
qualified upstream scientific state
        ↓
owner scientific reconstruction
        ↓
QualifiedScientificReconstructionV1
        ↓
deterministic disposition-evaluation replay
        ↓
ReplayVerifiedScientificDispositionV1
        ↓
future qualified scientific-view state
        ↓
CurrentWithinScientificViewDispositionV1
```

None of these objects imply scientific truth, consensus, recommendation, governance authority, medical authority, resource authority, or effect authority.

## 3. Required upstream state

The reconstruction owner must consume private-construction qualified objects, or exact owner-resolved immutable snapshots, for the scientifically material state used by the disposition policy.

Conceptually this includes:

```text
QualifiedPropositionTarget
QualifiedClosedEvidenceView
QualifiedLifecycleSnapshot
QualifiedExternalArgumentState
QualifiedDependencyGraph
QualifiedTargetCompatibilityState
QualifiedTriangulationState
QualifiedDispositionPolicy
QualifiedPredicateDerivationProfile
QualifiedEvaluatorExecutionLineage
QualifiedInformationCutoff
```

Exact type names should follow the actual implementations. Do not create parallel authority types when qualified equivalents already exist.

## 4. Closed evidence-view accounting

A reconstruction is not complete merely because every supplied contribution was processed.

The owner must begin from one exact closed evidence-view snapshot with explicit scope, discovery/query policy, registry/source generations, scientific use, and information cutoff.

Every candidate contribution in that view must terminate in exactly one auditable state:

```text
Admitted
Excluded(reason)
Unresolved(reason)
DuplicateAliasOf(contribution)
```

Require the accounting invariant:

```text
N(candidate)
  = N(admitted)
  + N(excluded)
  + N(unresolved)
  + N(explicit aliases)
```

This invariant proves accounting completeness relative to the declared evidence-view snapshot. It grants no scientific weight and does not claim that the declared source scope is globally complete.

No contribution may disappear merely because it would make a downstream predicate inconvenient.

## 5. Proof-carrying predicate derivation

A raw Boolean or enum value is not sufficient scientific input to a disposition policy.

Freeze:

```text
predicate value
    != qualified predicate derivation
```

Every predicate consumed by the policy must have an immutable predicate specification defining:

```text
predicate identity
semantic meaning
input domain / scientific-use scope
allowed source state
quantification scope
closed-world requirement, if any
unknown / unresolved semantics
not-applicable semantics
blocked semantics
derivation algorithm/profile identity
derivation implementation artifact identity
execution lineage identity
```

The owner produces a `PredicateDerivationReceipt` conceptually containing:

```text
predicate specification identity
exact upstream snapshot identities
exact reason-topology root(s) consulted
exact admitted/excluded/unresolved contribution identities consulted
result value
complete derivation trace / material commitment
implementation + execution lineage
```

Only predicate values produced by the owner derivation path may enter the lower disposition-evaluation replay boundary.

A caller-provided value such as:

```text
support-exists = Satisfied
```

must never be upgradable merely because it equals the owner-derived value.

## 6. Closed-world vs open-world predicates

Predicates that assert existence and predicates that assert absence have different epistemic requirements.

### 6.1 Existential predicates

A predicate such as:

```text
qualified-support-exists
```

may be satisfied by one exact qualified support receipt, subject to the predicate specification.

### 6.2 Negative-by-absence predicates

A predicate such as:

```text
no-qualified-opposition-exists
all-candidate-evidence-accounted-for
no-active-defeater-exists
```

must not be derived from failure to find a matching record in an open or incomplete search.

Require an exact closed universe or a policy-defined bounded universe:

```text
absence claim
    + closed evidence-view identity
    + complete candidate accounting
    + exact cutoff/generations
        -> bounded negative result
```

Without closure, the appropriate result is `Unknown`, `Unresolved`, or `Blocked`, according to the registered predicate profile.

Freeze:

```text
not found
    != does not exist
```

and:

```text
incomplete search
    != negative scientific evidence
```

## 7. Three-valued and partial knowledge discipline

The reconstruction layer must preserve epistemic partiality.

At minimum it must distinguish the policy-visible states already required by the disposition layer, such as:

```text
Satisfied
NotSatisfied
Unknown
NotApplicable
Blocked
```

No defaulting rule may coerce missing, unresolved, unavailable, or untestable state into `NotSatisfied` unless that coercion is explicitly part of a qualified predicate profile whose scientific semantics justify it.

In particular:

```text
unknown opposition
    != no opposition

untestable assumption
    != assumption satisfied

unavailable source
    != negative result
```

## 8. Reason topology reconstruction

The owner must reconstruct the complete reason topology used by every derived predicate and final scientific disposition.

A set of reason-node identities is insufficient. The topology itself is material.

Require an exact reason-topology root/snapshot binding that preserves, as applicable:

```text
support edges
opposition edges
rebutting defeaters
undercutting defeaters
scope defeaters
falsifier relations
lifecycle/adjudication relations
dependency relations
target compatibility / transform relations
triangulation relations
unresolved conflicts
```

Freeze:

```text
same reason node set
    + different edge topology
        != same scientific reconstruction
```

The reconstruction receipt must bind the exact graph/root used by predicate derivation and by the later disposition evaluation.

## 9. Predicate/reason correspondence

Every policy predicate must be explainable by the reason topology and qualified upstream state.

The owner should be able to produce a correspondence table conceptually like:

```text
Predicate P
    <- derivation receipt D
    <- reason subgraph R
    <- exact qualified upstream objects U1..Un
```

A predicate whose result cannot be traced to its declared inputs fails reconstruction.

Likewise, reason nodes that are declared scientifically material but have no path into any predicate, unresolved issue, or disposition reason must remain visible rather than silently discarded as “unused”.

## 10. No circular scientific self-authorization

The final disposition may not be an input that helps derive the predicates that select that same disposition.

Disallow cycles of the form:

```text
stored disposition
    -> predicate derivation
    -> policy evaluation
    -> same stored disposition
```

or:

```text
model says claim is supported
    -> support predicate
    -> SupportedWithinScope
```

unless the model output first enters as ordinary evidence and passes the same qualified admission/argument/reason pipeline as other evidence.

The scientific disposition must be a projection of independently reconstructed scientific state, not a self-confirming feature.

## 11. Exact correspondence with the lower evaluation seam

The bridge into the deterministic disposition-evaluation replay layer is owner-created.

Require exact correspondence for every material field, including conceptually:

```text
reconstructed proposition identity
    == evaluation proposition identity

reconstructed scientific-use identity
    == evaluation scientific-use identity

reconstructed evidence-view identity/snapshot
    == evaluation context evidence-view identity/snapshot

reconstructed lifecycle generation
    == evaluation context lifecycle generation

reconstructed argument/adjudication generation
    == evaluation context argument generation

reconstructed dependency generation
    == evaluation context dependency generation

reconstructed compatibility/triangulation state
    == evaluation context compatibility/triangulation state

reconstructed reason-topology root
    == evaluation context reason-topology root

reconstructed predicate-derivation profile/artifact/execution lineage
    == evaluation context predicate-derivation lineage

owner-derived predicate facts
    == evaluation predicate facts

qualified policy profile/artifact
    == evaluation policy profile/artifact

reconstructed information cutoff
    == evaluation information cutoff
```

No public API should accept a valid lower evaluation witness plus arbitrary matching strings and upgrade it to a full scientific replay witness.

## 12. Reconstruction identity

The reconstructed scientific material should itself have one exact immutable identity or material commitment.

Conceptually:

```text
ScientificReconstructionMaterialV1
    -> canonical representation
    -> content identity
```

The material commitment should cover every scientifically relevant reconstructed component, not merely the final predicate vector.

Two reconstructions that happen to produce the same predicates but differ in evidence-view snapshot, reason topology, lifecycle state, dependency state, compatibility state, triangulation state, or derivation lineage remain different reconstructions.

## 13. Historical replay vs currentness

Successful reconstruction and evaluation replay prove only historical/replay consistency for the exact inputs and cutoff.

They do not establish that those inputs still represent the current state of any scientific view.

Freeze:

```text
ReplayVerifiedScientificDispositionV1
    != CurrentWithinScientificViewDispositionV1
```

A later currentness verifier must compare the replayed scientific-state identities against an independently qualified namespace-scoped scientific-view state head.

State advancement makes the previous currentness witness stale but must not invalidate historical replay.

## 14. Legacy scientific-method engine boundary

The existing root cognitive `ScientificMethodEngine` may contribute ordinary evidence-shaped observations through an explicit adapter, but it may not satisfy reconstruction directly using:

```text
HypothesisStatus
raw posterior
raw evidence_count
rank_hypotheses()
find_contradictions()
```

These are heuristic cognitive outputs, not SCI-014 qualified predicate/reason capabilities.

## 15. Qualification vectors

A first executable SCI-014A qualification should prove at minimum:

1. caller-supplied predicate facts cannot mint the full scientific replay witness;
2. caller-supplied reason-topology root cannot mint it;
3. a valid lower disposition-evaluation replay witness under forged reconstruction context cannot mint it;
4. identical predicate values under a different evidence-view snapshot fail correspondence;
5. identical predicate values under a different lifecycle/dependency/triangulation state fail correspondence;
6. same reason-node identities with a rewired topology fail;
7. a different unqualified predicate-derivation artifact cannot be substituted;
8. omitted candidate evidence breaks closed-view accounting;
9. unresolved candidate evidence cannot be silently converted to absence/opposition;
10. a negative-by-absence predicate cannot become `NotSatisfied`/`Satisfied` without the required closed-world receipt;
11. a predicate receipt with the correct value but wrong source subgraph fails;
12. a predicate receipt with correct sources but wrong derivation profile/artifact/execution lineage fails;
13. a model/LLM/HDC output cannot directly set a disposition predicate without ordinary evidence admission;
14. a lower evaluation witness cannot be upgraded by caller-supplied matching IDs;
15. full replay remains historical evidence after the live scientific view advances;
16. no reconstructed/full replay witness grants currentness, truth, recommendation, governance, medical, resource, or execution authority.

## 16. First implementation shape

Do not begin with a Theory Atlas database or domain migration.

Prefer one tiny owner reconstruction module/crate with synthetic qualified fixture types that model the required upstream capabilities without claiming those upstream implementations already exist.

The first executable product should be conceptually:

```text
QualifiedScientificReconstructionV1
```

with private construction and read-only access to:

```text
exact reconstruction material identity
exact reason-topology root
exact predicate derivation receipts
exact owner-derived predicate facts
exact upstream scientific-state identities
```

Then compose it with the independently qualified lower disposition-evaluation replay seam to produce the reserved:

```text
ReplayVerifiedScientificDispositionV1
```

Do not implement current-view state in the same tranche.

## 17. Exit gate

SCI-014A is complete only when the repository can truthfully state:

> A full replay-positive scientific-disposition witness can be created only after the owner verifier reconstructs the exact reason topology and every policy predicate from a closed, qualified scientific evidence state, preserves all unresolved/unknown cases, and then passes the independent deterministic disposition-evaluation replay boundary.

Anything weaker remains evaluation replay, not full scientific replay.
