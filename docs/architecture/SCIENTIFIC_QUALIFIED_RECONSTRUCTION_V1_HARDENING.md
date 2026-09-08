# Scientific Qualified Reconstruction v1 — Derivation Graph Hardening

Status: architecture hardening companion to `SCIENTIFIC_QUALIFIED_RECONSTRUCTION_V1.md`.

This note tightens SCI-014A around predicate dependency semantics, fixed-point evaluation, coverage, and material identity. It does not expand authority.

## 1. New core distinction

The scientific reason graph may contain legitimate cycles:

```text
argument A attacks B
B attacks C
C attacks A
```

That does **not** imply that the predicate-derivation engine may recurse arbitrarily.

Freeze:

```text
scientific reason graph
    != predicate derivation dependency graph
```

and:

```text
cyclic scientific argument
    != implicit recursive predicate semantics
```

The reason graph records scientific relationships. The derivation graph records computational dependencies between exact policy-visible predicates.

## 2. Predicate dependency graph

Every predicate specification must declare the predicates, qualified upstream objects, reason subgraphs, and closure receipts on which it depends.

Conceptually:

```text
PredicateDerivationSpec {
    predicate_id
    upstream_object_kinds
    reason_subgraph_requirements
    predicate_dependencies
    recursion_semantics
    result_domain
    closed_world_requirements
    derivation_profile
}
```

The owner reconstructs one exact `PredicateDependencyGraph` for the assessment.

The graph itself is scientifically material and must be covered by the reconstruction material identity.

Changing only dependency edges while retaining the same predicate values is a different reconstruction.

## 3. Default rule: derivation must be well-founded

The default SCI-014A profile requires predicate dependencies to be acyclic and topologically evaluable.

A dependency cycle without an explicitly qualified recursion profile fails closed.

Freeze:

```text
predicate A -> predicate B -> predicate A
    != valid merely because evaluation happened to terminate once
```

and:

```text
implementation recursion behavior
    != registered scientific semantics
```

A runtime stack, hash-map visitation order, cached previous value, or implementation-specific iteration order must never define scientific meaning implicitly.

## 4. Explicit fixed-point semantics

Some future scientific policies may legitimately require recursive/non-monotonic derivation. Such cycles are allowed only through an explicit, immutable recursion/fixed-point profile.

That profile must bind at least:

```text
strongly-connected-component identity
value lattice / result domain
initial state
transition operator
update ordering semantics
termination / convergence rule
maximum iteration or resource policy
ambiguity semantics
oscillation semantics
multiple-fixed-point semantics
trace commitment
implementation artifact
execution lineage
```

No cycle receives an implicit fixed-point interpretation.

## 5. No false convergence authority

A recursive evaluator may only emit a qualified predicate value when the registered profile's convergence theorem/criterion is satisfied.

Otherwise it must yield an explicit non-positive state such as:

```text
Unknown
Blocked
Unresolved
NonConvergent
AmbiguousFixedPoint
```

according to the registered scientific profile.

Freeze:

```text
iteration limit reached
    != converged
```

```text
last observed value
    != qualified fixed point
```

```text
one implementation found a fixed point
    != unique scientific solution
```

If multiple admissible fixed points exist and the profile does not define a qualified selection theorem, the result remains ambiguous.

## 6. Negation and absence stratification

Negative-by-absence predicates are especially dangerous inside recursive derivation.

For example:

```text
no-active-defeater-exists
```

must never be inferred from a predicate graph whose own closure depends on that negative conclusion.

Require explicit stratification for negation/closed-world dependencies.

Conceptually:

```text
stratum 0: closed evidence-view accounting
stratum 1: positive existential scientific facts
stratum 2: qualified absence predicates over closed lower strata
stratum 3: disposition predicates
```

Exact strata are profile-defined, but negative dependencies may only consume a universe proven closed at an earlier/equivalent qualified layer.

Freeze:

```text
absence predicate helps establish the closure needed to justify itself
    -> reject
```

## 7. Stronger anti-circularity

The existing prohibition on:

```text
stored disposition -> predicates -> same disposition
```

extends to every material derived object.

The following must fail unless a specifically qualified recursive profile proves otherwise:

```text
final disposition
    -> reason topology
    -> predicate
    -> final disposition
```

```text
policy output label
    -> evidence eligibility
    -> support predicate
    -> policy output label
```

```text
current view disposition
    -> candidate evidence selection
    -> reconstructed predicates
    -> current view disposition
```

Scientific search/admission cannot be conditioned on the answer being reconstructed unless that adaptive protocol was preregistered as a separate scientific process whose selection effects are explicitly represented.

## 8. Policy-predicate coverage theorem

A reconstruction must prove exact coverage between the qualified policy and derivation receipts.

Let:

```text
RequiredLeafPredicates(policy)
```

be the predicate identities referenced by every rule/fallback path under the exact policy profile, including predicates needed only by lower-precedence rules.

Require:

```text
for every required leaf predicate:
    exactly one qualified derivation receipt
```

unless the policy profile explicitly defines a qualified multi-receipt aggregation rule.

No required predicate may be silently omitted because the winning rule did not consult it in that particular execution.

This preserves the complete reason topology required by #783/#825.

## 9. Auxiliary predicate discipline

A derivation profile may use internal/auxiliary predicates, but they must not become hidden epistemic state.

Every auxiliary predicate that influences a policy-visible result must be covered by:

```text
identity
specification
upstream dependencies
receipt / trace
result
material commitment
```

Auxiliary predicates with no path to any policy predicate, unresolved condition, or retained scientific reason may remain diagnostic, but they must be marked as such rather than silently altering evaluation.

## 10. Predicate derivation receipt coverage

A `PredicateDerivationReceipt` must bind not only its result but its exact dependency frontier.

Conceptually add:

```text
direct_upstream_object_ids
direct_predicate_dependency_ids
reason_subgraph_root
closed_world_receipts
recursion_profile_id, if any
SCC/fixed-point trace root, if any
```

Two receipts with identical result values but different dependency frontiers are not interchangeable.

## 11. No hidden ambient derivation state

Predicate derivation must be closed over declared scientific inputs.

Unless explicitly bound through a qualified execution/input artifact, derivation may not depend on:

```text
wall clock
live network
ambient filesystem
mutable global state
process environment
unbound cache
previous Atlas disposition
user/session preference
hidden model memory
non-deterministic RNG
```

If learned or stochastic computation is scientifically required, its exact model, seed/randomness policy, input artifact, execution capsule, and output evidence must be bound explicitly before the result participates in predicate derivation.

## 12. Reconstruction material identity hardening

`ScientificReconstructionMaterialV1` must not be identified by an ambiguous ad hoc serialization.

The future identity profile must bind:

```text
reconstruction material schema/profile identity
canonicalization profile identity
domain-separation tag
canonical bytes/content digest
```

The canonical material must include at minimum:

```text
qualified proposition/use identity
closed evidence-view snapshot/accounting
lifecycle + adjudication generations
dependency / compatibility / triangulation state
complete reason-topology root
predicate dependency graph
all predicate specifications
all derivation receipts
all closed-world receipts
all recursion/fixed-point receipts
policy profile/artifact
information cutoff
owner reconstruction profile/artifact/execution lineage
```

Friendly labels are not content identity.

Schema/profile drift creates a new reconstruction identity unless an explicit qualified equivalence/migration receipt exists.

## 13. Material replay must retain losing/unused dependencies

The scientific reconstruction is not merely the minimal path that selected the final disposition.

Dependencies belonging to lower-precedence policy rules, unresolved alternatives, defeaters, falsifiers, or scientific reasons that were materially in scope must remain represented even when they did not determine the summary label.

Freeze:

```text
not selected by winning rule
    != scientifically irrelevant
```

This preserves future auditability and allows a later policy profile to reinterpret the same historical scientific state without rewriting its evidence history.

## 14. Qualification vectors to add to SCI-014A

Add at minimum:

1. direct predicate self-cycle rejects;
2. two-predicate cycle rejects without a recursion profile;
3. same cycle under an explicit convergent fixed-point profile succeeds only when the registered criterion is met;
4. iteration-cap exhaustion does not count as convergence;
5. oscillating cycle becomes unresolved/non-convergent;
6. multiple fixed points do not silently select one;
7. negative-by-absence predicate cannot participate in the closure proof that justifies itself;
8. same final predicate vector with changed derivation edges yields a different reconstruction identity;
9. same predicate result with changed direct dependency frontier fails receipt correspondence;
10. every policy-required predicate has a qualified derivation receipt even when its rule loses precedence;
11. auxiliary hidden predicate influencing a policy-visible result fails qualification;
12. stored/final disposition cannot enter any upstream derivation dependency;
13. evidence selection conditioned on the desired/final disposition fails unless represented by an explicit qualified adaptive protocol;
14. ambient mutable/cache/model state cannot alter predicate derivation without entering the material lineage;
15. canonical container reordering preserves reconstruction identity while semantic dependency changes do not;
16. reconstruction schema/profile drift cannot silently preserve identity.

## 15. Revised implementation sequence

After #825 is hosted-qualified, SCI-014A should still begin synthetically, but in this order:

```text
A. closed qualified evidence-view fixture
B. reason-topology reconstruction
C. predicate dependency graph + coverage
D. proof-carrying non-recursive predicate derivation
E. lower #825 evaluation replay composition
F. full ReplayVerifiedScientificDispositionV1
```

Do **not** implement general recursive/fixed-point predicates in the first product tranche. The first implementation should reject cycles outright while the architecture reserves a later explicit profile for qualified recursion.

This keeps the first executable theorem small and fail-closed.

## 16. Authority ceiling

Nothing in this hardening grants:

```text
scientific truth
consensus
global literature completeness
recommendation
governance authority
medical authority
resource authority
effect authority
```

It only strengthens the theorem that a full replay witness corresponds to one exact, closed, owner-reconstructed scientific reasoning state rather than a self-consistent or accidentally recursive set of derived labels.
