# AOBench finite viability-localized model v0.1

**Status:** Initial finite surrogate implementation; requires exact-head Rust qualification  
**Tracking:** [AOBench finite viability/local-to-global issue #12045](https://github.com/Luminous-Dynamics/symthaea/issues/12045)  
**Implementation:** `crates/domains/symthaea-organizational-closure/src/viability.rs`

## 1. Research basis

The July 2026 BioSystems paper *Autopoiesis as viability-localized self-production in a topos* proposes a formal definition with four axioms: viability-local realization closure (AP1), boundary production (AP2), admissibility of production relative to constraints (AP3), and existence of a viable global regime (AP4). It also studies finite tissue/cellular-sheaf examples that expose local-to-global obstruction signals.

- https://doi.org/10.1016/j.biosystems.2026.105808
- https://pubmed.ncbi.nlm.nih.gov/42107485/

Segura (October 2026) separately formalizes the distinction between an organization, a formal surrogate and the response profile produced by a finite assay family. Restricted profiles justify equivalence only when the assay family is conservative over a declared candidate class:

- https://doi.org/10.1016/j.biosystems.2026.105928

This implementation is a deliberately narrow finite-data surrogate inspired by these ideas. It is not a categorical implementation of an elementary topos, a Lawvere–Tierney topology, a monad, or the complete AP1–AP4 definitions.

## 2. Why this is separate from the runtime verifier

The existing runtime verifier checks execution evidence: process contracts, exact component versions, stored artifact bytes, SHA-256 digests, producer/consumer lineage and external dependency observations.

This new module checks a different object: a frozen finite formal model. Its successful result means that the supplied finite maps and constraints satisfy the module's finite witness checks. It does not authenticate execution and it does not change `qualifies_functional_self_maintenance()`.

Keep at least four result planes separate:

1. Runtime evidence and actual execution.
2. Finite formal-model witnesses.
3. Experimental assay adequacy / model distinguishability.
4. Physical or biological measurements.

## 3. Finite encoding

The model contains:
- finite production states and their viability-equivalence classes;
- finite organization states and their viability-equivalence classes;
- a realization map from production states to organization states;
- finite boundary states and a boundary-production map;
- a frozen admissibility set over boundary states;
- finite variables, domains and local constraint relations;
- an explicit search-node budget.

### AP1-style finite realization witness

For viable production states, the realization map must:
- be total;
- target a declared viable organization state;
- map each production equivalence class consistently to exactly one organization class;
- be injective on the induced classes;
- cover all viable organization classes.

This is a finite bijection on the declared viability classes. It is not a proof that an arbitrary left-exact localization functor has been implemented.

### AP2-style finite boundary-production witness

Every viable production state must map to a declared viable boundary state. The induced boundary class must be well-defined for each production viability class, and every required viable boundary class must be represented.

### AP3-style finite admissibility witness

Every boundary produced by a viable production state must belong to the frozen admissibility set. Admissibility is a hard condition; it is not averaged into a health score. A policy/authority root is not made internally authoritative merely because an enforcement mechanism is modeled as a produced component.

### AP4-style finite global-regime witness

Variables have finite, non-empty domains. Each local constraint is a relation over an ordered scope. The checker searches for an assignment satisfying every registered local relation.

The search is deterministic and uses ordered maps/sets. Its budget counts expanded search nodes. Outcomes:
- `Pass`: a global assignment was found;
- `Fail`: the finite search exhausted the declared search space without a global assignment;
- `Inconclusive`: the node budget was reached before a conclusion.

Budget exhaustion is never converted into proof of non-existence.

## 4. Local-to-global obstruction fixture

The test suite contains three individually satisfiable binary constraints over variables (x,y,z\in\{0,1\}):

- (x=y)
- (y=z)
- (x\ne z)

Each local relation has allowed assignments, but there is no global assignment satisfying all three simultaneously. The evaluator reports:
- all three local patches individually satisfiable;
- no global viable regime;
- a finite local-to-global obstruction signal.

This is a constraint-system illustration of the kind of obstruction investigated through finite sheaf/tissue models. It is not, by itself, a proof that this encoding is equivalent to a particular cellular-sheaf construction.

## 5. Deterministic negative controls

The module includes tests for:
- valid finite witness across all four checks;
- a realization map that is not well-defined on viability classes;
- missing boundary production;
- inadmissible produced boundary;
- locally satisfiable patches with no global regime;
- search-budget exhaustion returning inconclusive;
- empty state universes not passing vacuously;
- malformed local relation tuple arity.

## 6. Interpretation and limits

The checker reports AP1/AP2/AP3/AP4-style statuses independently. The convenience predicate `qualifies_finite_surrogate()` is true only when all four pass and no findings remain.

The result is relative to:
- the finite candidate states;
- the supplied viability-class partition;
- the realization and boundary maps;
- the admissibility set;
- the finite variables and local relations;
- the search budget and encoding.

It does not establish that the state space is complete, that the equivalence relation matches biology, that the finite model is physically realized, or that a system is alive, sentient, conscious or strictly autopoietic.

## 7. Qualification status

The source and deterministic tests have been committed on the research branch. No local Rust compiler is available in this execution environment. Exact-head format, compile, tests and strict Clippy must be checked on the exact commit; until then, implementation is committed but not compiler-qualified.
