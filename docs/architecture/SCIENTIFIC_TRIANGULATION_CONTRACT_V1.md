# Scientific Triangulation Contract v1

**Status:** architecture refinement only; non-authorizing; non-qualifying.

**Parent audit:** `#648@ea0f794142802486549f0b33eb5aeb1495e747a7`

## 1. Purpose

SCI-001 already separates evidence contribution, dependency/replication analysis, and scientific disposition. SCI-006 proposes a shared evidence-dependency graph, while SCI-014 proposes a proposition-centered Theory Atlas.

The missing join is **triangulation**: how multiple results may inform one scientific proposition without turning publication count, model count, or numerical agreement into independent confirmation.

This refinement freezes the contract that should sit between dependency topology and Theory Atlas disposition.

It does not introduce Rust product types, change #648, assign evidence authority, or claim that any current economics/Futures/science branch is qualified.

---

## 2. Core theorem

A future kernel must preserve:

```text
same proposition label
    != same scientific target
    != compatible estimand / target
    != poolable result
    != independent evidence
    != triangulated causal support
    != scientific truth
```

and:

```text
agreement
    != independence

disagreement
    != failed science
```

Triangulation is the structured comparison of evidence paths with explicit target compatibility, dependency topology, assumptions, methods, and result semantics.

No v1 scalar `triangulation_score`, `replication_score`, or `evidence_strength` is proposed.

---

## 3. Target compatibility precedes evidence aggregation

Two contributions must not be pooled or described as replications merely because they use similar labels.

A generic kernel should not infer target compatibility from names, units, textual similarity, or ontology proximity. A domain-owned comparator should produce a typed compatibility receipt.

Candidate common relation vocabulary:

```text
TargetCompatibility {
    ExactSameTarget,
    ExplicitlyTransformable {
        transform_artifact_id,
        assumptions,
        loss_semantics,
    },
    RelatedButNotPoolable {
        relation_id,
    },
    Incomparable {
        reason_id,
    },
}
```

`ExplicitlyTransformable` means the transformation is declared and reviewable; it does not mean the transformation is scientifically valid merely because an ID exists.

The transform itself is a scientific dependency and must enter lineage analysis.

### Economics conformance example

For a causal estimand, target identity may include:

```text
treatment
outcome
outcome unit
population
intervention contrast
horizon
effect target
```

Therefore:

```text
national 12-month ATE
    != youth 12-month ATE
    != national 3-month ATE
    != national 12-month ATT
```

unless an explicit domain-qualified transformation/relation says otherwise.

### Cross-domain rule

The shared kernel should carry the compatibility receipt but should not own the domain equation that decides compatibility.

---

## 4. Poolability is not triangulation

Even two results targeting the exact same estimand may not be eligible for numerical pooling.

A future aggregation decision should be separate:

```text
PoolingDisposition {
    NotAssessed,
    NotPoolable { reasons },
    PoolableUnderDeclaredModel {
        pooling_model_artifact_id,
        assumptions,
        heterogeneity_policy_id,
        dependence_handling_id,
    },
}
```

No default inverse-variance average, vote, majority rule, or model averaging should occur at the kernel boundary.

The pooling model and heterogeneity/dependence policy are themselves dependencies.

Triangulation may be scientifically useful even when pooling is forbidden.

---

## 5. Dependency topology must project assumptions explicitly

SCI-006/#634-style dependency inventories correctly retain source data, vintages, measurement specifications, sampling designs, transformations, models, estimators, identification strategies, outcome policies, and evaluation protocols.

The generic graph should additionally be able to represent dependencies such as:

```text
AssumptionDeclaration
DiagnosticArtifact
ExecutionCapsule
LearnedPriorArtifact
LearnedGrammarArtifact
```

because different strategy/model IDs can still share the same critical assumption or learned prior.

For example:

```text
strategy:A != strategy:B
```

while both may depend on:

```text
assumption:no-unmeasured-confounding-v1
```

Therefore:

```text
different method identity
    != different failure mode
```

Known assumption overlap should be visible even when model/data implementations differ.

This extension does not imply that every assumption is testable or that an assumption ID proves its truth. The #620 direction remains relevant:

```text
assumption declaration
    != diagnostic result
    != assumption truth
```

---

## 6. Failure-mode diversity is multidimensional

A triangulation assessment should expose overlap and diversity rather than collapse them to one independence bit.

Candidate dimensions include:

```text
data/vintage overlap
measurement overlap
sampling overlap
transformation overlap
model/prior overlap
estimator overlap
identification-strategy overlap
assumption overlap
diagnostic overlap
execution/toolchain overlap
outcome-policy overlap
evaluation-protocol overlap
learned-grammar/prior overlap
```

The strongest generic no-overlap statement should remain scoped:

```text
DeclaredDisjointWithinScope
```

not:

```text
IndependentReplication
```

unless a separate authority has qualified inventory completeness and the relevant independence semantics.

---

## 7. Result relation is domain-owned

The common kernel should not decide that two arbitrary numerical results agree.

A domain-owned result comparator should produce a typed receipt under a predeclared comparison protocol.

Illustrative generic vocabulary:

```text
ResultRelation {
    ConcordantWithinDeclaredCriterion,
    DirectionallyConcordantButHeterogeneous,
    Discordant,
    MixedOrIndeterminate,
    NotComparable,
}
```

The comparison criterion, tolerance, sign convention, uncertainty handling, and missing-data policy must be explicit dependencies.

A result must not become `Concordant` merely because point estimates have the same sign while their estimands, populations, horizons, or uncertainty semantics differ.

---

## 8. Disagreement is first-class evidence

Triangulation must preserve disagreement rather than treating it as a failed aggregation.

A discordance may indicate:

```text
regime dependence
measurement mismatch
population heterogeneity
assumption failure
model misspecification
implementation defect
unmodeled intervention differences
time-varying mechanism strength
true falsification of one hypothesis
```

The Theory Atlas should therefore retain discordant evidence, unresolved defeaters, and proposed discriminating experiments.

A future planner may use disagreement to propose experiments that distinguish these explanations, but the disagreement itself does not identify which explanation is correct.

---

## 9. Proposed triangulation object

A future shared object may have a shape similar to:

```text
TriangulationAssessment {
    target_claim_id,
    contribution_ids,
    target_compatibility_receipts,
    dependency_graph_receipt,
    shared_assumption_receipts,
    result_comparison_receipts,
    pooling_disposition,
    known_dependency_components,
    declared_disjoint_relations,
    incomplete_inventories,
    concordances,
    discordances,
    unresolved_defeaters,
    discriminating_experiment_candidates,
    limitations,
}
```

Important omissions are intentional:

```text
confidence: f64
truth_probability
independent_count
replication_count
triangulation_score
winner
causal = true
```

Those shortcuts would erase the structures the object is meant to preserve.

---

## 10. Triangulation dispositions should remain scoped

Illustrative bounded conclusions include:

```text
MultipleConcordantLineagesWithKnownSharedDependencies
ConcordantAcrossDeclaredDisjointScopes
DiscordantAcrossDeclaredDisjointScopes
EvidenceTargetMismatch
InsufficientDependencyInventory
RelatedTargetsNotPoolable
AssumptionOverlapDominatesApparentMethodDiversity
```

These labels are examples, not a proposed total ordering.

A disposition must retain the receipts that justify it.

---

## 11. Replication and triangulation are different questions

Replication asks whether a specified result/experiment can be reproduced under a declared scope.

Triangulation asks whether differently structured evidence paths converge or diverge on a proposition or family of related targets.

Therefore:

```text
replication
    != triangulation
```

and:

```text
successful replication
    != independent triangulation
```

A replication that reuses the exact same data, measurement, code, estimator, and assumptions is valuable reproducibility evidence but may add little failure-mode diversity.

Conversely, a different design may be valuable triangulation evidence while not being a literal replication because its estimand or measurement differs.

---

## 12. Theory Atlas integration

SCI-014 should consume triangulation structures rather than perform paper-count voting.

For one proposition, the Atlas should be able to display:

```text
exact supporting targets
related-but-nonpoolable targets
opposing/discordant targets
known dependency components
assumption overlap
method/design diversity
diagnostic states
measurement and sampling differences
result-comparison receipts
unresolved defeaters
open discriminating experiments
```

The Atlas should preserve original contributions and never replace them with only an aggregate summary.

A new contribution may change the dependency topology or expose a hidden shared assumption without changing any numerical result; that is still a scientifically meaningful update.

---

## 13. Falsification and triangulation

SCI-007/SCI-013 falsifier structures should be able to participate directly in triangulation.

For example:

```text
Claim A predicts outcome X under condition R.
Claim B predicts outcome Y under condition R.

Experiment E is preregistered to discriminate X from Y.
```

The resulting contribution should retain:

```text
which claim predictions were tested
which falsifiers were activated
which assumptions were shared
which evidence lineages were reused
which outcome was observed
```

A claim surviving multiple attacks should accumulate an explicit attack graph, not a larger scalar confidence number.

---

## 14. Historical and learned-model contamination

Triangulation over historical-discovery benchmarks must include model/tool/grammar/embedding ancestry.

Two apparent rediscoveries are not disjoint if both use a learned primitive derived from the same post-cutoff discovery.

Therefore learned artifacts should flow into SCI-006 dependency inventories before SCI-014 interprets apparent convergence.

This is consistent with the SCI-001 learned-grammar provenance theorem:

```text
learned primitive use
    -> inherited scientific dependency
```

---

## 15. Economics as a conformance suite, not the owner

The Economic Science branches are a strong conformance case because they already distinguish:

```text
measurement specification
sampling design
measurement uncertainty
causal estimand
identification strategy
assumption diagnostics
evidence dependency topology
```

A shared triangulation contract should be tested against these semantics without moving economics-specific estimand or causal rules into the common kernel.

The generic kernel should ask domains for:

```text
target compatibility receipt
result comparison receipt
dependency inventory
assumption/dependency projection
```

and preserve them without reinterpretation.

---

## 16. Implementation gate

Do not implement a common triangulation engine merely from this document.

A production implementation should wait until at least:

1. SCI-001 common/non-common boundaries are accepted;
2. SCI-006 dependency-graph semantics have an independently qualified common candidate;
3. domain conformance cases demonstrate that target compatibility can be expressed without semantic weakening;
4. result-comparison criteria are preregisterable and provenance-bound;
5. the implementation proves it does not auto-assign independent-replication, causal, novelty, or action authority.

A likely sequence is:

```text
SCI-006 evidence dependency graph core
    -> target-compatibility adapter contract
    -> result-comparison adapter contract
    -> triangulation assessment
    -> SCI-014 Theory Atlas v1
```

No qualification credit should transfer from #634 or any economics branch into the future common implementation.

---

## 17. Review checklist

A future implementation should be rejected if it:

- infers target equivalence from labels or shared units;
- automatically pools compatible-looking results;
- calls no-overlap `independence` without a separate qualification authority;
- ignores shared assumptions because strategy/model IDs differ;
- counts authors, institutions, papers, or model architectures as independent evidence;
- hides discordant/null/refuted contributions;
- allows post-outcome selection of comparison tolerances or pooling models;
- collapses triangulation to one mandatory scalar score;
- lets a scientific disposition mint action/safety/execution authority;
- moves domain-specific estimand or causal semantics into the generic kernel.

The desired theorem is narrower and stronger:

```text
qualified evidence structures
    + explicit target compatibility
    + explicit dependency topology
    + explicit result-comparison semantics
        -> auditable triangulation assessment
```

not:

```text
many agreeing studies -> truth
```
