# SYM-FV-INFRA-003B — Verifier Outcome Mapping v1

`SYM-FV-INFRA-003` defines the canonical operational result vocabulary:

```text
Pass
Fail
Blocked
EnvironmentFailure
```

This child defines the reviewed semantic layer between raw tool/lane outcomes and those operational states **without confusing workflow disposition with the semantic meaning of the evidence**.

## Rule

A raw verifier outcome does not become an evidence result directly.

```text
raw tool outcome
      ↓
reviewed outcome disposition
      ├──> operational result
      └──> semantic polarity
```

The mappings are closed-world. Unknown or ambiguous outcomes default to:

```text
operational result = Blocked
semantic polarity  = NoSemanticConclusion
```

never `Pass` and never semantic counterevidence.

## Operational mapping

```text
SemanticSuccess             -> Pass
SemanticCounterexample      -> Fail
ProofOrQualificationFailure -> Fail
UnsupportedBoundary         -> Blocked
InsufficientBound           -> Blocked
ResourceExhaustion          -> Blocked
MissingPrerequisite         -> Blocked
StaleSubjectOrDependency    -> Blocked
AmbiguousOrUnknownOutcome   -> Blocked
UnclassifiedToolCrash       -> Blocked
EnvironmentUnavailable      -> EnvironmentFailure
ToolInstallationFailure     -> EnvironmentFailure
RunnerInfrastructureFailure -> EnvironmentFailure
```

## Semantic polarity

```text
SemanticSuccess             -> PositiveSupport
SemanticCounterexample      -> SemanticCounterevidence
ProofOrQualificationFailure -> QualificationNegative
UnsupportedBoundary         -> NoSemanticConclusion
InsufficientBound           -> NoSemanticConclusion
ResourceExhaustion          -> NoSemanticConclusion
MissingPrerequisite         -> NoSemanticConclusion
StaleSubjectOrDependency    -> NoSemanticConclusion
AmbiguousOrUnknownOutcome   -> NoSemanticConclusion
UnclassifiedToolCrash       -> NoSemanticConclusion
EnvironmentUnavailable      -> NoSemanticConclusion
ToolInstallationFailure     -> NoSemanticConclusion
RunnerInfrastructureFailure -> NoSemanticConclusion
```

The critical boundary is:

```text
workflow Fail
!= property false
```

A proof or qualification failure means the attempted evidence did not establish its obligation. It does **not** by itself establish the negation of the subject property.

Only an explicit `SemanticCounterexample` carries `SemanticCounterevidence` polarity in this mapping generation.

## Examples

Lean:

```text
exact theorem elaborates + axiom policy passes
  -> SemanticSuccess
  -> Pass + PositiveSupport

exact theorem fails elaboration
  -> ProofOrQualificationFailure
  -> Fail + QualificationNegative
```

Kani / model checking:

```text
property satisfied within exact harness/bound
  -> SemanticSuccess
  -> Pass + PositiveSupport for that bounded evidence class

real assertion/property counterexample
  -> SemanticCounterexample
  -> Fail + SemanticCounterevidence

unwind/solver/resource exhaustion
  -> InsufficientBound / ResourceExhaustion
  -> Blocked + NoSemanticConclusion
```

## Hostile controls

A hostile mutant is considered semantically rejected only when the lane observes the expected semantic counterexample/failure class required by the lane contract.

```text
mutant + genuine semantic counterexample
= candidate valid negative control

mutant + proof harness failed to elaborate
!= evidence the mutant property is false

mutant + unwind exhaustion
!= valid negative control

mutant + solver exhaustion
!= valid negative control

mutant + runner failure
!= valid negative control
```

This prevents verifier incompleteness, proof-authoring failure, or environmental failure from masquerading as mutation sensitivity.

## Receipt binding

Receipts consuming this mapping should record:

```text
result
outcome_disposition
semantic_polarity
raw_tool_outcome
mapping_schema_version
mapping_schema_digest
```

Changing either operational disposition or semantic polarity creates a new evidence generation. Historical receipts are not rewritten.

## Refutation boundary

```text
SemanticCounterevidence
!= automatic refutation of any same-named claim
```

The assurance calculus still needs an explicit subject/claim/refutation relation and admissibility checks before counterevidence can defeat a closure claim.

This mapping only classifies the semantic polarity of the observed verifier outcome.

## Non-equivalences

```text
canonical outcome mapping
!= verifier correctness
!= theorem truth
!= completeness of the tool taxonomy
!= implementation refinement
!= evidence-class promotion
!= automatic claim refutation
!= runtime authority
```

`Pass` remains evidence-class-specific. For example, Kani `Pass` is still only bounded model evidence within the recorded harness and bounds; it does not become an unbounded theorem.
