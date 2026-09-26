# SYM-FV-INFRA-003B — Verifier Outcome Mapping v1

`SYM-FV-INFRA-003` defines the canonical formal-evidence result vocabulary:

```text
Pass
Fail
Blocked
EnvironmentFailure
```

This child defines the semantic mapping layer between raw tool/lane outcomes and those four states.

## Rule

A raw verifier outcome does not become an evidence result directly.

```text
raw tool outcome
      ↓
outcome disposition
      ↓
canonical result
```

The mapping is closed-world. Unknown or ambiguous outcomes default to `Blocked`, never `Pass`.

## Canonical dispositions

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

## Hostile controls

A hostile mutant is considered successfully rejected only when the lane observes the expected semantic failure/counterexample class.

```text
mutant + semantic counterexample
= valid negative control

mutant + unwind exhaustion
!= valid negative control

mutant + solver exhaustion
!= valid negative control

mutant + runner failure
!= valid negative control
```

This prevents model-checker incompleteness or environmental failure from masquerading as mutation sensitivity.

## Receipt binding

Receipts consuming this mapping should record:

```text
result
outcome_disposition
raw_tool_outcome
mapping_schema_version
mapping_schema_digest
```

Changing the mapping creates a new evidence generation. Historical receipts are not rewritten.

## Non-equivalences

```text
canonical outcome mapping
!= verifier correctness
!= theorem truth
!= completeness of the tool taxonomy
!= implementation refinement
!= evidence-class promotion
!= runtime authority
```

`Pass` remains evidence-class-specific. For example, Kani `Pass` is still only `BoundedModelSafety` within the recorded harness and bounds; it does not become an unbounded theorem.
