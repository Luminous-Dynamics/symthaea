# Workbench Provenance CI Scope v1

Status: **superseded for deployment; retained as a regression and migration artifact only**

Schema: `symthaea-workbench-provenance-ci-scope-v1`

## Historical purpose

This tranche was created during CI incident #787 after measuring severe repository-wide PR fan-out. It proved that the six Workbench provenance PR surfaces could be classified conservatively as an exact 28-file set and that a deterministic `paths-ignore` transformation of the monolithic `ci.yml` could be generated without unrelated workflow edits.

Its locally executed qualification surface remains useful:

```text
classifier contracts  28/28 pass
renderer contracts    18/18 pass
total                  46/46 pass
```

The classifier and renderer are therefore retained as regression evidence and as a reconciliation oracle for the initial Workbench registry.

## Why it must not be deployed

Subsequent provenance work established two stronger requirements that path-only omission cannot prove:

```text
TheoremQualificationRoot = exact PR head
MergeCompatibilityRoot   = synthetic PR merge commit

TheoremQualification != MergeCompatibility
```

and:

```text
RegisteredJudgeBlob
    =
PRHeadJudgeBlob
    =
ExecutingFocusedWorkflowJudgeBlob
```

The six current Workbench workflows were authored with default `actions/checkout` pull-request semantics and have not yet qualified the exact-PR-head and executing-workflow-definition contracts.

Therefore an exact 28-file `paths-ignore` block could reduce runner load while allowing the global matrix to disappear before the focused qualification path has the source/judge guarantees required by the newer evidence model.

That is no longer acceptable.

## Current authority

The narrow router may still establish only:

```text
this changed-path set belongs to the reviewed 28-file Workbench surface
this exact historical ci.yml transformation is deterministic
```

It does not establish:

```text
global_ci_may_skip in production
exact_pr_head_source_verified
workflow_definition_identity_verified
focused_theorem_passed
scientific_execution_qualified
```

## Long-term replacement

Deployment responsibility has moved to:

```text
Focused CI Scope Registry v1
+
CI Source Tree Identity v1
+
CI Workflow Definition Identity v1
```

A scope may eventually become cheap-route eligible only after:

```text
qualification_source_identity_qualified = true
executing_workflow_identity_qualified   = true
global_ci_eligible                       = true
```

and its exact PR-head workflow blob still matches the registered judge.

## 28-file migration identity

The original exact Workbench deployment set remains valuable as a migration invariant. The focused-scope registry must continue to reconcile its six initial scope ownership sets to exactly those 28 files, with no additions or omissions, until a reviewed registry revision deliberately changes that boundary.

Thus the narrow tranche has become:

```text
historical routing experiment
        +
exact migration corpus
        +
renderer regression theorem
```

rather than a production routing policy.

## Scientific boundary

As before:

```text
CI routing
    != focused theorem qualification
    != Workbench execution qualification
    != transform execution
    != FMQ-010
    != neural alignment
```

No scientific authority is granted by this artifact.
