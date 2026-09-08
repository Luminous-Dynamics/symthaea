# CI Workflow Definition Identity v1

Status: **candidate CI judge-identity theorem only; not deployed and not scientific authority**

Schema: `symthaea-ci-workflow-definition-identity-v1`

## Purpose

An exact PR-head source checkout does not by itself prove which workflow definition GitHub executed.

For a focused theorem, v1 therefore treats these as separate evidence:

```text
TheoremSourceIdentity
    !=
WorkflowDefinitionIdentity
```

The workflow-definition theorem compares three objects:

```text
RegisteredJudgeBlob
PRHeadJudgeBlob
ExecutingJudgeBlob
```

and verifies only when:

```text
RegisteredJudgeBlob
    =
PRHeadJudgeBlob
    =
ExecutingJudgeBlob
```

The executing copy is obtained by the focused workflow from its own:

```text
github.workflow_sha
```

The PR-head copy is obtained from the exact theorem checkout:

```text
github.event.pull_request.head.sha
```

## Why the check belongs inside each focused workflow

A separate meta-admission workflow has its own `github.workflow_sha`. That value identifies the meta-workflow definition, not the definition that a different focused workflow executed.

Therefore:

```text
MetaWorkflow.workflow_sha
    != proof of
FocusedWorkflow.workflow_sha
```

The focused workflow must perform its own executing-definition observation.

The registry may record that this capability has been qualified, but it must not manufacture the runtime observation itself.

## Pure verifier

`scripts/verify_ci_workflow_definition_identity.py` receives:

```text
workflow
expected_git_blob
head_root
executing_root
```

It performs no Git command, network request, Nix operation, Workbench invocation, or scientific transform.

It:

1. requires a canonical `.github/workflows/*.yml` path;
2. requires a canonical 40-lowercase-hex registered Git blob;
3. requires both roots to exist;
4. rejects symlinked workflow files;
5. requires each workflow to resolve inside its supplied root;
6. requires regular files;
7. computes Git blob identity from exact bytes;
8. requires both observed blobs to equal the registered blob.

Git blob identity is:

```text
SHA1("blob " || decimal_length || NUL || bytes)
```

## Authority boundary

A successful result establishes only:

```text
workflow_definition_identity_verified = true
```

It does not establish:

```text
focused_theorem_passed
merge_compatibility_passed
scientific_execution_qualified
transform_executed
fmq010_established
neural_alignment_established
consciousness_evidence
```

Thus:

```text
CorrectExecutingJudge != PassingTheorem
```

## Workbench migration

The six current Workbench focused workflows have not yet implemented this self-check. Their registry entries therefore remain:

```text
executing_workflow_identity_qualified = false
global_ci_eligible = false
```

A safe migration is:

```text
1. change one focused workflow under full CI
2. checkout exact PR head
3. checkout that workflow's own github.workflow_sha separately
4. invoke this verifier against both roots and the reviewed registered blob
5. execute the theorem only after judge identity succeeds
6. separately retain merge-compatibility evidence
7. hosted-qualify the migration
8. only then update the registry capability flag/blob
```

The registry update is itself protected control-plane state and therefore also requires full CI.

## Root of trust

This verifier is repository code and cannot be its own administrative root of trust.

Production deployment still requires repository branch/ruleset policy above the CI control plane.
