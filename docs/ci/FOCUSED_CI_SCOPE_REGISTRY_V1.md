# Focused CI Scope Registry v1

Status: **candidate CI admission architecture only; not deployed and not scientific authority**

Schema: `symthaea-focused-ci-scope-registry-v1`

## Motivation

Incident #787 established a repository-wide CI scheduling failure: at the measured point, 570 of 571 queued workflow runs were the monolithic global `CI` workflow.

The long-term fix should not accumulate ad hoc `paths-ignore` exceptions. v1 instead defines a registry of exact, reviewed focused qualification scopes.

```text
changed paths
    ↓
exact PR-head source
    ↓
scope ownership
    ↓
registered judge bytes
    ↓
qualified source/judge capabilities
    ↓
focused qualification manifest
    ↓
focused route OR full-CI fallback
```

The default is always full CI.

## Scope contract

Each scope declares:

```text
id
composition_group
composable
global_ci_eligible
qualification_source_mode = exact-pr-head
qualification_source_identity_qualified
executing_workflow_identity_qualified
merge_compatibility_separate = true
focused_workflow
focused_workflow_git_blob
scientific_authority = false
files[]
```

Files are exact paths, not globs. Every owned path has exactly one owner. Ambiguous ownership invalidates the registry.

## Two different judge questions

The registry distinguishes:

```text
HeadJudge
    =
workflow bytes in the exact PR-head checkout

ExecutingJudge
    =
workflow bytes GitHub actually executed for that focused workflow
    at that focused workflow's own github.workflow_sha
```

These are separate objects.

A meta-admission workflow cannot prove another workflow's `ExecutingJudge`, because the meta workflow has a different `github.workflow_sha`.

Therefore the responsibilities are split.

### Per-PR admission

The admission controller checks that:

```text
HeadJudgeBlob = RegisteredJudgeBlob
```

for every touched focused scope.

### Focused-workflow self-check

Each focused workflow must separately prove:

```text
ExecutingJudgeBlob = RegisteredJudgeBlob
```

using its own `github.workflow_sha`.

The generic pure verifier for that theorem is:

```text
scripts/verify_ci_workflow_definition_identity.py
```

A scope can become cheap-route eligible only after the registry records both:

```text
qualification_source_identity_qualified = true
executing_workflow_identity_qualified = true
```

and therefore:

```text
global_ci_eligible = true
    =>
qualification_source_identity_qualified = true
AND
executing_workflow_identity_qualified = true
```

This prevents metadata from promoting an unqualified source or judge.

## Source identity

Focused theorem qualification is defined over the exact PR head:

```text
TheoremQualificationRoot = PRHeadSHA
```

Merge compatibility is separate:

```text
MergeCompatibilityRoot = PRMergeSHA
```

and:

```text
TheoremQualification != MergeCompatibility
```

The generic source contract is defined in `CI_SOURCE_TREE_IDENTITY_V1.md`.

Current Workbench workflows have not yet qualified exact-head source checkout semantics, so their pilot entries remain ineligible.

## Fail-closed admission

Let `D` be the changed-path set.

Focused admission requires:

```text
D is non-empty
every changed path is canonical
no changed path is CI/build control-plane state
every changed path has one registered owner
all touched scopes are global-CI eligible
all touched scopes have qualified exact-head source identity
all touched scopes have qualified executing-workflow identity
all touched scopes share one composition_group
multi-scope mixes are composable
no touched focused workflow is itself changed
every touched focused workflow is a regular non-symlink head file
HeadJudgeBlob = RegisteredJudgeBlob
```

Otherwise:

```text
global_ci_required = true
```

There is no confidence score, majority rule, or "mostly docs" exception.

## Changed-path acquisition

The admission workflow uses the exact PR base/head commits and obtains changed paths with:

```text
git diff --name-only -z --no-renames BASE...HEAD
```

Rename detection is deliberately disabled. A rename is therefore represented conservatively as deletion of the old path plus addition of the new path, ensuring both ownership surfaces are considered.

`scripts/collect_ci_changed_paths.py` parses the NUL-delimited byte stream into deterministic JSON.

It rejects missing terminal NUL, empty path records, duplicate paths, non-UTF-8 paths, and overwrite of an existing output. Path canonicality is then enforced by the registry verifier.

## Bootstrap / self-authorization guard

A focused workflow may not cheaply qualify a change to itself.

```text
focused_workflow in changed_paths
    -> full-ci-required
```

The intended lifecycle is:

```text
workflow/source-identity migration
    -> full CI

hosted qualification of that migration
    -> evidence

registry capability/blob promotion
    -> full CI

later theorem-only changes under stable judge
    -> focused admission eligible
```

Thus the cheap route exists only after an expensive onboarding event.

## Composition groups

Multiple scopes may compose only when they share one composition group and every touched scope is composable. Otherwise full CI is required.

The initial Workbench pilot uses `workbench-provenance-v1`, but all six scopes are currently ineligible.

## Protected control plane

The registry cannot own or cheaply route changes to global CI/build/admission state, including global workflows/actions, Cargo and Nix roots, toolchain roots, the source/judge identity contracts, the changed-path collector, registry/verifier/tests, and governance classifiers.

Any such change receives full CI.

## Initial Workbench pilot

The registry currently contains the six reviewed Workbench provenance scopes corresponding to #624, #629, #638, #667, #681, and #690.

Their exact owned-file union remains the same reviewed 28-file surface used by the earlier narrow Workbench routing experiment.

The current state is intentionally:

```text
global_ci_eligible = false
qualification_source_identity_qualified = false
executing_workflow_identity_qualified = false
```

for every pilot scope.

Their existing workflow blobs are retained as migration references, not as permission to skip global CI.

## Narrow router status

The earlier 28-file Workbench `paths-ignore` renderer is retained only as a regression/migration artifact.

It is **superseded for deployment** because path-only exclusion cannot prove exact PR-head theorem source or executing focused-workflow definition identity.

Deploying it now would reduce runner pressure by weakening the evidence model. v1 forbids that shortcut.

## Admission versus qualification

The admission controller only answers which checks are required. It does not answer whether those checks passed.

Therefore:

```text
FocusedCiAdmission
    != FocusedQualificationPassed
    != WorkbenchExecutionQualified
    != TransformExecuted
    != AtlasCorrectness
    != FMQ010
    != NeuralAlignment
```

All registry and admission outputs keep `scientific_authority = false`.

## Root of trust

Repository code cannot recursively certify its own administrative authority.

The intended hierarchy is:

```text
repository branch/ruleset policy
        ↓
CI admission control plane
        ↓
qualified source/judge identity capabilities
        ↓
registered exact judge bytes
        ↓
focused theorem qualification
        ↓
scientific interpretation
```

`main` is currently unprotected, so this candidate does not claim production enforcement. Branch/ruleset protection and required checks remain prerequisites for relying on cheap admission.

## Qualification status

Current locally executed candidate surfaces:

```text
source-tree identity contracts           55/55 pass
workflow-definition identity contracts   13/13 pass
changed-path collector contracts         10/10 pass
focused registry/admission contracts     45/45 pass
```

These are local construction results, not hosted qualification.

## Deployment sequence

Preferred sequence:

```text
1. preserve frozen scientific/provenance heads
2. drain/cancel obsolete global-CI backlog operationally
3. qualify this control-plane tranche
4. establish branch/ruleset protection
5. migrate focused workflows one at a time to:
       exact PR-head checkout
       source-identity proof
       own github.workflow_sha judge proof
6. hosted-qualify each migration under full CI
7. promote corresponding registry capability flags/blobs under full CI
8. integrate admission into global CI as a cheap first job
9. condition expensive matrices on global_ci_required
10. retain scheduled full-repository qualification as an interaction backstop
```

No existing neuroscience PR needs to be restacked merely to define this architecture.
