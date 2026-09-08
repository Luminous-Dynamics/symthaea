# CI Source Tree Identity v1

Status: **candidate CI provenance theorem only; not deployed and not scientific authority**

Schema: `symthaea-ci-source-tree-identity-profile-v1`

## Problem

A GitHub Actions run associated with a pull-request head does not by itself prove that the job executed that head commit. For the `pull_request` event, GitHub's default workflow ref/SHA represent the synthetic pull-request merge ref. A default `actions/checkout` therefore answers a merge-compatibility question unless the workflow explicitly selects the pull-request head SHA.

The v1 model separates the two questions:

```text
TheoremQualificationRoot = PRHeadSHA
MergeCompatibilityRoot   = PRMergeSHA

TheoremQualification != MergeCompatibility
```

A success in one lane cannot substitute for a success in the other.

## Exact-head theorem lane

For pull-request theorem qualification, checkout must use exactly:

```text
${{ github.event.pull_request.head.sha }}
```

and the observed checkout commit must satisfy:

```text
ObservedCheckoutSHA = EventHeadSHA
```

This lane answers only:

```text
Did this theorem run against the exact PR head source identity?
```

It does not establish that the head merges cleanly with the current base.

## Merge-compatibility lane

For explicit integration compatibility, checkout may use:

```text
${{ github.sha }}
```

and must satisfy:

```text
ObservedCheckoutSHA = EventMergeSHA
```

This lane answers only:

```text
Did this test run against the synthetic PR merge source identity?
```

It does not establish that the frozen PR head alone passed the theorem.

## Lane-minimal identity roots

The identity digest deliberately excludes contextual fields that would make one source commit acquire multiple identities.

For an exact-head observation:

```text
IdentityPreimage = {
  schema,
  lane = exact-pr-head,
  base_repository_id,
  source_repository_id = head_repository_id,
  checkout_commit_sha = event_head_sha
}
```

For a merge-compatibility observation:

```text
IdentityPreimage = {
  schema,
  lane = pr-merge-compatibility,
  base_repository_id,
  source_repository_id = base_repository_id,
  checkout_commit_sha = event_merge_sha
}
```

and:

```text
IdentitySHA256 = SHA256(CanonicalJSON(IdentityPreimage))
```

### Exact-head independence from base movement

The exact-head identity does not contain `event_merge_sha`.

Therefore:

```text
same head SHA + changed synthetic merge SHA
    -> same exact-head identity
```

This is required for a frozen head to remain the same theorem subject while its base branch moves.

### Merge independence from redundant head metadata

The merge identity does not contain `event_head_sha`; the selected merge commit already commits its parent graph.

Therefore a retained head-SHA field cannot rename an otherwise identical merge source identity.

## Repository identity

Display names are retained for human audit but do not define identity.

The identity root uses immutable GitHub repository IDs:

```text
base_repository_id   <- ${{ github.repository_id }}
head_repository_id   <- ${{ github.event.pull_request.head.repo.id }}
```

This avoids case/rename aliases in `owner/name` from creating multiple identities for the same repository.

For fork pull requests, the exact-head root therefore binds both:

```text
base_repository_id
head/source_repository_id
```

rather than pretending the head commit originated in the base repository.

## Executing workflow definition is a separate provenance object

The source tree being tested and the workflow definition GitHub is executing are not assumed to be the same commit.

GitHub exposes:

```text
github.workflow_ref
github.workflow_sha
```

and v1 retains both in every PR source observation.

They are explicitly excluded from the source identity root:

```text
workflow_definition_identity_binding = separate-qualification-envelope
```

because a base-branch update may legitimately change the synthetic merge/workflow commit while the frozen theorem head remains unchanged.

However, exclusion from the source identity does **not** mean the workflow definition is trusted implicitly. The CI admission layer must separately prove:

```text
RegisteredQualifierBlob
    = HeadQualifierBlob
    = ExecutingWorkflowCommitQualifierBlob
```

before a stable focused scope can be eligible to skip global CI.

The candidate source-identity workflow demonstrates this by checking out `${{ github.workflow_sha }}` into a separate directory, hashing the actual workflow file with Git blob semantics, and requiring it to equal the workflow file bytes in the exact-head checkout.

Thus:

```text
CorrectSourceIdentity
    + DriftedExecutingJudge
    -> qualification NOT admissible
```

This prevents base-branch movement from silently changing the judge while preserving the theorem subject.

## Tree SHA boundary

The workflow retains:

```text
git rev-parse HEAD^{tree}
```

but v1 marks that value:

```text
tree_sha_identity_binding = diagnostic-only
```

The pure verifier does not independently reconstruct Git commit objects, so an arbitrary reported tree SHA must not rename an otherwise identical verified commit identity.

The selected Git commit SHA is the authoritative source identity in v1. A future Git-object reconstruction theorem may separately verify the explicit tree object if that extra decomposition becomes useful.

Thus:

```text
same selected commit + changed diagnostic tree report
    -> same source identity
```

not:

```text
unverified tree report
    -> new source identity
```

## Optional merge context for exact-head qualification

`event_merge_sha` may be absent in the exact-head lane. Head qualification must not depend on a synthetic merge object being available.

The merge lane requires a valid merge SHA.

## Workflow-dispatch boundary

`workflow_dispatch` may exercise the static contracts diagnostically, but it cannot establish either:

```text
exact PR-head qualification
PR-merge compatibility
```

because there is no pull-request event authority for those identities.

## Authority state

The profile may claim only:

```text
source_tree_identity_contract_defined = true
```

A valid observation may establish only:

```text
source_tree_identity_verified = true
exact_pr_head_source_verified = true|false
pr_merge_source_verified      = true|false
```

according to its lane.

It cannot establish:

```text
focused_theorem_passed
merge_compatibility_passed
scientific_execution_qualified
transform_executed
fmq010_established
neural_alignment_established
consciousness_evidence
```

Therefore:

```text
CorrectSourceIdentity != PassingTheorem
CorrectSourceIdentity != ScientificResult
```

## Workbench migration implication

The existing Workbench focused workflows #624/#629/#638/#667/#681/#690 use default `actions/checkout` semantics and have not yet qualified exact PR-head source identity or executing-workflow-definition identity.

Their current queued runs remain potentially useful as merge-ref compatibility evidence, but must not be promoted to exact-head theorem evidence without an explicit checkout/source observation proving the head SHA.

The focused-scope registry therefore keeps all six current Workbench pilot scopes at:

```text
global_ci_eligible = false
qualification_source_identity_qualified = false
```

until a deliberate workflow migration qualifies both the source checkout and stable judge bytes.

The earlier 28-file emergency Workbench `paths-ignore` proposal must therefore not be deployed as-is.

## CI source trust hierarchy

The intended production hierarchy is:

```text
branch/ruleset policy
       ↓
CI admission control plane
       ↓
registered qualifier blob
       ↓
executing workflow blob == registered qualifier blob
       ↓
verified checkout source identity
       ↓
focused theorem execution
       ↓
scientific interpretation
```

The repository is currently unprotected, so the repository-internal verifier is not claimed as an administrative root of trust. Production deployment should eventually pair the admission model with branch/ruleset protection and required checks.

## Qualification contracts

The dependency-free adversarial suite covers at least:

- exact-head versus merge checkout substitution;
- merge SHA independence of exact-head identity;
- tree-report independence of both identity roots;
- workflow ref/SHA retention without source-identity contamination;
- optional merge SHA for exact-head observations;
- required merge SHA for merge observations;
- immutable repository-ID identity;
- fork head repository identity;
- repository display-name non-authority;
- canonical SHA and repository-ID spelling;
- workflow-ref and workflow-SHA syntax;
- path-like repository-name rejection;
- closed-world profile and observation schemas;
- duplicate JSON-key rejection;
- boolean/integer laundering rejection;
- workflow-dispatch non-promotion;
- lane-meaning and checkout-expression drift;
- workflow-definition separation drift;
- authority non-escalation.

No Nix, Workbench, scientific transform, network client, or scientific result is required by this theorem.
