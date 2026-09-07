# CI Lifecycle Tiering v1

## Purpose

Symthaea's repository-wide `CI` workflow is a large integration suite. It is appropriate as a premerge/main/release guarantee but disproportionately expensive as feedback for hundreds of simultaneously open draft research PRs.

The v1 lifecycle theorem is:

```text
draft iteration
    !=
full premerge qualification
```

and:

```text
fewer draft integration jobs
    !=
weaker merge qualification
```

The intended behavior is to run a small fail-safe Tier 1 during explicit draft iteration, while retaining the complete existing integration surface before merge and on main.

## Current repository pressure

At the 2026-09-07 observation interval used to refine this contract, GitHub reported:

```text
open PRs             507
open drafts          491
open non-drafts       16
queued workflow runs 622
in-progress runs       3
```

This is scheduler state, not evidence that any scientific/software candidate passed or failed.

## Current GitHub merge-policy state

For exact `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`, current repository metadata reports:

```text
main protected                         false
branch protection enabled              false
required status-check enforcement      off
required status contexts/checks        []
repository rulesets                     []
```

This resolves the earlier #217 uncertainty about accidentally leaving an unknown required check permanently pending.

It does **not** establish that Tier 1 is sufficient to merge.

```text
GitHub currently permits merge without full CI
    !=
Symthaea policy permits merge without full CI
```

Until repository protection is deliberately introduced, full-premerge qualification remains a governance/review obligation rather than an enforcement guarantee supplied by GitHub settings.

## Routing modes

`scripts/ci/qualification_scope.py` defines exactly five v1 modes:

```text
iteration
full-premerge
main
scheduled
manual-full
```

Only `iteration` suppresses the full matrix.

### iteration

Requires an ordinary `pull_request` event whose payload contains exactly:

```text
pull_request.draft == true
```

This is the only reduced-verification route.

### full-premerge

Used for every non-draft pull request and as the fail-safe mode for malformed/unknown states.

A ready PR synchronized to a new exact head receives full premerge verification again.

### main

Every push to `refs/heads/main` receives the complete integration surface.

### scheduled

The existing weekly schedule remains full. Schedule-specific jobs such as stress tests and psych-bench retain their own narrower internal conditions, but ordinary integration jobs are not reduced merely because the trigger is scheduled.

### manual-full

Every `workflow_dispatch` is a deliberate full qualification.

The existing unique manual concurrency grouping must remain non-superseding in effect: a deliberate manual run must not be cancelled merely because an ordinary PR update occurs.

## Fail toward more verification

The resolver uses this invariant:

```text
unknown event -> full-premerge
missing pull_request object -> full-premerge
missing draft field -> full-premerge
non-boolean draft field -> full-premerge
unexpected push ref -> full-premerge
malformed event JSON -> full-premerge
```

No parser error may produce `iteration`.

The future YAML integration must preserve the same rule when router output is missing:

```text
needs.qualification-scope.outputs.run_full != 'false'
```

rather than:

```text
needs.qualification-scope.outputs.run_full == 'true'
```

The former means missing/failed output tends toward full execution. The latter would accidentally skip the full matrix if the router failed to emit an output.

A dependency failure also needs deliberate `always()` handling so a router problem does not suppress the full jobs. Existing non-router dependency semantics (for example `needs: test`) must still be preserved rather than making downstream jobs run after their substantive prerequisite failed.

## Tier 1 allowlist

The checked-in registry is:

`scripts/ci/ci_lifecycle_jobs_v1.json`

V1 allows these existing jobs during draft iteration:

```text
governance
fmt
cls-field-count
workspace-targets
embodiment-safety-composition
orphan-modules
secrets-scan
```

These protect inexpensive source/repository structure and immediate safety/governance properties.

They do not constitute merge qualification.

## Everything else defaults to full-only

The registry is allowlist-based rather than denylist-based:

```text
new/unrecognized CI job -> full-only
```

This matters because a future expensive matrix must not accidentally become draft-safe merely because someone forgot to add it to a list of heavy jobs.

Current full-only examples include root tests/Clippy, Muse, hardened regression jobs, all feature matrices, feature interactions, psych-bench, dependency/security audits, SBOM, subcrate matrices, Genesis benchmarks, stress tests, and compliance suites.

Focused subsystem/scientific workflows outside the monolithic `CI` workflow remain independent. A draft may still execute its exact focused theorem gate when that workflow's own trigger says it should.

## Registry structural validation

`scripts/ci/validate_ci_lifecycle_registry.py` checks the registry against the actual workflow without requiring a YAML dependency.

It fails if:

- an allowlisted Tier-1 job no longer exists;
- the Tier-1 list contains duplicates;
- a Tier-1 job has a job-level `needs:` dependency and therefore is no longer independently runnable;
- a documented full-only example disappears unexpectedly;
- the workflow no longer has the simple top-level `jobs:` shape the conservative parser understands;
- the registry stops defaulting unknown jobs to full.

The validator does not mutate the workflow.

## Pull-request trigger lifecycle

The eventual `ci.yml` patch should make the PR activity surface explicit so lifecycle transitions themselves can activate the appropriate tier.

Candidate v1 activity types:

```yaml
pull_request:
  types:
    - opened
    - synchronize
    - reopened
    - ready_for_review
    - converted_to_draft
```

The essential transitions are:

```text
open as draft              -> iteration
synchronize draft          -> iteration
reopen as draft            -> iteration
convert ready -> draft     -> iteration
ready_for_review           -> full-premerge
synchronize ready PR       -> full-premerge
reopen non-draft PR        -> full-premerge
```

Do not rely on activity type alone; `pull_request.draft` remains the authoritative routing field for ordinary PR events.

## Candidate-controlled workflow boundary

Ordinary `pull_request` workflows execute from PR-associated workflow content. Therefore a candidate can potentially change the workflow/router code in the same branch being evaluated.

Consequently:

```text
CI router says iteration/full
    !=
trusted merge authority
```

and:

```text
candidate-owned green check
    !=
base-owned policy proof
```

Lifecycle tiering is a **resource-scheduling architecture**, not a cryptographic/security capability.

Until a separately reviewed base-owned/ruleset/required-check boundary exists, reviewers must verify that any intended merge head received the full integration suite under the expected workflow generation.

Do not solve this by moving broad candidate execution into `pull_request_target`: that trigger has a different trust model and can expose privileged base context to untrusted code/caches if misused.

## Future stronger enforcement

A later repository-administration tranche may introduce a base-owned merge rule such as:

```text
exact candidate head
+ trusted full-integration admission/check
    -> merge eligible
```

Possible GitHub mechanisms include deliberately configured required checks/rulesets or a merge-queue policy. That is separate from this scheduling patch and requires administration access plus its own failure-mode review.

## Tier Q relationship

Tier Q from #217 is complementary and potentially higher leverage:

```text
issue/theorem
    -> branch-only candidate
    -> focused exact qualification
    -> PASS/FAIL/INCONCLUSIVE
    -> product PR only if independently justified
```

Tier Q prevents some qualification experiments from creating a product PR at all. Lifecycle tiering reduces the cost of PRs that still legitimately exist.

Neither one replaces #75's runner-capacity work.

## Intended YAML integration shape

The final implementation should add one small `qualification-scope` job and gate only full-only jobs on it.

Conceptually:

```text
qualification-scope
    -> mode + run_full

Tier-1 jobs
    -> independent; run on both iteration and full events

full-only jobs
    -> run unless scope explicitly says run_full=false
```

Tier-1 jobs should not acquire a dependency on the router. This prevents a router problem from hiding cheap safety feedback.

For a full-only job with no other prerequisite, the desired semantics are conceptually:

```text
if router succeeded and run_full=false:
    skip
else:
    run
```

For a full-only job that already depends on `test`, preserve both facts:

```text
router did not explicitly select iteration
AND
test succeeded
```

Do not use a blanket `always()` that causes substantive downstream tests to execute after their real parent test failed.

## Evidence language

Keep the following non-equivalences explicit:

```text
Tier-1 green != full CI green
focused theorem green != repository integration green
full CI green != scientific truth
queued != PASS
cancelled != FAIL
workflow skipped by lifecycle policy != test passed
GitHub branch policy off != integration optional
```

## Staging rule

The v1 work remains on unopened branch:

`ci/lifecycle-tiering-v1`

Do not open it as a PR merely to test the concept while the repository is experiencing extreme Actions pressure.

First qualify the pure router/registry locally or through an already-available queue-neutral mechanism. Only then generate/review the exact `ci.yml` transformation.

## Acceptance criteria for the eventual workflow patch

1. draft PR open/synchronize/reopen executes exactly the registered Tier-1 jobs from monolithic CI;
2. focused external workflows remain governed by their own contracts;
3. draft -> ready launches the full existing matrix for that exact head;
4. later ready-PR synchronize launches full CI for the new head;
5. ready -> draft returns future ordinary synchronizations to Tier 1;
6. manual dispatch remains full and effectively non-superseding;
7. main push remains full;
8. schedule behavior remains full plus existing schedule-specific conditions;
9. unknown/malformed routing state tends toward full verification;
10. newly added CI jobs default to full-only;
11. existing downstream `needs:` semantics remain intact;
12. Tier-1 green cannot be represented as merge qualification;
13. a workflow/router change in the candidate cannot be silently treated as trusted base-owned merge authority;
14. measured focused-gate queue latency improves after deployment.
