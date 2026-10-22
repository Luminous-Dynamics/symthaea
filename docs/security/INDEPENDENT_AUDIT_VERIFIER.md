# Independent Security Audit Verifier

The workflow `.github/workflows/security-audit-independent-verifier.yml` is a default-branch-owned verifier for this repository's shared audit workflow. It does not execute pull-request code. It queries the authoritative run and PR APIs, rejects stale runs and fork-originated runs, checks the exact caller workflow blob at both the audited PR-head commit and the commit identified by `verdict.workflow_sha`, and pins the Luminous Platform engine source before validating the aggregate verdict artifact's SHA-256 digest and required lane coverage. It requires the API workflow-run `path` to exactly match the base-owned workflow path, ties the verdict workflow reference to the independently matched PR number, and compares `verdict.workflow_sha` to the independently fetched current PR `merge_commit_sha`. That comparison is accepted only while GitHub reports `mergeable: true`; the caller workflow blob is then checked at that exact test-merge commit. Missing verifier self-test outcome is a failure, never an implicit success.

The verifier publishes `Security Audit / Independent Verifier`. The status and workflow job do not become a merge gate until repository rules explicitly require them.

## Qualification sequence

1. This PR introduces the verifier, so it cannot independently certify its own bootstrap merge: GitHub only runs a `workflow_run` receiver after that workflow exists on the default branch. Use existing protections and review for the bootstrap merge.
2. After merge, open a new test PR and confirm the receiver runs from the default branch, its self-tests pass, its artifact digest matches GitHub metadata, and the exact-head status is updated.
3. Then require the exact verifier status and workflow job in repository rules. A missing, queued, skipped, stale, malformed, expired, `FAIL`, or `INCOMPLETE` result is never a pass.
4. Changes to the caller workflow or engine pin require a separate trusted policy update. Do not let a PR update both the audited source and the trusted expectation that authorizes it.

The workflow is privilege-separated from PR code, but the repository's default branch and rules remain the local trust root. A truly external trust root requires a separately governed verifier or GitHub App.




## Pre-merge verifier tests

The unprivileged `.github/workflows/security-audit-verifier-tests.yml` workflow runs the verifier's Python compilation and adversarial unit tests on the exact PR head, with read-only repository access, no secrets, and no write permissions. It is test evidence only: it neither publishes an authorization status nor replaces the default-branch `workflow_run` trust anchor. Hosted results must complete and be inspected; a local unit-test pass is not a hosted CI pass.



## Fail-closed status integrity

A verifier exception can otherwise leave a previous green commit status visible, but unconditionally writing `failure` is also unsafe: another workflow with the same display name could trigger the receiver and poison the status. The verifier now only attempts an exception-path failure update when the authenticated-event fields identify the exact policy-pinned workflow ID, a `pull_request` run, and matching base/head repository IDs. Once the run has been authenticated through the API, failures in its result or evidence still publish a failure. An inability to reach GitHub is reported as incomplete; no software can guarantee a remote status update while the status API itself is unavailable.

Historical snapshot (2026-10-09): the verifier suite contained 23 test methods. This was superseded by the subsequent exact-head hardening documented below; use the latest inventory and hosted-run status instead.

## Enforcement verification snapshot (2026-10-09)

GitHub's `GET /repos/{owner}/{repo}/branches/main` response reports `protected: false` for this repository, and the repository-level `/rulesets` endpoint returned an empty list. The connected integration's branch-protection detail request returned HTTP 403, and organization-level ruleset policy could not be established from this connection. Thus the available evidence does **not** show an active required-status merge gate on `main`. This is a release blocker for enforcement, not a reason to treat the verifier as passed. A repository administrator must configure and verify the exact `Security Audit / Independent Verifier` status and verifier job as required checks, define controlled bypasses, and confirm organization policy if present.





The verifier's exception-path status update is additionally bound to the triggering event's expected workflow ID, `pull_request` event type, same base/head/current repository IDs, and default-branch PR target. Malformed API payloads or unexpected ordinary exceptions are normalized to a verifier failure, while status-write failures themselves are reported without recursively crashing. A failed or untrusted trigger cannot poison the authoritative status merely by sharing the workflow display name.



## Verdict contract and freshness

The external verifier requires the exact luminous.security-audit.verdict.v1 field set; rejects duplicate JSON object keys; requires the caller workflow reference and SHA to be canonical; matches the workflow run URL to GitHub's API record; and rejects timestamps more than five minutes in the future or older than seven days. The policy deliberately requires a new audit before merging a PR whose last audit evidence is stale. The corresponding unit tests are present, but hosted execution has not completed, so behavior remains unqualified until those runs finish.

## Required-check semantics for reruns

GitHub documents that `workflow_run: requested` is not emitted for a re-run; `in_progress` is the early invalidation event for reruns. If a rerun is queued, an older commit status might remain until the trusted receiver can publish its next state. Therefore a production ruleset must require the repository's **producer audit workflow check** as well as the `Security Audit / Independent Verifier` commit status, and must not treat the receiver's own default-branch job as a substitute for a PR-head check. The producer check holds the PR while a new audit attempt is queued/running; the custom status only turns green after independent verification. Test-run status on the PR head before authorizing merge.


## Latest exact-head base-freshness and permission guard — 2026-10-11

- The verifier performs a second authoritative PR API read after validating the verdict artifact and immediately before success publication. The merge SHA must remain identical and the API merge state must be recognized; `behind`, `unknown`, missing, and unrecognized values fail closed.
- Synchronized verifier Git blob: `47536b70be613967497764a8b2b43def295d8b03`. Synchronized verifier-test blob: `4a637e5fe9e4332fafe5ba471b3d51b6f216e9cf`. The current suite contains 35 test methods, including a regression that requires the workflow-level token permissions to remain read-only and `statuses: write` to occur only in the `verify` job.
- The receiver workflow's top-level permission block is read-only. The only `statuses: write` occurrence is the explicit permission on the job that publishes the independent status.
- These guards reduce intra-run TOCTOU and permission-creep risk but do not revoke an already-published green status after `main` advances. Require the producer audit check and `Security Audit / Independent Verifier` with strict/up-to-date branch protection, then perform the base-advance acceptance test tracked in [Platform issue #24](https://github.com/Luminous-Dynamics/luminous-platform/issues/24).
- Hosted runs on the updated exact heads remain queued/pending. Authored test counts and static source checks are not a test pass. These PRs are drafts; the verifier is not an active merge gate.
