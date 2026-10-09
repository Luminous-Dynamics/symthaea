# Independent Security Audit Verifier

The workflow `.github/workflows/security-audit-independent-verifier.yml` is a default-branch-owned verifier for this repository's shared audit workflow. It does not execute pull-request code. It queries the authoritative run and PR APIs, rejects stale runs and fork-originated runs, checks the exact caller workflow blob and pinned Luminous Platform engine source, then validates the aggregate verdict artifact's SHA-256 digest and required lane coverage.

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

The Python test file currently contains 20 test methods. The corresponding hosted runs have not completed yet, so their result remains unverified.

## Enforcement verification snapshot (2026-10-09)

GitHub's `GET /repos/{owner}/{repo}/branches/main` response reports `protected: false` for this repository, and the repository-level `/rulesets` endpoint returned an empty list. The connected integration's branch-protection detail request returned HTTP 403, and organization-level ruleset policy could not be established from this connection. Thus the available evidence does **not** show an active required-status merge gate on `main`. This is a release blocker for enforcement, not a reason to treat the verifier as passed. A repository administrator must configure and verify the exact `Security Audit / Independent Verifier` status and verifier job as required checks, define controlled bypasses, and confirm organization policy if present.



## Required-check semantics for reruns

GitHub documents that `workflow_run: requested` is not emitted for a re-run; `in_progress` is the early invalidation event for reruns. If a rerun is queued, an older commit status might remain until the trusted receiver can publish its next state. Therefore a production ruleset must require the repository's **producer audit workflow check** as well as the `Security Audit / Independent Verifier` commit status, and must not treat the receiver's own default-branch job as a substitute for a PR-head check. The producer check holds the PR while a new audit attempt is queued/running; the custom status only turns green after independent verification. Test-run status on the PR head before authorizing merge.
