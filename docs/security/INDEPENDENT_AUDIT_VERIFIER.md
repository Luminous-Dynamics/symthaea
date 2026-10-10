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

The Python test file currently contains 23 test methods. The corresponding hosted runs have not completed yet, so their result remains unverified.

## Enforcement verification snapshot (2026-10-09)

GitHub's `GET /repos/{owner}/{repo}/branches/main` response reports `protected: false` for this repository, and the repository-level `/rulesets` endpoint returned an empty list. The connected integration's branch-protection detail request returned HTTP 403, and organization-level ruleset policy could not be established from this connection. Thus the available evidence does **not** show an active required-status merge gate on `main`. This is a release blocker for enforcement, not a reason to treat the verifier as passed. A repository administrator must configure and verify the exact `Security Audit / Independent Verifier` status and verifier job as required checks, define controlled bypasses, and confirm organization policy if present.





The verifier's exception-path status update is additionally bound to the triggering event's expected workflow ID, `pull_request` event type, same base/head/current repository IDs, and default-branch PR target. Malformed API payloads or unexpected ordinary exceptions are normalized to a verifier failure, while status-write failures themselves are reported without recursively crashing. A failed or untrusted trigger cannot poison the authoritative status merely by sharing the workflow display name.



## Verdict contract and freshness

The external verifier requires the exact luminous.security-audit.verdict.v1 field set; rejects duplicate JSON object keys; requires the caller workflow reference and SHA to be canonical; matches the workflow run URL to GitHub's API record; and rejects timestamps more than five minutes in the future or older than seven days. The policy deliberately requires a new audit before merging a PR whose last audit evidence is stale. The corresponding unit tests are present, but hosted execution has not completed, so behavior remains unqualified until those runs finish.

## Required-check semantics for reruns

GitHub documents that `workflow_run: requested` is not emitted for a re-run; `in_progress` is the early invalidation event for reruns. If a rerun is queued, an older commit status might remain until the trusted receiver can publish its next state. Therefore a production ruleset must require the repository's **producer audit workflow check** as well as the `Security Audit / Independent Verifier` commit status, and must not treat the receiver's own default-branch job as a substitute for a PR-head check. The producer check holds the PR while a new audit attempt is queued/running; the custom status only turns green after independent verification. Test-run status on the PR head before authorizing merge.


## Latest TOCTOU hardening — 2026-10-11

After the verdict artifact's ZIP digest, schema, manifest, and member hashes have been validated, the verifier fetches the authoritative PR state again immediately before publishing success. The second read must still identify the same open same-repository PR head/base and the same canonical `merge_commit_sha`; a changed test-merge SHA fails closed. The regression suite now contains 33 test methods. Current synchronized Git blobs are verifier `78a096eebc98f6aeb51a5b6773bcffb90e9189f8` and tests `7d72105363ab2c538bdbebbeeab5313bddae994d`.

This narrows the intra-run time-of-check/time-of-use window; it does not expire a previously published status if the base branch changes after verification. Ongoing enforcement still needs current producer checks and a trusted base-update/status-reconciliation design. Hosted qualification for the latest code head is queued, so no test pass is claimed.
