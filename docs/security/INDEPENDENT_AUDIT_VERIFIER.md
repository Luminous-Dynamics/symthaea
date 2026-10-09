# Independent Security Audit Verifier

The workflow `.github/workflows/security-audit-independent-verifier.yml` is a default-branch-owned verifier for this repository's shared audit workflow. It does not execute pull-request code. It queries the authoritative run and PR APIs, rejects stale runs and fork-originated runs, checks the exact caller workflow blob and pinned Luminous Platform engine source, then validates the aggregate verdict artifact's SHA-256 digest and required lane coverage.

The verifier publishes `Security Audit / Independent Verifier`. The status and workflow job do not become a merge gate until repository rules explicitly require them.

## Qualification sequence

1. This PR introduces the verifier, so it cannot independently certify its own bootstrap merge: GitHub only runs a `workflow_run` receiver after that workflow exists on the default branch. Use existing protections and review for the bootstrap merge.
2. After merge, open a new test PR and confirm the receiver runs from the default branch, its self-tests pass, its artifact digest matches GitHub metadata, and the exact-head status is updated.
3. Then require the exact verifier status and workflow job in repository rules. A missing, queued, skipped, stale, malformed, expired, `FAIL`, or `INCOMPLETE` result is never a pass.
4. Changes to the caller workflow or engine pin require a separate trusted policy update. Do not let a PR update both the audited source and the trusted expectation that authorizes it.

The workflow is privilege-separated from PR code, but the repository's default branch and rules remain the local trust root. A truly external trust root requires a separately governed verifier or GitHub App.


## Enforcement verification snapshot (2026-10-09)

The repository-level rulesets API returned an empty list for this repository during implementation. Reading branch-protection settings was denied to the connected integration, so that result does **not** prove that no organization-level ruleset or branch protection applies. Before claiming merge enforcement, a repository administrator must verify in GitHub that the exact commit status `Security Audit / Independent Verifier` and the verifier job check are required, bypasses are controlled, and the policy applies to the default branch.
