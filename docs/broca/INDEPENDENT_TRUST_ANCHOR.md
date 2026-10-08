# Broca Independent Trust Anchor

## Purpose

The Broca Feature Matrix is intentionally PR-controlled because it must compile and exercise the proposed Broca compiler.

That makes its own verifier and workflow an insufficient final trust root.

This companion workflow is maintained on the repository's default branch and runs on GitHub's `workflow_run` event. Its verifier is therefore outside the pull request's source-control boundary.

## Verification model

The verifier never checks out, imports, builds, or executes pull-request source. It treats the PR as API-readable data and independently validates:

- the triggering run's exact head SHA and current open same-repository PR;
- allowed source-change scope;
- immutable Action references and security-sensitive workflow settings;
- exact-head checkout configuration and Rust 1.96.0 selection;
- freeze receipt and attestation plumbing;
- compiler build-context and native-package invalidation hooks;
- the frozen UniMorph semantic namespace;
- all eight Broca Feature Matrix jobs;
- successful Workflow Syntax and PR Governance runs for that exact head.

A successful verification publishes the commit status context `Broca / Independent Trust Anchor` on the exact PR head. Before doing so, it requires each of the eight expected jobs to be successfully completed and verifies the critical qualification steps inside those jobs were actually executed, not merely skipped.

The status only becomes a merge gate when repository branch-protection or ruleset policy explicitly requires that exact context.

## Why workflow_run

GitHub documents that `workflow_run` workflows execute from the default branch. This is the intended privilege boundary here: the policy code lives on the trusted base branch while pull-request source is fetched only through read-only API calls.

The verifier deliberately does not consume or execute artifacts from the triggering workflow. This follows GitHub's security guidance that artifacts from preceding workflows must be treated as untrusted data.

For the rare case where all required runs completed before this trust-anchor workflow existed on the default branch, the same verifier exposes a constrained `workflow_dispatch` replay. The caller supplies only a workflow-run ID; the base-owned verifier fetches the authoritative run record, requires a completed same-repository pull-request run, and still subjects its head to the independent snapshot and all exact-head gate checks. The workflow itself refuses manual execution from any ref other than the repository's current default branch, because GitHub permits API-driven dispatches against other refs after the workflow has run once.

The workflow also observes `requested` and `in_progress` activity types. Those early events publish a pending commit status, invalidating any earlier success before a new qualifying run can complete. GitHub documents these activity types for `workflow_run`; `requested` is not emitted for a re-run, so `in_progress` remains the early invalidation path for re-runs.

## Evidence chain

The resulting control-plane chain is:

`default-branch verifier -> API-only PR inspection -> exact PR head -> independent workflow/job checks -> trusted commit status`

The verifier also self-audits the base-owned trust workflow definition at its trusted checkout SHA, including its exact Action pins, permissions, early-invalidation triggers, default-branch replay restriction, and manual-run input plumbing. This is an accidental-drift guard; repository branch/ruleset governance remains the ultimate external trust root.

The trust anchor complements, rather than replaces, the Broca Feature Matrix's local compiler replay and artifact provenance checks.

## Claim ceiling

A passing independent trust-anchor receipt establishes that the declared repository-control-plane contract and required qualification runs were independently inspected.

It does not establish linguistic validity, source correctness, semantic adequacy, pronunciation correctness, naturalness, or speaker appropriateness.


## Independent snapshot lock

The verifier consumes `docs/broca/independent_trust_policy_v1.json`, which is maintained outside the Broca PR. The policy records the approved PR number, base ref/base SHA, the current live base-branch tip SHA, exact approved head SHA, and the Git blob SHA of every changed file. Any new Broca commit or movement of the stacked base branch therefore invalidates the independent status until a separate base-branch policy update explicitly re-approves the new snapshot.


## Causal run binding

The verifier does not trust the `workflow_run` event payload by itself. It fetches the authoritative workflow-run record by run ID and requires agreement on repository identity, run ID, workflow name, event, head SHA, head branch, run attempt, and conclusion before qualifying anything. Repository identity is bound both to the GitHub repository's stable numeric ID and its full name. Workflow and pull-request enumerations are paginated with explicit safety bounds; truncation is a hard verification error rather than an implicit partial result. Workflow-job enumeration is likewise paginated, with an exact eight-job contract and a hard 1000-job safety bound. Commit-to-PR association is paginated with the same bounded, fail-closed treatment. Both the approved snapshot and the live PR file response must also contain unique file paths; duplicate identities are rejected rather than normalized away. The verifier then performs a final exact-head reconciliation immediately before publishing a successful status, so a newly superseding run is not silently grandfathered by an earlier observation.

## Final publication revalidation

Immediately before a successful status is published, the verifier re-reads the authoritative PR record and complete changed-file list. It re-applies the independent snapshot lock and rejects any head, base, repository, draft/state, or file/blob change observed after the earlier qualification checks. This closes the time-of-check/time-of-use gap between evidence admission and trust-status publication.

## Trust-policy schema boundary

The independent policy is treated as a typed admission record, not arbitrary JSON. Every approved entry must have a non-empty path, an added/modified status, and a canonical 40-hex Git blob SHA; duplicate paths and malformed live PR file records are rejected. The trust workflow also rejects additional trigger classes such as pull_request_target, push, schedule, repository_dispatch, and workflow_call, preserving the intended default-branch workflow_run trust boundary.

The checked-in policy is a bootstrap artifact for the initial compiler qualification. After the trust anchor is established on the default branch, changing the policy itself requires an independently authorized base-branch change; a target PR cannot redefine its own admission policy and then satisfy that same policy.

## Workflow identity and result binding

The verifier binds Broca Feature Matrix, Workflow Syntax, and PR Governance to their GitHub workflow IDs as well as their names and paths. Recreating one of those workflows therefore requires an explicit trust-root change before its runs become admissible. Non-PASS and STALE verifier results also exit non-zero, while WAITING remains a successful control-plane wait state with a pending commit status; stale outcomes explicitly replace any prior successful trust status with failure.
