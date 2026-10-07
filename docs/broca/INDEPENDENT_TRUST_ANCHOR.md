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

A successful verification publishes the commit status context `Broca / Independent Trust Anchor` on the exact PR head.

The status only becomes a merge gate when repository branch-protection or ruleset policy explicitly requires that exact context.

## Why workflow_run

GitHub documents that `workflow_run` workflows execute from the default branch. This is the intended privilege boundary here: the policy code lives on the trusted base branch while pull-request source is fetched only through read-only API calls.

The verifier deliberately does not consume or execute artifacts from the triggering workflow. This follows GitHub's security guidance that artifacts from preceding workflows must be treated as untrusted data.

## Evidence chain

The resulting control-plane chain is:

`default-branch verifier -> API-only PR inspection -> exact PR head -> independent workflow/job checks -> trusted commit status`

The trust anchor complements, rather than replaces, the Broca Feature Matrix's local compiler replay and artifact provenance checks.

## Claim ceiling

A passing independent trust-anchor receipt establishes that the declared repository-control-plane contract and required qualification runs were independently inspected.

It does not establish linguistic validity, source correctness, semantic adequacy, pronunciation correctness, naturalness, or speaker appropriateness.


## Independent snapshot lock

The verifier consumes `docs/broca/independent_trust_policy_v1.json`, which is maintained outside the Broca PR. The policy records the approved PR number, base ref/base SHA, exact approved head SHA, and the Git blob SHA of every changed file. Any new Broca commit therefore invalidates the independent status until a separate base-branch policy update explicitly re-approves the new snapshot.


## Causal run binding

The verifier does not trust the `workflow_run` event payload by itself. It fetches the authoritative workflow-run record by run ID and requires agreement on run ID, workflow name, event, head SHA, head branch, run attempt, and conclusion before qualifying anything. Workflow and pull-request enumerations are paginated with explicit safety bounds; truncation is a hard verification error rather than an implicit partial result.
