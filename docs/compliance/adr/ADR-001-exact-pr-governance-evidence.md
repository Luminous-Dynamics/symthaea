# ADR-001: Exact PR Governance Evidence

**Date**: 2026-09-21
**Status**: Proposed
**Change Class**: A (Safety-Critical)

## Context

The pull-request governance check was named `Governance Check (Class A/B Changes)` but its legacy CI helper compared against the mutable `origin/$GITHUB_BASE_REF` branch tip, silently fell back to `HEAD~1` if that comparison failed, and reported Class A policy violations as warnings while still exiting successfully.

Separately, #4960 demonstrated that a workflow run associated with a raw PR head can execute a synthetic PR merge subject instead. #5325 therefore binds execution identity explicitly. Inspection of that stronger design exposed a second authority boundary: an ordinary `pull_request` workflow is selected from the PR integration subject, so a PR that edits its own governance workflow can influence the code producing that status. A future required merge root therefore should not rely solely on PR-head-controlled workflow code.

A source-coverage audit exposed a third defect: the legacy Class A/B path list describes an older repository layout. For example, threshold constants now live under `src/cognitive_loop/thresholds/`, `SafetyAgentConfig` lives at `src/safety/agent.rs`, and the old `crates/mycelix-bridge-common/...` paths are not present in this repository. Copying that list into a stricter gate would create false confidence rather than stronger governance.

The remaining governance problem is therefore three-dimensional:

1. bind the *change-set being classified* to the exact immutable event base/head pair;
2. bind Class A coverage to current, verified repository roots and govern deletion/rename escape paths; and
3. ensure the authoritative evaluator itself comes from trusted `main` code and never executes untrusted PR-head code.

## Decision

Add a machine-readable governance policy and dedicated validator that:

- computes the PR change set from the exact event base SHA and raw head SHA;
- has no mutable branch-name comparison and no `HEAD~1` fallback;
- includes additions, copies, deletions, modifications, renames, and type changes in governed path classification;
- records both old and new paths for rename/copy records so a protected root cannot escape by moving;
- freezes minimum safety/governance roots inside the validator so the policy cannot remove its own protection;
- fails if a required protected root is absent from both the event base and head without an explicit policy migration;
- protects the current threshold module tree, threshold override definitions/promotion path, EthicsEngine, SafetyAgent, governance charter, policy, validator, ordinary governance workflow, and base-controlled governance-root workflow;
- requires a changed Class A ADR for every Class A PR;
- rejects unresolved ADR template placeholders or missing required sections;
- requires approved Class A commit prefixes for unique non-merge commits that touch Class A surfaces;
- leaves Class B enforcement explicitly deferred to #5363 rather than preserving a stale or contradictory list;
- emits explicit nonclaims that path coverage is a versioned policy, structural validation does not establish scientific adequacy, test execution, full-CI success, or merge authorization.

Stage `.github/workflows/pr-governance-root.yml` as the stronger future merge-authority context:

- trigger with `pull_request_target` lifecycle events so the workflow definition comes from trusted base/default-branch context;
- admit work only for non-draft pull requests targeting `main`;
- use only `contents: read` permissions;
- check out the exact event `main` base SHA with credentials persistence disabled;
- fetch the PR head only as Git objects into an isolated ref;
- verify the fetched object equals the event head SHA;
- never check out the PR head into the worktree;
- never execute PR-head scripts, Actions, or generated code;
- run the trusted `main` validator over the exact event base/head Git objects;
- retain stable per-PR concurrency and `cancel-in-progress: true` for lifecycle revocation.

The ordinary `pull_request` Governance check remains useful integration evidence, but the staged `Governance Root Check` is the intended eventual required-status authority once it has been merged into `main`, behaviorally qualified, and bound into the trusted-root ruleset.

### Scientific Basis

The basis is empirical repository and platform evidence rather than a claim of scientific novelty. Historical #4960 execution showed that PR association and checkout subject can differ. Inspection of the existing governance helper showed a mutable `origin/$GITHUB_BASE_REF...HEAD` comparison, a silent `HEAD~1` fallback, and an always-zero CI exit path even when Class A changes were detected. Current-tree inspection showed that several inherited path assumptions were stale. GitHub's documented `pull_request_target` model also establishes the relevant trust distinction: the event runs in the base/default-branch context and becomes dangerous when untrusted PR code is checked out and executed, which this design deliberately avoids.

The stronger design follows the repository's established evidence discipline: bind authority to exact immutable identities, keep the trusted evaluator outside the untrusted subject it evaluates, fail closed when identity or protected-root continuity cannot be established, and keep structural evidence distinct from execution and scientific evidence.

## Impact Analysis

### Downstream Systems Affected

- [x] Pull-request governance admission
- [x] Workflow Syntax changed-file selection
- [x] Trusted-root v2 promotion semantics
- [x] Future required-status-check authority source
- [x] Class A path migration/deletion/rename handling
- [ ] Product runtime behavior
- [ ] Scientific model behavior
- [ ] Optimizer execution authority

The change affects repository governance only. It does not alter Symthaea product code, model parameters, scientific preregistration subjects, Cargo/Nix inputs, or optimizer authority.

### Risk Register Impact

No existing product-runtime risk entry is changed by this governance-only patch. The change reduces repository-process risk: a Class A detection can no longer silently produce a green governance status; changed-file classification no longer depends on a moving branch ref; protected files cannot escape classification merely by deletion or rename; and the staged authoritative root does not execute PR-head governance code.

Class B remains intentionally deferred because `GOVERNANCE_CHARTER.md` §3.3 and `docs/compliance/adr/README.md` disagree about whether every Class B change requires an ADR. That policy conflict is tracked separately as #5363 and must be reconciled before Class B is promoted to fail-closed merge evidence.

The Class A path set is itself a versioned policy. Presence checks prove the declared v1 roots exist across the event base/head boundary; they do **not** prove that every safety-sensitive code path in the repository has been exhaustively classified. Expanding that inventory remains a governance-review responsibility rather than a claim made by this structural checker.

## Test Evidence

- [x] The Python validator has embedded positive/negative `--self-test` coverage for path classification, authority-specific commit prefixes, required ADR structure, template-placeholder rejection, policy self-root removal, and delete/rename path parsing.
- [x] The policy JSON is validated against hard-coded minimum protected roots, current ADR roots, explicit Class B deferral, and expected commit-prefix semantics.
- [x] Current-tree audit verified the v1 safety roots use the current repository layout rather than the stale legacy path prefixes.
- [x] Workflow Syntax selects PR workflow/script changes using exact event base/head SHAs rather than a mutable branch name.
- [x] The base-controlled workflow is structurally read-only, restricted to PRs targeting `main`, and contains no PR-head checkout or PR-head execution step.
- [ ] The new `pull_request_target` Governance Root cannot behaviorally execute from this introduction PR because it is not yet present on trusted `main`; qualification must occur only after merge into the base/default lineage.
- [ ] Hosted-runner execution remains blocked by #4276 at the time of this ADR.
- [ ] Full CI success is explicitly not established by this ADR or structural validator.

The change does not affect the cognitive loop, so a 100+ cycle cognitive-loop soak test is not applicable to this governance-only patch.

## Rollback Plan

Revert the single governance-hardening commit. That restores #5325's execution-identity-only workflow state without modifying #5325 itself, the frozen DE scientific lineage, product code, or trusted-root repository settings. If a false positive is found in the new validator or root workflow, keep the PR draft and repair the child subject rather than weakening or bypassing the check on `main`.

## Consequences

Positive consequences: immutable change-set identity, fail-closed Class A structural policy, current-root continuity checks, delete/rename governance, self-protection of the governance surface, a `main`-controlled read-only merge-authority path, stronger evidence for any future required Governance context, and clearer separation between structural compliance and execution/scientific evidence.

Negative consequences: Class A PRs now require an ADR and approved commit prefix before the strict governance gate can pass; the stronger root requires a two-stage activation sequence because it cannot govern the PR that first introduces it; hosted-runner qualification remains pending; the finite Class A path inventory still requires periodic human review; and Class B remains advisory until #5363 resolves the documentation conflict.
