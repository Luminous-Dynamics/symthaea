# ADR-003: Exact Commit History Topology for Governance Authority

**Date**: 2026-09-29
**Status**: Proposed
**Change Class**: A (Safety-Critical)
**Authors**: Tristan Stoltz, Claude (AI pair)

## Context

The Governance Root binds authority to the exact event base SHA and exact event head SHA. That boundary must remain meaningful when Git history contains merge commits, rewritten or force-pushed heads, divergent histories, or incomplete object availability. Git's merge-base and range mechanisms have distinct semantics, so governance must not silently substitute a merge-base-derived endpoint or a branch name for the immutable event SHAs.

## Decision

Add deterministic self-test coverage using a temporary Git repository. The topology fixture verifies that:

- a distinct descendant head is accepted;
- identical base/head SHAs are rejected;
- a merge commit descending from the exact base is accepted;
- reverse ancestry is rejected;
- a rewritten/non-descendant history is rejected;
- a shallow repository cannot establish the required ancestry relation;
- a partial clone or promisor-remote repository is rejected before topology-sensitive validation;
- configured alternate-ref commands and external object-database environment overrides are rejected;
- Git object-database environment overrides are rejected;
- replacement refs under refs/replace/ are rejected;
- the trusted Governance Root workflow contract rejects a checkout filter that could introduce a partial object graph;
- the direct endpoint diff remains the changeset authority even when the head is a merge commit.

Production validation first rejects an explicitly shallow or partial/promisor repository, alternate-ref commands, replacement refs, or Git object-database environment overrides, while the trusted workflow contract separately rejects a checkout filter; it then requires the exact 40-hex event SHAs, requires the base to be an ancestor of the head, and computes the governed path set from the exact base/head endpoints. Branch names, reflog-derived fork points, and merge-base substitutions are not authority inputs.

## Scientific Basis

The basis for this change is adversarial software assurance and Git object-model semantics, not a claim about Symthaea's scientific models. The control depends on immutable commit identity, explicit parent reachability, and a complete local object graph. Git documents merge-base --is-ancestor as the direct ancestry test; Git's partial-clone model explicitly permits missing objects and on-demand retrieval from promisor remotes, which is unsuitable as an implicit authority source for this boundary.

## Impact Analysis

This change strengthens governance-history validation and its deterministic self-test fixture. It does not modify cognitive thresholds, ethics behavior, safety-agent runtime behavior, scientific/product lineage, or merge authorization.

### Risk Register Impact

The change addresses governance-integrity risk from merge topology, rewritten heads, and endpoint ambiguity. It reduces the chance that history shape causes governance classification to evaluate a different changeset than the event supplied. It does not establish complete repository safety coverage, scientific adequacy, test execution, runner qualification, Actions event-policy qualification, or merge authorization.

## Test Evidence

The self-test now creates a temporary Git repository and exercises descendant, identical, merge, reverse, rewritten-history, shallow-history, partial-clone-extension, promisor-remote, alternate-object-environment, alternate-ref-command, and replacement-ref cases. It also checks that a merge head's exact endpoint diff contains changes introduced on both sides of the merge. GitHub Actions execution remains a separate evidence layer until an eligible hosted runner executes the workflow.

## Rollback Plan

Revert the history-topology validator test change and this ADR together. Retain the preceding exact base/head ancestry check as the fallback structural control. Do not activate the Governance Root based solely on this ADR; event-policy, runner, and independent-review gates remain separate.
