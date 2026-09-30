# ADR-006: Protect privileged-workflow governance surfaces with the Governance Root

**Date**: 2026-09-30  
**Status**: Proposed  
**Change Class**: A (Governance-Critical)

## Context

The privileged-workflow inventory and its fail-closed detector are repository-side enforcement mechanisms for elevated GitHub Actions surfaces. They must not become an independent source of authority.

The intended authority direction is:

**trusted Governance Root → protects detector/inventory contract → detector validates privileged workflow surfaces → inventory records expected privileged surfaces**

GitHub documents `pull_request_target` and `workflow_run` as privileged workflow contexts and warns that untrusted pull-request content must not be checked out and executed in those contexts. The repository therefore needs a stronger authority layer around the detector itself.

## Decision

The Governance Root Class A protected-root contract is extended to include:

- `.github/scripts/check-privileged-workflow-inventory.py`
- `.github/governance-privileged-workflow-inventory-v1.json`
- `docs/compliance/adr/ADR-005-privileged-workflow-inventory.md`
- `.github/workflows/workflow-syntax.yml`

The existing Governance Root workflow executes the detector only from the trusted event-base repository state. It verifies that the protected privileged-workflow surfaces exist there, runs the detector self-test and inventory validation, and records that the privileged inventory authority comes from the trusted main base.

The governance validator also treats deletion, rename, copy, and other protected-root transitions as changes to the protected authority surface. Its self-test explicitly removes each protected surface from a synthetic policy and requires fail-closed rejection.

## Scientific Basis

This ADR governs CI authority separation rather than scientific behavior. No cognitive, threshold, ethics-engine, consciousness, or product-scientific semantics are changed.

The security basis is GitHub's documented trust model for privileged Actions events: `pull_request_target` runs with elevated repository-token/secrets authority, while `workflow_run` can also execute in a privileged context. GitHub recommends avoiding execution of untrusted pull-request content in these contexts.

## Impact Analysis

The change makes the Governance Root the stronger authority for the privileged-workflow detector/inventory contract. A pull request cannot rely on a modified detector or inventory from its own head to establish that those modifications are safe.

The Workflow Syntax workflow remains an enforcement surface and is itself protected by the Governance Root Class A contract.

This does not authorize GitHub's external Actions event policy and does not establish hosted-runner qualification.

### Risk Register Impact

Reduced risk:

- detector deletion or weakening;
- inventory omission;
- joint detector/inventory weakening;
- weakening of Workflow Syntax enforcement;
- Governance Root and privileged-workflow enforcement being changed together without structural Class A scrutiny.

Remaining risks:

- external GitHub Actions event-policy configuration;
- runtime workflow behavior;
- hosted-runner qualification;
- artifact/cache runtime safety;
- scientific adequacy and full CI success.

## Test Evidence

The embedded Governance Root validator self-test covers:

- protected policy-root removal;
- detector/inventory/ADR/Workflow Syntax protected-root removal;
- protected-root deletion/rename/copy state transitions;
- exact base/head ancestry;
- object-graph integrity;
- trusted-base Governance Root workflow requirements;
- forbidden untrusted checkout/download patterns.

The privileged-workflow detector has its own deterministic adversarial self-tests for privileged trigger discovery, inventory coverage, runtime provenance, artifact boundaries, cache authority, and reusable-workflow cache caps.

These are structural self-tests; their successful execution in GitHub Actions is not established by this ADR.

## Rollback Plan

Revert the Governance Root protection commits as a coordinated governance change. Do not independently remove the protected detector/inventory surfaces or weaken their inventory contract.

If the protection introduces an operational incompatibility, preserve the detector and inventory files while correcting the Governance Root contract through a new Class A governance change.

## Nonclaims

This ADR does not establish:

- GitHub external Actions event-policy authorization;
- hosted-runner availability or qualification;
- workflow execution success;
- full CI success;
- artifact or cache runtime safety;
- scientific adequacy;
- merge authorization;
- any change to frozen scientific/product lineage #4813.
