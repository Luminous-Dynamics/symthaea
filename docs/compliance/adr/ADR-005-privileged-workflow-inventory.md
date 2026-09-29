# ADR-005: Privileged Workflow and Event Inventory

**Date:** 2026-09-29  
**Status:** Proposed  
**Change Class:** A — Safety-Critical Governance

## Context

GitHub Actions has multiple trust boundaries that cannot be reduced to token permissions alone. In particular, pull_request_target executes with a privileged context, while workflow_run can start a workflow with access to secrets and write-capable tokens even when the upstream workflow does not have those privileges.

The repository therefore needs a machine-readable inventory that distinguishes trigger trust from permission surface and from untrusted-code execution.

## Decision

Maintain `.github/governance-privileged-workflow-inventory-v1.json` as a repository-side declaration of currently known privileged workflow/event surfaces on main.

The inventory records:
- exact workflow path;
- trigger and activity types;
- trust classification;
- top-level and job-level token permissions;
- secret consumption;
- untrusted pull-request checkout/execution;
- third-party action pins;
- trust guards and purpose;
- migration/retirement target where applicable;
- external event-policy attestation requirements.

The inventory is **not authoritative by itself**. A future validator must discover privileged trigger usage from the trusted base tree and require exact inventory coverage. An undocumented privileged trigger must fail closed.

## Security properties

The inventory explicitly separates:
1. trigger trust boundary;
2. token permission surface;
3. secret consumption;
4. untrusted-code checkout;
5. untrusted-code execution.

This prevents a workflow with ordinary write permissions, such as a trusted deployment workflow, from being incorrectly classified solely because it has a write permission.

## External policy boundary

GitHub's effective Actions event policy is an external control-plane fact. Repository source cannot prove that an applicable enterprise, organization, or repository policy permits a privileged event.

Accordingly, this ADR records the requirement for independent policy attestation rather than attempting to infer authorization from repository code.

## Relationship to existing governance work

- #6493 remains the external pull_request_target policy qualification boundary.
- #4606 remains the preferred migration path for retiring the transitional draft governor.
- #6559 remains the exact history-topology authority boundary.
- #6584 tracks the inventory/validator hardening tranche.

## Test evidence

The initial inventory is a declarative governance artifact. It does not claim successful workflow execution or test execution.

A later validator must test both positive coverage and fail-closed cases, including:
- undocumented pull_request_target;
- undocumented workflow_run;
- inventory entry for a missing workflow;
- trigger/event mismatch;
- permission mismatch;
- changed action pin;
- newly introduced privileged workflow.

## Rollback Plan

Remove the inventory and ADR only through the same governance process that protects other Class A governance artifacts. Existing workflow behavior must not be weakened as a rollback mechanism.

## Nonclaims

This ADR does not establish:
- effective GitHub Actions event-policy authorization;
- hosted-runner qualification;
- Governance Root execution;
- scientific adequacy;
- full CI success;
- merge authorization.
