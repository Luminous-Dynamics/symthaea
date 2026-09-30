# ADR-005: Privileged Workflow and Event Inventory

**Date:** 2026-09-29  
**Status:** Proposed  
**Change Class:** A — Safety-Critical Governance

## Context

GitHub Actions has multiple trust boundaries that cannot be reduced to token permissions alone. In particular, `pull_request_target` executes with a privileged context, while `workflow_run` can start a workflow with access to secrets and write-capable tokens even when the upstream workflow does not have those privileges.

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
- external event-policy attestation requirements;
- cross-workflow data-flow contract for `workflow_run` consumers, including upstream workflow identity, artifact consumption, artifact handling, cache influence, and privileged side effects.

The inventory is **not authoritative by itself**. `.github/scripts/check-privileged-workflow-inventory.py` now discovers the repository's supported privileged trigger surfaces from the workflow tree and requires exact inventory coverage for the mechanically observable contract. An undocumented privileged trigger, stale inventory entry, trigger mismatch, permission mismatch, or action-pin mismatch fails closed.

The detector intentionally scopes v1 to `pull_request_target` and `workflow_run`, the two elevated trust boundaries currently declared by this governance tranche. It does not classify every workflow with write permissions as privileged. Ambiguous YAML constructs in the inspected trigger/permission surfaces fail closed rather than being guessed.

## Security properties

The inventory and detector explicitly separate:
1. trigger trust boundary;
2. token permission surface;
3. secret consumption;
4. untrusted-code checkout;
5. untrusted-code execution.

This prevents a workflow with ordinary write permissions, such as a trusted deployment workflow, from being incorrectly classified solely because it has a write permission.

The detector compares mechanically observable fields: trigger/event configuration, workflow-run upstream workflow names, declared permissions, pinned third-party action references, artifact-download surfaces, cache authority, and local reusable-workflow calls. For privileged workflows, an explicit cache-mode disposition is recorded; write-capable `cache-mode` values are treated as an explicit trust-boundary override rather than inferred away. A privileged workflow that calls a local reusable workflow without an explicit cache cap fails closed, because GitHub permits cache authority to propagate from the caller and otherwise allows the called workflow to request write access. For `workflow_run`, the inventory must additionally declare a cross-workflow data-flow contract and the workflow source must explicitly bind runtime provenance: repository identity, configured workflow name, main-branch identity, expected subject SHA, conclusion, and run identity (`id` + `run_attempt`). If artifacts are consumed, the privileged consumer must select the triggering `workflow_run.id`, use an exact artifact `name`, and inventory that exact observed name; broad `pattern` and artifact-ID selection are rejected in v1. Artifact extraction and execution remain explicit inventory fields, with execution required to remain false. Artifact extraction is restricted to an explicit path under `runner.temp` (or `$RUNNER_TEMP`) rather than the workspace. The detector also inspects privileged consumer commands for artifact-to-code transitions: shell/interpreter invocation, `source`, compilation/build/package-manager execution, `chmod`/executable preparation, or copying from the isolated artifact path into the workspace, `PATH`, or system executable directories fails closed. It additionally tracks simple shell/environment indirection from the isolated artifact path into execution sinks such as `bash "$ARTIFACT_DIR/..."`, `source "$ARTIFACT_DIR/..."`, and `find ... -exec` so a consumer cannot evade the direct path patterns merely by assigning the artifact directory to a variable. Data-only reads such as `cat` remain structurally admissible. This is a static boundary check; it does not prove that arbitrary runtime interpretation cannot occur through an unrecognized command, variable transformation, composite action, or reusable workflow. The detector fails closed when that contract is absent, inconsistent with the observed trigger/action surface, or when the privileged consumer lacks these explicit runtime guards. Human-purpose, guard, secret-consumption, and untrusted-code claims remain declarative evidence and are not silently inferred by the detector.

## External policy boundary

GitHub's effective Actions event policy is an external control-plane fact. Repository source cannot prove that an applicable enterprise, organization, or repository policy permits a privileged event.

Accordingly, this ADR records the requirement for independent policy attestation rather than attempting to infer authorization from repository code.

## Fail-closed validation

The detector has a deterministic self-test covering:
- valid `workflow_run` inventory;
- trigger/activity mismatch;
- stale inventory after privileged-trigger removal;
- missing cross-workflow data-flow contract;
- false artifact-consumption declaration;
- missing upstream workflow identity in the data-flow contract;
- missing runtime provenance guards in a privileged `workflow_run` consumer;
- artifact execution/interpretation, executable preparation, privileged workspace/PATH copy attempts, and simple variable-indirection execution attempts after artifact download;
- explicit `cache-mode: write` detection;
- privileged reusable-workflow calls without an explicit cache cap;
- accepted explicit read-only cache caps on reusable-workflow calls.

The Workflow Syntax gate compiles changed `.github/scripts/*.py`, executes the detector self-test, validates the current privileged inventory, and JSON-parses changed `.github/*.json` files.

This is structural validation only. It does not prove that a GitHub Actions workflow actually executed successfully.

## Relationship to existing governance work

- #6493 remains the external pull_request_target policy qualification boundary.
- #4606 remains the preferred migration path for retiring the transitional draft governor.
- #6559 remains the exact history-topology authority boundary.
- #6584 tracks the inventory/validator hardening tranche.
- #6586 introduces the initial privileged-workflow inventory.
- #6604 tracks cache-write override hardening.
- This follow-on adds independent repository-side detection and fail-closed coverage checking.

## Test Evidence

The detector's self-test is embedded in `.github/scripts/check-privileged-workflow-inventory.py`. Workflow Syntax also runs `py_compile` on changed Python helpers.

No GitHub Actions execution is claimed by this ADR unless a corresponding run receipt independently establishes it.

## Rollback Plan

Remove the detector and inventory only through the same governance process that protects other Class A governance artifacts. Existing workflow behavior must not be weakened as a rollback mechanism.

## Nonclaims

This ADR does not establish:
- effective GitHub Actions event-policy authorization;
- hosted-runner qualification;
- Governance Root execution;
- scientific adequacy;
- full CI success;
- merge authorization.
