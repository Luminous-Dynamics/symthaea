# ADR-004: External Actions Event-Policy Attestation Boundary

**Date**: 2026-09-29
**Status**: Proposed
**Change Class**: A (Safety-Critical)
**Authors**: Tristan Stoltz, Claude (AI pair)

## Context

The Governance Root intentionally uses `pull_request_target` so the workflow definition is taken from trusted base-repository context while the pull request head is handled only as untrusted Git objects. GitHub documents that `pull_request_target` has elevated trust and that public repositories may be subject to a default event policy blocking the trigger; GitHub has announced enforcement of that default policy for affected public repositories on 2026-11-02. The repository source tree cannot, by itself, prove the effective enterprise, organization, or repository Actions execution policy currently applied by GitHub.

The governance boundary therefore has two distinct layers:

```
trusted source-level workflow contract
!=
effective external Actions event policy
```

Conflating these layers would turn a source-code inspection into an unsupported runtime authorization claim.

## Decision

Make the external event-policy dependency explicit and machine-visible in the trusted Governance Root receipt. The workflow must state:

- `actions_event_policy_requirement=pull_request_target_must_be_explicitly_allowed`;
- `actions_event_policy_attestation=not_established_by_repository_code`;
- `contents: read` remains the only declared token permission;
- no repository secrets are consumed by the Governance Root;
- activation remains blocked as an operational matter until the applicable GitHub Actions policy is independently inspected and recorded.

The structural validator treats those receipt fields as part of the trusted workflow contract. Removing either field therefore becomes a governed change rather than a silent weakening of the evidence boundary.

This ADR does not grant, infer, or simulate the external Actions permission. The actual policy must be verified through GitHub's Actions policy controls or API by an appropriately authorized administrator.

## Scientific Basis

The basis is software supply-chain and authorization-boundary analysis. GitHub's documented security model distinguishes the trusted base workflow of `pull_request_target` from the elevated event trust it receives. The repository can verify the source-level invariants it controls, but the effective event allowlist is an external control-plane property.

## Impact Analysis

This change affects governance evidence and the declaration of the Governance Root's external execution dependency. It does not modify cognitive thresholds, ethics behavior, safety-agent runtime behavior, scientific/product lineage, or merge authorization.

### Risk Register Impact

The change addresses the risk of treating a source-level workflow contract as proof that GitHub will permit the privileged event to execute. It preserves a hard boundary between repository-controlled evidence and platform-controlled policy.

### External Execution Policy

The required external attestation is: inspect the effective Actions workflow-execution policy at the applicable repository, organization, and enterprise scopes; determine whether `pull_request_target` is permitted for `.github/workflows/pr-governance-root.yml`; record the policy scope and enforcement mode; and do not activate Governance Root as required merge authority until that evidence exists.

The repository validator intentionally reports this layer as not established by repository code.

## Test Evidence

The structural validator now requires the two explicit event-policy boundary fields in the trusted Governance Root workflow. Its self-test exercises the trusted-workflow contract and continues to reject forbidden untrusted-checkout patterns and unpinned action references. This change does not establish that GitHub's external policy permits the event, and no hosted workflow execution is claimed by this ADR.

## Rollback Plan

Revert the workflow receipt-field change, validator contract extension, policy nonclaim, and this ADR together. Retain the earlier source-level `pull_request_target` hardening and exact base/head controls. Do not activate Governance Root based on repository source alone.
