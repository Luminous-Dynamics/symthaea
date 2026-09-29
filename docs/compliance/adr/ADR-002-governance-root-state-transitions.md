# ADR-002: Adversarial Governance Root State-Transition Coverage

**Date**: 2026-09-29
**Status**: Proposed
**Change Class**: A (Safety-Critical)
**Authors**: Tristan Stoltz, Claude (AI pair)

## Context

The Governance Root validator evaluates an exact event base/head changeset and protects a finite, versioned set of Class A roots. Structural protection must remain fail-closed across repository state transitions, not only ordinary modified-file records. Protected roots can be deleted, renamed, copied, replaced, or introduced on only one side of the event boundary. A Class A changeset can also lose its required ADR, and mixed Safety/Governance authority can be represented with an incorrectly scoped commit prefix.

## Decision

Extend the governance validator self-test with an explicit adversarial state-transition matrix. The matrix tests the protected-root existence invariant for base-only, head-only, both-sides, and neither-side states; verifies that Class A changes without a changed ADR fail closed; and retains negative mixed-authority prefix cases. The production validator continues to evaluate the actual event base/head Git objects rather than synthetic state.

The protected-root invariant remains: every required Class A root must exist in at least one of the exact event base or exact event head commits. Exact changeset parsing separately retains both sides of rename/copy records, while ancestry validation requires the head to descend from the base.

## Scientific Basis

The basis for this change is adversarial software assurance rather than a claim about Symthaea's scientific models. The relevant correctness property is that governance controls must not silently disappear when repository objects transition between states. Explicit state coverage makes the structural invariant auditable and regression-testable.

## Impact Analysis

This change strengthens only the structural governance validator and its deterministic self-tests. It does not change cognitive thresholds, ethics behavior, safety-agent runtime behavior, scientific/product lineage, or merge authorization.

### Risk Register Impact

The change addresses governance-integrity risk associated with protected-root deletion, replacement, and authority-boundary regression. It does not establish that all safety-sensitive code is classified, nor does it resolve the deferred Class B policy.

## Test Evidence

The validator self-test exercises protected-root base-only, head-only, both-side, and neither-side states; Class A changes with and without a changed ADR; mixed Safety/Governance prefix acceptance and rejection; protected-root deletion, rename, and copy path representations; forbidden privileged-workflow patterns; and unpinned actions/checkout rejection.

Execution of these tests in GitHub Actions remains a separate evidence layer until an eligible hosted runner executes the workflow.

## Rollback Plan

Revert the validator and this ADR together. The previous exact base/head validator remains the fallback structural governance implementation. Do not activate or remove the Governance Root based solely on this ADR; Actions event-policy qualification, runner qualification, and independent review remain separate gates.