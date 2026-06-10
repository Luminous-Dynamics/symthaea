# ADR-NNN: [Title]

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded
**Change Class**: A (Safety-Critical) | B (Consciousness-Affecting) | C (Behavioral) | D (Non-Behavioral)

## Context

What is the issue or decision that needs to be made?

## Decision

What is the change being made?

### Parameter Changes

| Parameter | Old Value | New Value | File |
|-----------|-----------|-----------|------|
| EXAMPLE_THRESHOLD | 0.3 | 0.25 | `thresholds.rs` |

### Scientific Basis

Citation(s) justifying the change.

## Impact Analysis

### Downstream Systems Affected
- [ ] Safety monitoring (SafetyAgent)
- [ ] Ethics evaluation (EthicsEngine)
- [ ] Consciousness scoring (ConsciousnessEquationV2)
- [ ] Governance permissions (Mycelix consciousness gating)
- [ ] Learning dynamics (FEP, CfC)
- [ ] Other: ___

### Risk Register Impact
- Does this change affect any risk in AI_RISK_REGISTER.md? Which ones?
- Does this introduce a new risk?

## Test Evidence

- [ ] Unit tests verify new behavior
- [ ] Proptest shows stability across perturbation
- [ ] Soak test (100+ cycles) if cognitive loop affected
- [ ] All existing tests pass

## Rollback Plan

How to revert this change if problems are discovered.

## Consequences

What are the positive and negative consequences of this decision?
