# ADR-001: Fix SafetyAgent Trend Detection Self-Reinforcement

**Date**: 2026-03-06
**Status**: Accepted
**Change Class**: A (Safety-Critical)
**Authors**: Tristan Stoltz, Claude (AI pair)

## Context

The SafetyAgent's trend detection mechanism (lines 207-218 of `src/safety/agent.rs`) checked `a.level` (the final level after trend escalation) in the history window. When a Green assessment got trend-escalated to Yellow, it was stored as Yellow in history, keeping the window permanently degraded. This created infinite Yellow lock-in — the agent could never recover to Green after any degradation period.

A safety system that can escalate but never de-escalate is fundamentally broken. Operators must be able to confirm that corrective action has succeeded, and the system must reflect genuine improvement in underlying metrics.

## Decision

Added a `raw_level: SafetyLevel` field to `SafetyAssessment` that captures the metrics-only level before trend escalation. Changed trend detection to check `a.raw_level` instead of `a.level` so trend-escalated assessments do not pollute the history window.

### Parameter Changes

| Parameter | Old Value | New Value | File |
|-----------|-----------|-----------|------|
| SafetyAssessment fields | `level` only | `level` + `raw_level` | `src/safety/agent.rs` |
| Trend window check | `a.level >= Yellow` | `a.raw_level >= Yellow` | `src/safety/agent.rs` |

### Scientific Basis

NRC (Nuclear Regulatory Commission) safety monitoring requires that safety systems can both escalate AND de-escalate. A system that can only escalate violates the recovery principle — operators must be able to confirm that corrective action has succeeded. The self-reinforcing Yellow lock-in violated this principle.

## Impact Analysis

### Downstream Systems Affected
- [x] Safety monitoring (SafetyAgent)
- [ ] Ethics evaluation (EthicsEngine)
- [ ] Consciousness scoring (ConsciousnessEquationV2)
- [ ] Governance permissions (Mycelix consciousness gating)
- [ ] Learning dynamics (FEP, CfC)
- [ ] Other: SafetyAuditReport serialization gains the `raw_level` field (additive, backward compatible)

### Risk Register Impact
- Recovery after degradation now works correctly (Green achievable within `escalation_window` cycles of normal metrics).
- Trend detection still functions for genuine sustained degradation (raw metrics that are Yellow+).
- No change to escalation behavior — only de-escalation is affected.
- Does not introduce a new risk. Reduces existing risk of permanent safety-level lock-in.

## Test Evidence

- [x] Unit tests verify new behavior
- [ ] Proptest shows stability across perturbation
- [x] Soak test (100+ cycles) if cognitive loop affected
- [x] All existing tests pass

Specific tests:

- `soak_recovery_returns_to_green` — verifies recovery to Green after 20 cycles of Orange-level degradation followed by 50 normal cycles.
- `soak_1000_cycle_full_lifecycle` — 1000-cycle test covering normal, degradation, collapse, recovery, spike, and recovery phases; verifies final state is Green.
- All 15 soak tests pass, all 28 existing safety agent unit tests pass.

## Rollback Plan

Revert the commit. The `raw_level` field is additive; removing it is safe. Deserialization of assessments that include `raw_level` will fail against old code — add `#[serde(default)]` to the field if a graceful rollback without data loss is needed.

## Consequences

**Positive:**
- The SafetyAgent can now correctly de-escalate after a degradation period ends, matching NRC recovery principles.
- Trend detection remains effective for genuine sustained degradation where raw metrics are Yellow or above.
- The separation of `raw_level` and `level` provides clearer audit trails — reviewers can see exactly what metrics indicated vs. what trend escalation applied.

**Negative:**
- `SafetyAssessment` gains one additional field, marginally increasing memory per assessment in the history window.
- Any external consumers deserializing `SafetyAssessment` must handle the new `raw_level` field (mitigated by `serde(default)` if needed).
