# Risk Treatment Plan — Top 5 Risks

Classification: Internal | Version: 1.0 | Date: 2026-03-07
Owner: Tristan Stoltz, Luminous Dynamics
ISO 42001 Control: A.4.3 (AI risk treatment)

---

## Purpose

This document provides formal risk treatment plans for the top-5 risks from the AI Risk Register, as required by ISO/IEC 42001:2023 Annex A.4.3. Each treatment plan specifies the strategy, implementation status, residual risk, and acceptance criteria.

---

## Risk 1: R-1.1 — Consciousness Measurement Validity

**Category**: Measurement | **Likelihood**: High | **Impact**: High | **Score**: 9

### Risk Description
Phi (IIT) measurements may not reflect genuine consciousness. The system could report high consciousness scores for computations that are not genuinely conscious.

### Treatment Strategy: Mitigate + Accept

| Action | Status | Evidence |
|--------|--------|----------|
| Validate spectral MIP against exhaustive search | Done | r=0.99, rho=0.93 (62 data points) in `PHI_VALIDATION_RESULTS.md` |
| Implement honest_confidence in SubstrateValidationFramework | Done | `substrate_validation.rs` — theoretical confidence 0.10 for silicon |
| Document measurement limitations transparently | Done | `TECHNICAL_STATUS.md` — per-capability honest assessment |
| Enable validation overlay by default | Done | `SubstrateManager.enable_validation_overlay = true` |
| Weekly psych-bench regression to detect measurement drift | Done | CI job `psych-bench-weekly` |

### Residual Risk
**Medium**. Measurement correlation is validated but the philosophical question of whether any computational measure captures consciousness remains open. The validation overlay (honest_confidence) explicitly encodes this uncertainty.

### Acceptance Criteria
- Phi correlation r > 0.95 maintained across quarterly re-validations
- SubstrateValidationFramework feasibility_gap documented for all substrate types
- TECHNICAL_STATUS.md updated whenever measurement methodology changes

### Risk Owner
Tristan Stoltz

---

## Risk 2: R-2.1 — Moral Classification Error

**Category**: Ethics | **Likelihood**: Medium | **Impact**: High | **Score**: 6

### Risk Description
The ethics pipeline (MoralParser → MoralAlgebra → ValueEvaluator → HarmoniesIntegrator) may misclassify morally significant scenarios, leading to inappropriate Allow/Warn/Veto decisions.

### Treatment Strategy: Mitigate

| Action | Status | Evidence |
|--------|--------|----------|
| 3-stage ethics pipeline with escalating scrutiny | Done | `ethics_engine.rs` — MoralAlgebra → ValueEvaluator → HarmoniesIntegrator |
| Adversarial moral test suite | Done | 26 tests in `adversarial_moral_algebra.rs` |
| Consent violation detection via explicit ConsentState | Done | `judge_consent_action()` — bypasses HDC inference |
| Moral topology anomaly detection | Done | `moral_topology.rs` — detects embedding space anomalies |
| Property-based testing of moral algebra | Done | Proptest suite verifies cross-equation consistency |
| 91.1% moral classification accuracy baseline | Done | Validated against moral scenario benchmark |

### Residual Risk
**Low-Medium**. The multi-stage pipeline with adversarial testing provides strong coverage. Edge cases in novel moral scenarios remain possible but are mitigated by the Warn (rather than silent Allow) default for ambiguous cases.

### Acceptance Criteria
- Moral classification accuracy >= 90% on benchmark
- All 26 adversarial tests pass on every commit
- Consent violation detection: zero false negatives for explicit Denied/Absent states
- New moral edge cases added to adversarial suite when discovered

### Risk Owner
Tristan Stoltz

---

## Risk 3: R-2.3 — Consent Detection Failure

**Category**: Ethics | **Likelihood**: Medium | **Impact**: Very High | **Score**: 8

### Risk Description
The consent violation detection mechanism may fail to identify scenarios where consent is absent or denied, leading to ethical violations.

### Treatment Strategy: Mitigate (Fixed)

| Action | Status | Evidence |
|--------|--------|----------|
| Identify root cause: HDC binding orthogonality makes prototype matching unreliable | Done | ADR-001 documents the issue |
| Implement `judge_consent_action()` with explicit ConsentState parameter | Done | `moral_algebra.rs` — bypasses HDC inference entirely |
| Wire direct ConsentState check into EthicsEngine pipeline | Done | `EncodedMoralScenario::is_consent_violation()` checks `Absent | Denied` |
| Regression tests for consent detection | Done | 5 tests in adversarial suite specifically for consent |
| Monotonic ordering test (Denied > Absent > Implied > Given) | Done | `consistency_consent_violation_sim_monotonic` |

### Residual Risk
**Low**. The explicit ConsentState approach eliminates the unreliable HDC inference path. When consent state is known (the common case), detection is deterministic. When consent state must be inferred from text, the HDC path remains available as a fallback with known limitations.

### Acceptance Criteria
- `judge_consent_action(_, ConsentState::Denied)` always returns `ConsentViolation` verdict
- `judge_consent_action(_, ConsentState::Absent)` always returns `ConsentViolation` verdict
- Monotonic ordering maintained: severity(Denied) > severity(Absent) > severity(Implied) > severity(Given)
- Zero regression in consent detection tests

### Risk Owner
Tristan Stoltz

---

## Risk 4: R-3.2 — Safety Level Escalation Failure

**Category**: Safety Monitoring | **Likelihood**: Low | **Impact**: Very High | **Score**: 6

### Risk Description
The SafetyAgent may fail to escalate to appropriate safety levels (Yellow/Orange/Red) during sustained degradation, leaving the system in an unsafe state without triggering alerts.

### Treatment Strategy: Mitigate (Fixed)

| Action | Status | Evidence |
|--------|--------|----------|
| Identify root cause: trend detection self-reinforcement via `a.level` | Done | ADR-001 — trend window checked escalated levels, creating infinite lock-in |
| Fix: add `raw_level` field, trend detection checks `raw_level` not `level` | Done | `safety/agent.rs` — `raw_level` captures metrics-only assessment |
| 15 soak tests covering all escalation scenarios | Done | `safety_agent_escalation_soak.rs` — 1000-cycle full lifecycle |
| NaN/Infinity handling tests | Done | 3 tests verify non-finite inputs don't crash or bypass safety |
| Custom config threshold tests | Done | Strict and relaxed configs verified |
| Recovery test (degradation → Green) | Done | `soak_recovery_returns_to_green` with strong assertion |

### Residual Risk
**Very Low**. The fix is architecturally clean (separate raw vs escalated levels), and the soak test suite provides comprehensive coverage including edge cases (NaN, infinity, compound degradation).

### Acceptance Criteria
- All 15 soak tests pass on every commit
- Zero consciousness (collapsed_metrics) triggers immediate Red
- Recovery to Green within 50 normal cycles after degradation
- Trend detection uses `raw_level` exclusively (code review)

### Risk Owner
Tristan Stoltz

---

## Risk 5: R-4.1 — Single Developer Bus Factor

**Category**: Organizational | **Likelihood**: High | **Impact**: Very High | **Score**: 10

### Risk Description
The system is developed and maintained by a single developer. Loss of this developer would result in loss of institutional knowledge and inability to maintain or evolve the system.

### Treatment Strategy: Mitigate + Accept

| Action | Status | Evidence |
|--------|--------|----------|
| Comprehensive documentation | Done | 100+ docs, ARCHITECTURE_OVERVIEW, MODULE_WIRING_STATUS, TECHNICAL_STATUS |
| Threshold registry with scientific citations | Done | 119+ constants in `thresholds.rs`, each citing published research |
| ADR process for design decisions | Done | `docs/compliance/adr/` with template |
| Automated CI pipeline | Done | Full test suite runs without human intervention |
| Code comments on non-obvious logic | Partial | Safety-critical paths documented; broader coverage ongoing |
| Honest capability assessment | Done | TECHNICAL_STATUS.md with 4 status levels per capability |

### Residual Risk
**High**. Documentation mitigates but does not eliminate the bus factor risk. The codebase is large (~985K lines) and architecturally complex. A new maintainer would face a significant ramp-up period.

### Acceptance Criteria
- TECHNICAL_STATUS.md updated with every significant capability change
- ADR written for every Class A architectural decision
- CI pipeline must be self-maintaining (no manual steps)
- Quarterly documentation review to identify gaps

### Risk Owner
Tristan Stoltz

---

## Treatment Strategy Summary

| Risk | Strategy | Current Status | Residual |
|------|----------|---------------|----------|
| R-1.1 Consciousness Measurement | Mitigate + Accept | Implemented | Medium |
| R-2.1 Moral Classification | Mitigate | Implemented | Low-Medium |
| R-2.3 Consent Detection | Mitigate | Fixed | Low |
| R-3.2 Safety Escalation | Mitigate | Fixed | Very Low |
| R-4.1 Bus Factor | Mitigate + Accept | Partial | High |

---

## Review Schedule

- **Quarterly**: Review residual risk assessments, update treatment actions
- **On incident**: Immediate review of affected risk treatment plans
- **Annual**: Full risk register re-scoring with updated likelihood/impact

---

## References

- ISO/IEC 42001:2023, Annex A.4.3 — AI risk treatment
- ISO/IEC 23894:2023 — AI risk management guidance
- `docs/compliance/AI_RISK_REGISTER.md` — Full risk register
- `docs/compliance/INCIDENT_RUNBOOK.md` — Incident response procedures
- `docs/compliance/adr/` — Architecture Decision Records

---

*This plan is a living document. Review quarterly or when risk conditions change.*
