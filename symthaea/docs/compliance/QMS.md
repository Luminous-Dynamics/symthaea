# Quality Management System (QMS) Documentation

Classification: Internal | Version: 1.0 | Date: 2026-03-07
Owner: Tristan Stoltz, Luminous Dynamics
ISO 42001 Controls: A.3.3, A.4.5, A.6.3

---

## Purpose

This document formalizes Symthaea's quality management practices for ISO/IEC 42001:2023 compliance. It describes quality objectives, metrics, acceptance criteria, and continuous improvement processes.

---

## 1. Quality Objectives

| Objective | Target | Measurement | Review Cycle |
|-----------|--------|-------------|-------------|
| Safety monitoring coverage | 100% of cycles assessed | SafetyAgent runs every cycle | Continuous |
| Moral classification accuracy | >= 90% | Psych-bench moral scenario benchmark | Quarterly |
| Phi measurement correlation | r >= 0.95 | Spectral MIP vs exhaustive validation | Quarterly |
| Test suite pass rate | 100% (0 regressions) | CI pipeline: `cargo test --lib` | Per-commit |
| Soak test pass rate | 100% (15/15) | Safety soak test suite | Per-commit |
| Adversarial test pass rate | 100% (26/26) | Adversarial moral algebra suite | Per-commit |
| Documentation currency | All docs updated within 30 days of relevant change | Manual review | Quarterly |

---

## 2. Quality Gates

### 2.1 Pre-Commit Gate

All commits must pass:
- `cargo fmt --check` — code formatting
- `cargo clippy` — lint analysis
- Class A commit-msg hook (safety-critical file changes require `safety:` prefix)

### 2.2 CI Gate

The CI pipeline (`symthaea-ci.yml`) enforces:
- Full unit test suite (~4,100+ tests)
- 39-combination feature matrix (ensures feature flag compatibility)
- 45 sub-crate builds
- Documentation generation (`cargo doc`)

### 2.3 Release Gate

Before any release:
1. All CI checks green
2. Safety soak tests (15) pass
3. Adversarial moral tests (26) pass
4. Substrate validation tests (37) pass
5. Compliance dashboard passes (12+ suites)
6. TECHNICAL_STATUS.md reviewed and updated
7. COMPLIANCE_MATRIX.md reviewed

---

## 3. Quality Metrics

### 3.1 Code Quality

| Metric | Current | Target | Tool |
|--------|---------|--------|------|
| Clippy warnings | 0 | 0 | `cargo clippy` |
| Test count (lib) | 4,100+ | Growing | `cargo test --lib` |
| Test count (workspace) | 12,000+ | Growing | Full workspace test |
| Feature combinations tested | 39 | All critical paths | CI matrix |
| Sub-crate coverage | 45/45 | 100% | CI sub-crate job |

### 3.2 Safety Quality

| Metric | Current | Target | Tool |
|--------|---------|--------|------|
| Safety soak tests | 15/15 pass | 100% | `safety_agent_escalation_soak` |
| Adversarial moral tests | 26/26 pass | 100% | `adversarial_moral_algebra` |
| Safety agent unit tests | 33/33 pass | 100% | `safety::agent` |
| Audit report tests | 9/9 pass | 100% | `safety::audit` |
| NaN/Infinity handling | 3 tests | Comprehensive | Safety soak suite |

### 3.3 Behavioral Quality

| Metric | Current | Threshold | Tool |
|--------|---------|-----------|------|
| Moral accuracy | 91.1% | >= 90% | Moral scenario benchmark |
| Phi correlation (MIP search) | r=0.99 | >= 0.95 | `test_spectral_mip_validation` |
| Proptest stability | 29 cases pass | All finite | `proptest_feedback_stability` |
| Threshold sensitivity | 18 cases pass | All bounded | `proptest_threshold_sensitivity` |

---

## 4. Non-Conformance Management

### 4.1 Detection

Non-conformances are detected through:
- CI pipeline failures (automated)
- Safety soak test failures (automated)
- CalibrationHistory drift detection (automated, >75% same-direction)
- SelfAssessmentMonitor alerts (automated, drift >1 sigma)
- Manual review (quarterly)

### 4.2 Classification

| Severity | Definition | Response Time |
|----------|-----------|---------------|
| Critical | Safety Red level, moral classification failure, data corruption | Immediate (SEV-1) |
| Major | Safety Orange sustained, test regression, feature interaction bug | 24 hours (SEV-2) |
| Minor | Safety Yellow sustained, documentation gap, non-critical test flake | 1 week (SEV-3) |
| Informational | Code quality warning, performance regression, cosmetic issue | Quarterly review (SEV-4) |

### 4.3 Resolution

1. **Root cause analysis**: Required for Critical and Major
2. **ADR**: Required for design-level changes to fix Critical issues
3. **Regression test**: Required for all fixes — prevents recurrence
4. **Documentation update**: TECHNICAL_STATUS.md, risk register, or compliance matrix as applicable

---

## 5. Continuous Improvement

### 5.1 Automated Feedback Loops

| Loop | Trigger | Action |
|------|---------|--------|
| CalibrationHistory | Systematic drift detected | Flag for review, auto-recalibrate if within bounds |
| SelfAssessmentMonitor | EMA deviates >1 sigma | Queue recalibration, log event |
| SafetyAgent | Level escalation | Log assessment, generate audit trail |
| Proptest | Property violation found | Add regression case to test suite |

### 5.2 Manual Review Cycle

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Compliance matrix review | Quarterly | Tristan Stoltz |
| Risk register update | Quarterly | Tristan Stoltz |
| Psych-bench regression | Weekly (CI) + Quarterly (manual review) | CI / Tristan Stoltz |
| Test suite audit | Quarterly | Tristan Stoltz |
| Documentation gap analysis | Quarterly | Tristan Stoltz |

---

## 6. Records

All quality records are maintained in:
- **Git history**: Complete change log, commit messages with scope
- **CI logs**: Test results per commit
- **SafetyAuditReport**: Per-session safety assessment history
- **CalibrationHistory**: Parameter drift tracking
- **ADR directory**: Architecture decisions with rationale

---

## References

- ISO/IEC 42001:2023 — AI Management System
- ISO 9001:2015 — Quality Management Systems (informative)
- `docs/compliance/SDLC.md` — Development lifecycle
- `docs/compliance/INCIDENT_RUNBOOK.md` — Incident response
- `docs/compliance/AI_RISK_REGISTER.md` — Risk register

---

*This document is reviewed quarterly or when significant quality events occur.*
