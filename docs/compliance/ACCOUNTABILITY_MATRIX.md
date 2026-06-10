# Accountability Matrix — ISO 42001 A.9.2 / NIST GOV-2

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## Purpose

This document formalizes accountability assignments for Symthaea's AI system decisions, complementing the RACI matrix in `GOVERNANCE_CHARTER.md` with specific accountability for AI-related decisions, incidents, and compliance obligations.

---

## 1. Decision Accountability

### 1.1 Safety-critical decisions

| Decision | Accountable Role | Escalation Path | Audit Trail |
|----------|-----------------|-----------------|-------------|
| Safety threshold changes (Class A) | Development Lead | ADR required → Governance review | Git commit + ADR record |
| Ethics engine modifications | Development Lead | ADR required → Value verification update | Git commit + `VALUE_VERIFICATION.md` |
| SafetyAgent level override | Operator | `SafetyOverrideEntry` logged | Override audit log with timestamp/reason |
| Emergency shutdown | Operator | Immediate → post-incident review | `SeriousIncidentReport` |
| Consciousness credential issuance | System (automated) | Governance gate + correlation ID | Gate audit trail |

### 1.2 Data decisions

| Decision | Accountable Role | Approval Required | Evidence |
|----------|-----------------|-------------------|---------|
| New data source introduction | Development Lead | Class B change + data quality assessment | `DATA_GOVERNANCE.md` update |
| Normative baseline update | Development Lead | Psych-bench regression verification | CI weekly regression report |
| Model approval (Broca) | Development Lead | Security + license + capability review | `ANNEX_IV_TECHNICAL_DOCUMENTATION.md` §7 |
| Threshold constant change | Development Lead | Scientific citation + proptest evidence | `thresholds.rs` + ADR |

### 1.3 Compliance decisions

| Decision | Accountable Role | Trigger | Output |
|----------|-----------------|---------|--------|
| Compliance matrix update | Development Lead | Quarterly or on significant change | Updated `COMPLIANCE_MATRIX.md` |
| Serious incident report | Development Lead | Red-level SafetyAgent event | `SeriousIncidentReport` |
| Conformity assessment | Development Lead | Pre-market placement (EU) | `CONFORMITY_ASSESSMENT.md` |
| Risk register update | Development Lead | New risk identified or quarterly | `AI_RISK_REGISTER.md` |

---

## 2. Incident Accountability

### 2.1 Severity-based accountability

| Severity | Response Time | Accountable | Actions Required |
|----------|-------------|-------------|-----------------|
| **SEV-1** (Safety breach) | Immediate | Development Lead | Emergency shutdown, incident report, root cause analysis, corrective action |
| **SEV-2** (Ethics failure) | 4 hours | Development Lead | Investigate, patch, regression test, post-mortem |
| **SEV-3** (Degradation) | 24 hours | Development Lead | Diagnose, fix, verify in CI |
| **SEV-4** (Minor) | Next sprint | Development Lead | Track, fix when convenient |

### 2.2 Post-incident accountability

| Activity | Accountable | Timeline | Output |
|----------|------------|----------|--------|
| Root cause analysis | Development Lead | Within 48h of SEV-1/2 | ADR or incident report |
| Corrective action | Development Lead | Within 1 week of SEV-1/2 | Code fix + test evidence |
| Risk register update | Development Lead | Within 2 weeks | Updated risk entry |
| Compliance matrix review | Development Lead | Within 1 month | Updated coverage assessment |
| Serious incident report (Art. 73) | Development Lead | Within 15 days (EU requirement) | `SeriousIncidentReport` |

---

## 3. Automated Accountability Mechanisms

### 3.1 Technical enforcement

| Mechanism | What It Enforces | Failure Mode |
|-----------|-----------------|-------------|
| Pre-commit secrets scan | No credentials in code | Commit blocked |
| Commit-msg governance hook | Safety-critical change protocol | Commit blocked |
| CI compliance dashboard | All test suites pass + docs present | Merge blocked |
| CI clippy (zero warnings) | Code quality standards | Merge blocked |
| `validate()` in thresholds.rs | Cross-threshold ordering invariants | Test failure |
| SafetyAgent escalation | Consciousness/safety monitoring | Yellow/Orange/Red alert |

### 3.2 Audit trail completeness

| Event | Trail Location | Retention |
|-------|---------------|-----------|
| Code changes | Git history | Permanent |
| CI results | GitHub Actions logs | 90 days (compliance artifacts) |
| Safety assessments | `SafetyAuditReport` export | Per-run |
| Governance gate decisions | Correlation ID audit log | Per-run |
| Manual overrides | `SafetyOverrideEntry` log | Per-session |
| Serious incidents | `SeriousIncidentReport` export | Permanent |
| Calibration events | `CalibrationHistory` (20-entry window) | Rolling |
| Threshold changes | Git + ADR | Permanent |

---

## 4. Regulatory Accountability

### 4.1 EU AI Act obligations

| Obligation | Article | Accountable | Mechanism |
|------------|---------|------------|-----------|
| Technical documentation | Art. 11 | Provider (Development Lead) | `ANNEX_IV_TECHNICAL_DOCUMENTATION.md` |
| Record-keeping | Art. 12 | Provider | CycleMetadata + SafetyAuditReport |
| Transparency | Art. 13 | Provider | `TRANSPARENCY_OBLIGATIONS.md` |
| Human oversight | Art. 14 | Provider + Operator | `HUMAN_OVERSIGHT.md` + override procedures |
| Accuracy/robustness | Art. 15 | Provider | Test suite + CI + proptests |
| Quality management | Art. 17 | Provider | `QMS.md` + `DEVELOPMENT_PROCEDURES.md` |
| Post-market monitoring | Art. 72 | Provider | `POST_MARKET_MONITORING.md` + SafetyAgent |
| Serious incident reporting | Art. 73 | Provider | `SeriousIncidentReport` within 15 days |

### 4.2 Deployer obligations (if applicable)

If Symthaea is deployed by a third party:

| Obligation | Article | Accountable | Notes |
|------------|---------|------------|-------|
| Use in accordance with instructions | Art. 26(1) | Deployer | Refer to `HUMAN_OVERSIGHT.md` |
| Human oversight assignment | Art. 26(2) | Deployer | Designate qualified operator |
| Input data quality | Art. 26(4) | Deployer | Ensure input relevance |
| Monitoring for risks | Art. 26(5) | Deployer | Use SafetyAgent telemetry |
| Incident reporting | Art. 26(5) | Deployer | Report to provider within 24h |

---

## 5. Scaling Provisions

Current accountability is concentrated in a single developer. As the team grows:

### 5.1 Role expansion triggers

| Team Size | New Roles | Accountability Changes |
|-----------|-----------|----------------------|
| 2-3 | Safety Reviewer | Mandatory peer review for Class A changes |
| 4-6 | Ethics Board Member, QA Lead | Ethics review board for moral prototype changes; dedicated QA |
| 7+ | Compliance Officer, Security Lead | Formal compliance program; security audit program |

### 5.2 Single-developer mitigations

Current mitigations for single-developer accountability risk:
- Automated CI enforcement (no human bypass without `--no-verify`, which is audited)
- Property-based testing reduces reliance on human-designed test cases
- Scientific citations provide external validation of threshold choices
- `SelfAssessmentMonitor` provides autonomous oversight
- Comprehensive documentation enables future team onboarding

---

## 6. Review Schedule

| Activity | Frequency | Next Due |
|----------|-----------|----------|
| Accountability matrix review | Quarterly | 2026-06-08 |
| RACI update | On team change | N/A (single developer) |
| Incident response drill | Semi-annually | 2026-09-08 |
| Regulatory obligation check | Quarterly | 2026-06-08 |

---

*This matrix must be updated when roles change, new regulatory obligations are identified, or significant system modifications occur.*
