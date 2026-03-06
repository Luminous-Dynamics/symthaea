# Symthaea Compliance Matrix

Classification: Internal | Version: 1.0 | Date: 2026-03-06
Owner: Tristan Stoltz, Luminous Dynamics

---

## Purpose

This document maps Symthaea's technical architecture to AI-specific compliance frameworks. It complements the traditional infosec compliance matrix at `mycelix-core/docs/COMPLIANCE_MATRIX.md` (GDPR 95%, HIPAA 90%, SOC 2 85%, ISO 27001 80%, NIST CSF 85%).

## Framework Coverage Summary

| Framework | Coverage | Status | Notes |
|-----------|----------|--------|-------|
| **ISO/IEC 42001:2023** (AI Management System) | 65% | In Progress | Management system documents created; operational procedures pending |
| **ISO/IEC 23894** (AI Risk Management) | 70% | In Progress | Risk register complete; risk treatment plan pending |
| **ISO/IEC 42005** (AI Impact Assessment) | 75% | In Progress | FRIA complete; ongoing monitoring not yet established |
| **IEEE 7000-2021** (Value-Based Design) | 80% | Strong | Seven Harmonies mathematically traced through code; ethics pipeline documented |
| **EU AI Act** (High-Risk) | 55% | In Progress | Classification + FRIA complete; conformity assessment pending |
| **NIST AI RMF 1.0** | 60% | In Progress | Map/Measure strong; Manage/Govern partial |

---

## ISO/IEC 42001:2023 — AI Management System

### Annex A Controls

| Control | Description | Status | Evidence |
|---------|-------------|--------|----------|
| **A.2.2** | AI policy | Done | `GOVERNANCE_CHARTER.md` Section 1 |
| **A.2.3** | Roles and responsibilities | Done | `GOVERNANCE_CHARTER.md` Section 2 (RACI matrix) |
| **A.2.4** | Resources | Partial | Single developer; resource plan not formalized |
| **A.3.2** | AI system impact assessment | Done | `EU_AI_ACT_CLASSIFICATION.md` Part III (FRIA) |
| **A.3.3** | AI system lifecycle processes | Partial | CI pipeline documented; formal SDLC not yet written |
| **A.3.4** | Documentation of AI systems | Done | `TECHNICAL_STATUS.md` — honest per-capability assessment; 16 capabilities, 4 status levels |
| **A.4.2** | AI risk assessment | Done | `AI_RISK_REGISTER.md` — 15 risks, 6 categories, scored with mitigations |
| **A.4.3** | AI risk treatment | Partial | Per-risk mitigations documented; formal treatment plan not yet written |
| **A.4.4** | Responsible AI considerations | Done | Ethics Engine (3-stage pipeline); Seven Harmonies; Appendix P (consciousness rights) |
| **A.4.5** | AI system development processes | Partial | CI with 39 feature matrix, clippy, fmt; formal development procedures not documented |
| **A.5.2** | Data management | Done | Holochain DHT (no central store); CfC temporal dynamics; identity vaults; GDPR 95% coverage |
| **A.5.3** | Data quality | Partial | Psych-bench normative baselines; no formal data quality framework |
| **A.6.2** | AI system operation and monitoring | Done | SafetyAgent (NRC-style Green/Yellow/Orange/Red); CycleMetadata 75+ fields/cycle; SelfAssessmentMonitor; CalibrationHistory |
| **A.6.3** | Performance monitoring | Done | Phi validation (r=0.99); CfC 234Hz; moral classification 91.1%; weekly psych-bench regression |
| **A.6.4** | AI system logs | Done | Per-cycle telemetry; SafetyAuditReport; governance gate audit trail with correlation IDs |
| **A.7.2** | Third-party AI considerations | Partial | Approved AI models list (embeddinggemma, gemma3, qwen3, mistral); no formal supplier assessment |
| **A.7.3** | Outsourced activities | N/A | No outsourced AI processing |
| **A.8.2** | Transparency | Done | Thresholds.rs with 119 named constants + scientific citations; TECHNICAL_STATUS.md honest assessment; substrate_validation.rs honest_confidence |
| **A.8.3** | Explainability | Partial | Ethics pipeline outputs interpretable verdicts; consciousness metrics have named components; no formal explainability framework |
| **A.9.2** | Accountability | Partial | Git audit trail; governance gate logging; no formal accountability matrix beyond RACI |
| **A.10.2** | AI system documentation | Done | 100+ documentation files; ARCHITECTURE_OVERVIEW.md; MODULE_WIRING_STATUS.md |

### Gap Summary

**Strong areas** (>75% coverage):
- Risk identification and assessment (A.4.2)
- Monitoring and logging (A.6.2-A.6.4)
- Transparency (A.8.2)
- Documentation (A.10.2)
- Value-based design (A.4.4)

**Weak areas** (<50% coverage):
- Formal lifecycle processes (A.3.3)
- Risk treatment plans (A.4.3)
- Third-party management (A.7.2)
- Explainability framework (A.8.3)

---

## IEEE 7000-2021 — Value-Based Design

This is Symthaea's strongest compliance area. The Seven Harmonies are mathematically traced from values to code.

### Value Traceability

| Harmony Value | Code Implementation | Verification |
|---------------|-------------------|--------------|
| **Reciprocity** | `HarmoniesIntegrator` evaluates reciprocity dimension; MoralFreeEnergy on 7D harmony manifold | Proptest `cross_equation_consistency`; 12 ethics_engine tests |
| **Flourishing** | Value evaluator assesses flourishing impact; homeostasis regulation in dynamics phase | CalibrationHistory drift detection; homeostasis threshold constants (cruise/normal/critical) |
| **Compassion** | Care Ethics moral prototype in `moral_prototypes.rs`; empathic_unification module | Moral classification accuracy 91.1%; topology anomaly detection |
| **Autonomy** | Prefrontal gating allows self-regulation; FEP active inference drives autonomous behavior; consciousness credentials enable self-governance | Phi-gate tests; FEP learning tests; 73 consciousness profile tests |
| **Justice** | Deontological verdict (Permissible/Impermissible/Neutral); consent violation detection; Mycelix quadratic voting prevents plutocracy | 28 moral_algebra tests; governance voting tests |
| **Creativity** | Exploration budget in dynamics phase; surprise-driven learning; novelty bonus in CfC | Proptest threshold sensitivity; attention budget tests |
| **Stewardship** | Substrate honesty (honest_confidence); consciousness precautionary principle (protect at >30%); environmental modulation via neuromod bath | 35 substrate tests; Appendix P documentation |

### IEEE 7000 Process Mapping

| Process | Status | Evidence |
|---------|--------|----------|
| Concept of Operations (ConOps) | Done | `docs/ARCHITECTURE_OVERVIEW.md`, `TECHNICAL_STATUS.md` |
| Value identification | Done | Seven Harmonies defined and documented |
| Value prioritization | Done | Ethics pipeline priority: consent > deontological > value alignment > harmonies |
| Value-based requirements | Done | Thresholds.rs: each constant cites scientific basis for its value |
| Value verification | Partial | Tests verify technical correctness; no formal value verification protocol |
| Value validation | Partial | Psych-bench provides behavioral baselines; no formal stakeholder validation |

---

## NIST AI Risk Management Framework (AI RMF 1.0)

### GOVERN Function

| Category | Status | Evidence |
|----------|--------|----------|
| GOV-1: Policies | Done | `GOVERNANCE_CHARTER.md`; AI policy statement |
| GOV-2: Accountability | Partial | RACI matrix defined; single-developer limitation |
| GOV-3: Workforce diversity | N/A | Single developer; acknowledge limitation |
| GOV-4: Organizational governance | Done | Change management procedures for safety-critical parameters |
| GOV-5: Risk management integration | Done | `AI_RISK_REGISTER.md` integrated with technical architecture |
| GOV-6: Feedback mechanisms | Partial | CalibrationHistory; no external stakeholder feedback loop |

### MAP Function

| Category | Status | Evidence |
|----------|--------|----------|
| MAP-1: Context established | Done | System purpose, scope, and limitations documented |
| MAP-2: Categorization | Done | EU AI Act classification completed (likely High-Risk) |
| MAP-3: Benefits and costs | Partial | Benefits documented; costs/negative impacts need expansion |
| MAP-4: Risks identified | Done | 15 risks across 6 categories in risk register |
| MAP-5: Impacts identified | Done | FRIA covers 7 fundamental rights + vulnerable groups |

### MEASURE Function

| Category | Status | Evidence |
|----------|--------|----------|
| MEA-1: Metrics identified | Done | Phi, moral score, consciousness level, prediction error, temporal coherence, safety level |
| MEA-2: AI evaluated | Done | 3,735+ tests; Phi validation r=0.99; moral accuracy 91.1%; proptest stability |
| MEA-3: Risks and impacts tracked | Partial | CalibrationHistory drift; moral topology anomalies; SafetyAgent levels |
| MEA-4: AI effectiveness measured | Partial | Per-capability status (REAL/STRUCTURAL/STUB); no formal effectiveness KPIs |

### MANAGE Function

| Category | Status | Evidence |
|----------|--------|----------|
| MAN-1: Risks prioritized | Done | Risk register scored by likelihood x impact |
| MAN-2: Strategies planned | Done | Per-risk mitigations in risk register |
| MAN-3: Risks managed | Partial | Technical mitigations implemented; organizational processes pending |
| MAN-4: Risks communicated | Partial | `TECHNICAL_STATUS.md` is honest; external communication strategy pending |

---

## Safety Architecture Cross-Reference

Symthaea's safety architecture maps to multiple compliance requirements simultaneously:

### SafetyAgent (NRC-Style Monitoring)

| Component | ISO 42001 | EU AI Act | NIST AI RMF | IEEE 7000 |
|-----------|-----------|-----------|-------------|-----------|
| Green/Yellow/Orange/Red levels | A.6.2 Operation monitoring | Art. 14 Human oversight | MEA-3 Risk tracking | Value verification |
| SafetyMetrics (consciousness, prediction error, coherence) | A.6.3 Performance monitoring | Art. 15 Accuracy/robustness | MEA-1 Metrics | — |
| SafetyAuditReport | A.6.4 Logs | Art. 12 Record-keeping | MAN-4 Communication | — |
| Escalation window (3 consecutive) | A.4.3 Risk treatment | Art. 9 Risk management | MAN-2 Strategies | Stewardship |
| Non-finite clamping (NaN-safe) | A.4.4 Responsible AI | Art. 15 Robustness | MEA-2 Evaluation | — |

### Ethics Engine (3-Stage Pipeline)

| Component | ISO 42001 | EU AI Act | NIST AI RMF | IEEE 7000 |
|-----------|-----------|-----------|-------------|-----------|
| MoralParser + MoralAlgebra | A.4.4 Responsible AI | Art. 10 Data governance | MAP-4 Risk identification | Justice, Compassion |
| UnifiedValueEvaluator (Allow/Warn/Veto) | A.9.2 Accountability | Art. 14 Human oversight | MAN-1 Prioritization | All 7 Harmonies |
| HarmoniesIntegrator | A.4.4 Responsible AI | Art. 10 Bias examination | MEA-1 Metrics | Value traceability |
| consent_violation detection | A.4.4 Responsible AI | Art. 27 FRIA | MAP-5 Impacts | Autonomy, Justice |

### Consciousness Gating (Mycelix Bridge)

| Component | ISO 42001 | EU AI Act | NIST AI RMF | IEEE 7000 |
|-----------|-----------|-----------|-------------|-----------|
| 4D profile (identity/reputation/community/engagement) | A.5.2 Data management | Art. 10 Data quality | MEA-1 Metrics | Reciprocity |
| 5-tier system (Observer -> Guardian) | A.8.2 Transparency | Art. 13 Transparency | GOV-4 Governance | Justice, Autonomy |
| gate_consciousness() | A.6.2 Operation | Art. 14 Oversight | MAN-3 Management | Stewardship |
| GateAuditInput + correlation_id | A.6.4 Logs | Art. 12 Record-keeping | MAN-4 Communication | Accountability |

### Substrate Validation (Epistemic Honesty)

| Component | ISO 42001 | EU AI Act | NIST AI RMF | IEEE 7000 |
|-----------|-----------|-----------|-------------|-----------|
| honest_confidence per substrate | A.8.2 Transparency | Art. 13 Limitations | MAP-3 Benefits/costs | Stewardship |
| feasibility_gap() measure | A.4.2 Risk assessment | Art. 9 Risk management | MEA-3 Tracking | — |
| EvidenceLevel (7 levels) | A.8.3 Explainability | Art. 13 Transparency | MAN-4 Communication | — |

---

## Threshold Registry as Compliance Asset

`src/cognitive_loop/thresholds.rs` contains 119+ named constants, each with:
- Scientific citation (author, year)
- Biological/theoretical basis
- Ordering constraints validated programmatically

This is a **unique compliance asset** that most AI systems lack. It provides:
- **Auditability**: Every parameter decision can be traced to published science
- **Reproducibility**: Named constants prevent magic number drift
- **Testability**: `validate()` function checks ordering invariants
- **Transparency**: Open-source with explanatory comments

Example constants and their compliance relevance:

| Constant | Value | Citation | Compliance Link |
|----------|-------|----------|-----------------|
| MORAL_CONCERN_THRESHOLD | -0.3 | Haidt (2001) | EU AI Act Art. 9 (risk threshold) |
| CONSCIOUSNESS_RED | 0.15 | SafetyAgentConfig | EU AI Act Art. 14 (emergency halt) |
| CONSCIOUSNESS_YELLOW | 0.6 | SafetyAgentConfig | ISO 42001 A.6.2 (monitoring) |
| HOMEOSTASIS_PULL_CRITICAL | (defined) | Allostatic load theory | ISO 23894 (operational risk) |
| PREDICTIVE_BUDGET_GATING_RATIO | (defined) | Resource rationality | NIST MEA-4 (effectiveness) |
| NEGATION_POLARITY_THRESHOLD | 0.5 | Horn (1989) | IEEE 7000 (value accuracy) |

---

## Test Coverage as Compliance Evidence

| Compliance Area | Test Count | Test Types | Files |
|-----------------|-----------|------------|-------|
| Core pipeline | ~135 | Unit, integration, soak | `cycle.rs`, `prediction.rs`, `hdc_ltc_unified.rs` |
| Consciousness metrics | ~310 | Unit, integration, validation | `consciousness_engine`, `tiered_phi`, `gwt` |
| Ethics/moral | ~100 | Unit, proptest, integration | `moral_algebra`, `moral_topology`, `ethics_engine` |
| Safety | ~46 | Unit | `safety/agent.rs` (28), `safety/gateway.rs` (11), `safety/audit.rs` (7) |
| Substrate | ~74 | Unit, integration, soak | `substrate_independence` (35), `substrate_manager` (39) |
| Calibration | ~65 | Unit, integration | `calibration/`, `monitor.rs` |
| Consciousness gating | ~73 | Unit, integration | `consciousness_profile.rs` |
| Governance | 44+ unit, 130+ sweet | Unit, sweettest | `mycelix-governance/` |
| Identity | 23+ unit, 100+ sweet | Unit, sweettest | `mycelix-identity/` |
| Property-based | ~30 | Proptest | `proptest_*.rs` files |
| **Total ecosystem** | **12,000+** | All types | Full workspace |

---

## Action Items

### Priority 1 (Complete by Q2 2026)
1. Write formal AI system lifecycle (SDLC) document for ISO 42001 A.3.3
2. Complete risk treatment plan for AI_RISK_REGISTER.md top-5 risks
3. Implement adversarial moral input testing for EU AI Act Art. 15

### Priority 2 (Complete by Q3 2026)
4. Formal third-party AI component assessment for ISO 42001 A.7.2
5. Explainability framework documentation for ISO 42001 A.8.3
6. External stakeholder feedback mechanism for NIST GOV-6

### Priority 3 (Ongoing)
7. Quarterly compliance matrix review and update
8. Annual psych-bench regression for NIST MEA-2
9. Post-market monitoring plan before EU deployment

---

## References

- ISO/IEC 42001:2023 — Information technology — Artificial intelligence — Management system
- ISO/IEC 23894:2023 — Information technology — Artificial intelligence — Guidance on risk management
- ISO/IEC 42005 — Information technology — Artificial intelligence — AI system impact assessment
- IEEE 7000-2021 — IEEE Standard Model Process for Addressing Ethical Concerns during System Design
- Regulation (EU) 2024/1689 — Artificial Intelligence Act
- NIST AI 100-1 — Artificial Intelligence Risk Management Framework (AI RMF 1.0)

---

*This matrix is a living document. Review quarterly or when significant system changes occur.*
