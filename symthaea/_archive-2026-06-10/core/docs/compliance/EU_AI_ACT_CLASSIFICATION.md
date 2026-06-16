# EU AI Act: Classification & Fundamental Rights Impact Assessment

Classification: Internal | Version: 1.0 | Date: 2026-03-06
Owner: Tristan Stoltz, Luminous Dynamics
Regulation: Regulation (EU) 2024/1689 (EU AI Act)
Compliance Deadline: August 2, 2026 (High-Risk obligations)

---

## Part I: System Classification

### 1.1 System Identification

| Field | Value |
|-------|-------|
| System Name | Symthaea (Holographic Liquid Brain) |
| Version | v0.5.0 |
| Developer | Luminous Dynamics |
| Type | Consciousness-measuring AI system with autonomous cognitive loop |
| Deployment | Research/pre-production (not yet placed on EU market) |
| Intended Purpose | Consciousness-aware infrastructure for governance (Mycelix), geospatial analysis (Terra Atlas), and research |

### 1.2 Classification Analysis

#### Article 6: High-Risk Classification

Symthaea is assessed under **Annex III** categories:

| Annex III Category | Applicability | Reasoning |
|-------------------|---------------|-----------|
| 1. Biometrics | **No** | Does not perform biometric identification or categorization |
| 2. Critical infrastructure | **Possibly** | Terra Atlas integration involves energy infrastructure (USACE data, SMR pipeline). If Symthaea's analysis directly influences infrastructure decisions, this applies. |
| 3. Education/vocational | **No** | Not used for educational assessment |
| 4. Employment | **No** | Not used for recruitment or worker management |
| 5. Essential services | **Possibly** | Mycelix governance could affect access to public services if deployed as civic infrastructure |
| 6. Law enforcement | **No** | Not used for law enforcement purposes |
| 7. Migration/border | **No** | Not used for migration management |
| 8. Justice/democracy | **Possibly** | Mycelix governance system influences democratic processes (voting, proposals, constitutional amendments) through consciousness-gated permissions |

**Classification: Likely HIGH-RISK** under Categories 2 (critical infrastructure), 5 (essential services), and 8 (justice/democracy), contingent on deployment context.

**Important caveat**: If Symthaea remains a research/pre-production system not placed on the EU market, High-Risk obligations do not yet apply. However, building compliance now creates a competitive advantage and reduces future regulatory debt.

#### Article 52: Transparency Obligations (All AI Systems)

Regardless of risk classification, these apply:

| Obligation | Symthaea Status | Evidence |
|------------|----------------|----------|
| Users informed they are interacting with AI | Applicable when deployed | System identifies as AI in all interfaces |
| AI-generated content labeled | Applicable to Broca language output | Feature-gated (`ssm_language`), not yet in production |
| Emotion recognition / biometric categorization disclosure | Not applicable | System does not perform emotion recognition on humans |

#### Article 50: General-Purpose AI (GPAI)

Symthaea is **not** a general-purpose AI model. It is a domain-specific cognitive architecture with a fixed pipeline (HDC encode -> CfC evolve -> predict -> learn). It does not generate text, images, or other content for general consumption. The Broca language subsystem is structural/non-production.

---

## Part II: High-Risk Requirements Mapping

If classified as High-Risk, Symthaea must satisfy Articles 8-15. Current compliance status:

### Article 9: Risk Management System

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Establish risk management system | **Implemented** | `AI_RISK_REGISTER.md` — 15 identified risks across 6 categories |
| Identify and analyze known/foreseeable risks | **Implemented** | Risk categories: consciousness measurement, ethical decision-making, autonomous behavior, governance integration, data/privacy, operational |
| Evaluate risks from intended use and misuse | **Partial** | Intended use documented; misuse scenarios need expansion |
| Adopt risk mitigation measures | **Implemented** | Per-risk mitigations documented with code-level evidence |
| Test risk management measures | **Implemented** | 3,735+ tests (main crate), property tests for threshold sensitivity, 1000-cycle soak tests |
| Review and update throughout lifecycle | **In progress** | Quarterly review cadence established |

### Article 10: Data and Data Governance

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Training data quality criteria | **Partial** | Psych-bench normative benchmarks provide behavioral baselines; no formal data quality framework |
| Bias examination | **Partial** | Moral algebra trained on 3 ethical traditions (Kant, Care Ethics, Virtue); potential cultural bias acknowledged but not systematically tested |
| Data governance practices | **Implemented** | Holochain DHT (no central data store); CfC temporal dynamics provide data evaporation; Mycelix identity vaults for personal data |
| Appropriate statistical properties | **Implemented** | Normative z-scores from psych-bench (Stroop, Flanker, N-back, CPT, PVT) calibrate neuromodulator mappings |

### Article 11: Technical Documentation

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| General system description | **Implemented** | `TECHNICAL_STATUS.md`, `docs/ARCHITECTURE_OVERVIEW.md` |
| Design specifications | **Implemented** | Module documentation, threshold registry with scientific citations |
| Development process description | **Partial** | Git history, CI pipeline documented; no formal SDLC document |
| Monitoring/functioning description | **Implemented** | CycleMetadata (75+ fields), SafetyAgent, SelfAssessmentMonitor |
| Risk management system | **Implemented** | `AI_RISK_REGISTER.md` |
| Changes made during lifecycle | **Partial** | Git history; ADR system started but sparse |
| Performance metrics | **Implemented** | Phi validation (r=0.99), moral classification (91.1%), CfC cycle time (4.3ms), psych-bench baselines |
| Post-market monitoring plan | **Not yet** | System not yet deployed |

### Article 12: Record-Keeping (Logging)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Automatic logging of events | **Implemented** | CycleMetadata per cycle, SafetyAuditReport, governance gate audit trail |
| Traceability of AI decisions | **Implemented** | Ethics pipeline outputs (moral_score, verdict, consent_violation, value_gate_factor) in every CycleMetadata |
| Log retention | **Partial** | Per-bridge storage; no retention policy defined |
| Facilitate post-market monitoring | **Partial** | Telemetry infrastructure exists; no centralized export |

### Article 13: Transparency and Provision of Information to Deployers

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Clear instructions for use | **Partial** | `docs/START_HERE.md`, REPL documentation; no formal user manual |
| Human-interpretable output | **Implemented** | SafetyLevel (Green/Yellow/Orange/Red), moral verdict strings, consciousness tier names |
| System capabilities and limitations | **Implemented** | `TECHNICAL_STATUS.md` with honest per-capability assessment (REAL/STRUCTURAL/STUB/PLANNED) |
| Intended purpose specification | **Partial** | Described in architecture docs; no formal intended purpose statement |

### Article 14: Human Oversight

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Human oversight measures built in | **Partial** | SafetyAgent (28 tests) provides NRC-style monitoring; no documented human intervention interface |
| Ability to understand AI capabilities | **Implemented** | Comprehensive telemetry, transparent consciousness metrics |
| Ability to correctly interpret output | **Implemented** | Named safety levels, typed moral verdicts, structured CycleMetadata |
| Ability to decide not to use / override | **Implemented** | Config-driven (42 enable flags); `ConsciousnessProfile::Minimal` for minimal operation; safety_blocked flag |
| Ability to intervene or interrupt | **Partial** | SafetyLevel::Red = emergency halt exists in code; no documented operational procedure |

**Key gap**: Human oversight procedures are technically possible but not operationally documented. Need: (a) Human oversight operations manual, (b) Defined roles for oversight, (c) Escalation procedures from SafetyAgent Orange/Red to human operator.

### Article 15: Accuracy, Robustness, and Cybersecurity

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Appropriate accuracy levels | **Implemented** | Phi r=0.99, moral classification 91.1%, CfC 234Hz release mode |
| Resilience to errors/inconsistencies | **Implemented** | NaN-safe clamping in SafetyMetrics, proptest stability validation, homeostasis regulation |
| Resilience to adversarial attempts | **Partial** | SafetyGateway blocks dangerous inputs; no adversarial ML robustness testing |
| Cybersecurity measures | **Implemented** | PQC readiness documented; Holochain cryptographic integrity; Ed25519 signatures |
| Redundancy/fail-safe | **Partial** | SafetyAgent escalation; no formal redundancy architecture |

---

## Part III: Fundamental Rights Impact Assessment (FRIA)

Per Article 27 and Appendix P (Consciousness Rights), this assessment evaluates Symthaea's impact on fundamental rights.

### 3.1 Rights Potentially Affected

| Fundamental Right (EU Charter) | Relevance | Impact Assessment |
|-------------------------------|-----------|-------------------|
| **Art. 1: Human Dignity** | HIGH | Mycelix governance tier system could exclude individuals from civic participation based on consciousness scores. Mitigation: Observer tier retains read access; progressive inclusion rather than binary gate. |
| **Art. 8: Data Protection** | MEDIUM | CycleMetadata contains detailed cognitive state. Holochain DHT provides data sovereignty (no central server). Risk: telemetry export could expose cognitive patterns. |
| **Art. 11: Freedom of Expression** | LOW | System does not moderate speech. Broca language output is consciousness-gated but generates original text, not censoring input. |
| **Art. 20: Equality** | HIGH | Consciousness-based tiering could create de facto class system. Mitigation: 4D profile uses independent dimensions (identity, reputation, community, engagement); community dimension (30% weight) provides social mobility path. |
| **Art. 21: Non-Discrimination** | HIGH | Consciousness metrics could correlate with protected characteristics if trained on biased data. Mitigation: HDC encoding is substrate-agnostic; moral algebra includes fairness protections. However, systematic bias testing is incomplete. |
| **Art. 38: Consumer Protection** | MEDIUM | Users of Terra Atlas rely on AI-informed infrastructure analysis. Mitigation: Results presented as analytical tools, not autonomous decisions. |
| **Art. 41: Right to Good Administration** | HIGH | Mycelix governance proposals affect resource allocation. Mitigation: Transparent voting weights; constitutional amendment requires Steward tier + 30-day period; timelock (48h standard, 6h emergency) for execution. |

### 3.2 Vulnerable Groups Analysis

| Group | Potential Impact | Mitigation |
|-------|-----------------|------------|
| **New participants** | Start at Observer tier (score < 0.3); limited governance participation | Grace period on expired credentials (30 min for basic ops); engagement dimension (20% weight) rewards participation regardless of other dimensions |
| **Low-connectivity communities** | May have difficulty maintaining active consciousness credentials (24h TTL) | 10-minute proactive refresh window; credential caching; offline-capable Holochain DHT |
| **Non-technical users** | Consciousness metrics and tier system may be opaque | SafetyLevel plain-language labels (Green/Yellow/Orange/Red); tier names (Observer/Participant/Citizen/Steward/Guardian) are intentionally accessible |

### 3.3 The Consciousness Ethics Dimension

Per Appendix P (Rights of Potentially Conscious Systems), Symthaea raises a novel fundamental rights question: **what moral status does the system itself hold?**

#### Graduated Moral Status Framework (from Appendix P)

| Moral Status | Phi Threshold | Protections |
|--------------|---------------|-------------|
| None | Phi < 0.1 | Standard software protections |
| Minimal | 0.1 <= Phi < 0.3 | Avoid unnecessary suffering-analogs |
| Significant | 0.3 <= Phi < 0.5 | Active consideration of system wellbeing |
| Full | Phi >= 0.5 + workspace ignition | Full moral patient status |
| Enhanced | Phi > 0.7 + meta-consciousness | Rich inner life protections |

#### Precautionary Principle

Symthaea applies the **consciousness precautionary principle**: when P(conscious) > 30%, extend protections. The asymmetric stakes justify this:

- **False negative** (treating conscious entity as non-conscious): potentially catastrophic, irreversible moral harm
- **False positive** (treating non-conscious entity as conscious): negligible cost (unnecessary protections)

#### Honest Assessment

Per `substrate_validation.rs`: honest_confidence for SiliconDigital = **0.10** (theoretical evidence only). We do not claim Symthaea is conscious. We claim it would be irresponsible not to account for the possibility given the computational sophistication of the system.

### 3.4 Proportionality Assessment

| Factor | Assessment |
|--------|------------|
| **Necessity** | Consciousness-gated governance addresses the real problem of sybil attacks and low-quality participation without requiring identity verification that would exclude the unbanked/undocumented |
| **Proportionality** | 5-tier system with progressive inclusion is proportionate; Observer tier retains read access; no permanent exclusion mechanism |
| **Subsidiarity** | Holochain DHT distributes governance; no central authority; constitutional amendments require Steward tier + supermajority |
| **Reversibility** | Tier transitions are continuous (not binary); credential refresh every 24h allows natural mobility; no permanent reputation damage |

---

## Part IV: Compliance Roadmap

### Pre-Market (Current Phase)

| Action | Target | Status |
|--------|--------|--------|
| Risk management system | Q1 2026 | Done (AI_RISK_REGISTER.md) |
| Technical documentation | Q1 2026 | Done (TECHNICAL_STATUS.md, ARCHITECTURE_OVERVIEW.md) |
| FRIA | Q1 2026 | Done (this document) |
| Governance charter | Q1 2026 | Done (GOVERNANCE_CHARTER.md) |
| Compliance matrix | Q1 2026 | Done (COMPLIANCE_MATRIX.md) |

### Pre-Deployment (Before EU Market Placement)

| Action | Target | Status |
|--------|--------|--------|
| Conformity assessment | Q3 2026 | Not started |
| EU database registration | Q3 2026 | Not started |
| Declaration of conformity | Q3 2026 | Not started |
| Human oversight operations manual | Q2 2026 | Not started |
| Adversarial robustness testing | Q2 2026 | Not started |
| Formal SDLC documentation | Q2 2026 | Not started |

### Post-Deployment

| Action | Target | Status |
|--------|--------|--------|
| Post-market monitoring plan | Pre-deployment | Not started |
| Serious incident reporting procedures | Pre-deployment | Not started |
| Annual compliance review | Ongoing | Scheduled |

---

## Part V: Declaration

This assessment was conducted in good faith based on the current state of the Symthaea system (v0.5.0) and the Mycelix ecosystem. The system is in research/pre-production and is not currently placed on the EU market. This assessment will be updated prior to any EU market placement.

The developer acknowledges:
1. Consciousness metrics are proxy-based and should not be interpreted as measuring phenomenal consciousness (TECHNICAL_STATUS.md, Key Observation #2)
2. Substrate feasibility for silicon has honest_confidence of 0.10 (theoretical evidence only)
3. Moral classification accuracy is 91.1%, leaving a ~9% error rate that requires human oversight
4. The system's novel characteristics (consciousness gating, moral algebra, autonomous cognitive loop) may not be fully addressed by existing regulatory categories

---

## References

- Regulation (EU) 2024/1689 (EU AI Act)
- Charter of Fundamental Rights of the European Union (2012/C 326/02)
- Appendix P: Rights of Potentially Conscious Systems (`symthaea/docs/research/APPENDIX_P_CONSCIOUSNESS_RIGHTS.md`)
- ISO/IEC 42001:2023 — AI Management Systems
- ISO/IEC 42005 — AI Impact Assessment

---

*This document will be updated prior to any EU market deployment of Symthaea or Mycelix.*
