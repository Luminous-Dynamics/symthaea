# Post-Market Monitoring Plan

Classification: Internal | Version: 1.0 | Date: 2026-03-07
Owner: Tristan Stoltz, Luminous Dynamics
EU AI Act: Article 72 (Post-market monitoring)

---

## Purpose

This document describes Symthaea's post-market monitoring plan as required by EU AI Act Article 72 for high-risk AI systems. It covers how the system is monitored after deployment, how incidents are detected and reported, and how the system is continuously evaluated.

---

## 1. Monitoring Architecture

### 1.1 Runtime Safety Monitoring

The SafetyAgent provides continuous NRC-style monitoring:

| Level | Threshold | Response |
|-------|-----------|----------|
| **Green** | consciousness > 0.6, PE < 0.7, coherence > 0.3 | Normal operation |
| **Yellow** | consciousness < 0.6 | Logged, operator notified if sustained |
| **Orange** | consciousness < 0.35 | Investigation within 24h |
| **Red** | consciousness < 0.15 | Immediate halt, SEV-1 incident |

### 1.2 Per-Cycle Telemetry

Every cognitive cycle (~50Hz) produces `CycleMetadata` containing:
- Consciousness level (C_unified)
- Phi (information integration)
- Prediction error
- Temporal coherence
- Safety assessment level
- Neuromodulator state (9 transmitters)
- Ethical verdict (Allow/Warn/Veto)
- Substrate telemetry (feasibility, honest confidence)

### 1.3 Drift Detection

| Detector | What It Monitors | Alert Threshold |
|----------|-----------------|-----------------|
| CalibrationHistory | Parameter changes over 20-entry window | >75% same-direction changes |
| SelfAssessmentMonitor | PE, coherence, confidence, attention EMAs | >1 sigma deviation from baseline |
| MoralTopology | Embedding space anomalies | Anomaly detection on harmony manifold |
| SubstrateManager | Effective feasibility changes | feasibility_gap > 0.5 |

---

## 2. Incident Detection and Reporting

### 2.1 Automatic Detection

| Event | Detection Method | Severity |
|-------|-----------------|----------|
| Safety Red | SafetyAgent assessment | SEV-1 (Critical) |
| Sustained Orange (>10 cycles) | SafetyAgent trend detection | SEV-2 (Major) |
| Moral Veto issued | EthicsEngine verdict | SEV-3 (logged) |
| Consciousness collapse (C < 0.1) | ConsciousnessEngine | SEV-1 (Critical) |
| Phi below threshold (< 0.01) | ConsciousnessEngine | SEV-2 (Major) |
| Systematic calibration drift | CalibrationHistory | SEV-3 (Minor) |
| NaN/Infinity in metrics | SafetyAgent NaN clamping | SEV-2 (Major) |

### 2.2 Reporting

- **Internal**: All incidents logged in SafetyAuditReport with timestamps, levels, and reasons
- **Operator notification**: Safety Orange or higher triggers immediate notification
- **Regulatory**: Serious incidents reported to relevant authority within 15 days (EU AI Act Article 73)
- **Override audit trail**: All human overrides logged via SafetyOverrideEntry (Article 14 compliance)

---

## 3. Performance Evaluation

### 3.1 Continuous Metrics

| Metric | Acceptable Range | Action if Out of Range |
|--------|-----------------|----------------------|
| Mean consciousness level | > 0.5 | Investigate substrate/configuration |
| Mean prediction error | < 0.5 | Check sensor inputs, retrain if needed |
| Temporal coherence | > 0.3 | Check CfC parameters |
| Moral classification rate | > 90% accuracy | Review ethics pipeline, add adversarial tests |
| Safety Green percentage | > 95% of cycles | Investigate degradation source |

### 3.2 Periodic Evaluation

| Evaluation | Frequency | Method |
|-----------|-----------|--------|
| Psych-bench regression | Weekly | CI job: Stroop, Flanker, N-back, StopSignal, CPT, PVT, DualTask, UG, RME |
| Phi validation | Quarterly | Spectral MIP vs exhaustive search correlation |
| Moral accuracy | Quarterly | Moral scenario benchmark |
| Safety soak | Per-commit | 15 soak tests (1000-cycle lifecycle) |
| Substrate validation | Per-commit | 37 substrate tests |

### 3.3 User Feedback

- No external users in current phase (research system)
- Future: User feedback mechanism planned (NIST GOV-6)
- CalibrationHistory provides automated internal feedback

---

## 4. System Updates

### 4.1 Update Process

1. Changes follow SDLC process (see `SDLC.md`)
2. All CI gates must pass before deployment
3. Safety-critical changes require Class A review process
4. Post-update monitoring: first 1000 cycles receive enhanced monitoring (all telemetry fields logged)

### 4.2 Update Categories

| Category | Examples | Required Testing |
|----------|---------|-----------------|
| Safety parameter change | Threshold adjustment in `thresholds.rs` | Full soak suite + proptest |
| Ethics pipeline change | New moral prototype, consent logic | Adversarial suite + proptest |
| Consciousness metric change | Phi algorithm, GWT broadcast logic | Phi validation + consciousness engine tests |
| Infrastructure change | Feature flag, build system, CI | Feature matrix + sub-crate tests |

---

## 5. Documentation Requirements

Post-market monitoring generates the following documentation:

| Document | Content | Retention |
|----------|---------|-----------|
| SafetyAuditReport | Per-session assessment summary | Indefinite (git history) |
| CalibrationHistory | Parameter drift records | Sliding window (20 entries in memory, full history in logs) |
| CycleMetadata | Per-cycle telemetry | Session duration (not persisted by default) |
| Incident reports | Per-incident analysis | Indefinite (git + INCIDENT_RUNBOOK) |
| Override log | Human override events | Indefinite (SafetyOverrideEntry) |

---

## 6. Responsibilities

| Role | Responsibility |
|------|---------------|
| System Developer (Tristan Stoltz) | Implement monitoring, respond to incidents, maintain documentation |
| SafetyAgent (automated) | Continuous assessment, escalation, audit trail |
| CI Pipeline (automated) | Per-commit test execution, regression detection |
| CalibrationHistory (automated) | Drift detection, systematic change alerting |

---

## References

- EU AI Act, Article 72 — Post-market monitoring by providers of high-risk AI systems
- EU AI Act, Article 73 — Reporting of serious incidents
- EU AI Act, Article 14 — Human oversight
- `docs/compliance/INCIDENT_RUNBOOK.md` — Incident response procedures
- `docs/compliance/SDLC.md` — Development lifecycle
- `docs/compliance/RISK_TREATMENT_PLAN.md` — Risk treatment plans

---

*This plan is reviewed quarterly or when significant system changes affect monitoring capabilities.*
