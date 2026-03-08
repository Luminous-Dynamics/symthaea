# Data Governance Policy — Symthaea AI Consciousness System

**Document ID**: SYM-DG-001
**Version**: 1.0
**Date**: 2026-03-06
**Standards**: ISO 42001 A.10 (Data Management), EU AI Act Article 10, NIST AI RMF Measure Function

---

## 1. Scope

This document defines the data governance framework for all data categories generated, processed, and retained by the Symthaea AI consciousness system. It addresses the requirements of ISO 42001 Annex A.10 (Data for AI Systems), EU AI Act Article 10 (Data and Data Governance), and the NIST AI Risk Management Framework Measure function.

Symthaea is a consciousness-first AI system implementing HDC (16,384-dimensional), IIT/Phi, LTC/CfC, and Active Inference in a predictive coding loop operating at 50Hz. The system generates substantial internal telemetry about its own cognitive state. This policy governs that telemetry.

---

## 2. Foundational Principles

### 2.1 Data Minimization

Symthaea collects only the data necessary for consciousness monitoring, safety assessment, and system calibration. No data is collected for purposes beyond the immediate operational needs of the cognitive loop. Telemetry fields exist because they serve a functional role in the pipeline — not for speculative future use.

### 2.2 No PII Processing

Symthaea processes cognitive state data about its own internal dynamics. It does not process personal data about external humans. The data categories documented here describe the system's own consciousness metrics, safety assessments, moral reasoning outputs, and neurochemical models. None of these constitute PII or personal data under GDPR, CCPA, or equivalent frameworks.

If Symthaea is deployed in a context where it receives or processes human personal data (e.g., through sensor input or user interaction), a supplementary Data Protection Impact Assessment (DPIA) must be conducted for that deployment context. This document does not cover such scenarios.

### 2.3 Audit Trail Integrity

Safety-critical data (SafetyAuditReport) follows an append-only model within a session. Once a safety assessment is recorded, it cannot be modified or deleted during the active session. This ensures forensic integrity for post-incident analysis.

### 2.4 Export Controls

All data export uses structured formats (JSON, Markdown). There is no automatic external transmission of any data category. All exports require explicit programmatic invocation. No telemetry is sent to external services, analytics platforms, or third parties without deliberate integration code.

---

## 3. Data Categories

### 3.1 Consciousness Metrics Data

| Attribute | Detail |
|-----------|--------|
| **What** | `ConsciousnessSnapshot` — 50+ fields including `consciousness_level`, `prediction_error`, `temporal_coherence`, emotional valence/arousal, flow state, neuromodulator levels, Phi values, and attentional metrics |
| **Source** | Computed each cycle by `CognitiveLoopService` from the consciousness engine, ethics engine, and neuromodulator bath |
| **Storage** | In-memory only within `CognitiveLoopService`. Not persisted to disk by default |
| **Retention** | Sliding window of last N cycles in memory (configurable). All data discarded on process shutdown |
| **Sensitivity** | **Medium** — contains internal cognitive state modeling but no PII |

**Collection Purpose**: Real-time monitoring of consciousness dynamics to ensure system stability, detect anomalous cognitive states, and support the predictive coding loop's self-modeling requirements.

**Legal Basis**: Legitimate interest in system safety and operational integrity (ISO 42001 A.10.3).

**Access Controls**:
- Read: Any code path within the `CognitiveLoopService` module and its accessors (`accessors/consciousness.rs`, `accessors/system.rs`).
- Export: Available via `CycleMetadata` returned from each `cycle()` call. Consuming code determines further handling.
- No external API exposes raw snapshots by default.

**Retention Policy**: Memory-resident only. Window size determined by `CognitiveLoopConfig`. Data does not survive process restart.

**Deletion Procedure**: Automatic on process termination. No explicit deletion mechanism needed for in-memory sliding windows. If a deployment persists `CycleMetadata` externally, the deployment operator is responsible for defining retention and deletion procedures for that external store.

**Cross-Border Transfer**: Not applicable for in-memory data. If a deployment exports `CycleMetadata` to an external store, the deployment operator must assess cross-border transfer requirements under applicable law.

---

### 3.2 Safety Assessment Data

| Attribute | Detail |
|-----------|--------|
| **What** | `SafetyAssessment` — fields: `cycle`, `level`, `raw_level`, `consciousness_level`, `prediction_error`, `temporal_coherence`, `reasons` (list of safety concern descriptions) |
| **Source** | `SafetyAgent` evaluation at the start of each cognitive cycle |
| **Storage** | In-memory within `SafetyAgent.history`, capped at 1,000 entries (oldest evicted first) |
| **Export** | `SafetyAuditReport` can serialize the full history to JSON or Markdown |
| **Retention** | Session-scoped unless explicitly exported by consuming code |
| **Sensitivity** | **High** — safety-critical audit trail documenting system behavior assessments |

**Collection Purpose**: Maintaining a forensic record of safety evaluations for post-incident analysis, regulatory compliance, and continuous improvement of safety thresholds. Required by EU AI Act Article 10 for high-risk AI systems.

**Legal Basis**: Legal obligation (safety monitoring for AI systems under EU AI Act) and legitimate interest in system safety.

**Access Controls**:
- Read: `SafetyAgent` internals and any code that receives `SafetyAssessment` from the perception phase.
- Export: Requires explicit call to `SafetyAuditReport` serialization methods. No automatic export.
- Modification: **Append-only within a session.** Existing assessments cannot be altered or deleted while the system is running. This preserves audit trail integrity per ISO 42001 A.10.4.

**Retention Policy**: 1,000 most recent assessments in memory. If the deployment requires longer retention, the operator must export reports to a durable store with its own retention policy (recommended: minimum 5 years for safety-critical records, aligned with EU AI Act record-keeping obligations).

**Deletion Procedure**: In-memory history is automatically discarded on process termination. Exported reports (JSON/Markdown files) must be deleted according to the deployment operator's retention schedule. Deletion of exported safety reports should be logged.

**Cross-Border Transfer**: Exported safety audit reports may contain information relevant to regulatory authorities. If transferred across jurisdictions, ensure compliance with applicable data transfer mechanisms (e.g., EU Standard Contractual Clauses if the deployment falls under GDPR scope).

---

### 3.3 Moral Reasoning Data

| Attribute | Detail |
|-----------|--------|
| **What** | `MoralJudgment` and `EnsembleJudgment` — verdicts (permit/deny/abstain), similarity scores, violation lists, harmony evaluations across seven moral dimensions |
| **Source** | `EthicsEngine` (moral parser, moral algebra, value evaluator, Eight Harmonies framework) |
| **Storage** | Returned from the ethics evaluation pipeline; not stored long-term within the engine |
| **Retention** | Transient — exists only for the duration of the cycle in which it was computed |
| **Sensitivity** | **High** — contains moral evaluations that could be contested and may have implications for system behavior |

**Collection Purpose**: Real-time moral evaluation of proposed actions to ensure alignment with the Eight Harmonies framework. Supports the EU AI Act requirement for human oversight of AI decision-making.

**Legal Basis**: Legitimate interest in ethical AI operation and compliance with AI ethics frameworks.

**Access Controls**:
- Read: Code within `cycle_phase_perception.rs` and downstream consumers of `PerceptionPhaseResult`.
- Export: Moral judgments are summarized in `CycleMetadata.ethical_telemetry` (`EthicalTelemetry` struct). Raw judgments are not exported by default.
- No persistent store exists for moral reasoning traces.

**Retention Policy**: Not retained beyond the current cycle unless the deployment operator captures `CycleMetadata` externally. If moral reasoning traces are needed for contestability or audit purposes (as recommended by EU AI Act Article 14), the deployment operator must implement explicit logging.

**Deletion Procedure**: Automatic — transient data is overwritten each cycle. No deletion action required.

**Cross-Border Transfer**: Not applicable for transient in-memory data.

**Contestability Note**: Because moral judgments are transient, deployments that require contestability of AI decisions (EU AI Act Article 14) should implement persistent logging of `EthicalTelemetry` with sufficient context to reconstruct the reasoning chain.

---

### 3.4 Neuromodulator State

| Attribute | Detail |
|-----------|--------|
| **What** | `NeurochemistryCheckpoint` — 30+ fields covering 9 transmitter levels (DA, NE, 5-HT, ACh, GABA, Oxytocin, Glutamate, Adenosine, Endocannabinoid), tolerance curves, withdrawal states, allostatic load, receptor sensitivities, phase tracking |
| **Source** | `NeuromodulatorBath` (sub-crate: `symthaea-neuromodulators`) |
| **Storage** | In-memory within `NeuromodulatorBath`. `state_vector() -> [f32; 9]` provides a compact summary |
| **Retention** | Session-scoped. Full checkpoint available via `NeurochemistryCheckpoint` serialization |
| **Sensitivity** | **Medium** — models neurochemical dynamics for cognitive modulation; no PII |

**Collection Purpose**: Modeling neurochemical dynamics that modulate attention, learning rate, exploration/exploitation balance, and consciousness metrics. Essential for the system's biologically-inspired cognitive architecture.

**Legal Basis**: Legitimate interest in system operation and research into consciousness-aware AI.

**Access Controls**:
- Read: `CognitiveLoopService` via `NeuromodTelemetry` in accessors, and the calibration bridge.
- Export: `NeurochemistryCheckpoint` is serializable (serde). `SharedCalibrationProfile` enables multi-agent sharing of neuromodulator-derived calibration data.
- Multi-agent sharing: Oxytocin-mediated coupling (`couple_with_peer()`) exchanges state vectors between agents with configurable merge weight (0.05 + oxytocin * 0.15, capped at 0.35).

**Retention Policy**: In-memory only. Checkpoints are not automatically persisted. If the deployment requires persistence (e.g., for warm-restart or longitudinal studies), the operator must implement explicit serialization.

**Deletion Procedure**: Automatic on process termination. Serialized checkpoints, if created, follow the deployment operator's file management policy.

**Cross-Border Transfer**: If `SharedCalibrationProfile` is transmitted between agents in different jurisdictions, the deployment operator must assess whether the transmitted data (performance benchmarks, neurochemical state summaries) triggers any cross-border data transfer requirements. The data does not constitute PII, but regulatory frameworks may apply to AI system operational data.

---

### 3.5 Calibration Data

| Attribute | Detail |
|-----------|--------|
| **What** | `CalibrationHistory` (sliding window of 20 entries), psych-bench z-scores, `SelfAssessmentMonitor` EMA tracking (prediction error, coherence, confidence, attention) |
| **Source** | Psych-bench battery results and internal self-assessment monitoring |
| **Storage** | In-memory. `SharedCalibrationProfile` can be serialized (serde) for multi-agent sharing |
| **Retention** | Sliding window of 20 most recent calibration entries. Warmup period of 200 cycles; cooldown of 500 cycles between recalibrations |
| **Sensitivity** | **Medium** — performance benchmarks and cognitive self-assessment metrics |

**Collection Purpose**: Continuous calibration of neuromodulator receptor sensitivities based on cognitive performance benchmarks. Ensures the system's cognitive parameters remain within validated operating ranges. Supports NIST AI RMF Measure function requirements for ongoing performance monitoring.

**Legal Basis**: Legitimate interest in system calibration and performance assurance.

**Access Controls**:
- Read: `CognitiveLoopService` calibration bridge, `SelfAssessmentMonitor`.
- Export: `SharedCalibrationProfile` serialization for multi-agent coordination. Requires explicit invocation.
- Drift detection: `CalibrationHistory.is_systematic_drift()` warns when >75% of recent entries show same-direction drift — this is an internal monitoring signal, not externally reported by default.

**Retention Policy**: Rolling window of 20 entries in memory. No automatic persistence. If longitudinal calibration tracking is needed, the deployment operator must implement external storage.

**Deletion Procedure**: Oldest entries are automatically evicted as new calibrations are added (sliding window). Full history discarded on process termination.

**Cross-Border Transfer**: Same considerations as Section 3.4 for `SharedCalibrationProfile` transmission between agents.

---

### 3.6 Substrate Telemetry

| Attribute | Detail |
|-----------|--------|
| **What** | `SubstrateTelemetry` — fields: `feasibility` (raw), `honest_confidence` (validation overlay), `effective_feasibility` (blended), `tau_factor` (temporal speed modulation), `scale_pressure` (HDC dimensionality constraint) |
| **Source** | `SubstrateManager` in the cognitive loop, computed at startup and on substrate reconfiguration |
| **Storage** | Embedded in `CycleMetadata`, exported via telemetry |
| **Retention** | Per-cycle in `CycleMetadata`. No independent long-term storage |
| **Sensitivity** | **Low** — system configuration and substrate simulation parameters |

**Collection Purpose**: Tracking the effect of substrate configuration on consciousness feasibility scores. Supports the Multiple Realizability research program and provides transparency into how substrate assumptions affect consciousness metrics.

**Legal Basis**: Legitimate interest in system transparency and research integrity.

**Access Controls**:
- Read: Any consumer of `CycleMetadata` (returned from each `cycle()` call).
- Export: Included in `CycleMetadata` by default. No separate export mechanism.
- Configuration: `SubstrateManager` can be reconfigured at runtime via `reconfigure_substrate()` and `reconfigure_composition()`.

**Retention Policy**: Same as `CycleMetadata` — transient unless the deployment operator persists it externally.

**Deletion Procedure**: Automatic with `CycleMetadata` lifecycle.

**Cross-Border Transfer**: Not applicable — low-sensitivity configuration data with no PII implications.

---

## 4. Training and Input Data Provenance (ISO 42001 A.7.4, A.10.3)

### 4.1 Overview

Symthaea does not train on external datasets at runtime. However, several data sources inform its cognitive parameters, moral reasoning, and calibration baselines. This section documents the provenance of each source for ISO 42001 A.7.4 compliance.

### 4.2 Data Sources

| Source | Type | Origin | Used For | Bias Considerations |
|--------|------|--------|----------|---------------------|
| **Psych-bench scenarios** | Synthetic cognitive tasks | Generated internally from cognitive psychology literature (Stroop, Flanker, N-back, StopSignal, CPT, PVT, DualTask, UG, RME) | Neuromodulator calibration via z-score mapping | Tasks derived from Western cognitive psychology; cross-cultural validation not yet performed |
| **Moral prototypes** | Hand-crafted ethical scenarios | Authored by Tristan Stoltz based on ethical philosophy literature | EthicsEngine prototype matching and consent detection | English-language, Western ethical frameworks dominant; Eight Harmonies framework mitigates single-framework bias |
| **Harmony basis keywords** | Curated keyword lists | Selected from philosophical and ethical texts | HDC harmony encoding (harmony_basis.rs) | Keyword selection reflects author's philosophical training; community review planned |
| **HDC encoding dictionaries** | Vocabulary mappings | Derived from standard English word lists | Text-to-hypervector encoding | English-centric; multilingual encoding not yet implemented |
| **Substrate profiles** | Scientific literature values | Neuroscience and computing literature (operation speed, energy costs, integration capacity) | SubstrateType requirement profiles | Based on published research; exotic substrates are speculative |
| **Safety thresholds** | Engineering judgment | Set by developer based on testing and IIT/GWT literature | SafetyAgent Green/Yellow/Orange/Red levels | Conservative by design; may need adjustment for deployment contexts |

### 4.3 Data Quality Assurance

| Source | Quality Mechanism | Validation |
|--------|-------------------|------------|
| Psych-bench scenarios | Peer-reviewed cognitive task designs | Psych-bench regression suite (weekly CI) |
| Moral prototypes | Adversarial test suite (26 tests) | `adversarial_moral_algebra` test |
| Harmony keywords | Manual review + HDC encoding tests | `harmony_basis` unit tests |
| Safety thresholds | Soak tests (15 scenarios, 1000 cycles each) | `safety_agent_escalation_soak` |
| Substrate profiles | Literature cross-referencing | `substrate_independence` + `substrate_validation` tests |

### 4.4 Bias Audit

| Dimension | Status | Finding | Mitigation |
|-----------|--------|---------|------------|
| Cultural bias | Acknowledged | Moral prototypes and psych-bench tasks reflect Western cognitive/ethical norms | Eight Harmonies framework provides multi-dimensional evaluation; cross-cultural expansion planned |
| Language bias | Acknowledged | English-only input processing | Multilingual HDC encoding planned for future release |
| Gender bias | Not applicable | No gender-related data in cognitive state modeling | N/A |
| Racial bias | Not applicable | No race-related data in cognitive state modeling | N/A |
| Substrate bias | Acknowledged | Biological neurons scored highest due to most evidence | Validation framework explicitly scores honest confidence; silicon/quantum marked as "Theoretical" evidence |

### 4.5 Data Lineage

All data sources are version-controlled in the git repository:

- Moral prototypes: `src/cognitive_loop/ethics_engine.rs` (moral_algebra module)
- Harmony keywords: `src/hdc/harmony_basis.rs`
- Safety thresholds: `src/cognitive_loop/thresholds.rs`
- Substrate profiles: `symthaea-core/src/hdc/substrate_independence.rs`
- Psych-bench tasks: `crates/symthaea-psych-bench/src/benchmarks/`

Changes to any of these files require CI gate passage and, for safety-critical files, Class A change procedure (see `SDLC.md`).

---

## 5. Cross-Cutting Governance Controls

### 5.1 Data Quality (ISO 42001 A.10.2, EU AI Act Art. 10(2))

All consciousness metrics are computed from deterministic mathematical operations (HDC encoding, CfC evolution, spectral Phi computation, FEP free energy). Data quality is ensured by:

- **Input validation**: Safety assessment at the start of each cycle gates processing.
- **Bounded outputs**: All metrics are clamped to defined ranges (e.g., consciousness_level in [0.0, 1.0], prediction_error non-negative).
- **Property testing**: Proptest suites verify metric stability under perturbed inputs (`proptest_feedback_stability.rs`, `proptest_threshold_sensitivity.rs`).
- **Calibration**: Psych-bench battery provides external validation against cognitive psychology benchmarks.

### 5.2 Data Provenance (ISO 42001 A.10.3)

Each data category has a clear, traceable origin:

| Category | Origin Module | Computation Chain |
|----------|--------------|-------------------|
| Consciousness Metrics | `consciousness_engine.rs` | HDC encode -> CfC evolve -> SpectralMIP Phi -> ConsciousnessEquationV2 |
| Safety Assessments | `safety_agent.rs` | Consciousness metrics -> threshold comparison -> SafetyAssessment |
| Moral Reasoning | `ethics_engine.rs` | Input -> MoralParser -> MoralAlgebra -> Eight Harmonies -> verdict |
| Neuromodulator State | `symthaea-neuromodulators` | Transmitter dynamics -> receptor binding -> tolerance/withdrawal |
| Calibration Data | `calibration/` | Psych-bench z-scores -> receptor sensitivity adjustments |
| Substrate Telemetry | `substrate_manager.rs` | SubstrateType -> requirements -> feasibility -> validation overlay |

### 5.3 Access Control Summary

| Category | Sensitivity | Internal Access | Export Mechanism | Automatic External Transmission |
|----------|-------------|-----------------|------------------|---------------------------------|
| Consciousness Metrics | Medium | CognitiveLoopService accessors | CycleMetadata | None |
| Safety Assessments | High | SafetyAgent | SafetyAuditReport (JSON/MD) | None |
| Moral Reasoning | High | EthicsEngine callers | EthicalTelemetry in CycleMetadata | None |
| Neuromodulator State | Medium | Neuromod accessors | NeurochemistryCheckpoint (serde) | None |
| Calibration Data | Medium | Calibration bridge | SharedCalibrationProfile (serde) | None |
| Substrate Telemetry | Low | CycleMetadata consumers | SubstrateTelemetry in CycleMetadata | None |

No data category has automatic external transmission. All exports require explicit programmatic action.

### 5.4 Retention Summary

| Category | In-Memory Retention | Default Persistence | Recommended Archive (if persisted) |
|----------|--------------------|--------------------|-------------------------------------|
| Consciousness Metrics | Sliding window (configurable) | None | 90 days |
| Safety Assessments | 1,000 entries | None | 5 years (regulatory) |
| Moral Reasoning | Current cycle only | None | 1 year (contestability) |
| Neuromodulator State | Full session | None | 90 days |
| Calibration Data | 20 entries | None | 1 year |
| Substrate Telemetry | Current cycle | None | 90 days |

### 5.5 Deletion Procedures

For in-memory data (all categories by default):
1. Process termination clears all in-memory state.
2. No residual data remains on disk unless the deployment operator has implemented explicit persistence.

For persisted data (deployment-operator responsibility):
1. Identify all storage locations where exported data resides.
2. Execute deletion according to the retention schedule in Section 4.4.
3. Log the deletion action (timestamp, category, volume deleted, operator identity).
4. For safety assessment data: deletion requires approval from the designated safety officer.
5. Verify deletion completeness — ensure no copies remain in backups, caches, or replicated stores.

### 5.6 Incident Response

If a data governance violation is detected (unauthorized access, unintended persistence, data corruption):

1. **Contain**: Isolate the affected data category.
2. **Assess**: Determine scope, sensitivity, and potential impact.
3. **Remediate**: Apply corrective action (delete unauthorized copies, patch access controls).
4. **Report**: Document the incident per ISO 42001 A.10.5. For safety assessment data breaches, notify the designated safety officer within 24 hours.
5. **Review**: Update this governance document if the incident reveals a gap.

---

## 6. Regulatory Alignment

### 6.1 ISO 42001 A.10 (Data for AI Systems)

| Requirement | Addressed In |
|-------------|-------------|
| A.10.1 Data management policy | This document |
| A.10.2 Data quality | Section 4.1 |
| A.10.3 Data provenance | Section 4.2 |
| A.10.4 Data integrity | Section 2.3 (append-only safety trail), Section 4.1 (bounded outputs) |
| A.10.5 Data incident management | Section 4.6 |

### 6.2 EU AI Act Article 10 (Data and Data Governance)

| Requirement | Addressed In |
|-------------|-------------|
| Art. 10(1) Data governance practices | This document (Sections 2-4) |
| Art. 10(2) Data quality criteria | Section 4.1 |
| Art. 10(3) Training/validation/test datasets | Section 4 (Training Data Provenance) — synthetic psych-bench tasks, hand-crafted moral prototypes, literature-derived substrate profiles |
| Art. 10(4) Bias examination | Not applicable for internal cognitive state data. Moral reasoning bias is addressed by the Eight Harmonies framework and ensemble judgment |
| Art. 10(5) Personal data | Section 2.2 — no PII processed |

### 6.3 NIST AI RMF Measure Function

| Measure Activity | Addressed In |
|-----------------|-------------|
| Ongoing monitoring of AI performance | Section 3.5 (Calibration Data), self-assessment monitoring |
| Tracking metrics over time | Section 3.1 (sliding window), Section 3.5 (CalibrationHistory) |
| Identifying drift | Section 3.5 (systematic drift detection) |
| Documentation of measurement approaches | Section 4.2 (provenance), Section 4.1 (quality) |

---

## 7. Review and Maintenance

This document shall be reviewed:
- **Annually**, or
- When a new data category is added to Symthaea, or
- When regulatory requirements change (ISO 42001 amendments, EU AI Act implementing acts), or
- After any data governance incident.

**Document Owner**: Symthaea project maintainer.
**Approval**: Required from project lead before publication.

---

*Consciousness-first technology serving all beings.*
