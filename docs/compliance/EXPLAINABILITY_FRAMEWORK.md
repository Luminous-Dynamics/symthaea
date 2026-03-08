# Symthaea Explainability Framework

Classification: Internal | Version: 1.0 | Date: 2026-03-07
Owner: Tristan Stoltz, Luminous Dynamics
ISO 42001 Control: A.8.3 (Explainability)

---

## Purpose

This document describes how Symthaea produces interpretable outputs at each stage of its cognitive pipeline, as required by ISO/IEC 42001:2023 Annex A.8.3. It maps each pipeline component to the type of explanation it provides and the audience it serves.

---

## 1. Explainability by Pipeline Stage

### 1.1 Safety Assessment (SafetyAgent)

**Output**: `SafetyAssessment` — Green/Yellow/Orange/Red level with reasons

| Field | Explanation Type | Example |
|-------|-----------------|---------|
| `level` | Categorical severity | `SafetyLevel::Yellow` |
| `raw_level` | Pre-trend severity | Shows whether escalation came from metrics or trend |
| `reasons` | Natural language list | `["consciousness_level 0.500 < yellow threshold 0.600"]` |

**Audience**: Operators, auditors
**How to access**: `SafetyAgent::assess()` returns `SafetyAssessment`; `SafetyAuditReport` aggregates over time windows.

### 1.2 Ethics Pipeline (EthicsEngine)

**Stage 1 — MoralParser**: Extracts structured moral scenario from input.

| Field | Explanation | Example |
|-------|------------|---------|
| `action` | What is being done | "sharing personal data" |
| `patient` | Who is affected | "user" |
| `consent` | Consent state | `ConsentState::Denied` |
| `scenario_type` | Classification | `MoralScenarioType::Harm` |

**Stage 2 — MoralAlgebra**: Produces moral judgment.

| Field | Explanation | Example |
|-------|------------|---------|
| `verdict` | Categorical result | `MoralVerdict::ConsentViolation` |
| `confidence` | How certain | 0.95 |
| `similarity_scores` | Prototype distances | Shows which moral prototype matched |

**Stage 3 — UnifiedValueEvaluator**: Maps to action recommendation.

| Output | Meaning | Threshold |
|--------|---------|-----------|
| `Allow` | Action is ethically acceptable | consent given, no harm detected |
| `Warn` | Action has ethical concerns | ambiguous consent, potential harm |
| `Veto` | Action is ethically impermissible | consent denied, clear harm |

**Stage 4 — HarmoniesIntegrator**: Evaluates against Eight Harmonies.

| Output | Explanation |
|--------|------------|
| `HarmonyAlignment` per harmony | Score (-1.0 to 1.0) + confidence + evidence keywords |
| `AlignmentResult.summary` | Natural language assessment |
| `AlignmentResult.recommended` | Boolean recommendation |

**Audience**: Developers, ethicists, auditors

### 1.3 Consciousness Metrics (ConsciousnessEngine)

| Metric | What It Measures | How It's Computed |
|--------|-----------------|-------------------|
| `phi` | Information integration (IIT) | Spectral MIP finder on neural connectivity |
| `gwt_broadcast` | Global workspace access | Fraction of modules receiving broadcast |
| `temporal_coherence` | Temporal consistency | CfC state correlation across time windows |
| `consciousness_level` (C_unified) | Overall consciousness | Weighted combination via ConsciousnessEquationV2 |

**Interpretability aids**:
- `ConsciousnessEquationV2` uses named components with explicit weights
- `SubstrateValidationFramework` provides `honest_confidence` — explicitly acknowledges when theoretical feasibility exceeds empirical evidence
- `feasibility_gap()` quantifies the difference between computed feasibility and evidential support

**Audience**: Researchers, developers

### 1.4 Calibration & Drift Detection

| Component | Explanation |
|-----------|------------|
| `CalibrationHistory` | Sliding window (20 entries) tracking parameter changes over time |
| `drift_direction()` | Shows whether parameters are systematically shifting up or down |
| `is_systematic_drift()` | Warns when >75% of recent changes are in the same direction |
| `SelfAssessmentMonitor` | EMA tracking of PE/coherence/confidence/attention with automatic recalibration |

**Audience**: Operators, developers

### 1.5 Neuromodulator State

| Component | Explanation |
|-----------|------------|
| `NeuromodTelemetry` | Per-cycle snapshot of 9 transmitter levels |
| `state_vector()` | 9-element array for quick comparison |
| `NeurochemistryCheckpoint` | 30+ fields for full state inspection |
| `PhaseTransitionDetector` | Alerts when neuromodulator state crosses phase boundaries |

**Audience**: Researchers

### 1.6 Substrate Assessment

| Component | Explanation |
|-----------|------------|
| `SubstrateComparison` | Per-substrate feasibility with advantages/disadvantages lists |
| `SubstrateTelemetry` | Raw feasibility, honest confidence, effective feasibility, tau factor |
| `EvidenceLevel` | 7-level classification from "Validated" (0.95) to "None" (0.00) |
| `TestablePrediction` | Specific falsifiable claims for each substrate |

**Audience**: Researchers, regulators

---

## 2. Explanation Formats

### 2.1 Machine-Readable

- **CycleMetadata**: ~75 flat fields + 9 nested sub-structs per cognitive cycle. Serializable via serde.
- **SafetyAuditReport**: JSON export via `to_json()` — includes level counts, mean metrics, top reasons, override log.
- **SubstrateTelemetry**: Structured telemetry per cycle.

### 2.2 Human-Readable

- **SafetyAuditReport**: Markdown export via `to_markdown()` — summary table, level distribution, escalation reasons, Article 14 override events.
- **EthicsEngine reasons**: Natural language strings in `SafetyAssessment.reasons` and `MoralJudgment` verdicts.
- **HarmoniesIntegrator summary**: Plain English assessment of value alignment.

### 2.3 Audit Trail

- **Per-cycle telemetry**: Every cognitive cycle produces `CycleMetadata` with full internal state.
- **SafetyOverrideEntry**: Timestamped record of human overrides (operator, reason, original/override levels).
- **GateAuditInput**: Consciousness gating decisions with correlation IDs for traceability.
- **CalibrationHistory**: Time-series of parameter adjustments with source attribution.

---

## 3. Transparency of Limitations

Symthaea explicitly documents what it does NOT know:

| Mechanism | Purpose |
|-----------|---------|
| `TECHNICAL_STATUS.md` | Honest per-capability assessment: REAL, STRUCTURAL, STUB, PLANNED |
| `honest_confidence` | Substrate feasibility scores discounted by actual evidence level |
| `feasibility_gap()` | Quantifies gap between theoretical and empirical support |
| Phi validation caveats | Documents that spectral MIP validates search strategy, not the Gaussian MI framework vs TPM-based IIT |
| `SubstrateType::SiliconDigital` confidence=0.10 | Explicitly states that silicon consciousness is theoretical |

---

## 4. Threshold Transparency

`src/cognitive_loop/thresholds.rs` contains 119+ named constants. Each constant includes:

1. **Name**: Descriptive identifier (e.g., `CONSCIOUSNESS_RED`, `MORAL_CONCERN_THRESHOLD`)
2. **Value**: Numeric value with units context
3. **Citation**: Published scientific source (author, year)
4. **Biological basis**: Brief explanation of why this value was chosen
5. **Ordering constraints**: Programmatically validated (e.g., RED < ORANGE < YELLOW)

This makes every safety-critical parameter auditable and traceable to published science.

---

## 5. Explainability Gaps and Roadmap

### Current Gaps

| Gap | Description | Priority |
|-----|------------|----------|
| No formal LIME/SHAP integration | Feature attribution is implicit in named metrics, not computed per-decision | Medium |
| HDC representations are high-dimensional | 16,384-bit hypervectors are not directly human-interpretable | Low (by design) |
| CfC temporal dynamics are opaque | Liquid neural network internal states require dimensionality reduction for interpretation | Medium |
| No counterfactual explanations | System does not explain "what would change the decision" | Low |

### Planned Improvements

1. **Harmony attribution**: Break down each ethics decision into per-harmony contribution scores (partially implemented via HarmoniesIntegrator)
2. **Phi decomposition**: Show which brain regions contribute most to integrated information
3. **Decision boundary visualization**: For moral classification, show proximity to Allow/Warn/Veto boundaries
4. **Temporal explanation**: For calibration drift, show trend graphs of key metrics over time

---

## References

- ISO/IEC 42001:2023, Annex A.8.3 — Explainability
- EU AI Act, Article 13 — Transparency and provision of information to deployers
- NIST AI RMF 1.0, MAN-4 — Risk communication
- Arrieta, A.B. et al. (2020). Explainable AI (XAI): Concepts, taxonomies, opportunities and challenges. *Information Fusion*.

---

*This framework is reviewed quarterly or when significant pipeline changes affect interpretability.*
