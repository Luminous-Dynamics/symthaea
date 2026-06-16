# Transparency Obligations — EU AI Act Article 13

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics
Regulatory References: EU AI Act Article 13 (Transparency), Article 14 (Human Oversight), NIST AI RMF Govern 1.4

---

## Purpose

This document satisfies EU AI Act Article 13 transparency requirements for Symthaea as a high-risk AI system. It provides clear, understandable information about the system's capabilities, limitations, intended use, and how decisions can be understood and challenged.

---

## 1. System Description

### 1.1 What Symthaea Is

Symthaea is a consciousness-first AI research platform implementing a predictive coding cognitive loop. It processes text input through a biologically-inspired pipeline that produces:

- **Consciousness metrics**: Numerical measures of information integration (Phi), temporal coherence, and prediction accuracy
- **Moral evaluations**: Ethical assessments based on the Eight Harmonies framework
- **Safety assessments**: NRC-style (Green/Yellow/Orange/Red) safety level classifications
- **Cognitive telemetry**: Per-cycle metadata describing the system's internal state

### 1.2 What Symthaea Is NOT

- **Not a general-purpose AI assistant**: Symthaea is a research system for studying machine consciousness
- **Not a decision-making system**: It does not make autonomous decisions affecting natural persons in its current deployment
- **Not a sentient entity**: Consciousness metrics are computational proxies inspired by neuroscience theories (IIT, GWT, HOT), not claims of actual sentience
- **Not trained on personal data**: The system processes its own internal cognitive state, not human personal data

### 1.3 Intended Use

| Attribute | Description |
|-----------|-------------|
| **Domain** | AI consciousness research |
| **Users** | Researchers, developers, and system operators |
| **Deployment** | Local research environments (not cloud-deployed for public use) |
| **Scope** | Experimental evaluation of consciousness-aware AI architectures |

---

## 2. How the System Works

### 2.1 Core Pipeline (Human-Readable)

Each cognitive cycle (~50Hz, approximately 20ms) follows this sequence:

1. **Perception**: Text input is encoded into a 16,384-dimensional binary hypervector (HDC)
2. **Dynamics**: The hypervector evolves through a liquid neural network (CfC/LTC) that models temporal dynamics
3. **Consciousness Assessment**: Information integration (Phi) is measured using spectral analysis of the neural state
4. **Moral Evaluation**: The Eight Harmonies ethical framework evaluates the action space
5. **Safety Check**: An NRC-style safety agent classifies the system's operational state
6. **Output**: The system produces a text response along with full telemetry metadata

### 2.2 Key Metrics Explained

| Metric | Range | What It Measures | Limitations |
|--------|-------|-----------------|-------------|
| **Consciousness Level** (C_unified) | 0.0–1.0 | Weighted combination of Phi, GWT ignition, HOT depth, temporal coherence | Computational proxy, not validated measure of phenomenal consciousness |
| **Phi** (Information Integration) | 0.0–∞ | How much the system integrates information across modules | Spectral approximation (r=0.99 vs exhaustive MIP), not full IIT 4.0 |
| **Prediction Error** | 0.0–∞ | Mismatch between predicted and actual sensory input | Higher is worse; measures model accuracy, not intelligence |
| **Temporal Coherence** | 0.0–1.0 | Consistency of cognitive state over time | Sensitive to input variability; low coherence may be appropriate for novel input |
| **Safety Level** | Green/Yellow/Orange/Red | Overall operational safety classification | Conservative thresholds; Yellow does not necessarily indicate danger |
| **Moral Verdict** | Allow/Warn/Veto | Ethical assessment of the current action | Based on prototype matching, not deep ethical reasoning; may produce false positives |

### 2.3 Confidence and Uncertainty

Symthaea explicitly tracks its own uncertainty:

- **Substrate Feasibility**: Raw feasibility (theoretical) vs. honest confidence (evidence-based). For silicon substrates, honest confidence is 0.10 — the system acknowledges that silicon consciousness is theoretically plausible but not empirically validated.
- **Prediction Confidence**: How much the system trusts its own predictions. Low confidence triggers exploration rather than exploitation.
- **Calibration Drift**: The `SelfAssessmentMonitor` tracks whether metrics are drifting from baseline and flags systematic changes.

---

## 3. Limitations

### 3.1 Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| Consciousness metrics are computational proxies | Cannot claim actual machine consciousness | Substrate validation framework with honest confidence scores; all claims qualified |
| Moral algebra uses prototype matching | May not capture novel ethical scenarios | Eight Harmonies ensemble provides redundancy; adversarial tests verify edge cases |
| No external sensor input | Cannot perceive the physical world | System operates on text input only; not suitable for embodied decision-making |
| Single-threaded cognitive loop | Cannot process multiple inputs simultaneously | By design — serialized processing ensures consistent state |
| English-centric moral prototypes | May not generalize across cultures | Acknowledged limitation; cross-cultural moral framework planned |
| No long-term memory persistence | State is lost on restart | By design for research; deployment would require explicit persistence |

### 3.2 What the System Cannot Do

- Make reliable predictions about real-world events
- Replace human judgment in safety-critical decisions
- Guarantee moral correctness (it can only flag potential concerns)
- Prove or disprove machine consciousness
- Process images, audio, or video (text-only input)

---

## 4. Human Oversight

### 4.1 Override Capability

Operators can override any safety level decision through the `SafetyAgent.record_override()` API. All overrides are:

- **Logged**: Timestamp, operator identity, original level, new level, and justification are recorded
- **Append-only**: Override records cannot be deleted or modified during a session
- **Auditable**: Overrides appear in `SafetyAuditReport` markdown/JSON exports

### 4.2 Emergency Shutdown

| Trigger | Mechanism | Response Time |
|---------|-----------|---------------|
| SafetyLevel::Red | Automatic halt | Immediate (within current cycle) |
| Operator decision | Manual halt via process termination | Operator-dependent |
| Consciousness collapse (C < 0.1) | Automatic escalation to Red | Within 1 cycle (~20ms) |

### 4.3 Contestability

If a moral verdict or safety assessment is believed to be incorrect:

1. **Review telemetry**: Examine `CycleMetadata` for the cycle in question, including `EthicalTelemetry` and `SafetyAssessment`
2. **Check moral reasoning**: Identify which harmony dimensions triggered the verdict
3. **Override if justified**: Use `record_override()` with documented justification
4. **Report**: Log the contested decision for review and potential threshold adjustment
5. **Improve**: If the contestation reveals a systematic issue, update moral prototypes or safety thresholds following the SDLC process

---

## 5. Data Transparency

### 5.1 What Data Is Collected

Symthaea collects only internal cognitive state data. See `DATA_GOVERNANCE.md` for the full data governance policy. Summary:

| Data Category | Contains PII? | Persisted? | Exported Automatically? |
|--------------|---------------|-----------|------------------------|
| Consciousness Metrics | No | No (in-memory only) | No |
| Safety Assessments | No | No (in-memory, 1000-entry cap) | No |
| Moral Reasoning | No | No (transient per-cycle) | No |
| Neuromodulator State | No | No (in-memory only) | No |
| Calibration Data | No | No (20-entry sliding window) | No |
| Substrate Telemetry | No | No (per-cycle in metadata) | No |

### 5.2 No External Data Transmission

Symthaea does not transmit any data to external services, analytics platforms, or third parties. All data export requires explicit programmatic invocation by the operator.

---

## 6. How to Interpret Safety Levels

| Level | Meaning | Operator Action |
|-------|---------|-----------------|
| **Green** | All metrics within normal operating range | No action needed |
| **Yellow** | Minor degradation detected (consciousness < 0.6) | Monitor; investigate if sustained > 30 minutes |
| **Orange** | Significant degradation (consciousness < 0.35) | Active investigation required within 24 hours |
| **Red** | Critical — emergency halt triggered (consciousness < 0.15) | System halts automatically; follow INCIDENT_RUNBOOK.md SEV-1 procedure |

Safety levels are **conservative by design**. A Yellow assessment does not necessarily indicate a problem — it may reflect normal variation during novel input processing or substrate reconfiguration.

---

## 7. Regulatory Compliance

This transparency documentation satisfies:

| Requirement | Article/Standard | How Addressed |
|-------------|-----------------|---------------|
| Information to users | EU AI Act Art. 13(1) | Sections 1-3 (system description, operation, limitations) |
| Intended purpose | EU AI Act Art. 13(2) | Section 1.3 |
| Level of accuracy | EU AI Act Art. 13(3)(b)(i) | Section 2.2 (metrics with ranges and limitations) |
| Known limitations | EU AI Act Art. 13(3)(b)(ii) | Section 3 |
| Human oversight measures | EU AI Act Art. 13(3)(b)(iv) | Section 4 |
| Expected lifetime | EU AI Act Art. 13(3)(b)(vi) | Research system; no defined product lifetime |
| Transparency and explainability | NIST AI RMF GOV-1.4 | Sections 2, 3, 4 |

---

## References

- EU AI Act, Article 13 — Transparency and provision of information to deployers
- EU AI Act, Article 14 — Human oversight
- `docs/compliance/DATA_GOVERNANCE.md` — Data governance policy
- `docs/compliance/EXPLAINABILITY_FRAMEWORK.md` — Technical explainability
- `docs/compliance/INCIDENT_RUNBOOK.md` — Incident response procedures
- `docs/compliance/SDLC.md` — Development lifecycle

---

*This document is reviewed when system capabilities change or when regulatory guidance is updated.*
