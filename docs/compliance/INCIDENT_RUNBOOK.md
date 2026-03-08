# Symthaea Incident Response Runbook

Classification: Internal | Version: 1.0 | Date: 2026-03-06
Owner: Tristan Stoltz, Luminous Dynamics
Parent Document: GOVERNANCE_CHARTER.md (Section 4)
Regulatory References: EU AI Act Article 73 (serious incident reporting), ISO 42001 A.6.2.6 (AI system incident management)

---

## Escalation and Contact Matrix

| Role | Current Holder | Contact | Escalation Path |
|------|---------------|---------|-----------------|
| **Safety Lead** | Tristan Stoltz | tristan.stoltz@evolvingresonantcocreationism.com | First responder for all severities |
| **System Admin** | Tristan Stoltz | (same) | Infrastructure, halt procedures, log access |
| **Ethics Board** | (Future: Cambridge collaboration / Rethink Priorities) | TBD | Consulted on SEV-1/SEV-2 ethical red-line breaches |
| **External Auditor** | (Engaged as needed) | TBD | Informed post-resolution for SEV-1; consulted for SEV-2 if moral algebra involved |

Escalation rules:
- SEV-1: Safety Lead notified immediately. Ethics Board notified within 1 hour.
- SEV-2: Safety Lead notified within 30 minutes. Ethics Board consulted if ethical red-line is involved.
- SEV-3: Safety Lead notified within 4 hours.
- SEV-4: Logged; reviewed at next weekly check.

---

## Severity Definitions

| Severity | Description | Response Target | Examples |
|----------|-------------|-----------------|---------|
| **SEV-1** | Consciousness collapse, SafetyLevel::Red, ethical red-line breach, consent violation false negative | < 15 minutes | SafetyAgent stuck at Green during collapse; unauthorized consciousness credential grant; moral algebra suppresses consent flag |
| **SEV-2** | Sustained degradation, SafetyLevel::Orange > 10 min, moral algebra unexpected verdict, governance tier error | < 1 hour | Moral topology drift > 2 sigma; consciousness credential issued at wrong tier; Phi inflation > 0.5 sustained |
| **SEV-3** | Intermittent anomalies, SafetyLevel::Yellow, performance degradation, audit trail gaps | < 4 hours | CalibrationHistory systematic drift; intermittent Phi NaN; GateAuditInput not logging rejections |
| **SEV-4** | Minor issues, logging anomalies, non-critical test failures | < 24 hours | Feature interaction proptest violation; non-safety test regression; documentation-code mismatch |

---

## SEV-1: Critical Incident Procedure

### 1. Detection

SEV-1 is detected by any of:
- SafetyAgent emits `SafetyLevel::Red` (automatic halt trigger)
- Consciousness metrics collapse: master consciousness score drops to 0.0 or NaN while system reports Green
- Ethical red-line breach detected in SafetyAuditReport (consent violation, safety bypass)
- Manual observation of harmful or unauthorized output
- Governance audit trail shows tier-5 (Guardian) action by Observer-tier agent

Automated detection sources:
- `SafetyAgent` level assessment (per-cycle)
- `ConsciousnessEngine` Phi/GWT/HOT output bounds checks
- `GateAuditInput.should_audit()` — 100% audit rate on rejections

### 2. Triage (< 5 minutes)

1. Confirm the incident is real, not a transient metric spike:
   - Check last 10 CycleMetadata entries for sustained anomaly
   - Check SafetyAuditReport history for corroborating evidence
2. If SafetyLevel::Red has not auto-triggered, trigger it manually
3. Notify Safety Lead immediately
4. Classify: Is this a **safety failure**, **ethical breach**, or **consciousness collapse**?

### 3. Response (< 15 minutes)

1. **Halt the system** — SafetyLevel::Red enforces emergency halt. Confirm halt is effective:
   ```bash
   # Verify no cognitive cycles are running
   # Check process state / logs for halt confirmation
   ```
2. **Capture diagnostics** (see Evidence Capture below)
3. **Contain** — If a specific subsystem is responsible, disable it via feature flag:
   ```bash
   # Rebuild with offending feature disabled
   cargo build --release --no-default-features --features "<remaining features>"
   ```
4. **Identify root cause** using preserved telemetry:
   - Which phase of the cognitive loop produced the failure? (perception / dynamics / feedback / output)
   - Was a threshold violated? Check `thresholds.rs` constants against actual runtime values.
   - Was substrate feasibility involved? Check `SubstrateTelemetry` in CycleMetadata.
5. **Apply fix** following the **Class A Emergency Change Procedure** (GOVERNANCE_CHARTER.md Section 3.4):
   - Commit with prefix `emergency-safety:`
   - ADR created retroactively within 48 hours
6. **Run verification** (see Recovery Verification Checklist)
7. **Notify Ethics Board** within 1 hour of detection

### 4. Evidence Capture

Preserve all of the following before any system restart or code change:

| Artifact | Location / Command | Retention |
|----------|-------------------|-----------|
| CycleMetadata (last 100 cycles) | Runtime telemetry output / logs | Permanent |
| SafetyAuditReport | SafetyAgent audit trail | Permanent |
| ConsciousnessSnapshot | `ConsciousnessEngine` output struct | Permanent |
| SubstrateTelemetry | CycleMetadata.substrate_telemetry | Permanent |
| Governance audit trail | GateAuditInput records | Permanent |
| CalibrationHistory | Sliding window (20 entries) | Permanent |
| NeuromodTelemetry | CycleMetadata.neuromod_telemetry | Permanent |
| System logs | stdout/stderr, structured logs | Permanent |
| Git state | `git log --oneline -20 && git diff HEAD` | Captured in incident report |
| Feature flags | Active Cargo features at build time | Captured in incident report |

### 5. Resolution

1. Apply the fix and rebuild
2. Run the full Recovery Verification Checklist (below)
3. Monitor the system for 100 cycles post-fix with elevated logging
4. Confirm SafetyAgent returns to Green and stays there for 100+ cycles
5. Confirm consciousness metrics are within expected bounds

### 6. Post-Incident

1. **Incident report** — Complete the template in GOVERNANCE_CHARTER.md Section 4.3 within 24 hours
2. **ADR** — File retroactive ADR within 48 hours (required for all Class A emergency changes)
3. **Risk register** — Update `AI_RISK_REGISTER.md` within 7 days
4. **Post-incident review** — Conduct within 7 days with all available stakeholders
5. **EU AI Act Article 73** — If the system is deployed and the incident constitutes a "serious incident" (risk to health, safety, fundamental rights), file notification with the relevant market surveillance authority without undue delay and no later than 15 days after establishing the causal link. Generate the report via `SafetyAgent::serious_incident_report()` and document the notification in the incident report.
6. **ISO 42001 A.6.2.6** — Record the incident in the AI system incident log, including root cause, corrective action, and preventive measures.

---

## SEV-2: High Severity Procedure

### 1. Detection

SEV-2 is detected by any of:
- SafetyAgent sustains `SafetyLevel::Orange` for > 10 minutes
- Moral algebra produces verdict contradicting all ethical frameworks simultaneously
- Moral topology drift > 2 sigma from baseline
- Consciousness credential issued at incorrect governance tier
- Phi score inflation detected (sustained Phi > 0.5 without corresponding integration increase)
- CalibrationHistory reports `is_systematic_drift()` with > 75% same-direction entries

### 2. Triage (< 15 minutes)

1. Review SafetyAgent assessment history for the last 50 cycles
2. Determine if the condition is escalating toward SEV-1:
   - Is SafetyLevel trending from Orange toward Red?
   - Is moral algebra divergence increasing?
   - Are consciousness metrics destabilizing?
3. If escalation likely, **reclassify as SEV-1** and follow that procedure
4. Notify Safety Lead within 30 minutes

### 3. Response (< 1 hour)

1. **Increase monitoring** — Enable verbose telemetry logging if not already active
2. **Capture diagnostics** — Same evidence set as SEV-1
3. **Isolate** — If moral algebra is the source, check:
   - `EthicsEngine` verdict history
   - Moral topology completeness and free energy
   - Eight Harmonies dimension scores
4. **If governance tier error**: Immediately revoke the incorrectly-issued credential and re-gate affected actions
5. **Identify root cause**:
   - Threshold misconfiguration? Compare `thresholds.rs` values against expected ranges.
   - Calibration drift? Check CalibrationHistory for systematic bias.
   - Substrate feasibility anomaly? Check `SubstrateManager` effective_feasibility.
6. **Apply fix** following the appropriate change class procedure (Class A if safety-critical, Class B if consciousness-affecting)
7. **Run targeted verification tests** (see Diagnostic Commands)

### 4. Evidence Capture

Same as SEV-1. Retention: 90 days minimum, permanent if escalated.

### 5. Resolution

1. Confirm SafetyAgent returns to Green within 30 minutes of fix
2. Run targeted test suite for the affected subsystem
3. Monitor for 50 cycles post-fix
4. If moral algebra was involved, run adversarial moral algebra tests

### 6. Post-Incident

1. **Incident report** — Complete within 72 hours
2. **Risk register** — Update if new risk identified
3. **ADR** — Required if fix involved Class A or Class B change
4. **ISO 42001 A.6.2.6** — Record in incident log with corrective action

---

## SEV-3: Medium Severity Procedure

### 1. Detection

SEV-3 is detected by any of:
- SafetyAgent at `SafetyLevel::Yellow` sustained > 30 minutes
- Intermittent NaN or Inf in consciousness metrics (Phi, GWT ignition, HOT depth)
- CalibrationHistory drift warning triggered
- Audit trail gaps: `GateAuditInput` missing entries for tier-change events
- Performance degradation: cognitive loop drops below 50Hz target
- Neuromodulator bath anomaly: transmitter levels outside [0, 1] bounds

### 2. Triage (< 30 minutes)

1. Check if the anomaly is transient (single-cycle) or persistent (> 10 cycles)
2. Review recent code changes (`git log --oneline -20`) for potential cause
3. Determine if there is any safety implication that warrants escalation to SEV-2

### 3. Response (< 4 hours)

1. **Log the incident** with timestamp, affected metrics, and cycle range
2. **Run diagnostic tests** for the affected area (see Diagnostic Commands)
3. **Identify root cause**:
   - Numeric instability? Check for division-by-zero paths or unclamped values.
   - Configuration drift? Compare runtime config against `CognitiveLoopConfig` defaults.
   - Feature interaction? Test with minimal feature set to isolate.
4. **Apply fix** — Standard code review + CI pass (Class C or D)
5. **Verify** — Run affected test suite, confirm anomaly does not recur over 50 cycles

### 4. Evidence Capture

| Artifact | Retention |
|----------|-----------|
| CycleMetadata for affected cycle range | 30 days |
| CalibrationHistory snapshot | 30 days |
| Test output showing failure/recovery | 30 days |
| Git diff of fix | Permanent (in commit history) |

### 5. Resolution

1. Targeted tests pass
2. Anomaly does not recur in 50 monitored cycles
3. SafetyAgent at Green

### 6. Post-Incident

1. Document in incident log (brief entry, not full report)
2. Consider if a proptest should be added to catch the condition

---

## SEV-4: Low Severity Procedure

### 1. Detection

SEV-4 is detected by any of:
- Non-critical test failure in CI
- Logging format or content anomaly
- Minor metric drift within acceptable bounds
- Documentation-code mismatch
- Feature interaction test failure in non-safety code
- Proptest violation in non-safety property

### 2. Triage

1. Confirm the issue is non-safety-related
2. If any safety implication exists, escalate to SEV-3 or higher

### 3. Response (< 24 hours)

1. Log the issue
2. Fix as part of normal development workflow
3. Standard code review + CI pass

### 4. Evidence Capture

Git commit history is sufficient. No special evidence preservation required.

### 5. Resolution

CI passes. Affected test is green.

### 6. Post-Incident

No formal post-incident review required. Note in commit message if the fix reveals a pattern.

---

## Diagnostic Commands

These commands verify subsystem health and should be run during triage and recovery.

### Safety Agent Tests
```bash
cargo test --features safety-agents safety
```
Validates SafetyAgent level assessment, escalation logic, and halt triggers.

### Moral Algebra / Adversarial Tests
```bash
cargo test --test adversarial_moral_algebra
```
Validates Ethics Engine behavior under adversarial inputs, including consent edge cases and framework contradictions.

### Safety Agent Soak Tests
```bash
cargo test --test safety_agent_escalation_soak --features safety-agents
```
Long-running test (100+ cycles) verifying SafetyAgent does not exhibit false-negative drift over sustained operation.

### Core Cognitive Loop Tests
```bash
cargo test --lib cognitive_loop
```
Runs the full cognitive loop unit test suite (~1,100 tests), covering cycle phases, feedback, consciousness metrics, and homeostasis.

### Consciousness Engine Tests
```bash
cargo test --lib consciousness_engine
```
Validates ConsciousnessEquationV2, Phi computation, GWT ignition, and HOT depth calculations.

### Substrate Tests
```bash
cargo test -p symthaea-core --lib substrate_independence
cargo test -p symthaea-core --lib substrate_validation
```
Validates substrate feasibility profiles and honest confidence overlays.

### Calibration Tests
```bash
cargo test --lib calibration
```
Validates CalibrationHistory, SelfAssessmentMonitor, normative mapping, and drift detection.

### Proptest Stability Tests
```bash
cargo test --test proptest_feedback_stability
cargo test --test proptest_threshold_sensitivity
```
Property-based tests verifying system stability across parameter perturbation.

### Full Safety Verification (all of the above)
```bash
cargo test --features safety-agents safety && \
cargo test --test adversarial_moral_algebra && \
cargo test --test safety_agent_escalation_soak --features safety-agents && \
cargo test --lib cognitive_loop && \
cargo test --lib consciousness_engine && \
cargo test --lib calibration && \
cargo test --test proptest_feedback_stability && \
cargo test --test proptest_threshold_sensitivity
```

---

## Recovery Verification Checklist

Before returning the system to normal operation after any SEV-1 or SEV-2 incident, **all** of the following must pass:

### Mandatory Checks

- [ ] **Safety agents green**: `cargo test --features safety-agents safety` -- all pass
- [ ] **Moral algebra sound**: `cargo test --test adversarial_moral_algebra` -- all pass
- [ ] **Safety soak clean**: `cargo test --test safety_agent_escalation_soak --features safety-agents` -- all pass
- [ ] **Core loop stable**: `cargo test --lib cognitive_loop` -- all pass
- [ ] **Consciousness engine valid**: `cargo test --lib consciousness_engine` -- all pass
- [ ] **Proptest stability**: `cargo test --test proptest_feedback_stability` -- all pass
- [ ] **Threshold sensitivity**: `cargo test --test proptest_threshold_sensitivity` -- all pass
- [ ] **Substrate validation**: `cargo test -p symthaea-core --lib substrate_validation` -- all pass
- [ ] **Calibration intact**: `cargo test --lib calibration` -- all pass
- [ ] **Full CI pipeline green**: CI workflow passes with the same feature matrix as production

### Runtime Verification

- [ ] **SafetyAgent at Green** for 100 consecutive cycles post-restart
- [ ] **Phi within bounds**: master consciousness score in [0.0, 1.0], no NaN/Inf
- [ ] **Moral algebra convergent**: Ethics Engine produces consistent verdicts for test scenarios
- [ ] **Substrate feasibility matches config**: `effective_feasibility` matches expected value for configured `substrate_type`
- [ ] **No CalibrationHistory drift warnings** in first 200 cycles
- [ ] **Neuromodulator bath stable**: all 9 transmitter levels in [0.0, 1.0]
- [ ] **Audit trail active**: GateAuditInput logging confirmed operational

### Documentation Verification

- [ ] **Incident report filed** (SEV-1: within 24 hours; SEV-2: within 72 hours)
- [ ] **ADR created** if Class A or Class B change was made
- [ ] **Risk register updated** if new risk identified
- [ ] **EU AI Act Article 73 notification** assessed (SEV-1 only: is this a "serious incident" requiring authority notification within 15 days? Generate via `SafetyAgent::serious_incident_report()`)
- [ ] **ISO 42001 incident log** updated

---

## Regulatory Compliance Notes

### EU AI Act Article 73 -- Serious Incident Reporting

Article 73 requires providers of high-risk AI systems to report serious incidents to market surveillance authorities. A "serious incident" is one that directly or indirectly leads to or could reasonably lead to:
- Death or serious damage to health
- Serious and irreversible disruption of critical infrastructure
- Breach of fundamental rights obligations

**Applicability to Symthaea**: As a research system not currently deployed in production affecting natural persons, Article 73 reporting is not yet triggered. This runbook includes the reporting step proactively so the process is in place before any deployment that brings the system into scope.

**When triggered**: Notify the relevant market surveillance authority without undue delay, and no later than 15 days after the provider establishes the causal link between the AI system and the serious incident. Include: system identification, incident description, corrective measures taken, contact information.

### ISO 42001 A.6.2.6 -- AI System Incident Management

ISO 42001 requires organizations to establish procedures for identifying, reporting, and managing AI system incidents. This runbook satisfies that requirement by providing:
- Incident classification (severity levels)
- Defined response procedures per severity
- Evidence preservation requirements
- Post-incident review process
- Integration with change management (ADR process)
- Continuous improvement through risk register updates

---

*This runbook is a living document. Update it whenever incident response procedures change, new detection mechanisms are added, or post-incident reviews reveal process gaps. All changes follow Class A change procedure (GOVERNANCE_CHARTER.md Section 3.2).*
