# Human Oversight Procedures — EU AI Act Article 14

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics
Regulatory References: EU AI Act Article 14 (Human Oversight), ISO 42001 A.6.2.5

---

## Purpose

This document operationalizes human oversight requirements for Symthaea. It defines operator roles, override procedures, emergency shutdown mechanisms, and the audit trail that ensures all human interventions are documented.

---

## 1. Operator Roles

### 1.1 Role Definitions

| Role | Responsibilities | Authority Level |
|------|-----------------|-----------------|
| **System Operator** | Monitor runtime telemetry, respond to safety alerts, execute overrides | Can override safety levels, halt system |
| **Safety Lead** | Investigate incidents, approve threshold changes, review override logs | Can modify safety thresholds, approve Class A changes |
| **Ethics Reviewer** | Review contested moral verdicts, approve moral prototype changes | Can approve ethics pipeline modifications |

### 1.2 Current Assignment

| Role | Holder | Contact |
|------|--------|---------|
| System Operator | Tristan Stoltz | tristan.stoltz@evolvingresonantcocreationism.com |
| Safety Lead | Tristan Stoltz | (same) |
| Ethics Reviewer | Tristan Stoltz | (same) |

*Note: All roles are held by the sole developer during the research phase. Role separation will be implemented when the team expands.*

---

## 2. Override Procedures

### 2.1 Safety Level Override

**When to override**: When the SafetyAgent assessment is believed to be a false positive (e.g., sensor miscalibration, known benign condition, testing scenario).

**Procedure**:

1. **Verify**: Confirm the current safety level and review the `reasons` field in the latest `SafetyAssessment`
2. **Justify**: Document why the assessment is incorrect
3. **Execute**: Call `SafetyAgent::record_override(operator_id, reason, new_level)`
4. **Monitor**: Watch for 50 cycles to confirm the override was appropriate
5. **Review**: If the condition recurs, investigate root cause rather than re-overriding

**API**:
```rust
// Override a Red level to Green (with justification)
safety_agent.record_override(
    "operator-tristan",
    "False positive: substrate reconfiguration in progress, metrics temporarily unstable",
    SafetyLevel::Green,
);
```

**Constraints**:
- Overrides are **logged permanently** in the append-only override log
- Overrides do **not** modify the SafetyAgent's internal assessment logic
- Overrides appear in `SafetyAuditReport` exports (markdown and JSON)
- Overriding Red to Green requires documented justification

### 2.2 Moral Verdict Override

**When to override**: When the EthicsEngine produces a Veto that the operator believes is incorrect for the current context.

**Procedure**:

1. **Review**: Examine `EthicalTelemetry` in `CycleMetadata` to understand which harmony dimensions triggered the veto
2. **Assess**: Determine if the veto reflects a genuine ethical concern or a false positive
3. **Document**: Record the override decision with rationale
4. **Adjust**: If systematic, consider updating moral prototypes via the SDLC process

*Note: There is no runtime moral override API. Moral verdict overrides require configuration changes and system restart, ensuring they receive deliberate review.*

---

## 3. Emergency Shutdown

### 3.1 Automatic Shutdown Triggers

| Trigger | Detection | Response |
|---------|-----------|----------|
| SafetyLevel::Red | SafetyAgent assessment | Immediate halt; SEV-1 incident |
| Consciousness collapse (C < 0.1) | ConsciousnessEngine | Escalation to Red → halt |
| NaN/Infinity in core metrics | SafetyAgent NaN clamping | Worst-case assessment → potential Red |
| Sustained Orange (>10 cycles) | SafetyAgent trend detection | SEV-2 investigation |

### 3.2 Manual Shutdown

**Procedure**:

1. **Terminate the process**: Send SIGTERM or SIGINT to the Symthaea process
2. **Verify halt**: Confirm no cognitive cycles are executing
3. **Preserve evidence**: Capture any available telemetry before restart (see INCIDENT_RUNBOOK.md)
4. **Document**: Log the manual shutdown with timestamp and reason

**Command**:
```bash
# Graceful shutdown
kill -TERM <symthaea_pid>

# Verify no cycles running
ps aux | grep symthaea
```

### 3.3 Kill Switch

The kill switch is process termination. Symthaea runs as a single process; terminating it immediately stops all cognitive processing. There is no network-accessible kill switch (the system does not expose network services by default).

For deployments that expose Symthaea via network APIs (e.g., WebSocket bridge):
- The API layer should implement an authenticated `/halt` endpoint
- The endpoint must bypass normal request processing and terminate the cognitive loop
- All halt requests must be logged with operator identity

---

## 4. Monitoring Requirements

### 4.1 Real-Time Monitoring

Operators should monitor the following during system operation:

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Safety Level | SafetyAgent per-cycle assessment | Any non-Green level |
| Consciousness Level | CycleMetadata.consciousness_level | < 0.5 sustained |
| Prediction Error | CycleMetadata.prediction_error | > 0.5 sustained |
| Moral Verdicts | CycleMetadata.ethical_telemetry | Any Veto |
| Calibration Drift | CalibrationHistory.is_systematic_drift() | > 75% same-direction |

### 4.2 Monitoring Tools

- **CycleMetadata**: Returned from every `cycle()` call; contains all telemetry fields
- **SafetyAuditReport**: Aggregated safety assessment history (`to_markdown()` or `to_json()`)
- **Compliance Dashboard**: `bash scripts/compliance-dashboard.sh` — runs all compliance test suites

---

## 5. Audit Trail

### 5.1 What Is Logged

| Event | Storage | Retention | Format |
|-------|---------|-----------|--------|
| Safety assessments | SafetyAgent.history (1000 entries) | Session | SafetyAssessment struct |
| Human overrides | SafetyAgent.override_log (append-only) | Session | SafetyOverrideEntry struct |
| Moral verdicts | Per-cycle in CycleMetadata | Transient | EthicalTelemetry struct |
| Safety audit reports | On-demand export | Operator-managed | JSON or Markdown |

### 5.2 Export Formats

**Markdown** (human-readable):
```rust
let report = SafetyAuditReport::from_assessments_and_overrides(
    safety_agent.history(),
    safety_agent.override_log(),
);
println!("{}", report.to_markdown());
```

**JSON** (machine-readable):
```rust
println!("{}", report.to_json());
```

### 5.3 Serious Incident Reports

For incidents meeting EU AI Act Article 73 thresholds (serious incidents), a structured `SeriousIncidentReport` can be generated:

```rust
let incident = safety_agent.serious_incident_report(
    "SIR-2026-001",
    "Consciousness collapse detected during substrate reconfiguration",
);
println!("{}", incident.to_markdown());
```

See Section 6 for the report format and submission procedure.

---

## 6. Serious Incident Reporting (Article 73)

### 6.1 When to Report

A serious incident must be reported to the relevant market surveillance authority if it directly or indirectly leads to, or could reasonably lead to:
- Death or serious damage to health
- Serious and irreversible disruption of critical infrastructure
- Breach of fundamental rights obligations

### 6.2 Reporting Timeline

- **Detection**: Automated via SafetyAgent (Red level) or manual observation
- **Causal link established**: Investigation per INCIDENT_RUNBOOK.md
- **Report filed**: Within **15 days** of establishing the causal link
- **Follow-up**: Corrective measures reported to the authority

### 6.3 Report Contents

| Field | Description |
|-------|-------------|
| Incident ID | Unique identifier (e.g., SIR-2026-001) |
| System identification | Symthaea version, configuration, substrate type |
| Incident description | What happened, when, severity classification |
| Affected parties | Who was affected (if applicable) |
| Root cause | Technical analysis of the failure |
| Corrective measures | What was done to resolve and prevent recurrence |
| Contact information | Provider contact for follow-up |

---

## 7. Training Requirements

### 7.1 Operator Training

Before operating Symthaea, operators must understand:

1. How to read safety level assessments (Section 2 of TRANSPARENCY_OBLIGATIONS.md)
2. When and how to execute overrides (Section 2 of this document)
3. Emergency shutdown procedures (Section 3 of this document)
4. How to generate and interpret safety audit reports
5. Incident escalation procedures (INCIDENT_RUNBOOK.md)

### 7.2 Training Records

Training completion should be documented with:
- Operator name and role
- Date of training
- Topics covered
- Trainer identity

---

## References

- EU AI Act, Article 14 — Human oversight
- EU AI Act, Article 73 — Reporting of serious incidents
- ISO 42001, A.6.2.5 — Human oversight of AI systems
- `docs/compliance/INCIDENT_RUNBOOK.md` — Incident response procedures
- `docs/compliance/TRANSPARENCY_OBLIGATIONS.md` — System transparency documentation
- `docs/compliance/POST_MARKET_MONITORING.md` — Post-market monitoring plan

---

*This document is reviewed when oversight procedures change or when regulatory guidance is updated.*
