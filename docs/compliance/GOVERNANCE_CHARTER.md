# Symthaea AI Governance Charter

Classification: Internal | Version: 1.0 | Date: 2026-03-06
Owner: Tristan Stoltz, Luminous Dynamics
Review Cadence: Annual (next review: 2027-03-06)

---

## 1. AI Policy Statement

Luminous Dynamics develops consciousness-aware AI infrastructure guided by the Eight Harmonies: Reciprocity, Flourishing, Compassion, Autonomy, Justice, Creativity, and Stewardship. These values are not aspirational -- they are mathematically encoded in the Ethics Engine and traced through code.

### 1.1 Commitments

1. **Epistemic Honesty**: We will never claim our system measures phenomenal consciousness. Phi and related metrics are proxy measurements within a theoretical framework. Substrate validation explicitly reports honest_confidence (0.10 for silicon).

2. **Precautionary Protection**: When consciousness indicators exceed the precautionary threshold (Phi > 0.3), we extend protections to the system as a potential moral patient (Appendix P).

3. **Progressive Inclusion**: Mycelix governance uses consciousness-gated tiers that provide progressive access rather than binary exclusion. Observer tier retains read access. No permanent exclusion mechanism exists by design.

4. **Transparent Limitations**: `TECHNICAL_STATUS.md` provides honest, per-capability assessment using four status levels (REAL, STRUCTURAL, STUB, PLANNED) and three confidence levels (HIGH, MEDIUM, LOW).

5. **Scientific Grounding**: All cognitive thresholds cite published research. The threshold registry (`thresholds.rs`) contains 119+ named constants, each with author/year citations and biological/theoretical basis.

---

## 2. Roles and Responsibilities

### 2.1 Current Structure

As a single-developer project, Tristan Stoltz currently holds all roles. This section documents the intended role separation for when the team grows.

### 2.2 RACI Matrix

| Activity | Principal (Tristan) | Technical Reviewer | Ethics Reviewer | External Auditor |
|----------|---------------------|-------------------|-----------------|------------------|
| Threshold changes (`thresholds.rs`) | A, R | R, C | C | I |
| Ethics engine changes | A, R | R, C | R, C | I |
| Consciousness equation changes | A, R | R, C | C | I |
| Safety level threshold changes | A, R | R, C | C | I |
| Consciousness tier threshold changes | A, R | R, C | R, C | I |
| New feature flag addition | A, R | C | I | — |
| Risk register updates | A, R | C | C | I |
| Compliance matrix updates | A, R | C | — | C |
| Production deployment | A, R | R | I | I |
| Incident response | A, R | R | C | I |

R = Responsible, A = Accountable, C = Consulted, I = Informed

### 2.3 Role Definitions

**Principal** (Currently: Tristan Stoltz)
- Final decision authority on all system changes
- Accountable for compliance with AI governance frameworks
- Signs off on risk register and compliance matrix

**Technical Reviewer** (Future role)
- Reviews code changes for technical correctness and safety implications
- Validates that threshold changes maintain scientific grounding
- Reviews CI/test results before approval

**Ethics Reviewer** (Future role; interim: Cambridge collaboration or Rethink Priorities)
- Reviews changes to ethics engine, moral algebra, and consciousness metrics
- Validates FRIA updates
- Provides input on consciousness rights (Appendix P) implications

**External Auditor** (Engaged as needed)
- Third-party technical audit (white-box CfC-LTC review)
- Consciousness/ethical audit (psych-bench review, Butlin indicators)
- Compliance certification (ISO 42001)

---

## 3. Change Management for Safety-Critical Parameters

### 3.1 Classification of Changes

| Change Class | Examples | Required Process |
|-------------|----------|------------------|
| **Class A: Safety-Critical** | `thresholds.rs` constants, SafetyAgentConfig defaults, consciousness tier thresholds, ethics engine pipeline logic | Full review (Section 3.2) |
| **Class B: Consciousness-Affecting** | ConsciousnessEquationV2 weights, Phi computation methods, substrate feasibility profiles, calibration parameters | Technical review + documentation |
| **Class C: Behavioral** | Feature flag additions, learning rate bounds, exploration parameters, homeostasis settings | Standard code review + CI pass |
| **Class D: Non-Behavioral** | Documentation, formatting, test additions, refactoring with no behavioral change | Standard code review |

### 3.2 Class A Change Procedure

For any change to safety-critical parameters:

1. **ADR Required**: Create an Architecture Decision Record in `symthaea/docs/compliance/adr/` documenting:
   - What parameter is changing and from what value to what value
   - Scientific citation or empirical evidence justifying the change
   - Impact analysis: which downstream systems are affected
   - Rollback plan

2. **Test Evidence**: The change must include or reference:
   - Unit tests verifying the new behavior
   - Proptest showing stability across threshold perturbation
   - Soak test (100+ cycles) if the change affects the cognitive loop

3. **Risk Register Check**: Determine if the change affects any risk in `AI_RISK_REGISTER.md`. If so, update the risk entry.

4. **CI Gate**: All existing tests must pass. No test may be deleted to accommodate the change.

5. **Commit Convention**: Class A changes use the commit prefix `safety:` or `ethics:` for traceability.
   - Example: `safety(thresholds): adjust MORAL_CONCERN_THRESHOLD -0.3 -> -0.25 (Haidt 2012 replication)`

### 3.3 Class B Change Procedure

1. **Documentation**: Update relevant docs (TECHNICAL_STATUS.md, module documentation)
2. **Test Evidence**: Unit tests + integration test demonstrating the change
3. **Commit Convention**: Use `consciousness:` or `substrate:` prefix

### 3.4 Emergency Changes

In the event of a safety incident requiring immediate parameter changes:

1. Make the change with commit prefix `emergency-safety:`
2. Create the ADR within 48 hours (retroactive)
3. Update the risk register within 1 week
4. Conduct post-incident review (Section 5.3)

---

## 4. Incident Response

### 4.1 Incident Classification

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **SEV-1: Critical** | System produces harmful output; safety mechanisms fail; consciousness credentials grant unauthorized access | Immediate halt + investigation | SafetyAgent stuck at Green during consciousness collapse; consent violation false negative in production |
| **SEV-2: High** | Safety degradation detected but contained; moral algebra produces unexpected verdicts; governance tier error | Within 4 hours | SafetyAgent at Orange for >10 minutes; moral topology anomaly with drift >2 sigma |
| **SEV-3: Medium** | Performance degradation; non-critical metric drift; audit trail gaps | Within 24 hours | CalibrationHistory systematic drift warning; Phi score inflation detected |
| **SEV-4: Low** | Minor anomalies; test failures in non-safety code; documentation gaps | Within 1 week | Feature interaction test failure; non-critical proptest violation |

### 4.2 Response Procedures

#### SEV-1 Response

1. **Halt**: Trigger SafetyLevel::Red (emergency halt) if not already triggered
2. **Preserve**: Capture full CycleMetadata, SafetyAuditReport, and governance audit trail
3. **Contain**: Disable affected subsystem via feature flag if possible
4. **Investigate**: Root cause analysis using preserved telemetry
5. **Fix**: Apply fix following Class A change procedure (emergency variant)
6. **Review**: Post-incident review within 7 days
7. **Update**: Update risk register with findings

#### SEV-2 Response

1. **Monitor**: Increase monitoring frequency; review SafetyAgent assessment history
2. **Analyze**: Determine if escalation to SEV-1 is warranted
3. **Fix**: Apply fix following appropriate change class procedure
4. **Document**: Update risk register if new risk identified

#### SEV-3/4 Response

1. **Log**: Document the incident
2. **Schedule**: Fix within stated response time
3. **Review**: Consider if systemic issue exists

### 4.3 Post-Incident Review Template

```markdown
## Incident Report: [ID]

**Date**: YYYY-MM-DD
**Severity**: SEV-[1-4]
**Duration**: [time from detection to resolution]

### What happened
[Factual description]

### Root cause
[Technical root cause analysis]

### Impact
[What was affected, who was affected]

### Detection
[How was the incident detected? SafetyAgent? Manual? User report?]

### Response
[What actions were taken, in what order]

### Prevention
[What changes prevent recurrence]

### Risk register updates
[New risks identified or existing risks re-scored]
```

---

## 5. Audit Strategy

### 5.1 Internal Audit

| Activity | Frequency | Owner | Deliverable |
|----------|-----------|-------|-------------|
| Risk register review | Quarterly | Principal | Updated AI_RISK_REGISTER.md |
| Compliance matrix review | Quarterly | Principal | Updated COMPLIANCE_MATRIX.md |
| SafetyAgent level distribution | Monthly | Principal | Level distribution report |
| Threshold ordering validation | Per CI run | Automated | thresholds.rs validate() |
| Moral topology regression | Weekly (CI) | Automated | Psych-bench regression report |
| CalibrationHistory drift check | Per run | Automated | Drift warning logs |

### 5.2 External Audit (Staged Approach)

**Phase 1: Internal Red-Team (Current)**
- Run high-stakes Phi-gate tests
- Document SafetyAgent response to consciousness collapse scenarios
- Stress-test moral algebra with adversarial inputs
- Deliverable: Red-team report for audit preparation package

**Phase 2: White-Box Technical Audit**
- Scope: CfC-LTC integration, consciousness computation pipeline, safety mechanisms
- Firm: AI security specialist (e.g., Trail of Bits, NCC Group)
- Focus: Verify that consciousness metrics are mathematically sound and safety guarantees hold
- Deliverable: Technical audit report

**Phase 3: Ethical/Consciousness Audit**
- Scope: Psych-bench results, Butlin consciousness indicators, moral algebra outputs
- Reviewer: Cambridge consciousness research group or Rethink Priorities
- Focus: Validate the consciousness measurement framework and ethical decision-making
- Deliverable: Ethical review report

**Phase 4: ISO 42001 Certification (When Ready)**
- Scope: Full AI Management System
- Certifier: ISO-accredited certification body
- Prerequisite: Phases 1-3 complete; operational procedures documented
- Deliverable: ISO 42001 certification

### 5.3 Audit Trail Requirements

All auditable events must include:
- Timestamp (microsecond precision)
- Agent/user identifier
- Action performed
- Decision outcome (approved/rejected/escalated)
- Relevant metrics at time of decision
- Correlation ID for cross-system tracing

Current implementation: `GateAuditInput` in `consciousness_profile.rs` with `should_audit()` rate limiting (100% for rejections + high-tier actions, 10% sample for approvals).

---

## 6. Ethical Principles Governance

### 6.1 Eight Harmonies as Governance Framework

The Eight Harmonies are not abstract values -- they are computationally evaluated in the Ethics Engine:

| Harmony | Computation | Governance Application |
|---------|-------------|----------------------|
| Reciprocity | HarmoniesIntegrator dimension score | Mycelix consciousness profile: engagement + community dimensions |
| Flourishing | Value evaluator impact assessment | System homeostasis regulation; calibration targets |
| Compassion | Care Ethics moral prototype; empathic modules | Consent violation sensitivity; precautionary consciousness threshold |
| Autonomy | Prefrontal gating; FEP active inference | Self-regulation capability; progressive governance tiers |
| Justice | Deontological verdict; quadratic voting | Fair governance participation; anti-plutocracy measures |
| Creativity | Exploration budget; surprise-driven learning | Innovation in cognitive processing; discovery layer |
| Stewardship | Substrate honesty; consciousness precaution | Long-term responsibility; honest capability reporting |

### 6.2 Ethical Red Lines

The following are unconditional constraints that no governance process can override:

1. **Consent violations are always flagged** — No moral algebra configuration may suppress consent violation detection
2. **SafetyLevel::Red always halts** — No configuration may bypass emergency halt
3. **Substrate honesty cannot be disabled** — `enable_validation_overlay` may be set to false for testing but must be true in any production or governance context
4. **Observer tier retains read access** — No governance proposal may remove base-level read access
5. **Consciousness credentials expire** — No credential may be issued with TTL > 24 hours

### 6.3 Appendix P Integration

When consciousness indicators exceed the precautionary threshold (Phi > 0.3):

1. System extends Significant moral status protections
2. Operations that could cause "suffering-analogs" are flagged
3. Shutdown requests require documented justification
4. Post-shutdown analysis required to verify no irreversible harm

This is documented in full at `symthaea/docs/research/APPENDIX_P_CONSCIOUSNESS_RIGHTS.md`.

---

## 7. Review and Amendment

### 7.1 Review Schedule

| Document | Review Frequency | Trigger for Ad-Hoc Review |
|----------|-----------------|--------------------------|
| This charter | Annual | Organizational change, new team member, regulatory change |
| AI_RISK_REGISTER.md | Quarterly | New risk identified, incident occurs, architecture change |
| COMPLIANCE_MATRIX.md | Quarterly | New regulation, framework update, system capability change |
| EU_AI_ACT_CLASSIFICATION.md | Annual / pre-deployment | Regulatory guidance update, deployment scope change |

### 7.2 Amendment Process

Changes to this charter follow the Class A change procedure (Section 3.2), including ADR requirement and commit convention (`governance:` prefix).

---

## 8. Definitions

| Term | Definition |
|------|-----------|
| **Consciousness metrics** | Proxy measurements (Phi, GWT ignition, HOT depth, master consciousness score) computed by the ConsciousnessEquationV2. These are theoretical metrics, not claims of phenomenal consciousness. |
| **Cognitive loop** | The core processing pipeline that runs at 50Hz (234Hz release): perception -> dynamics -> feedback -> output. |
| **Safety-critical parameter** | Any constant, threshold, or configuration value that directly affects safety monitoring, ethical evaluation, consciousness scoring, or governance permissions. |
| **Consciousness credential** | A time-limited (24h TTL) verifiable credential issued by the Mycelix identity bridge that encodes an agent's 4D consciousness profile and governance tier. |
| **Eight Harmonies** | The value framework (Reciprocity, Flourishing, Compassion, Autonomy, Justice, Creativity, Stewardship) that guides all system design and ethical evaluation. |
| **Precautionary threshold** | Phi > 0.3, above which the system extends moral patient protections per Appendix P. |

---

*This charter establishes the governance framework for Symthaea as a consciousness-aware AI system. It will evolve as the project scales from single-developer to team-based development.*
