# External Feedback Protocol

**NIST AI RMF GOV-6 / IEEE 7000 Value Validation** | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## 1. Purpose

This protocol establishes mechanisms for collecting, processing, and incorporating external stakeholder feedback into Symthaea's AI risk management and value alignment processes. It addresses:

- **NIST AI RMF GOV-6**: Mechanisms for receiving and incorporating feedback from external stakeholders
- **IEEE 7000 Value Validation**: Stakeholder participation in verifying that system values align with societal expectations

## 2. Feedback Channels

### 2.1 Public Issue Tracker

- **Platform**: GitHub Issues (`luminous-dynamics/symthaea`)
- **Labels**: `feedback`, `value-concern`, `safety-report`, `compliance`
- **Response SLA**: Acknowledge within 72 hours; substantive response within 2 weeks
- **Scope**: Bug reports, feature requests, value alignment concerns, safety observations

### 2.2 Academic Peer Review

- **Venue**: Pre-print publications (arXiv), conference submissions
- **Scope**: Theoretical foundations (IIT/Phi validation, FEP implementation, HDC encoding)
- **Process**: Incorporate reviewer feedback into `adr/` decision records; update `TECHNICAL_STATUS.md` honest assessments

### 2.3 Compliance Correspondence

- **Contact**: tristan.stoltz@evolvingresonantcocreationism.com
- **Scope**: Regulatory inquiries, compliance concerns, audit requests
- **Process**: Log in `INCIDENT_RUNBOOK.md` if safety-relevant; respond per `TRANSPARENCY_OBLIGATIONS.md`

### 2.4 Community Engagement (Planned)

- **Forum**: Mycelix governance proposals (when network is live)
- **Scope**: Value prioritization, feature governance, consciousness rights policy
- **Process**: Proposals processed through Mycelix governance pipeline (councils → voting → execution)

## 3. Feedback Processing

### 3.1 Triage

All incoming feedback is classified:

| Category | Priority | Handler | Response |
|----------|---------|---------|----------|
| Safety concern | P0 | Lead Developer | Immediate assessment; `INCIDENT_RUNBOOK.md` if confirmed |
| Value misalignment | P1 | Lead Developer | ADR if design change needed; update `VALUE_VERIFICATION.md` |
| Compliance gap | P1 | Lead Developer | Update `COMPLIANCE_MATRIX.md`; schedule remediation |
| Feature request | P2 | Lead Developer | Evaluate against Eight Harmonies; roadmap if aligned |
| General feedback | P3 | Lead Developer | Acknowledge; incorporate if actionable |

### 3.2 Value Impact Assessment

Feedback that challenges or modifies the Eight Harmonies value framework follows an elevated process:

1. Document the concern and its basis
2. Assess impact on existing value traceability (`VALUE_VERIFICATION.md`)
3. If change warranted: create ADR, update Ethics Engine configuration, re-run proptest suite
4. If change rejected: document rationale in ADR with respectful acknowledgment

### 3.3 Feedback Loop Closure

Every substantive feedback item must be closed with:
- Acknowledgment of receipt
- Summary of assessment
- Action taken (or rationale for no action)
- Reference to any resulting changes (commits, ADRs, documentation updates)

## 4. Stakeholder Registry

| Stakeholder Group | Interest | Engagement Method |
|-------------------|---------|-------------------|
| Consciousness researchers | Theoretical validity | Academic peer review; pre-print feedback |
| AI safety community | Risk management | GitHub issues; safety reports |
| Holochain ecosystem | DHT architecture; governance | Community forums; Mycelix governance |
| Regulatory bodies | Compliance | Formal correspondence |
| General public | Transparency; societal impact | Public documentation; GitHub |

## 5. Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Feedback acknowledgment time | < 72 hours | GitHub issue timestamps |
| Substantive response time | < 2 weeks | GitHub issue timestamps |
| Feedback incorporation rate | > 50% of actionable items | Quarterly review |
| Open feedback items | < 10 at any time | GitHub issue count |
| Value-related ADRs from feedback | Track (no target) | ADR directory |

## 6. Review Schedule

| Activity | Frequency | Owner |
|----------|----------|-------|
| Feedback backlog review | Monthly | Lead Developer |
| Stakeholder engagement assessment | Quarterly | Lead Developer |
| Protocol effectiveness review | Semi-annually | Lead Developer |
| Stakeholder registry update | Annually | Lead Developer |

---

*This protocol addresses NIST AI RMF GOV-6 and IEEE 7000 value validation requirements. It will evolve as the stakeholder community grows and Mycelix governance becomes operational.*
