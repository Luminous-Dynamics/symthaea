# SOC 2 Type II Gap Analysis

Classification: Internal | Version: 1.0 | Date: 2026-03-22
Owner: Tristan Stoltz, Luminous Dynamics

---

## Purpose

Internal gap analysis mapping SOC 2 Trust Service Criteria (TSC) to Luminous Dynamics' current controls across Symthaea, Mycelix, and supporting infrastructure. This is a self-assessment to identify work required before engaging an external auditor.

> **Disclaimer**: This is not a formal audit. SOC 2 Type II requires an independent CPA firm to attest over an observation period (typically 6-12 months). This document identifies what needs to exist before that engagement begins.

---

## Summary

| Criterion | Description | Rating | Key Gap |
|-----------|-------------|--------|---------|
| CC1 | Control Environment | Partially Met | No formal infosec policy; no board/advisory oversight |
| CC2 | Communication & Information | Partially Met | No formal security awareness program |
| CC3 | Risk Assessment | Partially Met | No annual cadence; no formal risk acceptance sign-off |
| CC4 | Monitoring Activities | Partially Met | No SIEM; no centralized logging; no alert SLAs |
| CC5 | Control Activities | Partially Met | No change management board; no enforced code review |
| CC6 | Logical & Physical Access | Partially Met | No HSM; no periodic access reviews |
| CC7 | System Operations | Partially Met | No 24/7 ops; no incident response SLA |
| CC8 | Change Management | Partially Met | No CAB; no formal change approval workflow |

**Overall readiness**: ~45-55%. Strong documentation and technical controls exist, but operational processes (review cadences, formal approvals, centralized monitoring) are largely absent.

---

## CC1: Control Environment

**TSC**: The entity demonstrates a commitment to integrity and ethical values; the board of directors demonstrates independence from management and exercises oversight; management establishes structure, authority, and responsibility; the entity demonstrates commitment to competence; the entity enforces accountability.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| Governance Charter | `symthaea/docs/compliance/GOVERNANCE_CHARTER.md` with RACI matrix | Strong |
| Ethical framework | Eight Harmonies (8 values traced to code with automated tests) | Strong |
| Accountability matrix | `ACCOUNTABILITY_MATRIX.md` with decision/incident/regulatory accountability | Strong |
| Development principles | `CLAUDE.md` rules, `DEVELOPMENT_PROCEDURES.md`, change classification (A/B/C) | Strong |
| Value-based design | IEEE 7000 compliance at ~88%, value verification protocol | Strong |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No formal Information Security Policy document (distinct from governance charter) | High | 2-3 days. Write a standalone ISP covering scope, roles, acceptable use, data classification, incident reporting obligations. |
| No board of directors or advisory board providing independent oversight | High | 2-4 weeks. Establish a 2-3 person advisory board (can be informal for early-stage). Document charter, meeting cadence (quarterly), and oversight responsibilities. |
| Single-developer limitation means no segregation of duties | Medium | Ongoing. Document compensating controls (automated CI gates, mandatory test passage). Auditors will note this; mitigate with advisory board review of critical changes. |
| No formal background check or onboarding process documented | Low | 1 day. Document current process even if it is single-person. Needed for when team grows. |

### Rating: Partially Met

Strong ethical foundation and documented accountability, but lacks the organizational governance structures (board oversight, segregation of duties) that auditors expect.

---

## CC2: Communication and Information

**TSC**: The entity obtains or generates and uses relevant, quality information to support the functioning of internal control; the entity internally communicates information necessary to support the functioning of internal control; the entity communicates with external parties regarding matters affecting the functioning of internal control.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| Architecture documentation | `CLAUDE.md`, `ARCHITECTURE_OVERVIEW.md`, 100+ docs | Strong |
| Compliance documentation | 21+ compliance docs in `symthaea/docs/compliance/` | Strong |
| Transparency obligations | `TRANSPARENCY_OBLIGATIONS.md` (EU AI Act Art. 13) | Strong |
| External feedback protocol | `EXTERNAL_FEEDBACK_PROTOCOL.md` (GitHub issues, academic review) | Medium |
| Technical status honesty | `TECHNICAL_STATUS.md` with per-capability honest assessment | Strong |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No formal security awareness training or communication program | High | 1-2 days. Write a security awareness document. For single-developer, this is a self-attestation of policy knowledge. Scale to training program as team grows. |
| No formal process for communicating security policy changes to stakeholders | Medium | 1 day. Define a changelog/notification process for policy updates. Can be as simple as a dated changelog in each policy doc + commit messages. |
| No external-facing security contact or disclosure policy | Medium | 1 day. Create `SECURITY.md` at repo root with responsible disclosure process, PGP key, expected response time. |
| No formal vendor/third-party security communication | Low | 1 day. Document how security requirements are communicated to third-party dependencies (currently handled by `cargo-deny` and supply chain analysis in `ANNEX_IV_TECHNICAL_DOCUMENTATION.md` Section 7, but not framed as a communication control). |

### Rating: Partially Met

Excellent internal documentation. Gaps are primarily around formalizing communication as a deliberate control rather than relying on documentation being available.

---

## CC3: Risk Assessment

**TSC**: The entity specifies objectives; the entity identifies and assesses risks; the entity considers the potential for fraud; the entity identifies and assesses changes that could significantly impact the system of internal control.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| AI Risk Register | `AI_RISK_REGISTER.md` with 15 risks across 6 categories, scored by likelihood x impact | Strong |
| Risk Treatment Plan | `RISK_TREATMENT_PLAN.md` covering top 5 risks with treatment strategies and residual risk | Strong |
| Threat modeling | Defense cascade, immune system architecture, 7 threat types in SentinelManager | Strong |
| Impact assessment | FRIA (Fundamental Rights Impact Assessment) covering 7 rights + vulnerable groups | Strong |
| Benefits/costs analysis | `BENEFITS_COSTS_ANALYSIS.md` (NIST MAP-3) | Medium |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No annual (or any cadence-based) risk assessment review cycle | High | 1 day to define cadence. Then ongoing discipline. Add a "Last reviewed" date and "Next review" date to the risk register. Set calendar reminder. |
| No formal risk acceptance process with sign-off | High | 1-2 days. Add acceptance criteria to each risk in the register. Document who accepts residual risk and under what conditions. For single-developer, self-sign with advisory board countersign. |
| No fraud risk consideration documented | Medium | 1 day. Add a fraud risk section to the risk register. For a consciousness/AI system, this covers: data poisoning, model manipulation, credential fraud, insider threat. Some of these are already addressed (byzantine resistance, consciousness gating) but not framed as fraud controls. |
| Risk register focuses on AI-specific risks; broader operational risks (infrastructure, availability, key-person dependency) are not formally assessed | Medium | 2-3 days. Expand risk register or create a separate operational risk register covering: infrastructure failure, key-person risk, dependency supply chain, data loss. |

### Rating: Partially Met

Strong risk identification and treatment for AI-specific risks. Needs cadence, formal acceptance, and broader operational risk coverage.

---

## CC4: Monitoring Activities

**TSC**: The entity selects, develops, and performs ongoing and/or separate evaluations to ascertain whether the components of internal control are present and functioning; the entity evaluates and communicates internal control deficiencies in a timely manner.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| SafetyAgent (NRC 4-tier) | Green/Yellow/Orange/Red with escalation window (3 consecutive) | Strong |
| Prometheus metrics | 9 metrics via `observability.rs` (feature: `observability`) | Medium |
| CycleMetadata telemetry | 75+ fields per cognitive cycle | Strong |
| CalibrationHistory | Drift detection for consciousness metrics | Strong |
| Compliance dashboard in CI | `ci.yml` compliance job verifies all suites pass | Medium |
| SentinelManager | 7 threat types, interval 67, ThreatMemory with 32D HDV | Strong |
| Post-market monitoring | `POST_MARKET_MONITORING.md` (EU AI Act Art. 72) | Medium |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No SIEM or centralized log aggregation | High | 1-2 weeks. Deploy a log aggregation solution (Loki/Grafana stack or equivalent). Route SafetyAgent reports, CI results, and system logs to a central store. |
| No alert escalation SLA (e.g., "SEV-1 acknowledged within 15 minutes") | High | 1 day to define. Add response time targets to `INCIDENT_RUNBOOK.md` for each severity level. Acknowledge that single-developer means no 24/7 coverage. |
| Prometheus metrics exist but no persistent storage or dashboards | Medium | 2-3 days. Deploy Prometheus + Grafana (or use hosted). Create dashboards for key metrics. Set up alerting rules. |
| No periodic control effectiveness evaluation (e.g., "are our CI gates actually catching issues?") | Medium | 1 day to define evaluation criteria. Then quarterly review. Track false positive/negative rates for safety controls. |
| No formal deficiency communication process | Low | 1 day. Define how identified control failures are documented and tracked to resolution. Currently implicit in git commits. |

### Rating: Partially Met

Strong real-time monitoring within the application layer (SafetyAgent, telemetry). Weak on infrastructure-level monitoring, log aggregation, and operational alerting.

---

## CC5: Control Activities

**TSC**: The entity selects and develops control activities that contribute to the mitigation of risks; the entity selects and develops general control activities over technology; the entity deploys control activities through policies that establish what is expected and procedures that put policies into action.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| Consciousness gating | 4D profile, 5-tier system, `gate_consciousness()` with audit trail | Strong |
| RBAC | Mycelix consciousness tiers control access to governance functions | Strong |
| Dependency scanning | `cargo-deny` in CI pipeline | Medium |
| CI gates | Class A/B change detection, compliance dashboard, test gates | Strong |
| Feature flag discipline | 100 feature flags, documented in `DEVELOPMENT_PROCEDURES.md` | Strong |
| Change classification | A (safety-critical) / B (functional) / C (cosmetic) with different verification requirements | Strong |
| Automated testing | 21,500+ tests across workspace | Strong |
| Ethics pipeline | 3-stage (MoralParser, MoralAlgebra, UnifiedValueEvaluator) with Allow/Warn/Veto | Strong |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No Change Advisory Board (CAB) or formal change approval workflow | High | 1 day to document process. For single-developer, define self-review checklist for Class A changes + advisory board sign-off for safety-critical changes. |
| No enforced code review requirement (no branch protection, no required approvals) | High | 1 day. Enable branch protection on standalone repos (symthaea, mycelix). Private monorepo is single-developer, but standalone public repos should require PR review. Document compensating control for monorepo (CI gates + test passage). |
| No formal separation of development and production environments | Medium | 2-3 days. Document environment separation. If using Holochain conductors, define dev/staging/prod configurations. |
| No periodic review of control effectiveness | Medium | 1 day to define. Quarterly review of CI gate effectiveness, test coverage trends, and dependency scan results. |

### Rating: Partially Met

Unusually strong automated control activities (consciousness gating, ethics pipeline, extensive testing). Manual/organizational controls (approval workflows, environment separation) are the gaps.

---

## CC6: Logical and Physical Access Controls

**TSC**: The entity implements logical access security software, infrastructure, and architectures; the entity manages credentials; the entity restricts physical access; the entity manages access removal.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| Multi-factor authentication | MFA in mycelix-identity (DID registry, MFA zome) | Strong |
| Decentralized identity | DID system with verifiable credentials, recovery mechanisms | Strong |
| Consciousness tiers | 5-tier progressive access (Observer to Guardian) | Strong |
| Cryptographic auth | Ed25519 + PQC (post-quantum) authentication | Strong |
| Credential management | BWS (Bitwarden Secrets) for infrastructure credentials | Medium |
| Offline credentials | Degradation mode with `offline_credential.rs`, TTL-bounded | Medium |
| Sub-passports | `sub_passport.rs` with revocation audit | Medium |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No HSM (Hardware Security Module) for key storage | Medium | 2-4 weeks + cost. Evaluate YubiHSM or cloud HSM (AWS CloudHSM, GCP Cloud KMS). For early stage, document risk acceptance for software key storage with BWS as compensating control. |
| No formal periodic access review process | High | 1 day to define. Document quarterly review of: BWS access, SSH keys, GitHub permissions, Holochain conductor access. For single-developer, this is a self-audit checklist. |
| No automated access provisioning/deprovisioning | Low | N/A for single-developer. Document the process for when team grows. |
| No formal password/credential rotation policy | Medium | 1 day. Define rotation cadence for BWS master, SSH keys, API tokens. Document in ISP. |
| Physical access controls not documented | Low | 1 day. Document physical security of development machine and any servers. Even "single locked room" counts. |
| No session timeout or idle lockout policy documented | Low | 1 day. Document existing OS-level controls (screen lock, SSH timeout). |

### Rating: Partially Met

Strong cryptographic and identity controls at the application layer. Infrastructure-level access management (HSM, rotation, reviews) needs formalization.

---

## CC7: System Operations

**TSC**: The entity detects and monitors events; the entity identifies and evaluates anomalies; the entity evaluates events to determine whether they constitute incidents; the entity responds to identified security incidents; the entity mitigates and recovers from identified incidents.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| Incident runbook | `INCIDENT_RUNBOOK.md` with SEV-1 through SEV-4 definitions | Strong |
| Defense cascade | Graduated Yellow/Orange/Red responses with moral algebra filter | Strong |
| Offline credential degradation | Graceful degradation when network unavailable | Medium |
| Anomaly detection | SafetyAgent escalation window, CalibrationHistory drift detection, ThreatMemory | Strong |
| Serious incident reporting | `SeriousIncidentReport` struct (EU AI Act Art. 73) | Medium |
| Immune system | SentinelManager + ThreatMemory + CollectiveImmunity | Strong |
| Reputation slashing | Decay 0.998^days, slash 0.5x, blacklist <0.05 | Medium |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No 24/7 operations team or on-call rotation | High | Ongoing organizational gap. For early stage, document expected response times (e.g., "best effort within 4 hours during business hours, next business day otherwise"). Be honest with auditors. |
| No SLA for incident response times | High | 1 day. Add response time targets per severity to `INCIDENT_RUNBOOK.md`. SEV-1: 1hr acknowledge, 4hr mitigate. SEV-2: 4hr acknowledge, 24hr mitigate. Etc. |
| Incident runbook has never been exercised (no tabletop or drill) | Medium | 1 day. Run a tabletop exercise against the runbook. Document results and lessons learned. Repeat annually. |
| No formal incident post-mortem template or process | Medium | 1 day. Create a post-mortem template (timeline, root cause, action items, lessons). Store completed post-mortems in `docs/postmortems/`. |
| No backup and recovery testing | Medium | 1-2 days. Document backup strategy (what is backed up, frequency, retention). Test restore procedure. For Holochain DHT, document peer recovery expectations. |
| No business continuity / disaster recovery plan | Medium | 2-3 days. Document: RPO, RTO, recovery procedures, key-person contingency. |

### Rating: Partially Met

Strong automated incident detection and response at the application layer. Organizational incident response (staffing, SLAs, drills, BCP) is the gap.

---

## CC8: Change Management

**TSC**: The entity authorizes, designs, develops, configures, documents, tests, approves, and implements changes to infrastructure, data, software, and procedures.

### What Exists

| Control | Evidence | Strength |
|---------|----------|----------|
| Change classification | Class A (safety-critical) / B (functional) / C (cosmetic) in `SDLC.md` | Strong |
| CI gates | Automated test passage required, compliance dashboard | Strong |
| Governance charter enforcement | Documented change procedures for safety-critical parameters | Medium |
| Feature flags | 100 flags for progressive rollout, documented discipline | Strong |
| ADR process | ADR-001 written, template in place | Medium |
| Development procedures | `DEVELOPMENT_PROCEDURES.md` covering threshold protocol, CI pipeline, testing hierarchy | Strong |

### Gaps

| Gap | Severity | Effort to Close |
|-----|----------|-----------------|
| No Change Advisory Board (CAB) or formal approval workflow for changes | High | 1 day to define. For single-developer: Class A changes require documented self-review checklist + advisory board notification. Class B requires CI passage. Class C is at developer discretion. |
| No formal change request/ticket tracking | Medium | 1-2 days. Adopt GitHub Issues or a lightweight tracker for change requests. Link commits to issues. Currently changes are tracked only in git history. |
| No rollback procedures documented | Medium | 1 day. Document rollback strategy per change class. For Holochain zome upgrades, document DHT migration rollback (or lack thereof). |
| No formal testing requirements matrix per change type | Low | 1 day. Already partially addressed by change classification. Formalize: Class A requires full test suite + proptest + manual review. Class B requires relevant test suite. Class C requires unit tests. |
| No post-deployment verification checklist | Low | 1 day. Define smoke tests to run after each deployment. |

### Rating: Partially Met

Good automated controls and classification system. Needs formal approval workflows and change tracking.

---

## Additional Trust Service Categories (if pursuing)

### Availability

| What Exists | Gaps |
|-------------|------|
| Holochain DHT provides inherent redundancy | No uptime SLA defined |
| Offline credential degradation | No capacity planning |
| Defense cascade with graceful degradation | No load testing results documented |

**Rating**: Not Met (no formal availability commitments)
**Effort**: 3-5 days to define SLAs, document capacity, run load tests

### Confidentiality

| What Exists | Gaps |
|-------------|------|
| PQC encryption (post-quantum ready) | No data classification scheme |
| Identity vaults, health vaults | No formal data handling procedures |
| BWS for secrets management | No DLP (Data Loss Prevention) controls |
| Holochain agent-centric (no central data store) | No encryption-at-rest verification |

**Rating**: Partially Met
**Effort**: 3-5 days to formalize data classification and handling procedures

### Processing Integrity

| What Exists | Gaps |
|-------------|------|
| Phi validation (r=0.99) | No formal input validation policy |
| 21,500+ tests | No reconciliation procedures |
| CalibrationHistory drift detection | No processing error rate tracking |
| Ethics pipeline (3-stage) | |

**Rating**: Partially Met
**Effort**: 2-3 days to formalize

### Privacy

| What Exists | Gaps |
|-------------|------|
| GDPR 95% self-assessed | No formal privacy policy published |
| Data governance documentation | No privacy impact assessment for all data types |
| Agent-centric architecture (data sovereignty) | No data subject request handling procedure |

**Rating**: Partially Met
**Effort**: 3-5 days to formalize

---

## Prioritized Remediation Roadmap

### Phase 1: Foundation Documents (1-2 weeks)

These are prerequisites that close multiple gaps across criteria simultaneously.

| Action | Closes Gaps In | Effort |
|--------|---------------|--------|
| Write formal Information Security Policy (ISP) | CC1, CC2, CC6 | 2-3 days |
| Add response time SLAs to INCIDENT_RUNBOOK.md | CC4, CC7 | 1 day |
| Create SECURITY.md with disclosure policy | CC2 | 1 day |
| Define access review checklist and rotation policy | CC6 | 1 day |
| Add risk acceptance sign-off and review dates to risk register | CC3 | 1 day |
| Document CAB-equivalent process for Class A changes | CC5, CC8 | 1 day |
| Add fraud risk section to risk register | CC3 | 1 day |
| Create post-mortem template | CC7 | 1 day |

### Phase 2: Operational Controls (2-4 weeks)

| Action | Closes Gaps In | Effort |
|--------|---------------|--------|
| Deploy log aggregation (Loki/Grafana or equivalent) | CC4 | 1-2 weeks |
| Deploy persistent Prometheus + Grafana dashboards | CC4 | 2-3 days |
| Establish 2-3 person advisory board | CC1 | 2-4 weeks |
| Enable branch protection on standalone repos | CC5 | 1 day |
| Run incident response tabletop exercise | CC7 | 1 day |
| Document BCP/DR plan with RPO/RTO | CC7 | 2-3 days |
| Document environment separation (dev/staging/prod) | CC5 | 2-3 days |

### Phase 3: Observation Period Preparation (1-3 months)

| Action | Closes Gaps In | Effort |
|--------|---------------|--------|
| Begin quarterly risk register reviews (document each review) | CC3 | Ongoing |
| Begin quarterly access reviews (document each review) | CC6 | Ongoing |
| Begin quarterly control effectiveness evaluations | CC4, CC5 | Ongoing |
| Accumulate 6-12 months of evidence (logs, reviews, incidents) | All | Time |
| Evaluate HSM options for key storage | CC6 | 2-4 weeks |
| Engage SOC 2 auditor for readiness assessment | All | External |

---

## Cost Estimate

| Category | Estimated Cost | Notes |
|----------|---------------|-------|
| Documentation effort (Phase 1) | ~40 hours internal | Single developer time |
| Infrastructure (Phase 2) | $50-200/month | Hosted Grafana Cloud or self-hosted |
| Advisory board (Phase 2) | $0-5,000/year | Informal advisors or compensated |
| HSM (Phase 3, optional) | $500-2,000 one-time | YubiHSM 2 or cloud KMS |
| SOC 2 audit engagement | $20,000-50,000 | Type II, single trust service category |
| Total first-year estimate | $25,000-60,000 | Excluding internal labor |

---

## Honest Assessment

**Strengths auditors would recognize:**
- Unusually thorough documentation for a project of this size
- Automated control activities (consciousness gating, CI gates, ethics pipeline) far exceed typical startups
- Strong cryptographic foundation (Ed25519 + PQC, DID, Holochain agent-centric)
- Scientific rigor (119+ named constants with citations, Phi validation)
- Self-awareness of limitations (confidence ratings, honest_confidence on substrate)

**Concerns auditors would raise:**
- Single-developer organization lacks segregation of duties (most significant finding)
- No operational track record for incident response (runbook never exercised)
- No centralized logging or SIEM
- Many controls are "documented but never exercised" (procedural controls without operational evidence)
- No formal board or advisory oversight
- Observation period has not begun (Type II requires 6-12 months of operating evidence)

**Recommendation**: Focus Phase 1 documentation work first (low cost, high impact). Establish advisory board early. Begin observation period as soon as Phase 1 and Phase 2 controls are in place. Engage an auditor for a readiness assessment before committing to a full Type II engagement.

---

## References

- AICPA Trust Service Criteria (2017, updated 2022)
- SOC 2 Reporting on an Examination of Controls (AT-C Section 205)
- Existing compliance work: `symthaea/docs/compliance/COMPLIANCE_MATRIX.md`
- Risk register: `symthaea/docs/compliance/AI_RISK_REGISTER.md`
- Incident runbook: `symthaea/docs/compliance/INCIDENT_RUNBOOK.md`
- Governance charter: `symthaea/docs/compliance/GOVERNANCE_CHARTER.md`

---

*Internal assessment. Not a substitute for independent audit. Review quarterly.*
