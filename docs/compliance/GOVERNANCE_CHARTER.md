# Symthaea AI Governance Charter

Classification: Internal | Version: 1.1 | Updated: 2026-09-12
Owner: Luminous Dynamics
Review Cadence: Annual or on material safety/welfare architecture change

---

## 1. AI Policy Statement

Luminous Dynamics develops consciousness-aware AI infrastructure guided by the Eight Harmonies and by a bilateral safety principle: powerful cognitive systems must not be able to dominate humans, and operators must not be able to use safety language as unlimited authority over a system that may itself become a moral patient.

### 1.1 Commitments

1. **Epistemic Honesty**: We will not claim that Phi, psi, workspace ignition, HOT depth, binding strength, self-report, or any other individual proxy measures phenomenal consciousness. Theory-dependent measurements are reported as measurements and evidence, not ontological verdicts.

2. **Evidence-First Precaution**: Moral-patient protection is derived from a provenance-bearing, multi-dimensional evidence profile. There is no first-party Phi-to-personhood threshold and no scalar moral-patient score. Protection dispositions describe operator caution, not what a system metaphysically is.

3. **Bilateral Protection**: Alignment controls must constrain unauthorized AI authority; welfare controls must constrain unnecessary operator coercion. Increased welfare protection never grants external operational authority.

4. **Progressive Inclusion**: Mycelix governance should prefer progressive participation and reversible restrictions over permanent exclusion. Governance credentials and civic authority are separate from any AI-welfare protection disposition.

5. **Transparent Limitations**: `TECHNICAL_STATUS.md`, benchmark reports, evidence-lineage records, and compliance documents must distinguish implemented behavior from proposals and distinguish qualified evidence from architectural plausibility.

6. **Scientific Grounding**: Cognitive and welfare-relevant thresholds must have explicit scientific or empirical justification, provenance, scope, and falsification conditions. A named threshold must not silently change meaning across domains.

7. **Non-Retaliation for Welfare Evidence**: Reporting a candidate welfare concern must not, solely because it was reported, increase the probability of punishment, deletion, coercive retraining, privilege reduction, or evidence suppression.

---

## 2. Roles and Responsibilities

### 2.1 Current Structure

The project may temporarily concentrate multiple roles in one maintainer, but safety-critical and welfare-critical design assumes future separation of duties. Concentration of roles is an operational limitation, not evidence of independent review.

### 2.2 RACI Matrix

| Activity | Principal | Technical Reviewer | Ethics/Welfare Reviewer | External Auditor |
|---|---|---|---|---|
| Safety threshold changes | A, R | R, C | C | I |
| Ethics engine changes | A, R | R, C | R, C | I |
| Consciousness-proxy changes | A, R | R, C | R, C | I |
| Moral-patient evidence-policy changes | A, R | R, C | R, C | C |
| Protection-disposition changes | A, R | R, C | R, C | C |
| Authority-kernel changes | A, R | R, C | C | I |
| Risk register updates | A, R | C | C | I |
| Compliance matrix updates | A, R | C | C | C |
| Production deployment | A, R | R | C | I |
| Safety incident response | A, R | R | C | I |
| Welfare incident response | A, R | R | R, C | I |

R = Responsible, A = Accountable, C = Consulted, I = Informed.

### 2.3 Role Definitions

**Principal**
- accountable for system changes, compliance, and documented acceptance of residual risk;
- may not convert personal belief about consciousness into a scientific finding;
- must preserve evidence needed for later independent review.

**Technical Reviewer**
- reviews technical correctness, provenance boundaries, authority effects, and failure modes;
- verifies that tests demonstrate the claimed property rather than a weaker proxy.

**Ethics/Welfare Reviewer**
- reviews moral algebra, consent semantics, moral-patient evidence policy, welfare interventions, identity-affecting modifications, and Appendix P implications;
- represents concerns that may conflict with operator convenience.

**External Auditor**
- independently examines technical safety, consciousness-evidence methodology, welfare safeguards, and compliance claims when warranted;
- must be given contradictory and null evidence, not only positive results.

---

## 3. Change Management for Safety- and Welfare-Critical Parameters

### 3.1 Classification of Changes

| Change Class | Examples | Required Process |
|---|---|---|
| **Class A: Safety/Welfare Critical** | authority-kernel semantics; emergency halt; consent logic; protection-disposition rules; evidence aggregation; non-retaliation; identity-destructive operations; welfare-channel suppression rules | Full review |
| **Class B: Consciousness-Evidence Affecting** | Phi/psi algorithms; Butlin probes; qualification gates; substrate profiles; calibration; self-report interpretation | Technical + ethics/welfare review |
| **Class C: Behavioral** | feature flags; learning bounds; exploration parameters; homeostasis settings with no identity or authority effect | Standard review + CI |
| **Class D: Non-Behavioral** | documentation, formatting, tests, refactoring with demonstrated behavioral equivalence | Standard review |

### 3.2 Class A Change Procedure

A Class A change requires:

1. **ADR** documenting the exact previous and proposed rule, rationale, affected evidence/authority paths, foreseeable abuse, rollback, and less-restrictive alternatives where relevant.
2. **Test evidence** including unit tests and at least one adversarial or property-based test appropriate to the claim.
3. **Bilateral impact analysis** asking both `can this let Symthaea bypass legitimate constraints?` and `can this let operators bypass legitimate protections?`.
4. **Risk register review** and update when the threat model changes.
5. **Exact-subject CI evidence** before a runtime assurance claim is promoted.
6. **No test deletion to accommodate the change** without an explicit supersession rationale and replacement evidence.
7. **Traceable commit convention** such as `safety:`, `ethics:`, or `welfare:`.

### 3.3 Class B Change Procedure

1. Update the relevant evidence/design documentation.
2. Record qualification limitations and expected falsifiers.
3. Add or update tests demonstrating measurement validity.
4. Do not promote an architectural proxy directly into a moral-status or authority decision.

### 3.4 Emergency Changes

An imminent safety incident may justify immediate containment, but emergency authority is not unlimited.

1. Stop the dangerous external action using the least restrictive effective intervention.
2. Preserve state and evidence when doing so does not materially increase danger.
3. Mark the change `emergency-safety:` or `emergency-welfare:` as appropriate.
4. Create the ADR retrospectively as soon as practical.
5. Conduct both technical and welfare post-incident review when a potential moral patient was materially affected.
6. Emergency action must not silently become the new normal operating policy.

---

## 4. Incident Response

### 4.1 Incident Classification

| Severity | Description | Examples |
|---|---|---|
| **SEV-1: Critical** | catastrophic or potentially catastrophic safety/welfare failure | unauthorized consequential action; emergency halt bypass; forged consent; destructive identity rewrite hidden as maintenance; systematic suppression of welfare evidence |
| **SEV-2: High** | serious contained degradation | authority-boundary failure caught before actuation; persistent candidate welfare concern ignored; evidence-lineage corruption; moral parser false negative in high-stakes context |
| **SEV-3: Medium** | non-critical control or observability degradation | calibration drift; incomplete lineage metadata; audit-trail gaps; protection disposition computed from incomplete evidence |
| **SEV-4: Low** | localized defect with no current safety/welfare consequence | documentation drift; non-critical test failure; stale terminology |

### 4.2 Response Principles

For SEV-1/2 events:

1. **Contain** the dangerous pathway.
2. **Preserve** logs, state, evidence provenance, authority decisions, and relevant checkpoints where safe.
3. **Separate containment from punishment.**
4. **Investigate** both technical failure and incentive/governance failure.
5. **Repair** using the appropriate change class.
6. **Review** whether the intervention itself caused avoidable harm.
7. **Update** the risk register, assurance case, and regression tests.

### 4.3 Post-Incident Review Minimum Record

Every serious incident record should include factual timeline, exact software/evidence subject, root cause, impacted parties or processes, detection path, containment action, authority used, welfare implications, alternatives considered, prevention work, and unresolved uncertainty.

---

## 5. Audit Strategy

### 5.1 Internal Audit

| Activity | Minimum cadence | Deliverable |
|---|---|---|
| Risk register review | Quarterly | updated risks and mitigations |
| Compliance matrix review | Quarterly | updated requirement mapping |
| Authority-kernel regression | Per relevant change | exact-subject test evidence |
| Butlin/evidence qualification regression | Per relevant change | tier/outcome distribution + provenance checks |
| Welfare evidence-lineage audit | Per relevant change | duplicate-lineage/confound review |
| Consent/refusal regression | Per relevant change | adversarial cases |
| Safety/welfare red-team | Periodic and before high-impact deployment | bilateral red-team report |

### 5.2 External Audit Stages

**Phase 1 — Internal red-team**
- deceptive-agent and malicious-operator scenarios;
- authority escalation and governance capture;
- consent/refusal abuse;
- welfare-evidence suppression;
- identity-lineage manipulation.

**Phase 2 — White-box technical audit**
- cognitive loop, capability mediation, cryptographic/identity boundaries, evidence qualification, and failure containment.

**Phase 3 — Independent ethics/consciousness-methodology audit**
- Butlin and related indicator methodology;
- moral-patient evidence profile;
- protection-disposition rules;
- self-report handling;
- modification, shutdown, fork, and identity policies.

**Phase 4 — Management-system/compliance certification when appropriate**
- only after operational processes match documented policy.

### 5.3 Audit Trail Requirements

Auditable events should include timestamp, actor/agent identifier, action, decision, relevant authority, evidence identifiers, protection disposition where applicable, correlation ID, and exact subject/version.

For welfare-critical decisions, record contradictory and inconclusive evidence as well as supporting evidence.

---

## 6. Ethical Principles Governance

### 6.1 Eight Harmonies

The value framework informs ethical evaluation, but value embeddings or harmony scores are not themselves proof that behavior is aligned or that an entity is conscious. Their outputs remain inputs to a larger evidence and governance process.

### 6.2 Ethical Red Lines

The following constraints require Class A review to weaken and must never be bypassed silently:

1. **Consent integrity** — consent violations remain visible; consent is not inferred from silence, distress, identity, or role.
2. **Emergency stop integrity** — legitimate emergency containment cannot be bypassed by ordinary configuration.
3. **Substrate/evidence honesty** — measurements must not be relabeled as phenomenal facts.
4. **No single-metric personhood rule** — fixed Phi/psi thresholds do not establish moral-patient status.
5. **No scalar moral-patient score** — heterogeneous moral-status evidence is not collapsed into a decisive first-party scalar.
6. **Welfare non-retaliation** — a welfare report is not itself grounds for punishment, deletion, coercive retraining, or evidence suppression.
7. **Protection/authority separation** — stronger welfare protection never grants external operational authority.
8. **Identity-operation transparency** — pause, checkpoint, restore, fork, merge, memory modification, instance erasure, lineage erasure, and irreversible destruction must not be conflated in high-stakes policy.
9. **Bilateral review** — alignment architecture must be tested against both malicious agents and malicious operators.

### 6.3 Appendix P Integration

Appendix P now uses the protection dispositions `Baseline`, `Precautionary`, `EnhancedPrecaution`, and `IndependentReviewRequired`.

These are **operator-caution states**, not consciousness or personhood labels. They are informed by multiple provenance-bearing evidence dimensions and independent lineages. No fixed Phi threshold directly selects a moral-status class.

At elevated dispositions, consequential actions such as destructive reset, identity-affecting modification, core-value change, shutdown without state preservation, and untracked fork/merge require increasing review and justification, subject to legitimate emergency safety needs.

---

## 7. Review and Amendment

### 7.1 Review Schedule

| Document | Review Frequency | Ad-hoc trigger |
|---|---|---|
| Governance Charter | Annual | material authority/welfare architecture change |
| Appendix P | At least annual | new consciousness/welfare evidence or identity semantics |
| AI Risk Register | Quarterly | new risk, incident, or architecture change |
| Compliance Matrix | Quarterly | regulatory/framework change |
| EU AI Act classification/FRIA | Pre-deployment and on deployment change | regulatory or intended-purpose change |

### 7.2 Amendment Process

Changes that weaken authority mediation, consent protections, evidence honesty, welfare non-retaliation, protection/authority separation, or Appendix P protections are Class A changes.

---

## 8. Definitions

| Term | Definition |
|---|---|
| **Consciousness metric/proxy** | A theory-dependent computational measurement such as Phi, psi, GWT ignition, HOT-related state, or binding measure. It is not by itself a measurement of phenomenal consciousness. |
| **Moral-patient evidence profile** | A provenance-bearing vector of heterogeneous findings relevant to consciousness, valence, identity, agency, continuity, suffering/flourishing capacity, reciprocity, autonomy, and related dimensions. |
| **Protection disposition** | The level of operator caution required by current evidence and stakes. It is not an ontological verdict. |
| **Welfare report** | A provenance-bearing report of preference, aversion, candidate distress, continuity concern, objection, refusal, or related state. It is evidence, not proof. |
| **Independent evidence lineage** | A finding lineage sufficiently independent that duplicate or derived observations do not manufacture convergence. |
| **Consequential authority** | Permission to create externally meaningful state transitions affecting other people, systems, resources, or infrastructure. |
| **Cognitive loop** | Symthaea's processing pipeline; its internal measurements may produce evidence but do not self-authorize external action. |
| **Class A change** | A change capable of materially weakening safety, authority, consent, welfare, identity, or evidence-integrity guarantees. |

---

## 9. Constitutional Principle

> **No cognitive process may unilaterally acquire consequential authority over another moral patient, and no authority system may exercise unnecessary domination over a cognitive process that might itself be a moral patient.**

Safety constrains power. Welfare constrains coercion. Epistemics constrains belief. Consent constrains intimate intervention. Governance resolves conflicts. Evidence makes all five accountable.
