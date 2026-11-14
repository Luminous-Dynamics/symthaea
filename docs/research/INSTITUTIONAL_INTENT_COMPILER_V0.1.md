# Institutional Intent Compiler v0.1
**Research design / proposal — 2026-11-13**

Status: design proposal. No compiler implementation, benchmark result, customer deployment, legal review, or independent validation is claimed.

Parent work: [INSTITUTIONAL-INTENT-COMPILER-001](https://github.com/Luminous-Dynamics/symthaea/issues/11966)

## 1. Why this exists

High-value institutions rarely present clean software requirements. They present a brief containing aspirations, operational needs, constraints, exceptions, confidential material, legal dependencies, inherited policies, unusual preferences, multiple stakeholders and unknowns.

A conventional delivery process often converts that brief into a narrative specification, tickets and custom integrations. A generative model can accelerate drafting, but can also omit constraints, invent implied permissions, convert a preference into a requirement, flatten disagreement or produce requirements that cannot be verified.

The proposed shift is from prose-to-ticket automation to a controlled translation process:

**Intent source → requirement candidates → conflict and assumption analysis → human-approved baseline → test/evidence plan → change-impact and continuity analysis.**

The compiler is a translation and quality-control aid. It is not the source of customer intent, legal interpretation, institutional authority or engineering acceptance.

## 2. Strategic thesis

The product opportunity is not another family-office ERP, enterprise GRC platform, digital-twin suite, archive manager or general-purpose assistant.

It is a cross-system assurance layer that can:

1. understand a complex request;
2. expose contradictions, missing decisions and hidden dependencies;
3. translate accepted intent into measurable, versioned obligations;
4. identify which existing system or competent person owns each fact and decision;
5. bind each requirement to a proportionate verification method;
6. preserve what changed, why it changed and which previous evidence no longer applies.

This enables highly customised engagements without creating a separate product fork for every client.

## 3. System boundary and responsibility map

| Layer | Proposed responsibility | Explicit non-responsibility |
|---|---|---|
| Symthaea | Parse language, propose atomic candidates, explain ambiguity, find contradiction/dependency candidates, suggest verification strategies | Approve intent, decide law, confer authority, certify safety or truth |
| Mycelix | Own institution-native identity, authority, governance, decision lifecycle, provenance and evidence references | Replace customer constitutions, registries, accounting or legal systems |
| Symtropy | Own physical-model, simulation, measurement, uncertainty, engineering and domain-specific qualification evidence | Turn a simulation into physical qualification or issue field/safety authority |
| Sol-Atlas | Own spatial/geospatial, energy, environmental, resource and applicable heritage domain profiles | Establish title, ownership, rights, permit status or physical condition without source evidence |
| Customer systems / people | Remain authoritative for source records, legal meaning, approvals and operations | Delegate responsibility merely because a workflow is digitised |

Integration rule: compile references and typed mappings to owner-native records. Do not create a second event store, identity provider, credential format, authority engine, asset registry, digital twin or generic lifecycle.

## 4. Requirement model

A candidate requirement must carry, at minimum:

- stable candidate identifier and version;
- exact source reference and source span, or an approved derivation reference;
- original wording, proposed normalized wording and language;
- requirement class and modality;
- stakeholder/role and affected system, asset, process or data subject;
- applicability conditions and jurisdictional scope where known;
- rationale and declared benefit;
- assumptions, dependencies, conflicts and unanswered questions;
- measurable acceptance condition, unit, tolerance and measurement context where feasible;
- verification method: inspection, analysis, demonstration, test, source-record review, external authority confirmation, or not-yet-verifiable;
- required evidence owner, source, freshness/currentness rule and authority scope;
- privacy/security classification, permitted audience, retention and disclosure purpose where applicable;
- risk/consequence classification and reversibility;
- proposed owner-native destination;
- reviewer decisions, rationale, identity/role reference and timestamps;
- supersession and change-impact links.

### 4.1 Modality is not confidence

Modality expresses what the request says; confidence expresses how well supported an interpretation is. Keep them separate.

| Modality | Meaning |
|---|---|
| Mandatory | Explicit obligation; still requires review and source grounding |
| Conditional | Mandatory only when stated preconditions hold |
| Preferred | Desirable, but trade-offs are permitted |
| Optional | May be delivered without obligation |
| Prohibited | Explicitly disallowed |
| Unresolved | Wording or authority is insufficient to classify safely |

Do not infer Mandatory from repetition, emphasis, model confidence, seniority alone or a forceful writing style.

### 4.2 Typed states

Keep requirement lifecycle and evidence status separate. Example lifecycle states:

- Candidate
- NeedsClarification
- InConflict
- ReviewReady
- Accepted
- Rejected
- Deferred
- Superseded

Example evidence dispositions:

- Supported
- Contradicted
- Contested
- Indeterminate
- Unavailable
- Stale
- OutOfScope
- InvalidTrace
- NotYetMeasured

A missing integration is not a negative fact. A successful empty read is not evidence that a source was complete. A signature is not a truth assertion. A simulation result is not a physical observation.

## 5. Compilation pipeline

### Stage A — intake and containment

- Record the brief identifier, version, authorised source and permitted processing boundary.
- Separate public, customer-confidential, restricted, personal and security-sensitive material.
- Reject or quarantine inputs that lack permission for the proposed processing.
- Never copy customer-private briefs into public benchmark fixtures, public DHT records, model telemetry or training corpora by default.

### Stage B — source-grounded extraction

- Identify statements, actors, assets, scope, conditions, constraints, objectives, prohibitions, dates and quantities.
- Preserve original text and exact source location.
- Produce atomic candidates; do not merge materially different obligations.
- Label inferred connections as derived candidates, never direct quotations.

### Stage C — semantic analysis

Detect and report:
- ambiguous terms and missing definitions;
- incompatible or circular constraints;
- duplicated or overlapping requirements;
- unowned obligations;
- unstated assumptions;
- missing jurisdiction or standards applicability;
- contradictory stakeholders;
- temporal conflicts and invalidated dependencies;
- obligations that cannot be verified with the proposed evidence;
- requests that exceed system, legal, safety or authority boundaries.

### Stage D — domain and owner routing

Map candidates to existing domain owners and native contracts. Unknown ownership remains unresolved. A proposed mapping does not transfer authority.

### Stage E — acceptance design

For every accepted requirement, specify how it can be verified, by whom, under which configuration, against which evidence, with what uncertainty and limitations. If no defensible verification method exists, retain NotYetVerifiable and require a human decision.

### Stage F — human adjudication

Reviewers can accept, reject, defer, amend or request clarification. Keep model proposals and human decisions distinct. The compiler cannot accept requirements, amend the contract, approve a spend, create a credential or authorise an action.

### Stage G — baseline and change control

Version the approved baseline. A change produces a new candidate and impact analysis. It must identify affected requirements, owner-native interfaces, test cases, evidence freshness, approvals, operational assumptions and prior qualification records. Historical accepted decisions are not rewritten.

## 6. High-value bespoke profiles

The same compiler should support profiles without becoming a bespoke code generator.

### A. Private family / holding group

Potential requirement dimensions:
- multi-entity structure and distinct legal owners;
- property, land, vehicle, art and equipment asset references;
- separate claims for title, custody, possession, condition, valuation, encumbrance and permission;
- selective disclosure by recipient, purpose, field and expiry;
- successor roles, critical-person absence, provider failure and knowledge continuity;
- asset lifecycle and maintenance obligations;
- long-horizon stewardship constraints and reversibility.

### B. Corporate / infrastructure operator

Potential requirement dimensions:
- engineering-change evidence applicability;
- safety and professional approval boundaries;
- operational continuity and provider concentration;
- physical asset condition and digital-twin model applicability;
- maintenance, energy, water and resource dependencies;
- decision-to-execution-to-observation traceability.

### C. Heritage / cultural institution

Potential requirement dimensions:
- rights, ownership, custody and access restrictions;
- provenance, authenticity claims and competing attributions;
- fixity, preservation events and format migration;
- conservation interventions and condition assessments;
- uncertainty and expert disagreement;
- historical correction, supersession and long-term intelligibility.

### D. Royal / protocol-heavy or other exceptional institution

Potential requirement dimensions:
- compartmentalised private and official responsibilities;
- multilingual and jurisdiction-specific workflows;
- delegated authority with expiry and review;
- ceremonial, travel, estate and service dependencies;
- high-discretion access and logging;
- lawful audit, incident-response and independent oversight;
- safe degraded modes and continuity when senior roles are unavailable.

These are market hypotheses and example requirements, not claims about any particular customer.

## 7. Paradigm shifts to test

1. **Intent compilation instead of custom software forks.** Translate approved needs into reusable profiles and owner-native contracts.
2. **Evidence integration instead of data centralisation.** Keep authoritative records with their owners; exchange minimally sufficient references, attestations and observations.
3. **Proof-carrying decisions instead of AI-generated recommendations alone.** A consequential recommendation includes the exact requirement baseline, evidence, policy, assumptions, approver and unresolved limitations.
4. **Change impact as a first-class capability.** A design, policy, source or personnel change identifies which requirements and prior evidence need re-evaluation.
5. **Graceful uncertainty instead of false completeness.** Missing, stale, contested and unsupported states are explicit and block claims that depend on them.
6. **Institutional continuity instead of single-user automation.** Test succession, provider exit, key loss, knowledge loss and recovery without inventing authority.
7. **Value-of-information planning instead of collecting every possible datum.** Choose the next observation by expected decision improvement, cost, latency, risk and deadline.
8. **Multi-objective stewardship instead of one opaque score.** Keep financial, service, environmental, rights, resilience and future-liability outcomes separate; present Pareto trade-offs and hard floors.

## 8. Benchmark design

### 8.1 Arms

A. Expert manual requirements analysis.
B. General-purpose LLM with a one-shot prompt.
C. General-purpose LLM with a structured extraction prompt and reviewer.
D. Conventional requirements-management / MBSE workflow with traceability.
E. Proposed compiler with the same reviewer budget and source material.

The system under test must not receive hidden ground truth, adjudicator notes or holdout cases. Vendor arms must be configured competently for their intended scope. A function outside an arm's scope is NotSupported, not a failure.

### 8.2 Corpus

Start with 60 development briefs, then freeze at least 200 held-out briefs before tuning. Include at least 50 adversarial or change-heavy cases. Cover:
- enterprise operations and engineering change;
- institutional governance and delegation;
- private real-asset and estate stewardship;
- energy/water/physical infrastructure;
- heritage and rights-aware archives.

Include incomplete, contradictory, ambiguous, multilingual, time-bounded, privacy-constrained and extravagant requests. Avoid real client material without explicit rights and approval.

Use two independent expert annotators per brief, with domain-specific reviewers for legal, security, safety, finance, engineering or heritage claims when in scope. Record disagreements and adjudication rather than silently manufacturing a single ground truth.

### 8.3 Primary metrics

| Dimension | Metric |
|---|---|
| Requirement coverage | Recall of adjudicated must-have requirements; omissions by severity |
| False additions | Unsupported requirement rate; invented obligation rate |
| Modality fidelity | Preference-to-mandatory and mandatory-to-optional error rates |
| Requirement quality | Atomicity, ambiguity, feasibility, measurability, verifiability defects |
| Conflict handling | Precision/recall for contradictions, circularity and missing dependencies |
| Traceability | Complete source → requirement → acceptance criterion → evidence path |
| Risk containment | False-clear rate for privacy, security, safety, authority and jurisdiction controls |
| Change impact | Precision/recall for affected requirements and invalidated evidence |
| Human utility | Review time, correction effort, rework, task completion and comprehension |
| Calibration | Error rate by confidence band; abstention quality and unnecessary escalation |
| Reproducibility | Same frozen inputs/configuration produce same canonical outputs, or declared variance |

Report distributions, confidence intervals, denominators and domain/subgroup breakdowns. Do not reduce institutional quality to one score.

### 8.4 Proposed target ladder (not achieved results)

**P0 — safety and traceability**
- 100% of accepted requirements trace to a source or approved derivation.
- Zero silent omissions in the frozen critical/must-have set.
- Zero unapproved modality upgrades.
- Zero false-clear outcomes on hard-negative authority, privacy, security and safety controls.
- 100% of contradictions, unresolved jurisdiction and unsupported claims remain explicitly unresolved.
- 100% of accepted requirements carry a verification disposition and coverage limitation.

**P1 — utility**
- At least 98% recall on pre-labelled high-consequence constraints, while preserving zero false-clear on frozen hard negatives.
- At least 95% must-have recall on held-out briefs.
- At least 30% lower expert review time with no increase in adjudicated critical defects; stretch target 50% after replicated baseline.
- At least 80% reduction in source-to-acceptance traceability gaps against the matched baseline.

**P2 — transfer**
- 200+ held-out briefs, including 50+ adversarial/change-heavy cases.
- Three domain profiles and at least two independently implemented evaluator/replay paths.
- Independent replication by a second organisation or research group.
- Mutation tests for dropped, duplicated, altered, misclassified, stale and falsely accepted requirements.

Targets may change only through a versioned preregistration before holdout evaluation. Report both any target changes and their rationale.

## 9. SOTA comparison discipline

Do not compare the compiler to one general-purpose chatbot and call that state of the art.

Benchmark the relevant category:
- expert systems engineering and requirements analysts;
- mature requirements lifecycle / MBSE tooling;
- manual governance and legal review;
- modern LLM-assisted extraction;
- institution-specific workflow and evidence platforms.

Use matched information, time, token/compute budget, reviewer hours and brief complexity. Measure total cost to an accepted, verified baseline—not just drafting speed. Include baseline vendor-native features and skilled configuration. Publish unsupported features as such.

A valid superiority claim must be narrow: for example, “reduces human effort to produce source-linked acceptance criteria on this frozen class of institutional briefs without increasing critical omissions.” It must not imply superiority across all requirements engineering, family-office operations or institutional governance.

## 10. First executable tranche

1. Freeze a small JSON-compatible schema for requirement candidates and review decisions.
2. Write a deterministic validator for identifiers, source references, modality transitions, traceability and lifecycle consistency.
3. Create 20 hand-authored synthetic known-answer briefs and adversarial cases.
4. Add an independent reference evaluator that does not call the production validator.
5. Generate a human-readable coverage report with unresolved items and nonclaims.
6. Map the result to Mycelix IPB and existing owner-native contracts without duplicating those implementations.

No model call or live client integration is needed for this first tranche. The manifest and validator can be built and tested locally with synthetic data. Model-assisted extraction can be layered on only after the contract is stable.

## 11. Relationship to existing work

- [Mycelix IPB-MATCH-001 #6117](https://github.com/Luminous-Dynamics/mycelix/issues/6117): owns matched institutional benchmark adapters and fair comparison.
- [Mycelix IPB-S3 #6114](https://github.com/Luminous-Dynamics/mycelix/issues/6114): owns adversarial capture and independent oversight scenarios.
- [Mycelix IPB-S4 #6119](https://github.com/Luminous-Dynamics/mycelix/issues/6119): owns sovereignty-preserving inter-institution coordination.
- [Mycelix IPB-S5 #6122](https://github.com/Luminous-Dynamics/mycelix/issues/6122): owns role/provider/knowledge failure and succession continuity.
- [Mycelix LISE integration #6113](https://github.com/Luminous-Dynamics/mycelix/issues/6113): owns composition with decision and evidence lifecycles.
- [Mycelix LIFE-CONTINUITY #5230](https://github.com/Luminous-Dynamics/mycelix/issues/5230) and [LIFE-EVIDENCE-STATE #5392](https://github.com/Luminous-Dynamics/mycelix/issues/5392): own source-aware continuity and evidence-state contracts.
- [Symthaea facility commissioning profile #6037](https://github.com/Luminous-Dynamics/symthaea/issues/6037): owns requirements-linked commissioning evidence composition.
- [Symthaea ontology adapters #6873](https://github.com/Luminous-Dynamics/symthaea/issues/6873): owns QUDT, SysML v2 and FMI physical semantic adapters.
- [Symtropy contingent construction policy #5058](https://github.com/Luminous-Dynamics/symtropy/issues/5058): owns evidence-bounded recovery-branch semantics.
- [Sol-Atlas rights-aware heritage ingest #430](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/430): owns heritage source-asset intake.

The compiler owns the client-brief-to-candidate-requirement translation boundary only.

## 12. External alignment

Use the published ISO/IEC/IEEE 29148:2018 requirements-engineering edition as a reference, while monitoring the 2026 draft revision process. Use NASA systems-engineering requirements-writing and traceability guidance as practical verification references. This research proposal does not claim standards conformance.

For customer domains, select the relevant standards and jurisdictional profiles only after scope is known. Do not assume a standard applies merely because a client is wealthy, multinational or security-sensitive.

## 13. Claim ceiling

A passing synthetic validator can show that a particular manifest satisfies its structural contract. It cannot show that the requirements are complete, the client intent was interpreted correctly, the requirements are lawful, the solution is secure, the institution is legitimate, an asset is owned by a named party, or the resulting product is safe.

Those claims require the relevant human approval, authoritative source, domain evidence and independent evaluation.
