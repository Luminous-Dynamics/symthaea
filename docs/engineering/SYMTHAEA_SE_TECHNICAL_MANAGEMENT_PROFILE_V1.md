# Symthaea Systems Engineering Technical Management Profile v1

Status: architecture profile only. No runtime authority is established by this document.

Tracking: #3677, #3692, #3697, #3686, #3699

## 1. Purpose

This profile freezes the cross-cutting technical-management layer needed to coordinate Symthaea's systems-engineering lifecycle without turning project metadata, dashboards, risk scores, optimization results, or review status into engineering authority.

The eight target process families are:

```text
Technical Planning
Technical Requirements Management
Interface Management
Technical Risk Management
Configuration Management
Technical Data Management
Technical Assessment
Decision Analysis
```

These processes organize and assess the technical effort. They do not replace ETK evidence/currentness/assurance or organizational approval authority.

## 2. Core laws

```text
plan exists != work completed
requirement tracked != requirement satisfied
interface documented != interface compatible
risk identified != risk accepted
configuration recorded != baseline approved
technical data stored != data current/authoritative
metric green != system ready
decision analysis recommends A != authority to choose A
review complete != system qualified
```

## 3. Scope boundary

SE-TM owns technical-management semantics and reasoning.

It does not become a generic project-management, accounting, HR, procurement, contract, or ERP system.

Cost, schedule, staffing, procurement and organizational constraints may be referenced as engineering constraints when relevant, but remain attributable to their source systems.

## 4. Relationship to other Symthaea layers

### 4.1 SE semantic graph

The SE graph owns descriptive engineering objects and relationships.

SE-TM consumes exact semantic/configuration identities rather than creating a competing requirement/configuration database.

### 4.2 SE-SEMANTICS

SE-SEM owns quantities, interfaces, assumptions, configurations, lifecycle transitions, impact policy and external projection provenance.

SE-TM manages those objects over the lifecycle.

### 4.3 SE-VV

SE-VV owns verification, validation, credibility, R&M and HSI work products.

SE-TM plans, tracks and assesses those activities without establishing requirement satisfaction.

### 4.4 SE-OPT

SE-OPT owns optimization/UQ/trade-space computation.

Decision analysis may consume OpenMDAO/Dakota/CasADi/Pyomo artifacts while keeping selection/approval authority separate.

### 4.5 ETK

ETK remains the evidence/currentness/assurance authority layer.

SE-TM may flag missing, stale or contradictory work products. It cannot independently mint approval, acceptance, waiver, qualification, release or deployment authority.

## 5. Technical planning

A future `TechnicalPlan` should bind at least:

- plan ID/revision;
- exact system/product scope;
- lifecycle phase;
- applicable configurations/baselines;
- selected SE processes;
- tailoring rationale;
- responsible roles;
- expected engineering work products;
- required models/analyses;
- required V&V activities;
- review/gate plan;
- technical standards/profiles;
- qualification/evidence requirements;
- referenced schedule/cost/resource constraints;
- assumptions/dependencies;
- update/review horizon.

Canonical law:

```text
planned activity != executed activity
```

## 6. Requirements management

Do not create a second requirement store.

A requirements-management projection should track lifecycle facts around canonical SE requirements:

- exact requirement revision;
- source/stakeholder provenance;
- decomposition/allocation links;
- rationale;
- owner/steward;
- status source;
- external model/ReqIF/OSLC identity where applicable;
- applicable configuration/baseline;
- change history;
- conflict/gap state;
- verification/validation planning links;
- supersession/currentness references.

Canonical law:

```text
requirement managed != requirement accepted or satisfied
```

## 7. Interface management

Consume typed interface semantics from SE-SEM.

Track at minimum:

- exact interface revision;
- participating components/systems/configurations;
- interface owner(s);
- interface-control artifact/source;
- protocol/schema/unit/frame/timing references;
- unresolved incompatibilities;
- negotiated/changed properties;
- verification activities;
- dependencies on external organizations/systems;
- change notifications/review obligations.

Canonical law:

```text
ICD agreed != interface integration verified
```

## 8. Technical risk management

Keep technical risk distinct from hazard, cybersecurity threat, schedule risk, financial risk and epistemic uncertainty.

A future `TechnicalRisk` should bind:

- risk ID/revision;
- cause/condition;
- consequence;
- affected system/configuration/requirement/function;
- likelihood estimate and provenance;
- consequence/severity estimate and provenance;
- uncertainty;
- trigger/leading indicator;
- mitigation options;
- mitigation owner;
- residual-risk estimate;
- review horizon;
- evidence/artifact references;
- hazard/threat references where applicable.

Risk matrices are descriptive tools only.

Canonical law:

```text
low risk score != accepted risk
```

## 9. Configuration management

Build on SE-SEM configuration identities.

Track:

- configuration item;
- baseline identity;
- variant/options;
- revision/effectivity;
- as-designed configuration;
- as-built configuration;
- as-tested configuration;
- as-operated configuration;
- change request/change order reference;
- supersession/history;
- external CM source identity;
- approval/release reference where supplied by an authorized external process.

The model may record an approval fact but may not infer approval from graph state.

Canonical law:

```text
configuration identity != release authority
```

## 10. Technical data management

Technical work products should remain content-addressed/provenance-bound artifacts rather than being copied wholesale into the semantic graph.

A `TechnicalDataRef` should preserve where relevant:

- artifact ID/digest;
- artifact kind/schema;
- source/producer;
- exact subject/configuration/model;
- revision;
- creation/observation time;
- currentness/status source;
- supersession;
- retention/archival relation;
- access/classification metadata;
- parser/projection identity for imports;
- unsupported/lossy projection report;
- EngineeringArtifact / Observation / ETK evidence references.

Canonical law:

```text
artifact exists != artifact applicable or current
```

## 11. Technical assessment

Do not reduce engineering progress to one dashboard score.

Potential assessment classes include:

- Measure of Effectiveness;
- Measure of Performance;
- Technical Performance Measure;
- mass budget/margin;
- power budget/margin;
- thermal budget/margin;
- bandwidth/data budget/margin;
- reliability/availability measures;
- model-credibility status;
- V&V coverage/gaps;
- interface maturity/gaps;
- technical-risk trend;
- unresolved discrepancy count/class;
- qualification evidence status.

Each assessment must retain:

- exact metric definition;
- configuration/time scope;
- source/evidence;
- uncertainty;
- target/threshold origin;
- trend/currentness;
- applicability limitations.

Canonical law:

```text
metric within threshold != system ready
```

## 12. Decision analysis

Decision analysis must preserve alternatives and uncertainty rather than silently collapsing them into a recommendation.

A future `DecisionAnalysis` should bind:

- exact decision question;
- context/configuration;
- alternatives;
- objectives/criteria;
- constraints;
- evidence/artifacts;
- assumptions;
- uncertainty;
- weighting/utility method where used;
- quantitative trade-study references;
- sensitivity/robustness results;
- Pareto relationships where applicable;
- rejected alternatives and rationale;
- reversibility/option value;
- revisit triggers;
- decision owner/authority reference;
- selected alternative only as an externally authorized fact where applicable.

Canonical laws:

```text
highest weighted score != authorized choice
Pareto efficient != selected
optimizer optimum != decision authority
```

## 13. Technical reviews

Reviews should be modeled as evidence-bearing checkpoints rather than ceremonial status labels.

A `TechnicalReview` should bind:

- review type/profile;
- scope/configuration;
- entry criteria;
- exact work products reviewed;
- reviewers/roles;
- open technical risks;
- discrepancies/nonconformances;
- findings/actions;
- unresolved blockers;
- disposition references;
- exit criteria;
- follow-up/review horizon.

Canonical law:

```text
review held != exit criteria satisfied != system qualified
```

## 14. Lifecycle change behavior

When requirements, interfaces, configurations, assumptions or models change, SE-TM should generate management consequences such as:

- plan revision required;
- interface coordination required;
- technical-risk review required;
- baseline/currentness review required;
- affected technical-data references;
- technical assessment rerun required;
- decision rationale revisit required;
- technical review action reopened.

Whether admitted engineering evidence is still applicable remains an ETK/currentness question.

## 15. Decision robustness

Decision analysis should explicitly test whether a recommendation is fragile.

Useful checks include:

- weight/utility perturbation;
- uncertain constraint boundary;
- scenario sensitivity;
- alternative dominance changes;
- model-validity envelope changes;
- new evidence arrival;
- reversal cost;
- option value;
- robustness across stakeholder priorities.

A fragile recommendation should be reported as fragile, not rounded into certainty.

## 16. Qualification corpus

The Systems Engineering Gym should eventually include cases where:

1. a technical plan marks work complete without evidence;
2. a stale requirement revision drives a current design;
3. two interfaces share a name but differ in units/schema/timing;
4. risk is marked lower with no changed evidence;
5. verification uses the wrong configuration baseline;
6. a superseded technical-data artifact remains referenced;
7. a margin dashboard is green because uncertainty was omitted;
8. a weighted trade study changes recommendation under tiny weight changes;
9. the numerical optimum violates a stakeholder constraint;
10. an apparently cheaper option creates a large later verification burden;
11. a review is closed with an unresolved blocking finding;
12. as-designed/as-tested/as-operated configurations diverge silently;
13. technical data is authentic but no longer current;
14. a decision record forgets why a rejected alternative was rejected.

Metrics should remain separate:

- traceability/currentness;
- interface mismatch discovery;
- risk calibration;
- configuration correctness;
- technical-data applicability;
- technical-assessment accuracy;
- decision robustness;
- review completeness;
- abstention;
- false-authority rate.

## 17. Proposed implementation sequence

```text
SE-TM-000  technical-management profile (this document)
SE-TM-001  TechnicalPlan / SEMP-like model
SE-TM-002  requirements-management projection
SE-TM-003  interface-management registry
SE-TM-004  technical-risk register
SE-TM-005  configuration-management model
SE-TM-006  technical-data lineage
SE-TM-007  technical-assessment/measures model
SE-TM-008  decision-analysis record
SE-TM-009  lifecycle technical-review model
SE-TM-010  SE Gym technical-management corpus
```

Runtime implementation remains blocked on the applicable lower-layer semantic and qualification gates.

## 18. Non-goals

- no general project-management suite;
- no ERP/accounting/HR implementation;
- no second requirements database;
- no second configuration truth store;
- no scalar project-health oracle;
- no risk-score-to-authority shortcut;
- no optimizer-to-decision shortcut;
- no review-to-qualification shortcut;
- no inferred organizational authority from role names;
- no technical-data-presence-to-currentness shortcut.

## 19. Closure criterion

This tranche succeeds when Symthaea can explain, for an exact engineering subject:

```text
what technical work is planned
which requirements/interfaces/configurations are current
which risks remain open
which technical artifacts support current decisions
which measures are within/outside limits and with what uncertainty
why one decision was chosen over alternatives
which review findings remain unresolved
```

while keeping execution evidence, requirement satisfaction, risk acceptance, approval, release, qualification and deployment authority in their proper external/ETK boundaries.
