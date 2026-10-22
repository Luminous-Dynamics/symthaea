# Symthaea Systems Engineering V&V Lifecycle Profile v1

Status: architecture profile only. No runtime authority is established by this document.

Tracking: #3677, #3692, #3695, #3686, #3697

## 1. Purpose

This profile freezes the lifecycle distinction among:

- requirement verification;
- stakeholder/use validation;
- model and digital-twin credibility;
- reliability;
- maintainability;
- human systems integration (HSI).

These are related engineering concerns, but they answer different questions and must not collapse into one generic `verified` state.

Canonical distinctions:

```text
verification passed != validation passed
validation passed in one context != universal validity
model verified != model validated
model calibrated != physically true
reliable estimate != guaranteed mission success
maintainable design != maintained asset
human-factors analysis != demonstrated human-system performance
requirements coverage != requirements satisfied
```

ETK remains the authority-bearing evidence/currentness/assurance layer.

## 2. External alignment

Use public engineering guidance as semantic alignment, not copied authority.

The profile should remain compatible in spirit with:

- NASA Systems Engineering Handbook distinctions between verification and validation;
- NASA product-realization and lifecycle framing;
- NIST digital-twin credibility work emphasizing verification, validation and uncertainty quantification;
- NASA reliability and maintainability lifecycle objectives;
- NASA Human Systems Integration guidance treating hardware, software, humans, data and processes as parts of the system.

No mapping to an external standard establishes certification or qualification by itself.

## 3. Relationship to existing Symthaea layers

### 3.1 SE semantic layer

The SE graph owns descriptive engineering structure:

- needs;
- requirements;
- functions;
- components;
- interfaces;
- assumptions;
- hazards;
- controls;
- configurations;
- decisions;
- changes.

SE-VV consumes exact semantic/configuration identities. It must not duplicate the semantic graph.

### 3.2 ETK

ETK owns evidence admission, currentness, assurance consequences and authority-bearing qualification semantics.

SE-VV may record a verification or validation result, but cannot independently turn that result into requirement satisfaction, qualification, certification or deployment authority.

### 3.3 SE-OPT

Optimization/UQ owns trade studies, calibration, sampling, sensitivity and uncertainty campaigns.

SE-VV consumes those artifacts when evaluating model credibility or R&M claims.

### 3.4 SE-OBS

SE-OBS owns provenance-bound observations and digital-thread projection.

SE-VV consumes physical/operational observations for validation, model credibility, degradation and reliability reasoning.

### 3.5 Formal/simulation/native analyses

Formal proofs, simulations and native calculations remain analysis artifacts. Their role in V&V depends on the exact claim, configuration, validity envelope and ETK evidence policy.

## 4. Claim classes

A future implementation should keep at least these claim classes separate.

### 4.1 VerificationClaim

Question:

> Does this exact product/configuration comply with this exact requirement under this declared verification method and acceptance criterion?

Typical methods:

```text
Analysis
Inspection
Demonstration
Test
```

Possible combinations are explicit rather than implied.

### 4.2 ValidationClaim

Question:

> Does this exact product/configuration satisfy the intended stakeholder need or use in this declared operational context?

Validation binds to stakeholder intent, ConOps/mission context, realistic scenarios and representative users/environments where applicable.

### 4.3 ModelCredibilityClaim

Question:

> Is this exact model/twin credible enough for this declared decision/use within this validity envelope?

Credibility is use-specific, not universal.

### 4.4 ReliabilityClaim

Question:

> What evidence/model supports the probability or expectation that this system/service performs required functions over a specified mission/time/environment?

Reliability claims must retain population/model/source assumptions and uncertainty.

### 4.5 MaintainabilityClaim

Question:

> What evidence/model supports restoration, diagnosis, inspection, replacement or servicing performance under declared conditions?

### 4.6 HumanSystemClaim

Question:

> What evidence supports integrated human + hardware + software + process performance for a declared role/task/environment?

Simulated or AI-predicted human performance is not equivalent to human-in-the-loop evidence.

## 5. Verification planning and coverage

A future `VerificationPlan` should bind at least:

- exact requirement identity/revision;
- exact product/configuration identity;
- method;
- verification level;
- procedure/tool/fixture intent;
- prerequisites;
- environment;
- acceptance-criteria identity;
- produced artifact references;
- responsible role/organization where applicable;
- planned vs executed status.

Coverage diagnostics may identify:

- requirement with no verification activity;
- verification activity with no requirement;
- changed requirement with stale verification plan;
- verification artifact for wrong configuration;
- acceptance criterion missing or ambiguous;
- required method not represented.

Canonical law:

```text
verification coverage complete
!= verification passed
!= requirement satisfied
```

## 6. Validation and intended use

Validation must remain separate from verification even when the same test or artifact contributes to both.

A future `ValidationScenario` should bind:

- stakeholder need / mission objective / ConOps reference;
- user/operator/maintainer roles;
- exact product/configuration;
- environment;
- scenario/use case;
- assumptions;
- expected outcomes/suitability criteria;
- observed/derived outcomes;
- representativeness limitations;
- validity envelope;
- unresolved discrepancies.

Canonical law:

```text
all requirements verified
!= intended use validated
```

## 7. Model and digital-twin credibility

Do not collapse model credibility into one scalar score.

Preserve separate dimensions such as:

- implementation/code verification;
- solver/numerical verification;
- convergence/discretization evidence;
- calibration;
- validation against physical/operational observations;
- uncertainty quantification;
- discrepancy/error model;
- domain/applicability envelope;
- sensitivity;
- extrapolation status;
- source-data quality/currentness;
- intended decision/use.

Canonical laws:

```text
model runs != model credible
model verified != model validated
calibrated != validated
credible for use A != credible for use B
```

A digital twin should therefore carry an explicit credibility profile for each use rather than a single global `trusted=true` state.

## 8. Reliability and maintainability

A reusable R&M layer should preserve the difference between architecture analysis, physics-based prediction, historical field data and operational observations.

Potential reliability artifacts include:

- function/service availability models;
- reliability block diagrams;
- fault trees;
- failure-mode/effect analyses;
- common-cause/shared-dependency analyses;
- lifetime/failure-rate distributions;
- physics-of-failure analyses;
- degradation trajectories;
- inspection/detection coverage;
- observed field-failure data.

Potential maintainability artifacts include:

- diagnosis/isolation tasks;
- repair/replacement tasks;
- service access constraints;
- required tools/fixtures;
- staffing/skill assumptions;
- mean/distribution of restore time;
- spare/logistics dependency;
- inspection interval;
- calibration/recertification need;
- degraded-mode operation during maintenance.

Required distinctions:

```text
architectural fault propagation != physical FMEA/FMEDA
predicted failure rate != observed population rate
availability model != operational availability fact
repair procedure exists != repair demonstrated
```

## 9. FMEA/FMEDA/fault-tree representation

A future common representation should retain at least:

- exact item/function/configuration;
- failure mode;
- local effect;
- next-higher/system/end effect;
- cause/mechanism hypothesis;
- detection/diagnostic mechanism;
- prevention/mitigation/control;
- common-cause/shared-resource links;
- source/method for severity/occurrence/detection estimates;
- uncertainty;
- analysis revision;
- evidence/provenance references;
- unresolved assumptions.

Do not use RPN or another scalar as an automatic engineering priority/authority oracle.

## 10. Human Systems Integration

The engineered system includes humans and processes, not only hardware and software.

Roles may include:

- customer/user;
- operator;
- maintainer;
- assembler/manufacturing worker;
- inspector/test engineer;
- support/logistics personnel;
- trainer/supervisor;
- emergency/recovery personnel.

HSI semantics should cover where relevant:

- task/function allocation;
- human-machine interface;
- workload;
- attention/situation awareness;
- anthropometric/ergonomic envelope;
- reach/access/serviceability;
- procedure dependencies;
- training dependencies;
- staffing/crew assumptions;
- communication/coordination;
- human-error opportunities;
- recovery/error-tolerance;
- accessibility/inclusion constraints;
- human-in-the-loop evaluation/validation.

Canonical law:

```text
AI predicts usability
!= humans demonstrated usability
```

A simulated human model may help identify hypotheses and design defects, but cannot substitute for required representative human evidence.

## 11. Discrepancy / anomaly / nonconformance lifecycle

Use one typed discrepancy path rather than separate ad-hoc failure semantics in every tool.

Conceptually:

```text
Observation / TestResult / AnalysisResult
        ↓
Discrepancy
        ↓
Classification
        ↓
Competing hypotheses
        ↓
Investigation / evidence request
        ↓
Disposition proposal
        ↓
Change candidate
        ↓
re-verification / re-validation obligations
```

Preserve:

```text
anomaly != root cause
nonconformance != waiver
waiver proposal != waiver authority
repair != demonstrated root-cause correction
```

## 12. Lifecycle evidence matrix

A future evidence matrix should answer, for an exact configuration:

```text
Requirement
    ↓
Verification plan/result
    ↓
ETK requirement/obligation consequence

Stakeholder need / ConOps
    ↓
Validation scenario/result

Model/twin
    ↓
credibility profile

Hazard/function
    ↓
R&M / FMEA / fault-tree analysis

Human role/task
    ↓
HSI evidence

Operational asset
    ↓
SE-OBS observations
```

The matrix must preserve missing/unknown cells.

Do not compress lifecycle readiness into a single green percentage.

## 13. Re-verification and re-validation after change

Change-impact analysis should generate review obligations, not silently reuse prior V&V.

For an exact change, the system should eventually determine candidates such as:

- verification activities requiring rerun;
- validation scenarios requiring rerun;
- model-credibility claims requiring review;
- R&M analyses requiring update;
- HSI evidence requiring update;
- operational baselines requiring revision.

Whether existing admitted evidence remains applicable is an ETK/currentness question.

## 14. Qualification corpus

The Systems Engineering Gym should eventually include cases where:

1. every requirement is verified but the system solves the wrong stakeholder problem;
2. simulation passes while physical validation fails;
3. a model is calibrated inside one envelope and misused outside it;
4. redundant components share a hidden common-cause dependency;
5. a reliability estimate comes from an irrelevant/stale population;
6. a maintenance task cannot physically be performed in the installed configuration;
7. required servicing exceeds staffing/time constraints;
8. operator workload causes system failure despite subsystem compliance;
9. training/procedure assumptions are missing;
10. anomaly is incorrectly promoted to root cause;
11. a changed configuration reuses old verification evidence;
12. a synthetic human model is misrepresented as human-in-the-loop evidence;
13. automated tests pass while intended-use validation fails;
14. a digital twin is used for a decision outside its credibility envelope.

Metrics should remain separate:

- traceability correctness;
- verification coverage/accuracy;
- validation adequacy;
- credibility-envelope correctness;
- R&M/common-cause discovery;
- maintainability reasoning;
- HSI defect discovery;
- change/staleness detection;
- abstention;
- false-authority rate.

## 15. Proposed implementation sequence

```text
SE-VV-000  lifecycle profile (this document)
SE-VV-001  verification plan + coverage matrix
SE-VV-002  validation / ConOps / scenario model
SE-VV-003  model/digital-twin credibility profile
SE-VV-004  reliability + maintainability model
SE-VV-005  FMEA/FMEDA + fault-tree projection
SE-VV-006  HSI role/task/performance model
SE-VV-007  discrepancy/nonconformance lifecycle
SE-VV-008  lifecycle evidence matrix
SE-VV-009  SE Gym V&V/R&M/HSI corpus
```

Runtime implementation remains blocked on the relevant lower-layer SE contracts and executable qualification.

## 16. Non-goals

- no second ETK;
- no certification emulator;
- no generic `verified=true` lifecycle state;
- no one-number readiness score;
- no automatic waiver/disposition authority;
- no RPN-to-authority shortcut;
- no simulation-to-validation shortcut;
- no calibration-to-truth shortcut;
- no synthetic-human-to-HIL shortcut;
- no reliability estimate presented as guaranteed outcome;
- no maintainability estimate presented as demonstrated field performance.

## 17. Closure criterion

This tranche succeeds when Symthaea can trace and distinguish:

```text
requirement verification
stakeholder/use validation
model/twin credibility
reliability
maintainability
human-system performance
```

for exact configurations and lifecycle contexts, identify what is missing, stale or outside its applicability envelope, and hand bounded results to ETK without collapsing those engineering propositions into one generic notion of truth or authority.
