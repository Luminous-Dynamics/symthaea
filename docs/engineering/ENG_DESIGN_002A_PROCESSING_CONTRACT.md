# ENG-DESIGN-002A — Executable Engineering Design-Processing Contract

Status: source contract only. Parent: ENG-DESIGN-002 (#6090). Process root: ENG-DESIGN-001 (#6005). Formal-proof admission: ENG-FORMAL-001 (#6089). Optional human/aesthetic intent profiles: ENG-DESIGN-003 (#6091) and ENG-DESIGN-004 (#6096).

## Purpose

Freeze the first machine-checkable state/data semantics for processing an engineering design from intent through bounded release-evidence eligibility without creating a CAD/CAE/PLM engine, solver, standards authority, procurement workflow, fabrication controller, or physical execution path.

The processor is orchestration over canonical owners. It carries references and process dispositions; it does not become the source of truth for materials, loads, quantities, geometry, site state, solver results, BIM state, manufacturing state, commissioning state, human preference, or physical authority.

## Core laws

```text
idea submitted != engineering design
model exists != design verified
verification != validation
formal proof != FIELD evidence
release evidence eligible != physical article conforms
release evidence eligible != external approval
release evidence eligible != physical execution authority
```

No scalar `readiness_score`, `maturity_score`, `design_score`, `poetry_score`, `beauty_score`, `meaning_score`, or universal ranking is permitted.

## EngineeringDesignPackageV1

The processing root should bind references for:

- exact package identity and configuration generation;
- stakeholder/intended-use needs;
- requirements and constraints;
- assumptions and invalidation conditions;
- architecture/decomposition;
- mandatory interfaces;
- canonical geometry/configuration refs;
- quantity/unit/frame refs;
- material-state/property refs;
- load/support/environment/duty refs;
- domain solver/model/profile refs;
- risks/hazards/failure modes;
- alternatives/decision history;
- admitted formal proof obligations and receipts;
- verification cases/evidence;
- validation cases/evidence;
- DfX/manufacturing/metrology/service refs;
- optional aesthetic/poetic intent refs;
- external authority/professional/regulatory refs where required;
- unresolved items;
- change-impact/currentness state;
- exact claim ceiling.

These are references; canonical domain owners remain authoritative.

## Fail-closed processing precedence

The reference processor derives one categorical disposition from raw facts in this order:

1. physical execution authority boundary;
2. append-only history integrity;
3. intent and intended-use completeness;
4. requirements and verification-method completeness;
5. architecture availability;
6. mandatory interface resolution;
7. assumption currentness;
8. risk review/open blockers;
9. preference-vs-hard-requirement conflict;
10. analysis-plan completeness;
11. requested-analysis support;
12. admitted proof obligations;
13. evidence collection;
14. repair/as-built requalification;
15. configuration currentness;
16. required FIELD evidence;
17. verification completion/result;
18. required external authority/currentness;
19. intended-use validation;
20. release-evidence completion.

A later-looking state may never bypass an earlier blocker.

## Process stages and blocking dispositions

The contract uses categorical states including:

- `NeedsClarification`
- `IntentStructured`
- `RequirementsBlocked`
- `RequirementsBound`
- `InterfaceBlocked`
- `AssumptionBlocked`
- `RiskBlocked`
- `PreferenceConflictBlocked`
- `InterfacesBound`
- `AnalysisUnsupported`
- `ProofObligationOpen`
- `AnalysisPlanned`
- `EvidenceInProgress`
- `VerificationFailed`
- `PhysicalEvidenceBlocked`
- `DesignVerified`
- `ValidationFailed`
- `ConfigurationStale`
- `RequalificationRequired`
- `ExternalAuthorityRequired`
- `UseValidated`
- `HistoryIntegrityBlocked`
- `ReleaseEvidenceEligible`
- `AuthorityBoundaryBlocked`

These are process dispositions, not maturity ranks.

## Evidence separation

Formal proof may satisfy only an admitted proof obligation for the exact proposition/subject/assumptions it covers. If a requirement explicitly needs FIELD/physical evidence, a proof receipt cannot substitute for it.

Likewise:

```text
verification PASS != intended-use validation PASS
validation PASS != release evidence complete
external approval required != approval may be inferred internally
```

## Human/aesthetic intent

Aesthetic, experiential, symbolic, or poetic intent is optional. When declared, the processor may carry references to #6091/#6096 and participate in change impact. Such preference-class intent cannot weaken safety-, accessibility-, maintenance-, inspection-, code-, or other hard requirements.

No poetic/aesthetic evidence may mint fabrication, procurement, operation, or physical execution authority.

## Configuration and history

Design/configuration changes preserve prior evidence as history but may invalidate its current applicability. Repair or as-built discrepancy creates a requalification/applicability obligation when claim-relevant.

Rejected alternatives and superseded decisions remain append-only history; deletion is a process-integrity failure rather than a cleanup step.

## External authority

Where professional, regulatory, permitting, certification, or other external authority is required, the processor records an exact current external reference. The process may continue when that dependency is satisfied, but the external authority does not become internal Symthaea authority.

## Reference corpus

The canonical corpus is `eng-design-002a-processing-reference-v1` with 26 synthetic raw-fact cases. An independent qualifier must derive each disposition without trusting `expected` as an oracle.

The cases cover vague intent; missing requirements; missing verification methods; architecture/interface/assumption/risk blockers; poetic-preference conflict; analysis planning/support; proof obligations; evidence collection; failed verification; proof-vs-FIELD separation; pending/failed validation; configuration drift; repair/requalification; external authority; incomplete release evidence; history deletion; complete positive release route; physical-authority laundering; proof plus separately present FIELD evidence; and satisfied external authority dependency.

## Formal-method boundary

ENG-FORMAL may later prove surrounding state-machine invariants such as:

```text
no mandatory gate bypass
preference-class intent cannot weaken a hard requirement
FIELD-required evidence cannot be discharged by proof-only evidence
stale configuration cannot reach strict release eligibility
append-only history cannot be erased by a forward transition
release eligibility cannot mint execution authority
```

Formal proof of those process invariants still does not establish any real engineering requirement as physically satisfied.

## Claim ceiling

This contract may establish repository design-processing semantics and exact synthetic process dispositions for a frozen package model. It establishes no unmodeled physical truth, product safety, design adequacy, certification, regulatory compliance, professional approval, manufacturing/process capability, commercial viability, procurement/resource allocation, or physical execution authority.
