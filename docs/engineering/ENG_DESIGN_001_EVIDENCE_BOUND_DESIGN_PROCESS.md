# ENG-DESIGN-001 — Evidence-Bound Engineering Design Process

Status: Source contract  
Issue: #6005  
Source base: `main@eae17187e199e3a53d108b437c0215b5ff812261`  
Authority: repository process semantics only; no physical execution authority

## 1. Purpose

This document defines a reusable engineering-design process for Symthaea's physical, manufacturing, infrastructure, robotics, sensing, semiconductor, and other engineering work.

It complements, rather than replaces:

- `docs/compliance/SDLC.md` for the software/AI lifecycle;
- `docs/compliance/DEVELOPMENT_PROCEDURES.md` for day-to-day development procedures;
- the existing ADR system for architectural decisions;
- canonical observation, quantity/unit, configuration, calibration, process, lifecycle, service, closure, Mycelix operational, and physical-authority owners;
- exact-head source/qualifier evidence workflows already used throughout the repository.

The process exists to make design intent, requirements, assumptions, interfaces, design decisions, risks, verification, validation, configuration, and evidence form one inspectable thread.

It must not become a second simulation engine, PLM/ERP database, observation store, risk engine, qualification engine, lifecycle engine, or machine-control path.

## 2. Governing distinction

The core design thread is:

```text
stakeholder / intended-use need
→ requirement + constraint
→ assumption + interface
→ alternatives + decision
→ risk / failure-mode treatment
→ verification plan
→ validation plan
→ implementation / as-built configuration
→ evidence
→ claim ceiling
→ change impact / maintenance / requalification
```

The process preserves the following distinctions:

```text
design intent != requirement satisfaction
requirement written != requirement verified
simulation/model PASS != physical verification
verification != validation
prototype works != production/process capability
one tested article != all produced articles
nominal interface compatibility != integrated-system qualification
configuration changed != prior evidence remains current
risk mitigated on paper != mitigation verified
repair performed != repaired item requalified
operational fact != engineering qualification
engineering recommendation != physical execution authority
```

## 3. Typed design-thread references

Use stable typed identifiers in claim-bearing design packets.

| Prefix | Meaning | Notes |
|---|---|---|
| `NEED-*` | stakeholder/intended-use need | describes intended purpose, not proof of satisfaction |
| `REQ-*` | verifiable requirement | should trace to one or more needs |
| `CON-*` | constraint | boundary condition, regulation, environmental or design constraint |
| `ASM-*` | assumption | must state source and invalidation condition |
| `IFC-*` | interface | identity and contract at a system boundary |
| `DEC-*` | design/trade decision | preserves alternatives, evidence and rationale |
| `RISK-*` | risk/failure-mode item | mitigation claim remains distinct from verified mitigation |
| `VER-*` | verification case | proves conformance to a declared requirement only |
| `VAL-*` | validation case | evaluates fitness for declared intended use |
| `CFG-*` | configuration subject/generation reference | points to canonical configuration owner |
| `EVD-*` | evidence/receipt reference | points to canonical evidence owner |
| `CHG-*` | change-impact record | identifies invalidated or requalification-required evidence |

These identifiers are references. They do not create new canonical stores for the referenced facts.

## 4. Minimum design packet

A claim-bearing engineering design packet must contain the following sections or machine-readable equivalents.

### 4.1 Design intent

Record:

- problem/intended use;
- stakeholders or beneficiary context where applicable;
- explicit non-goals;
- expected operating context;
- claim ceiling;
- authority ceiling.

A design packet may describe a desired function without claiming that function has been achieved.

### 4.2 Requirements and constraints

Every claim-bearing requirement should have:

- unique `REQ-*` identity;
- source/need references;
- rationale;
- clear statement;
- applicability/scope;
- verification method;
- expected evidence plane or evidence class;
- acceptance criterion or externally-owned criterion reference where appropriate.

Constraints should be separately identified as `CON-*`; do not hide constraints inside prose-only rationale.

No universal scalar readiness, priority, self-sufficiency, closure or civilization score is required or permitted by this process.

### 4.3 Assumptions

Every `ASM-*` must state:

- assumption;
- source/basis;
- affected requirements/decisions;
- owner or responsible review point;
- invalidation condition;
- currentness state.

If an assumption becomes stale or false, dependent evidence remains historically retained but cannot silently remain current.

### 4.4 Interfaces

Every claim-bearing `IFC-*` should identify, as applicable:

- participating subjects;
- direction;
- information/material/energy/mechanical relationship;
- quantity/unit/type/frame semantics;
- timing/synchronization semantics;
- tolerance or envelope references;
- configuration/version binding;
- error/failure behavior;
- authority boundary;
- unresolved items.

An unresolved interface is a blocker when the claim depends on that interface. It is not permission to infer missing semantics.

### 4.5 Design alternatives and decisions

A `DEC-*` record must preserve:

- decision question;
- alternatives considered;
- objectives;
- hard constraints;
- evidence used;
- uncertainty and unresolved evidence;
- tradeoffs;
- selected alternative;
- rationale;
- rejected alternatives;
- supersession history.

A scalar score may be used locally if a domain genuinely requires it, but it must not erase the underlying dimensions, uncertainty, hard constraints, or rejected alternatives. This process never requires a universal weighted score.

### 4.6 Risks and failure modes

`RISK-*` items should preserve:

- failure mode or technical risk;
- cause/context;
- affected requirement/interface;
- consequence category;
- detection/observability route;
- mitigation or design control;
- residual uncertainty;
- verification route for the mitigation.

A planned mitigation is not a verified mitigation.

### 4.7 Verification plan

Verification answers whether the specified design/product satisfies its declared requirements.

Permitted top-level method labels are:

- `Analysis`
- `Inspection`
- `Demonstration`
- `Test`

A `VER-*` case binds:

- requirement refs;
- exact subject/configuration;
- method;
- required evidence plane/class;
- procedure or owner reference;
- acceptance criterion;
- evidence refs;
- result;
- assumptions and environmental conditions;
- currentness.

A model or simulation may verify a requirement whose declared method/evidence permits model evidence. It may not silently satisfy a requirement declared to require physical/FIELD evidence.

### 4.8 Validation plan

Validation answers whether the system satisfies the declared intended use or stakeholder need.

A `VAL-*` case binds:

- need/intended-use refs;
- exact subject/configuration;
- validation method;
- use context;
- success/acceptance criteria;
- evidence;
- residual limitations;
- result.

Verification may be complete while validation remains incomplete. That state must be representable directly.

### 4.9 Configuration and as-built binding

Every claim-bearing verification or validation result must bind the exact applicable `CFG-*` generation or equivalent canonical configuration identity.

A design revision, component substitution, remount, firmware change, calibration epoch change, material lot change, repair, or other claim-relevant configuration change must trigger `CHG-*` impact analysis.

Historical evidence is append-only. Requalification produces new current evidence; it does not rewrite the old event as if the configuration never changed.

### 4.10 DfX review

Physical designs should explicitly review the dimensions relevant to their domain rather than assuming buildability from geometry alone.

At minimum consider:

- manufacturability;
- assembly/integration;
- inspectability and metrology;
- testability;
- calibration/currentness;
- maintainability and repairability;
- tooling and fixture dependencies;
- spares and replaceability;
- supply/import dependencies;
- common-mode dependencies;
- safe failure/recovery boundary;
- packaging/logistics where relevant;
- end-of-life, remanufacture, circular recovery and requalification where relevant.

Absence of a relevant DfX route is an unresolved dependency, not evidence that the route is unnecessary.

## 5. Right-sized review gates

These gates are internal engineering reviews. They are not certification events.

### 5.1 Design Intent Review (DIR)

Entry:

- intended use or need exists;
- initial requirements and constraints exist;
- major non-goals and claim ceiling are explicit.

Success criteria:

- every claim-bearing requirement traces to an intended use/need or justified constraint;
- every requirement has an identified verification method;
- scope is bounded;
- no physical authority is introduced by the packet.

### 5.2 Interface & Risk Review (IRR)

Entry:

- DIR success;
- candidate architecture exists;
- assumptions/interfaces/alternatives are identified.

Success criteria:

- claim-relevant assumptions are current;
- claim-relevant interfaces are resolved or explicitly blocking;
- alternatives and rejected alternatives are retained;
- major technical risks/failure modes and mitigation-verification routes are identified;
- no unresolved common mode is described as independent redundancy.

### 5.3 Qualification Readiness Review (QRR)

Entry:

- IRR success;
- design/configuration subject is frozen enough to test;
- verification plan is executable.

Success criteria:

- verification coverage exists for claim-bearing requirements;
- required evidence planes/classes are explicit;
- test/inspection/analysis/demonstration methods are bound to exact subjects;
- calibration/metrology/currentness dependencies are identified;
- production/process capability is not inferred from prototype function;
- physical work remains under external human/authorized control paths.

### 5.4 Release Evidence Review (RER)

Entry:

- verification execution complete for the declared release claim;
- validation state explicitly known;
- evidence and exact configuration refs available.

Success criteria:

- evidence belongs to the current applicable configuration;
- validation status is not conflated with verification status;
- repair/substitution/configuration changes have required requalification;
- applicability does not exceed the tested/qualified scope;
- operational/Mycelix facts have not been laundered into engineering qualification;
- engineering evidence has not been laundered into operational truth;
- claim ceiling and residual limitations remain explicit;
- no physical execution authority is minted.

## 6. Change-impact discipline

Every claim-relevant change should create or update a `CHG-*` record.

The change-impact check asks:

1. Which `REQ-*`, `CON-*`, `ASM-*`, `IFC-*`, `DEC-*` and `RISK-*` items changed?
2. Which `VER-*` and `VAL-*` results depended on those items?
3. Which exact `CFG-*` generations are affected?
4. Is existing evidence still current for the unchanged scope?
5. Which evidence is now stale, invalid, blocked, or requires requalification?
6. Does the change introduce a new external owner/interface?
7. Does it alter any authority boundary?

Default rule:

```text
uncertain impact
→ retain prior evidence historically
→ mark currentness unresolved
→ require explicit requalification decision
```

Do not default to silent evidence carry-forward.

## 7. Design-to-manufacturing digital thread

For physical/manufacturing work the design packet should remain linkable through:

```text
requirement
→ model/drawing/specification
→ manufacturing/process plan
→ as-built/configuration identity
→ inspection/metrology observation
→ verification result
→ validation result
→ service/maintenance event
→ repair/replacement
→ requalification
```

The links should reference canonical owners rather than copy their state.

This enables later compatibility with model-based systems engineering and manufacturing digital-thread tooling without making SysML, STEP, QIF, MTConnect, CAD, PLM, MES, or another external format the semantic root of Symthaea.

## 8. Relationship to external engineering practice

This process intentionally adopts several widely used systems-engineering ideas:

- requirements traceability and planned verification;
- explicit interface management;
- configuration/change control;
- technical risk management;
- decision analysis with retained alternatives;
- separate verification and validation;
- review entrance/success criteria;
- digital-thread feedback from manufacturing/inspection to design.

Informative references:

- NASA Systems Engineering Handbook and current NASA systems-engineering guidance;
- NIST Digital Thread for Manufacturing;
- OMG SysML v2 / KerML / Systems Modeling API.

External standards and tools remain informative/interoperability targets. They do not override Symthaea's canonical evidence and authority boundaries.

## 9. Reference corpus

The synthetic source corpus is:

`docs/release/evidence/eng-design-001-process-reference-v1.json`

Canonical compact sorted-key JSON plus final newline SHA-256:

`5767253ad7bbf722e3fbafb7033480515973871d06c4935c60f397b0fe1332cb`

It contains 16 known-answer cases spanning:

- missing intended-use need;
- missing requirement traceability;
- missing verification method;
- stale assumption;
- unresolved interface;
- deleted decision alternatives;
- unverified mitigation;
- incomplete verification;
- MODEL evidence used for a FIELD-required verification;
- verified-but-not-validated state;
- configuration drift;
- repair without requalification;
- common-mode redundancy;
- scope overgeneralization;
- cross-owner evidence laundering;
- physical execution-authority rejection;
- complete validated-release eligibility with zero execution authority.

A future independent validator may establish only faithful process-contract representation.

## 10. Claim ceiling

This source contract establishes no real:

- product safety;
- regulatory or standards certification;
- manufacturing readiness;
- process capability;
- machine or component rating;
- water/food/health/grid safety;
- operational service sufficiency;
- economic viability;
- workforce sufficiency;
- procurement or resource allocation;
- physical actuation or execution authority.

It defines how future design evidence should be organized and reviewed.
