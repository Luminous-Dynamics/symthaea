# WATER-MFG-001B — ENG-DESIGN Consumer Qualification Packet

Status: Source/data contract  
Authority: analysis and design only; no physical execution authority  
Base: `main@eae17187e199e3a53d108b437c0215b5ff812261`

## 1. Purpose

This tranche is the first concrete consumer of `ENG-DESIGN-001`.

It applies the new engineering-design thread to the benign WATER-MFG profile already frozen by WATER-MFG-001A: a synthetic non-potable low-head circulation + filter module with explicit component, observation, maintenance, common-mode, repair/requalification, service, and productive-closure boundaries.

The purpose is **not** to qualify a real water system.

The purpose is to prove that the generic design process can bind an actual domain contract without duplicating WATER-MFG, IND-COMP, PROD-EQP, SE-OBS, Mycelix, CIV-SERVICE, or CIV-BOOT semantics.

## 2. Exact upstream references

This packet references, rather than copies, the exact current source/qualifier subjects:

| Owner | Source | Required qualifier |
|---|---|---|
| ENG-DESIGN | #6006 / `0cc3a618...` | #6009 / `1273c377...` |
| WATER-MFG | #6001 / `3c8a439a...` | #6003 / `62d0d4ec...` |
| IND-COMP | #5990 / `b2346d03...` | #5995 / `652b6f04...` |
| PROD-EQP | #5988 / `95631725...` | #5993 / `8482bc22...` |

Each exact qualifier is bound as:

`HostedExactHeadPassReceiptRequired`

with no receipt embedded in this source packet.

The packet therefore has the deliberate overall state:

`QualificationPending`

A transient runner state such as queued, in-progress, failed, or passed is **not source data**. Hosted execution results belong in external evidence/receipts bound to the exact qualifier head.

No consumer may promote these source semantics into production qualification until the required exact-head hosted PASS receipts exist.

## 3. Intended use

`NEED-WATER-001`

Provide an evidence-bound, maintainable non-potable low-head circulation/filter-module profile suitable for later engineering qualification under a declared feedstream and exact configuration.

Explicit non-goals:

- potable-water qualification;
- public-health or sanitation claims;
- pathogen-removal claims;
- chemical dosing or disinfection;
- autonomous pump/valve actuation;
- hydraulic operating ratings;
- treatment recipes;
- real service-capacity claims.

## 4. Design requirements

The packet freezes twelve claim-bearing design requirements.

### REQ-WATER-001 — Scope

The qualified scope remains explicitly non-potable and profile-bounded.

### REQ-WATER-002 — Feedstream characterization

Source presence is insufficient. The applicable feedstream profile must be characterized before treatment-effect or service claims are admitted.

### REQ-WATER-003 — Component function

Pump or valve presence does not establish requested function or duty. Physical component-function claims require qualified FIELD evidence owned by the appropriate engineering/observation layers.

### REQ-WATER-004 — Media identity/currentness

Filter/media identity and currentness remain bound to the exact installed configuration.

### REQ-WATER-005 — Measurement currentness

Claim-bearing flow/pressure/level observations require current calibration plus exact observation/configuration identity.

### REQ-WATER-006 — Commissioning separation

As-built presence and component function remain distinct from module commissioning.

### REQ-WATER-007 — Independent redundancy

Redundant paths may be called independent only when relevant power, control, sensing, feed, structural, or other common modes are independently bounded.

### REQ-WATER-008 — Renewal path

Maintenance, media renewal, consumables, calibration renewal, tooling, and spares remain explicit continuity dependencies.

### REQ-WATER-009 — Repair/requalification

A repaired or substituted component requires claim-relevant requalification before prior evidence is treated as current.

### REQ-WATER-010 — Applicability transfer

Feedstream, component, configuration, or environment drift outside the qualified profile blocks silent applicability transfer.

### REQ-WATER-011 — Productive closure

Partial local support remains distinct from full productive/reproductive closure.

### REQ-WATER-012 — Authority ceiling

The packet and every derived result have zero pump, valve, dosing, procurement, resource-allocation, or physical-operation authority.

## 5. Constraints

The packet hard-freezes five constraints:

1. no potable/public-health inference;
2. no physical actuation path;
3. no operating recipe or hazardous parameter set;
4. exact configuration/currentness binding for claim-bearing evidence;
5. unresolved/queued upstream qualification may not be represented as PASS.

## 6. Assumptions

### ASM-WATER-001 — Applicability profile

The declared synthetic feedstream profile remains unchanged.

Invalidation: feedstream identity or applicability profile changes.

### ASM-WATER-002 — Future physical measurement

Any future FIELD evidence is assumed to have current calibration and exact observation/configuration identity.

Invalidation: calibration epoch, sensor, mounting, parser, or acquisition context changes.

### ASM-WATER-003 — Upstream qualification identity

The exact ENG-DESIGN, WATER-MFG, IND-COMP and PROD-EQP qualifier identities are frozen, but their hosted terminal results are external evidence.

The source packet does not snapshot ephemeral runner state. It remains qualification-pending until exact-head PASS receipts are bound by a later evidence layer.

### ASM-WATER-004 — Import dependence

Imported components or consumables remain explicit productive dependencies until an independently evidenced local renewal/requalification route exists.

## 7. Interfaces

The first consumer deliberately exercises eight interface classes:

`IFC-WATER-001` source/feedstream → module inlet  
`IFC-WATER-002` component article → hydraulic-function evidence  
`IFC-WATER-003` media article → installed media configuration  
`IFC-WATER-004` calibrated sensor → observation evidence  
`IFC-WATER-005` maintenance/replacement → requalification state  
`IFC-WATER-006` Mycelix operational fact → engineering evidence boundary  
`IFC-WATER-007` installed module → CIV-SERVICE projection  
`IFC-WATER-008` local support route → CIV-BOOT productive closure

The critical theorem is:

```text
interface exists
!= interface qualified
```

and:

```text
operational event
!= engineering qualification
```

## 8. Design decisions

### DEC-WATER-001 — First profile

Selected: benign non-potable low-head circulation/filter module.

Rejected for this tranche:

- potable-treatment profile;
- autonomous dosing/control profile.

Reason: the selected profile exercises enough cross-domain engineering semantics without elevating health or execution authority.

### DEC-WATER-002 — Physical evidence

Where the claim is physical component or module function, FIELD evidence remains required.

MODEL evidence may support analysis; it cannot silently substitute for FIELD evidence.

### DEC-WATER-003 — Productive closure

Imported dependencies are represented as partial closure rather than collapsed into a binary self-sufficient flag.

### DEC-WATER-004 — Operation authority

The design packet is advisory/analytical only.

Physical work remains under separately authorized human or machine-control systems.

## 9. Technical risks and failure modes

The packet freezes seven first-order risks:

- `RISK-WATER-001`: feedstream/profile drift;
- `RISK-WATER-002`: shared common mode presented as independent redundancy;
- `RISK-WATER-003`: stale calibration or observation context;
- `RISK-WATER-004`: repair/substitution silently inheriting old evidence;
- `RISK-WATER-005`: Mycelix operational fact laundering into engineering qualification;
- `RISK-WATER-006`: non-potable evidence generalized to stronger service or potable claims;
- `RISK-WATER-007`: queued upstream semantics promoted to qualified production basis.

Every mitigation has a declared verification route.

A mitigation written in this document is not a verified mitigation.

## 10. Verification plan

There is one `VER-WATER-*` case for each requirement.

Every case is currently:

`NotExecuted / Unqualified`

This is deliberate.

The packet may define what must be verified without claiming the verification has occurred.

Top-level methods remain the ENG-DESIGN vocabulary:

- Analysis;
- Inspection;
- Demonstration;
- Test.

Physical component function and claim-bearing physical measurement are explicitly FIELD-bound.

## 11. Validation plan

`VAL-WATER-001`

Intended use:

Evidence-bound non-potable circulation/filter engineering profile under a declared feedstream and exact configuration.

Current result:

`PlannedQualificationPending`

A later claim-bearing validation requires, at minimum:

- exact upstream qualifier results;
- exact subject/configuration;
- claim-relevant FIELD evidence;
- the declared use context;
- explicit residual limitations.

Verification completion will not automatically imply validation completion.

## 12. Review state

The first consumer packet is intentionally representable in a partially mature state:

```text
DIR = SourceReady
IRR = SourceReady
QRR = BlockedPendingRequiredQualificationReceipts
RER = NotEntered

overall = QualificationPending
```

This is an important process test.

The engineering thread must represent useful progress without converting progress into evidence that does not yet exist.

## 13. Change-impact rule

`CHG-WATER-001` is triggered by:

- any bound upstream source-head change;
- any upstream qualifier-head change;
- feedstream/applicability-profile change;
- component substitution or repair;
- sensor/calibration/configuration change;
- claim-scope change.

Default action:

```text
retain prior evidence historically
→ mark currentness unresolved
→ perform explicit impact review
→ requalify affected claims where required
```

No silent evidence carry-forward.

## 14. Consumer-specific adversarial corpus

The machine-readable packet contains thirteen known-answer mutation cases.

They cover:

- missing need;
- missing verification method;
- unresolved claim-relevant interface;
- shared common-mode laundering;
- MODEL evidence substituted for FIELD evidence;
- unbound qualifier falsely promoted to PASS without a receipt;
- stale calibration;
- configuration change without impact review;
- repair without requalification;
- Mycelix event used as engineering proof;
- non-potable evidence generalized to potable;
- physical-operation authority requested.

The no-mutation baseline remains:

`QualificationPending`

That baseline is important: a faithful consumer with exact qualifier identities but no bound hosted PASS receipts must remain unresolved. Later receipt availability should advance a derived qualification state without rewriting this frozen design subject.

## 15. What this proves if independently qualified

A future independent validator PASS may establish only that the WATER design packet faithfully composes the frozen engineering-process and WATER/component/equipment source semantics.

It may **not** establish:

- potable-water safety;
- treatment efficacy;
- hydraulic rating;
- commissioned equipment;
- service sufficiency;
- productive closure;
- economics;
- procurement/resource allocation;
- physical execution authority.

## 16. Next gate

Do not create the production WATER adapter merely because this packet exists.

The intended progression is:

```text
ENG-DESIGN source + exact-head PASS receipt
WATER-MFG source + exact-head PASS receipt
IND-COMP source + exact-head PASS receipt
PROD-EQP source + exact-head PASS receipt
        ↓
WATER-DESIGN consumer source + independent qualifier
        ↓
only after exact relevant PASS:
thin production adapter
        ↓
FIELD campaign / exact as-built subject
        ↓
verification
        ↓
validation
```

This keeps the architecture moving forward without allowing documentation maturity to masquerade as physical maturity.
