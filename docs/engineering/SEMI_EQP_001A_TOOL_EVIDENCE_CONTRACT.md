# SEMI-EQP-001A — Semiconductor Tool Configuration and Capability Evidence Contract

Status: frozen reference contract for #5910 under SEMI-EQP-001 #5889 / SEMI-FAB-000 #5887.

This document defines evidence semantics only. It contains no machine-construction instructions, process recipes, operating points, controller commands, or execution authority.

## Core theorem

```text
machine concept
!= machine design
!= as-built machine
!= commissioned configuration
!= calibrated/current capability
!= process-family compatibility
!= qualified process capability
!= execution authority
```

A semiconductor tool is treated as a composition of canonical subsystem/evidence references. SEMI-EQP does not become the canonical owner of vacuum, thermal, optical, motion, electrical/RF, diagnostics, safety, calibration, provenance, or productive-closure physics.

## V1 evidence subject

A future implementation may represent a semiconductor-tool evidence subject with canonical references for roles such as:

- exact tool design/configuration;
- as-built article/configuration;
- process-family capability;
- workpiece/wafer handling;
- process workspace/chamber;
- motion/positioning;
- vacuum;
- gas/fluid service;
- thermal management;
- electrical/RF/power;
- optical subsystem;
- diagnostics/instrumentation;
- contamination control;
- exhaust/abatement;
- control/software;
- safety/interlock;
- calibration/currentness;
- subfab/facility requirements;
- maintenance/spares;
- commissioning evidence;
- observed capability evidence.

References are roles, not embedded subsystem models.

## Frozen dispositions

V1 reserves the following evidence dispositions:

- `ConceptOnly`
- `DesignOnly`
- `AsBuiltUncommissioned`
- `CommissionedPartialCapability`
- `CapabilityUnresolved`
- `ObservedCapabilityUnderProfile`
- `CalibratedCapabilityUnderProfile`
- `ProcessCompatibilityUnresolved`
- `ExecutionNotAuthorized`

No universal `ready`, `qualified`, or `machine_quality` scalar is admitted.

## Identity rules

Tool evidence identity must distinguish at least:

```text
machine design
as-built article
configuration revision
commissioning subject
calibration/currentness subject
process-family compatibility subject
```

Changing an equipment/configuration subject is a semantic identity change for evidence that depends on that configuration.

For collections explicitly declared set-like, source ordering is non-semantic and must canonicalize deterministically. Duplicate required references fail closed where duplicates have no meaning.

## Capability rules

### Design and physical state

A design artifact cannot mint an as-built article or commissioning state.

```text
CAD exists
!= physical machine exists
```

### Commissioning

As-built identity without commissioning remains `AsBuiltUncommissioned`.

Commissioning is profile/configuration relative. A commissioned machine can still have only partial capability if required subsystem or observation evidence is missing.

### Calibration/currentness

Observed capability with stale or unresolved calibration/currentness cannot be promoted silently to a stronger calibrated capability.

Historical evidence remains historically attributable to the calibration state in force at that time; later renewal does not rewrite it.

### Process compatibility

A declaration that a process family is supported is not sufficient. Required subsystem, diagnostic, subfab, calibration, and evidence references must resolve for the declared compatibility profile.

### Bounded trial evidence

One process-trial result can support an `ObservedCapabilityUnderProfile` claim for the exact trial/profile. It cannot establish general machine qualification, repeatability, yield, or production capability.

### Productive closure

Operational capability and productive closure remain independent.

```text
machine operational with imported critical subsystem
!= locally reproducible tool
```

SEMI-EQP records the subsystem/evidence references. CIV-BOOT/SEMI-BOOT owns local/import/service dependence.

### Safety and authority

Safety/interlock evidence and physical execution authority remain external authority-plane concerns. Stale/missing authority evidence does not erase machine observations, but it must block any `execution authorized` inference.

No SEMI-EQP artifact may mint physical execution authority.

## Frozen synthetic corpus

The exact known-answer corpus is:

`docs/release/evidence/semi-eqp-001a-synthetic-corpus-v1.json`

Canonical SHA-256:

`828ff976be87f9968fe949da28c21e4accee59c2ea36d0959d702ee5a3912d69`

The corpus contains 16 benign synthetic cases spanning:

1. design without as-built evidence;
2. as-built without commissioning;
3. commissioned configuration with unresolved required subsystem;
4. partial workspace observation without whole-tool evidence;
5. stale calibration blocking stronger capability;
6. declared process compatibility without required diagnostics;
7. unresolved subfab dependency;
8. configuration identity change;
9. canonical ordering of set-like subsystem refs;
10. duplicate-reference rejection;
11. one bounded process trial without general qualification;
12. imported critical subsystem with external productive closure;
13. stale safety evidence blocking execution authority;
14. later calibration renewal preserving stale historical evidence;
15. design proposal unable to mint physical state;
16. qualification artifact unable to mint execution authority.

## Prohibited V1 content

This contract and corpus intentionally contain no:

- tool dimensions;
- pressure or vacuum targets;
- gas identities or flow values;
- chemical identities/concentrations;
- temperatures;
- electrical powers/voltages/currents;
- RF frequencies/powers;
- processing times;
- doses;
- feature dimensions/targets;
- exhaust sizing;
- controller commands;
- construction/fabrication instructions;
- hazardous operating procedures.

## Future implementation rule

A future SEMI-EQP software adapter must consume canonical generic subsystem/evidence owners where available and differentially reproduce this exact 16-case corpus before stronger semantic claims are admitted.

A PASS over this corpus would establish software evidence semantics only. It would not establish a real semiconductor tool, process compatibility, commissioned equipment, process safety, wafer transformation, process qualification, yield, or fab capability.
