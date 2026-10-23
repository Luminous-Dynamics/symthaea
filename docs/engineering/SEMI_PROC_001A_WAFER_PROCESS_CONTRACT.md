# SEMI-PROC-001A — Wafer-State / Process-Step Contract

Status: preregistered reference contract for issue #5894  
Parent: SEMI-PROC-001 #5888 / SEMI-FAB-000 #5887  
Corpus: `docs/release/evidence/semi-proc-001a-synthetic-corpus-v1.json`  
Corpus SHA-256: `ec2e05fdf668d7bb0a98688de5de44eae6642f0bd96662e8fa4300feea2b9c85`

## Purpose

Freeze the semantic boundary for semiconductor process-flow evidence before any production
implementation, process recipe, tool control, or physical wafer campaign exists.

This contract deliberately defines **references, lineage, dispositions, and claim ceilings**.
It does not define semiconductor process operating parameters.

Core theorem:

```text
process step specified
!= process executable
!= process executed
!= intended transformation observed
!= wafer state qualified
```

A process attempt can exist without a valid output-state claim. A modeled or intended output
state can never silently become an observed state.

## Ownership

SEMI-PROC is a semiconductor specialization over existing owners.

- MFG-PROC #5704 owns namespaced manufacturing process-family/profile representation.
- ENG-SEMI #5671 and REF-001 #5868/#5877 own semiconductor/device physics and TCAD evidence.
- SEMI-EQP #5889 owns semiconductor manufacturing-equipment architecture/capability.
- SEMI-SUBFAB #5890 owns facility/subfab capability dependencies.
- SEMI-MET #5891 owns semiconductor metrology-role composition.
- SEMI-QUAL #5892 owns machine→process→wafer qualification composition.
- FIELD/SE-OBS own physical observations, calibration/currentness, quantities, and measured truth.
- Mycelix owns actual article/lot/work/custody provenance.
- CIV-BOOT owns productive-capability closure.
- Existing safety/ETK/HAL/operator systems retain execution authority.

This contract must not duplicate those systems.

## V1 reference model

### WaferStateRef

`WaferStateRef` is an immutable reference to one exact wafer/sample state in a lineage.

A state may reference canonical evidence for:

- subject/article identity;
- substrate/material state;
- geometry/layer-state descriptions;
- surface/interface state;
- pattern/mask state;
- physical observations and derived quantities;
- process-history lineage;
- unresolved or unknown attributes.

SEMI-PROC does not own the underlying quantity/material/observation payloads.

### SemiconductorProcessStepV1

A process step is a **planned transformation subject**, conceptually containing:

```text
SemiconductorProcessStepV1 {
    step_id,
    input_state_ref,
    process_family_ref,
    intended_transformation_class,
    equipment_capability_ref,
    material_input_refs,
    subfab_requirement_refs,
    metrology_gate_refs,
    authority_profile_ref,
    expected_output_constraint_refs,
}
```

No field in this structure establishes that the physical transformation happened.

### ProcessAttemptEvidenceRef

An execution attempt, if one exists, must refer to the exact:

- input state;
- process-step identity;
- equipment/configuration subject;
- relevant material/input subjects;
- relevant facility/subfab context;
- authority/execution subject;
- time/evidence lineage.

The representation does not itself mint execution authority.

### Observed output state

An observed output state requires observation/metrology evidence. A command completion or
equipment status is insufficient.

```text
attempt completed
!= observed wafer transformation
```

A downstream process step may consume only a state admitted by its declared policy. It cannot
consume a merely intended output as though it had been observed.

## Initial dispositions

V1 freezes the following reference vocabulary:

- `PlannedOnly`
- `CapabilityUnresolved`
- `AttemptedObservationIncomplete`
- `ObservedOutOfProfile`
- `ObservedCandidateState`
- `AcceptedUnderProfile`
- `RejectedUnderProfile`
- `ReworkRequired`
- `ExecutionNotAuthorized`

These are profile-relative evidence dispositions. They are not universal manufacturing states.

There is deliberately no `success=true`.

## Identity semantics

Claim-relevant changes create distinct subjects, including at least:

- input-state identity;
- process-family/profile identity;
- equipment/configuration identity;
- material/input references;
- required facility/subfab references;
- metrology-gate profile;
- expected-output/acceptance profile.

Order may be canonicalized only for collections explicitly declared semantically unordered.
Duplicate references that are nonsensical under the declared role must fail closed.

## Lineage and rework

Historical states and attempts are immutable evidence.

Rework creates a new branch/revision:

```text
state A
  ↓ attempt 1
state B — rejected
  ↓ rework attempt
state C — candidate/accepted
```

`state B` and the failed attempt remain part of the lineage.

Later success never rewrites prior failure.

## Metrology and acceptance

A physical process attempt and a metrology gate are separate events.

Examples:

```text
process attempt exists
+ no qualifying observation
→ AttemptedObservationIncomplete
```

```text
observation exists
+ no declared acceptance profile
→ ObservedCandidateState
```

```text
observation current and compatible
+ exact profile evaluation passes
→ AcceptedUnderProfile
```

Acceptance under one profile does not imply acceptance under another.

Stale/invalid calibration can preserve the observation while blocking a stronger qualification.

## Negative evidence

The contract treats all of the following as first-class evidence:

- failed process attempts;
- rejected wafer states;
- missing observations;
- stale calibration;
- profile mismatch;
- unresolved equipment capability;
- missing material/input references;
- lineage mismatch;
- rework;
- process/model disagreement.

Negative evidence is never dropped merely because a later attempt succeeds.

## Synthetic reference corpus

The frozen corpus contains 16 cases:

1. planned step without equipment capability;
2. capability available but no attempt;
3. attempt without metrology;
4. observed candidate without acceptance profile;
5. accepted under exact profile;
6. rejected under exact profile;
7. wrong input-state lineage;
8. downstream consumption of an unobserved intended state;
9. rework branch preserving rejected history;
10. missing required material/input reference;
11. stale metrology blocking stronger acceptance;
12. process-profile change creating distinct identity;
13. equipment-configuration change creating a distinct execution subject;
14. canonicalization of explicitly unordered references;
15. duplicate nonsensical reference rejection;
16. later success preserving earlier failure and minting no authority.

The corpus is intentionally synthetic and contains **no semiconductor fabrication recipe
parameters**.

## Security / authority boundary

Nothing in SEMI-PROC-001A authorizes or instructs:

- chemical handling;
- process-gas introduction;
- plasma/high-voltage/high-temperature operation;
- wafer-handling robot motion;
- machine energization;
- equipment control;
- procurement;
- cleanroom/facility certification;
- physical process execution.

Execution authority remains external.

## Implementation rule

The first production implementation must be able to consume or reproduce this contract without
weakening it.

It may claim only representation/validation of these semantics until stronger evidence exists.

A future implementation PASS therefore means, at most:

```text
software faithfully represents the frozen synthetic contract
```

It does not mean:

```text
physical semiconductor process works
wafer was transformed
equipment is safe
process is qualified
fab capability exists
```

## Promotion sequence

```text
SEMI-PROC-001A frozen contract/corpus
        ↓
deterministic reference validator/oracle
        ↓
shared-owner placement audit
        ↓
thin production representation/adapter
        ↓
SEMI-EQP / SUBFAB / MET composition
        ↓
SEMI-QUAL
        ↓
synthetic process campaign
        ↓
only later: appropriately authorized physical evidence
```

No later rung should be inferred from an earlier one.
