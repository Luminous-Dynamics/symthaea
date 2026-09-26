# SEMI-PROC-001B — MFG-PROC Adapter Map

Status: architecture/data contract only. No production semiconductor process behavior is implemented by this document.

Parent: #5907
Generic manufacturing substrate: #5902 @ `f300ae3615a789faaeaab78e259d5817612a6e4c`
Frozen semiconductor reference semantics: #5895 / #5900

## Core ownership theorem

```text
symthaea-manufacturing-process
+ symthaea-manufacturing-contracts
    own generic process identity / state / capability / plan topology

SEMI-PROC
    owns only semiconductor-specific roles and evidence composition
```

SEMI-PROC must not introduce a second generic state identity, transformation DAG, capability matcher, recipe-commitment identity, or rework graph.

## Exact generic types to reuse

### Process state identity

Reuse `ProcessStateRefV1` and `ProcessStateClassV1`.

Semiconductor roles map as follows where possible:

| Semiconductor role | Generic state class |
| --- | --- |
| substrate/material state | `MaterialState` |
| geometry/layer-stack state | `GeometryState` |
| surface/interface state | `SurfaceState` |
| cleanliness/contamination state | `CleanlinessState` |
| thermal-history state | `ThermalHistoryState` |
| metrology/inspection state | `InspectionState` |
| pattern-specific state not faithfully represented above | canonical semiconductor `Custom(...)` tag only when justified |

Use `ProcessStateRefV1::state_id()` / `coordinate()` as the canonical state identity. Do not hash a second `WaferStateId` over the same semantic state.

A semiconductor wrapper may carry role/context but its identity must delegate to the generic state coordinate.

## Transformation contract

Reuse `ProcessTransformationContractV1` for exact input/output/preserved-state sets.

SEMI-PROC may add semiconductor evidence references around a transformation, but it must not implement a second input/output-set validator or canonicalizer.

Important distinction:

```text
transformation contract declares intended semantic state relation
!= process attempt happened
!= output state physically observed
```

## Capability

Reuse:

- `ProcessCapabilityProfileV1`
- `ProcessCapabilityRequirementV1`
- `CapabilityMatchV1`
- `CapabilityEvidenceClassV1`

SEMI-PROC must consume `ProcessCapabilityRequirementV1::evaluate()` rather than reimplement process/profile/currentness/evidence comparison.

Suggested disposition mapping for V1 composition:

```text
ExactAdmitted
    -> capability prerequisite admitted

CompatibleButWeakerEvidence
UnresolvedExternalRefs
ExpiredOrStale
Unknown
    -> CapabilityUnresolved / stronger disposition unavailable

ProcessMismatch
EnvelopeMismatch
    -> incompatibility error or CapabilityUnresolved under the exact adapter profile
```

The implementation must freeze exact mapping in tests rather than infer optimistic compatibility.

## Process-plan topology

Reuse:

- `ProcessPlanV1`
- `ProcessPlanNodeV1`
- `PlanNodeKindV1`
- `ProcessPlanEdgeV1`
- `PlanEdgeKindV1`

Relevant existing generic nodes/edges already cover:

- `ProcessStep`
- `Inspection`
- `HoldPoint`
- `ExternalProvider`
- `MaterialHandling`
- conditional accept/reject
- bounded `Rework`
- hold release
- canonical DAG/reachability validation

SEMI-PROC must not create a second graph/cycle/reachability/rework implementation.

## Semiconductor evidence specialization

The missing production surface should remain small. Conceptually:

```text
SemiconductorProcessEvidenceV1 {
    declared_input_state_ref,
    attempt_input_state_ref,
    process_definition_ref,
    capability_requirement_ref,
    capability_profile_ref,
    capability_match,
    equipment_configuration_ref,
    attempt_evidence_ref,
    metrology_observation_ref,
    observed_output_state_ref,
    acceptance_profile_ref,
    acceptance_evaluation_ref,
    disposition,
    authority,
}
```

These fields should normally be references to canonical owners. Do not embed generic material, measurement, provenance, equipment, or authority data inside SEMI-PROC.

## Disposition reduction

V1 production behavior must reproduce the frozen #5895/#5900 reference semantics.

### `CapabilityUnresolved`

Use when the required capability is absent, unresolved, stale, weaker than required, or otherwise not admitted under the exact generic capability profile.

It must not be upgraded to `PlannedOnly` merely because a process step exists.

### `PlannedOnly`

Requires an admitted capability prerequisite but no process-attempt evidence.

### `AttemptedObservationIncomplete`

Use when an attempt exists but required physical observation/metrology is missing, stale, malformed, or otherwise insufficient for an observed/accepted state claim.

### `ObservedCandidateState`

Requires an exact observed output-state reference with sufficient observation evidence, but no admitted exact acceptance profile/evaluation.

### `AcceptedUnderProfile`

Requires the exact observed state plus explicit exact-profile acceptance evidence.

### `RejectedUnderProfile`

Requires the exact observed state plus explicit exact-profile rejection evidence.

### `ReworkRequired`

Rework must compose the generic bounded `PlanEdgeKindV1::Rework` topology while retaining the rejected state as immutable historical evidence and producing a new state/attempt lineage.

## Lineage rules

The production adapter must fail closed when:

- declared input-state coordinate != attempt input-state coordinate;
- downstream processing references intended output that lacks an admitted observed state;
- a required input/material/reference is absent;
- duplicate set-like refs appear where the generic contract rejects them;
- a process definition/profile or equipment configuration changes but stale identity/evidence is reused.

## Canonicalization

Delegate canonical set/order semantics to the generic owners whenever they already exist.

Do not independently sort/hash the same semantic data inside SEMI-PROC and thereby create competing identities.

## Authority

SEMI-PROC authority is representation/evidence only.

```text
process evidence disposition
!= machine execution authority
```

No SEMI-PROC disposition, including `AcceptedUnderProfile`, may mint fabrication, robot, gas/fluid, plasma, thermal, high-voltage, chemical, procurement, or operator authority.

## Qualification gates

Before production implementation opens:

1. #5900 exact reference validator head must execute its dedicated 16-case workflow successfully;
2. #5902 exact MFG-PROC replay head must execute relevant repository/focused qualification successfully.

The first production implementation must then differentially check the exact frozen #5895 corpus SHA-256:

`ec2e05fdf668d7bb0a98688de5de44eae6642f0bd96662e8fa4300feea2b9c85`

against all 16 preregistered expected outcomes.

## Nonclaims

This adapter map does not define or validate semiconductor recipes, process parameters, wafer transformations, equipment performance, process safety, yield, reliability, cleanroom/fab capability, or physical execution authority.
