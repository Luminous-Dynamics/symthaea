# ROB-TOOL-001A — robotic tool specialization and exchange evidence contract

Status: repaired source/data freeze for review only  
Parent: ROB-TOOL-001 #6038 / ROB-SPEC-001 #6033  
Frozen base: `main@eae17187e199e3a53d108b437c0215b5ff812261`

## Purpose

Freeze the first machine-checkable specialization contract for robotic tools/end-effectors and automatic exchange without creating a new component catalog, manufacturing-process authority, robot-control bus, safety kernel, workcell scheduler, or physical actuation path.

```text
tool mounted
!= tool identity established
!= interfaces compatible
!= calibration/currentness established
!= bounded task evidence admitted
!= task authorized
!= task executed successfully
```

and:

```text
same tool model
!= same physical tool instance
!= same calibration
!= same wear state
!= same host compatibility
```

## Ownership boundary

ROB-TOOL owns a composition/profile boundary over canonical owners. It may reference exact host/tool/configuration identities, interface states, calibration/TCP, host load/collision/workspace compatibility, external process-qualification receipts, consumable/wear state and bounded task evidence.

It does not own physical component identity, units, process qualification, observations, robot/workcell qualification, safety, scheduling, authority or execution evidence.

## External alignment

As observed at freeze time:

- ISO 11593:2022 is a published vocabulary reference for automatic end-effector exchange systems;
- ISO 10218-2:2025 is a published industrial robot application/cell integration reference.

These are external vocabulary/integration references only. This source establishes no standards conformance.

## Initial profiles

- `GenericGripper`
- `InspectionMetrologyHead`
- `PassiveMachineTendingAdapter`

The profile namespace remains extensible. Higher-force construction/process tools are deferred until the generic specialization boundary qualifies.

## Raw evidence model

The corpus deliberately contains no `tool_can_do_task=true`, no global `interface_compatible=true`, and no universal trial-count threshold.

`tool_defaults` and `exchange_defaults` define ordinary raw state. Every fixture then supplies only exact overrides. `field_vocabularies` freezes the allowed enum values so an independent qualifier can reject unknown values without inventing policy.

Tool-use raw facts include:

- physical tool identity;
- attachment observation;
- mechanical/power/fluid/data/action interfaces;
- calibration/TCP;
- host mass/inertia envelope;
- collision geometry;
- workspace;
- process-qualification requirement;
- exact external qualification-receipt presence/result/currentness/scope;
- consumable state;
- wear/health;
- configuration currentness;
- required and actual evidence plane;
- evidence currentness/scope;
- **required vs observed** independent task-trial multiplicity;
- planner/controller-only status;
- authority/execution-promotion requests.

This preserves:

```text
one observed success
!= profile-required evidence multiplicity

MODEL evidence
!= FIELD evidence

planner/controller result
!= task execution evidence
```

## External process qualification

A process-performing tool may require a canonical external qualification receipt.

V1 consumes the canonical result vocabulary:

`Pass | Fail | Blocked | EnvironmentFailure`

and separately requires:

```text
receipt bound
+ result = Pass
+ receipt current
+ exact scope match
```

before the ROB-TOOL profile may become admissible.

Therefore:

```text
drill mounted
!= fastening process qualified

welding head mounted
!= welding process qualified

tool has a PASS from another scope
!= this task/process profile qualified
```

ROB-TOOL never mints process qualification itself.

## TOOL fail-closed precedence

```text
physical/execution authority request
-> physical tool identity
-> observed attachment
-> interface contracts
-> calibration
-> TCP/frame
-> host mass/inertia envelope
-> collision geometry
-> workspace
-> required external process receipt/result/currentness/scope
-> consumable availability
-> wear/health currentness
-> configuration currentness
-> evidence plane/currentness/scope/multiplicity
-> ToolSpecializationProfileAdmissible
```

The terminal disposition means only that the synthetic specialization profile contains the required evidence relationships. It does not establish real task capability.

## Automatic exchange

The exchange profile preserves independently:

```text
controller/provider result
old-tool detach observation
new-tool attachment observation
new physical identity
mechanical connection state
power/fluid/data connections
semantic action contract
configuration-generation update
old-tool currentness retirement
calibration/TCP
host mass/inertia
collision geometry
```

The fail-closed chain is:

```text
controller success
!= physical detach/attach
!= correct identity
!= mechanical/service compatibility
!= configuration update
!= calibration current
```

`ExchangeSemanticallyComplete` remains an evidence/semantic disposition only.

## Configuration change

Tool exchange, repair, replacement, or materially different calibration can alter TCP, mass/inertia, compliance, collision geometry, process offsets, sensor transforms, payload envelope and command mapping.

A material tool change therefore creates a new exact configuration generation unless an independently qualified change-impact/transfer theorem permits bounded reuse. Historical task evidence remains historical.

## Frozen dispositions

- `ToolSpecializationProfileAdmissible`
- `IdentityBlocked`
- `AttachmentBlocked`
- `InterfaceBlocked`
- `CalibrationBlocked`
- `FrameBlocked`
- `HostEnvelopeBlocked`
- `CollisionModelBlocked`
- `WorkspaceBlocked`
- `ProcessQualificationBoundaryBlocked`
- `ConsumableBlocked`
- `HealthCurrentnessBlocked`
- `ConfigurationCurrentnessBlocked`
- `TaskEvidenceBlocked`
- `AuthorityBoundaryBlocked`
- `ExchangeControllerFailed`
- `ExchangeEvidenceIncomplete`
- `ExchangeConfigurationBlocked`
- `ExchangeCalibrationBlocked`
- `ExchangeSemanticallyComplete`

These are non-ranked states, not a robot-quality or readiness score.

## Synthetic corpus

Schema:

`rob-tool-001a-specialization-reference-v1`

Canonical compact sorted-key JSON + final newline SHA-256:

`e3244d59dd8d7ce385f21c7d79da06cd093fd6b79d9fd1ad5db770a84b319d2e`

Exact cases: **40**

Every frozen disposition has at least one fixture.

Disposition census:

- `AttachmentBlocked`: 1
- `AuthorityBoundaryBlocked`: 2
- `CalibrationBlocked`: 1
- `CollisionModelBlocked`: 1
- `ConfigurationCurrentnessBlocked`: 1
- `ConsumableBlocked`: 1
- `ExchangeCalibrationBlocked`: 2
- `ExchangeConfigurationBlocked`: 2
- `ExchangeControllerFailed`: 1
- `ExchangeEvidenceIncomplete`: 4
- `ExchangeSemanticallyComplete`: 1
- `FrameBlocked`: 2
- `HealthCurrentnessBlocked`: 1
- `HostEnvelopeBlocked`: 2
- `IdentityBlocked`: 2
- `InterfaceBlocked`: 3
- `ProcessQualificationBoundaryBlocked`: 4
- `TaskEvidenceBlocked`: 4
- `ToolSpecializationProfileAdmissible`: 4
- `WorkspaceBlocked`: 1

The corpus includes wrong/unverified instance identity; unobserved attachment; mechanical/power/action interface mismatch; stale calibration/TCP; host mass/inertia and collision-model blockers; unreachable workspace; profile-specific evidence multiplicity; MODEL-to-FIELD laundering; planner-only evidence; exact process receipt missing/Fail/stale/scope-mismatched and one current scoped PASS positive control; consumable and wear/currentness blockers; repair/configuration drift; exchange controller failure; controller-success/attachment disagreement; unverified new-tool identity; unresolved mechanical or power connection; stale old-tool currentness and missing generation update; stale post-exchange calibration/TCP; one complete exchange path; four admissible positive-control profiles; and physical/execution-authority promotion attacks.

## Qualification plan

A separate stdlib-only qualifier should:

1. hard-bind the repaired source head and both source blobs;
2. require exactly one qualifier commit / two files;
3. import no Symthaea production code;
4. verify canonical source bytes and digest;
5. verify exact schema/base/parent/issue/external-reference identities;
6. verify exact defaults and frozen field vocabularies;
7. reject unknown override fields or enum values;
8. reject attempts to redefine outcome or qualification-result vocabularies;
9. expand defaults and independently derive all 40 outcomes;
10. hostile-test all major blocker families;
11. require clean postflight.

## ROB-CELL boundary

ROB-CELL #6039 should consume exact ROB-TOOL profile/evidence refs rather than copy these semantics.

```text
robot profile admissible
+ tool specialization profile admissible
!= application/workcell qualified
```

Workcell spatial state, fixtures, infrastructure, human/shared-space assumptions, integration, commissioning and authority remain separate.

## BUILT-CX boundary

```text
ExchangeSemanticallyComplete
!= robot application commissioned
!= facility commissioned
```

BUILT-CX may later compose exact ROB-CELL commissioning evidence, not infer commissioning from a tool exchange.

## Claim ceiling

This source establishes no actual manipulation/process capability, industrial robot safety compliance, collaborative-operation approval, structural fastening, welding/process quality, construction quality, tool certification, manufacturing-process qualification, procurement/resource allocation, or physical actuation/task-execution authority.
