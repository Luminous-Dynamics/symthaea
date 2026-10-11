# ROB-CELL-001A — robot application, workcell, infrastructure, and fleet composition contract

Status: source/data freeze for review only  
Parent: ROB-CELL-001 #6039 / ROB-SPEC-001 #6033  
Frozen base: `main@eae17187e199e3a53d108b437c0215b5ff812261`

## Purpose

Freeze the first machine-checkable **robot application/workcell/fleet composition contract** without creating another robot ontology, fleet scheduler, planner, safety kernel, facility model, commissioning engine, infrastructure controller, or authority system.

This tranche is source/data only.

```text
robot qualified
!= robot application qualified

robot + tool qualified independently
!= robot-tool-task application qualified

planner/scheduler output
!= authority
!= physical execution
!= successful outcome

application commissioned
!= manufacturing process qualified
```

## Ownership boundary

ROB-CELL-001A owns only composition semantics for exact references to:

- robot/platform qualification receipts;
- tool-specialization qualification receipts;
- facility/workcell configuration identity/currentness;
- spatial/reference network currentness;
- fixed-cell collision/fixture context;
- mobile localization/map context;
- infrastructure resource interactions;
- handoff/custody observations;
- common-mode dependency declarations;
- external authority references;
- application commissioning receipts;
- task evidence scope/plane/multiplicity;
- reassignment admissibility references.

It does **not** own the underlying robot/tool qualification, BIM/facility truth, physical observations, safety certification, commissioning execution, manufacturing process capability, work allocation authority, or actuation.

## Two application kinds

The frozen v1 corpus intentionally covers only:

- `FIXED_CELL`
- `MOBILE_FLEET`

A changing construction site is deliberately deferred. Static workcell evidence must not be generalized to a changing site.

### FIXED_CELL

Represents a bounded stationary industrial robot application/workcell context.

Additional source facts include:

- collision-scene currentness;
- fixture/machine configuration currentness;
- machine command acceptance;
- independently observed machine-cycle state.

### MOBILE_FLEET

Represents autonomous logistics/mobile platforms interacting with facility infrastructure.

Additional source facts include:

- localization currentness;
- environment/map currentness;
- coordination-provider state;
- local-fallback receipt state;
- required infrastructure resource state;
- infrastructure command vs observed state;
- handoff/article/custody state;
- common-mode dependencies;
- reassignment candidate evidence.

## External references are informative

Freeze-time external alignment:

- ISO 10218-2:2025 — informative reference for industrial robot applications/cells;
- ISO 3691-4:2023 — current published informative reference for driverless industrial trucks/AMR-like systems; it is under revision;
- Open-RMF — external interoperability/provider reference for heterogeneous robot fleets and physical infrastructure.

None of these references creates standards compliance, safety approval, or authority.

The corpus carries an exact integration-reference profile and whether a caller is using it merely informatively or as part of an admission claim.

Applicability is **derived**, not source-authored:

```text
FIXED_CELL + ISO10218_2_2025
-> reference applicable to this profile's informative/application context

MOBILE_FLEET + ISO3691_4_2023
-> reference applicable to this profile's informative/application context
```

A caller cannot set `applicable=true`.

## Receipt boundary

Robot, tool, and application-commissioning results are consumed as external receipts.

For a required receipt, this profile requires:

```text
receipt bound
+ result = Pass
+ current
+ exact scope match
```

Canonical consumed result vocabulary:

```text
Pass | Fail | Blocked | EnvironmentFailure
```

ROB-CELL does not mint any of these qualification results.

In particular:

```text
ROB-TOOL qualifier exists
!= hosted ROB-TOOL PASS receipt exists
```

## Common fail-closed distinctions

```text
facility configuration named
!= facility configuration current

frame name exists
!= transform/reference network current

external authority reference exists historically
!= current authority reference

commissioning receipt exists
!= current scoped application commissioning PASS

MODEL task evidence
!= FIELD-required task evidence

one observed success
!= profile-required independent evidence multiplicity
```

## Fixed-cell semantics

For a fixed cell:

```text
robot receipt Pass
+ tool receipt Pass
+ current facility/reference network
+ current collision scene
+ current fixture/machine config
+ current scoped application commissioning
!= manufacturing process qualification
```

Machine integration preserves:

```text
machine command Accepted
!= machine cycle observed complete

machine command Rejected/Failed
-> MachineCommandBlocked
```

## Mobile fleet semantics

For mobile/fleet applications:

```text
route assigned
!= route physically valid now

map loaded
!= environment still matches map

localization solution exists
!= localization current

door/lift/gate API Accepted
!= resource physically ready

coordination provider Available
!= robot authorized
```

## Infrastructure transaction

The semantic stages are:

```text
resource required
-> resource availability
-> command external system
-> external acceptance
-> observed physical/system state
-> traversal/use
-> release
```

This v1 source only freezes the evidence boundary through independently observed resource state.

```text
infrastructure command Rejected/Failed
-> InfrastructureCommandBlocked
```

`InfrastructureUseSemanticallyComplete` is therefore a semantic/evidence disposition, not proof of safety or physical task completion.

## Handoff / custody

A handoff requires independent identity and custody state:

```text
carrier arrives
+ exact article identity
+ exact receiver identity
+ transfer accepted
+ custody/load-state observation
-> HandoffSemanticallyComplete
```

Proximity alone is insufficient.

## Common-mode dependencies

Nominal redundancy is not independent redundancy.

```text
robot A available
+ robot B available
+ shared localization/power dependency
!= independent redundancy
```

If a redundancy claim is requested and the common-mode relation is `SharedDependency` or `Unknown`, the claim is blocked.

## Coordination outage and local fallback

External coordination availability and local protective fallback are distinct.

```text
coordination unavailable
+ no current local fallback receipt
-> CoordinationUnavailableBlocked

coordination unavailable
+ current local fallback receipt
-> CoordinationFallbackOnly
```

`CoordinationFallbackOnly` does not authorize continued goal-directed task execution.

## Reassignment

Reassignment is allowed only as a composition result over independently current candidate references.

The frozen v1 candidate requires:

- alternate robot qualification = `CurrentPass`;
- alternate tool qualification = `CurrentPass`;
- alternate configuration = `Current`;
- alternate external authority reference = `Current`.

Then:

```text
-> ReassignmentAdmissible
```

This means only that the alternate is admissible to the composition profile. It does not dispatch or authorize it.

## Human/shared-space state

This profile may consume observed restricted-zone/shared-space state.

```text
human zone Clear
!= industrial safety compliance

human zone Occupied/Unknown when Clear is required
-> HumanZoneBlocked
```

No robot speed, collaborative label, or safety standard reference can bypass the external safety owner.

## Task evidence and execution

Task evidence preserves:

- required plane: `MODEL | FIELD`;
- actual plane: `MODEL | FIELD`;
- currentness;
- exact scope match;
- profile-required independent trial count;
- observed independent trial count.

No universal trial count is embedded as sufficient truth.

Also:

```text
planner path
!= measured execution

scheduler assignment
!= measured execution

execution claim
+ no observed successful execution
-> ExecutionEvidenceBoundaryBlocked
```

## Commissioning and production boundary

Consume BUILT-CX-style application commissioning receipts rather than recreating a commissioning ladder.

```text
robot commissioned
!= application commissioned

application commissioned
!= production line qualified

machine tending task succeeds
!= manufacturing process capability
```

Any request to promote this composition state into manufacturing-process qualification is blocked.

## Frozen result vocabulary

The corpus uses non-ranked dispositions:

- `ApplicationCompositionAdmissible`
- `RobotQualificationBlocked`
- `ToolQualificationBlocked`
- `FacilityConfigurationBlocked`
- `SpatialReferenceBlocked`
- `CollisionSceneBlocked`
- `FixtureStateBlocked`
- `LocalizationBlocked`
- `EnvironmentMapBlocked`
- `ResourceAvailabilityBlocked`
- `InfrastructureCommandBlocked`
- `InfrastructureObservationBlocked`
- `MachineCommandBlocked`
- `MachineObservationBlocked`
- `HandoffIdentityBlocked`
- `CustodyEvidenceBlocked`
- `CommonModeBlocked`
- `ConfigurationCurrentnessBlocked`
- `HumanZoneBlocked`
- `CoordinationUnavailableBlocked`
- `CoordinationFallbackOnly`
- `AuthorityReferenceBlocked`
- `CommissioningBoundaryBlocked`
- `TaskEvidenceBlocked`
- `ProcessQualificationBoundaryBlocked`
- `AuthorityBoundaryBlocked`
- `ExecutionEvidenceBoundaryBlocked`
- `HistoryIntegrityBlocked`
- `ApplicabilityBoundaryBlocked`
- `InfrastructureUseSemanticallyComplete`
- `HandoffSemanticallyComplete`
- `ReassignmentBlocked`
- `ReassignmentAdmissible`

They are not a maturity ladder and are not ordered.

## Fail-closed precedence

A future independent oracle should derive results with explicit precedence equivalent to:

```text
history integrity
-> external reference applicability when relied on for admission
-> safety/physical authority promotion
-> manufacturing-process qualification promotion
-> configuration currentness
-> facility configuration
-> spatial/reference network
-> robot qualification receipt
-> tool qualification receipt
-> human-zone state
-> external authority reference
-> application commissioning receipt
-> fixed-cell or mobile-fleet specific blockers
-> task evidence
-> execution evidence
-> positive composition disposition
```

Focused case kinds such as infrastructure, handoff, and reassignment terminate in their own bounded semantic dispositions.

## Synthetic corpus

Schema:

`rob-cell-001a-application-composition-reference-v1`

Canonical compact sorted-key JSON + final newline SHA-256:

`cac94e64bacafd221d6fedf6c535be2842877d84edf02674a9ae32557cf24d4c`

Exact cases: **51**

Exact dispositions: **33**, every disposition exercised by at least one known-answer case.

Coverage includes:

- robot/tool receipt missing/fail/stale/scope mismatch;
- stale facility/reference/collision/fixture configuration;
- stale localization and environment maps;
- missing resources;
- accepted infrastructure command with contradictory/unknown observed state;
- exact handoff identity/custody;
- common-mode redundancy;
- coordination outage with and without current local fallback;
- stale/missing external authority refs;
- application commissioning missing/fail/scope mismatch;
- MODEL/FIELD evidence separation;
- evidence multiplicity;
- machine command vs observed cycle;
- planner/scheduler vs execution;
- machine/infrastructure command rejection and independent state observation;
- process-qualification and authority promotion attacks;
- fixed-cell reference misapplied to a mobile admission claim;
- mobile logistics profile with no tool requirement, proving tool qualification is conditional rather than universal;
- reassignment with stale/missing candidate evidence;
- positive fixed-cell, mobile-fleet, infrastructure, handoff, fallback, and reassignment controls.

## Initial implementation gate

Do not add production workcell/fleet adapters until a source-bound independent qualifier establishes faithful representation of this frozen synthetic contract.

After qualification, keep the first adapter tranche read-only/provider-facing:

1. exact fixed-cell projection;
2. exact mobile fleet/infrastructure projection;
3. no dispatch;
4. no motor lowering;
5. no resource command execution from this profile.

## Relationship to ROB-TOOL

ROB-CELL consumes ROB-TOOL qualification receipts.

It must not reimplement tool identity, interface, calibration, TCP, process receipt, or automatic-exchange semantics.

```text
tool profile admissible
!= workcell/application admissible
```

## Relationship to BUILT-CX

ROB-CELL consumes application commissioning evidence from BUILT-CX.

It must not infer commissioning from robot/tool qualification or planner/fleet state.

## Relationship to microfactory pilot

BUILT-ROB-PILOT-001 #6040 should consume hosted qualification receipts from ROB-CELL and its upstream sources.

A source PR or queued qualifier identity is not a PASS receipt.

## Qualification plan

After source review, open a separate qualifier that:

1. hard-binds this exact source head and both source blobs;
2. requires exactly one qualifier commit / two files;
3. imports no Symthaea production code;
4. verifies canonical JSON bytes, defaults, vocabularies, cases, and nonclaims;
5. rejects unknown overrides and enums before expanding defaults;
6. independently derives all 51 outcomes;
7. derives external-reference applicability from application kind + reference profile;
8. hostile-tests receipt, currentness, infrastructure observation, handoff, common-mode, commissioning, planner/execution, reassignment, process-promotion, and authority-promotion boundaries;
9. requires clean postflight.

A qualifier PASS would establish only faithful representation of this frozen synthetic composition contract.

## Claim ceiling

This source establishes no:

- industrial robot safety compliance;
- mobile robot safety compliance;
- collaborative-operation approval;
- construction-site safety;
- fleet deployment readiness;
- manufacturing-process capability;
- facility safety;
- human replacement claim;
- procurement/resource allocation;
- physical actuation or task-execution authority.
