# BUILT-CX-001A — commissioning evidence/source contract

Status: source/data freeze for review only  
Parent: BUILT-CX-001 #6037 / BUILT-ENV-001 #6032  
Frozen base: `main@eae17187e199e3a53d108b437c0215b5ff812261`

## Purpose

Freeze the first machine-checkable commissioning evidence contract for buildings, utilities, equipment, robot cells, and integrated facilities without creating a second V&V engine, work-order system, digital twin, building-control system, process-qualification engine, or execution-authority layer.

```text
installed
!= commissioned

command accepted
!= physical response observed

component commissioned
!= subsystem integrated
!= facility commissioned
!= production process qualified
```

## Ownership boundary

BUILT-CX-001A owns only the commissioning composition profile:

- exact scope level;
- physical vs virtual commissioning evidence kind;
- installation/configuration state;
- calibration/currentness prerequisites;
- lower-level evidence dependencies;
- command-vs-observation distinction;
- functional-test result;
- integration result;
- mode coverage;
- common-mode dependency state;
- issue/retest lineage;
- configuration-change/recommissioning obligation;
- synthetic adversarial fixtures proving anti-laundering.

It does not own canonical:

- requirement satisfaction;
- physical observations;
- calibration;
- safety approval;
- work-order truth;
- process capability;
- product conformity;
- building/facility control;
- physical execution authority.

Those remain with SE-VV, SE-OBS/FIELD, ETK, MFG-PROC, domain owners, Mycelix, and existing authority layers.

## Raw facts, not `commissioned=true`

The corpus deliberately contains raw commissioning facts such as:

```text
evidence_kind
scope_level
delivered
installed
configuration_match
calibration_state
prerequisites_resolved
lower_level_evidence_current
command_accepted
physical_response_observed
functional_result
integration_result
mode_coverage_complete
common_mode_resolved
configuration_current
issue_closed
retest_evidence
history_retained
virtual_result
workflow_complete
model_information_complete
process_qualification_requested
physical_authority_requested
```

A future qualifier must derive disposition from those fields.

There is no source-authored `commissioned=true` oracle input.

## Scope hierarchy

The frozen scope vocabulary is:

```text
Component
Subsystem
System
IntegratedFacility
ProductionLine
```

A lower-level result may be referenced by a higher-level campaign, but it cannot promote itself.

```text
component PASS
!= subsystem PASS

all components PASS
!= integration PASS

integrated line commissioned
!= manufacturing process qualified
```

## Evidence kinds

Keep physical and virtual commissioning separate:

```text
Virtual
Physical
```

Virtual commissioning may support sequence/interface rehearsal and expected-response generation.

```text
virtual NotExecuted -> VirtualEvidencePending
virtual Fail        -> VirtualEvidenceFailed
virtual Pass        -> VirtualEvidenceOnly

virtual PASS
!= physical PASS
```

## Fail-closed precedence

A future independent oracle should derive results with precedence equivalent to:

```text
physical-authority request
-> manufacturing/process qualification promotion request
-> history integrity
-> configuration currentness / recommissioning
-> issue closure without retest
-> virtual evidence result (pending / failed / pass-only)
-> installation
-> configuration match
-> calibration
-> prerequisites / lower-level currentness
-> command accepted but no observed response
-> functional-test result
-> declared-mode coverage
-> common-mode dependency
-> component completion
-> integration result
```

No later commissioning state may bypass an earlier blocker.

## Installation and configuration

Preserve:

```text
delivered
!= installed

installed
!= correct configuration

correct configuration
!= calibration current
```

A commissioning result belongs to one exact configuration generation.

Equipment replacement, control-software revision, service topology change, fixture revision, or other material configuration change may make historical commissioning evidence non-current without deleting it.

## Command vs observation

A controller/transport/workflow receipt is not a physical observation.

```text
command accepted
!= action occurred
!= measured response
!= functional criterion satisfied
```

If a command is accepted and required physical response is not observed, the contract derives `PhysicalResponseBlocked`.

## Functional testing

A physical component cannot become commissioned if its required functional test has not executed.

```text
NotExecuted -> FunctionalTestPending
Fail        -> FunctionalTestFailed
Pass        -> continue through coverage/dependency checks
```

For non-component scopes, functional PASS is still insufficient without integration evidence.

## Integration

Higher-level commissioning may produce:

```text
IntegrationPending
IntegrationFailed
IntegratedCommissioned
```

Only exact current lower-level evidence may compose upward.

An integration defect can exist while all components individually pass.

## Mode coverage

Passing one operating mode does not establish declared modes not tested.

Incomplete declared-mode coverage derives `CoverageBlocked`.

## Common modes

Nominally redundant equipment may still share:

- upstream power;
- network/time source;
- cooling;
- utility feed;
- controller;
- localization/sensing;
- physical access or other shared dependency.

Unresolved common modes derive `CommonModeDependencyBlocked`.

## Workflows and BIM information

Mycelix work-order completion and IFC/IDS information completeness are useful external facts but do not establish functional commissioning.

```text
work order complete
!= functional acceptance

IFC/IDS complete
!= commissioning complete
```

The corpus includes explicit cases where those workflow/model states are true while commissioning remains pending.

## Issue closure and retest

```text
issue closed
+ no retest evidence
-> IssueClosureUnsubstantiated
```

Closing an issue is workflow state.

A repaired/changed subject requires exact retest/recommissioning evidence when the affected configuration changes.

Historical negative evidence remains queryable.

Deleting negative history derives `HistoryIntegrityBlocked`.

## Factory boundary

Keep separate:

```text
facility commissioning
utility commissioning
equipment commissioning
robot-cell commissioning
line commissioning
manufacturing process qualification
```

A production line may be `IntegratedCommissioned` while no manufacturing process capability has been established.

Attempting to mint process qualification directly from commissioning derives `ProcessQualificationBoundaryBlocked`.

## Synthetic corpus

Schema:

`built-cx-001a-commissioning-reference-v1`

Canonical compact sorted-key JSON + final newline SHA-256:

`f6ab3a0e66be5072ed479dc554728ce15a9471f1ead2ca10c221fdcedb441d71`

Exact cases: **32**

The cases cover:

1. delivered but not installed;
2. wrong installed configuration;
3. missing calibration;
4. no functional test;
5. accepted command without observed physical response;
6. test evidence bound to stale configuration;
7. component functional commissioning;
8. subsystem integration failure;
9. cross-subsystem timing/interface integration failure;
10. unresolved redundancy common mode;
11. virtual commissioning PASS only;
12. incomplete operating-mode coverage;
13. issue closed without retest;
14. equipment replacement;
15. control-software revision;
16. building services commissioned while process remains separate;
17. machines pass but material-flow integration fails;
18. robot commissioned but cell application prerequisites unresolved;
19. nominal utility capacity while quality/continuity evidence unresolved;
20. work order complete but functional commissioning absent;
21. IFC/IDS information complete but commissioning absent;
22. repair creates new generation;
23. negative history deleted after repair;
24. complete synthetic building route;
25. complete synthetic factory-line commissioning route;
26. commissioning improperly promoted to process qualification;
27. commissioning improperly requests physical authority;
28. functional test explicitly fails;
29. higher-level functional test passes but integration is not executed;
30. virtual commissioning explicitly fails;
31. virtual commissioning is not executed;
32. issue closure has a retest reference but the retest itself is still not executed.

## Frozen result vocabulary

Non-ranked dispositions:

- `InstallationBlocked`
- `ConfigurationBlocked`
- `CalibrationBlocked`
- `PrerequisiteBlocked`
- `PhysicalResponseBlocked`
- `FunctionalTestPending`
- `FunctionalTestFailed`
- `CoverageBlocked`
- `CommonModeDependencyBlocked`
- `ComponentCommissioned`
- `IntegrationPending`
- `IntegrationFailed`
- `IntegratedCommissioned`
- `VirtualEvidencePending`
- `VirtualEvidenceFailed`
- `VirtualEvidenceOnly`
- `RecommissioningRequired`
- `IssueClosureUnsubstantiated`
- `HistoryIntegrityBlocked`
- `ProcessQualificationBoundaryBlocked`
- `AuthorityBoundaryBlocked`

These are commissioning-profile dispositions only. They are not safety, certification, production-readiness, or authority states.

## Relationship to BUILT-BIM

BUILT-BIM may supply:

- external model/projection identity;
- information-delivery results;
- issue/workflow references.

Those remain external model/workflow evidence.

```text
model says installed
!= physical installation observation

IDS PASS
!= functional test PASS

BCF Closed
!= recommissioning PASS
```

## Relationship to ROB-CELL

ROB-CELL may consume commissioning refs for exact robot/tool/workcell applications.

```text
robot commissioned
!= robot application commissioned
```

Application/workcell commissioning must bind exact current robot, tool, fixture, facility, infrastructure, and task profiles.

## First implementation gate

Do not add a production commissioning adapter until an independent qualifier reproduces the exact synthetic outcomes from raw fields.

The first qualifier should:

1. hard-bind this source head and both source blobs;
2. import no Symthaea production code;
3. validate canonical bytes/schema/vocabularies;
4. derive all 32 outcomes independently;
5. hostile-test command-vs-observation, virtual-vs-physical, configuration drift, common modes, issue closure, process promotion, and authority promotion;
6. require clean postflight.

## Claim ceiling

This source contract establishes no:

- code compliance;
- permit/occupancy approval;
- fire/life-safety approval;
- equipment certification;
- production-process capability;
- product conformity;
- facility safety;
- procurement/resource allocation;
- physical execution authority.
