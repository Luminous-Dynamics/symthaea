# BUILT-BIM-001A — openBIM projection/source contract

Status: source/data freeze for review only  
Parent: BUILT-BIM-001 #6036 / BUILT-ENV-001 #6032  
Frozen base: `main@eae17187e199e3a53d108b437c0215b5ff812261`

## Purpose

Freeze the first machine-checkable **external-model projection contract** for built-environment interoperability without creating a second BIM ontology, requirements database, observation store, configuration engine, issue tracker, commissioning engine, or authority system.

This tranche is deliberately read-only and source/data only.

```text
external model / information requirement / issue artifact
!= canonical engineering truth
!= current as-built state
!= commissioning
!= physical authority
```

## External alignment

The source profile records these external interoperability families as alignment references:

- buildingSMART IFC 4.3.2.0 / ISO 16739-1:2024 — open building/infrastructure model exchange;
- buildingSMART IDS 1.0 — machine-interpretable information delivery requirements/checking;
- BCF/openCDE-style issue/common-data-environment exchange — workflow projection only;
- bSDD — external classification/property dictionary references.

These remain projection sources. Their identities, validation results, workflow states, and data values do not become stronger Symthaea evidence merely by import.

## Ownership boundary

BUILT-BIM-001A owns only:

- supported external source-profile registry for this adapter contract;
- exact external artifact identity binding;
- exact projection-adapter/profile identity binding;
- preservation of external object identity;
- canonical-owner mapping state;
- projection-loss reporting;
- information-delivery result projection;
- issue/workflow-state projection;
- source-value authority preservation;
- synthetic adversarial fixtures proving non-laundering.

It does **not** own:

- canonical physical quantities/units/frames;
- canonical engineering configurations;
- requirements acceptance/satisfaction;
- physical observations/metrology;
- as-built truth;
- commissioning;
- structural/geotechnical adequacy;
- project work/execution provenance;
- evidence currentness authority;
- physical execution authority.

Those stay with SE-SEM / SE-VV / SE-OBS / FIELD / ETK / Mycelix / domain owners.

## Frozen theorem

```text
IFC object exists
!= canonical Symthaea engineering fact
!= physical article/as-built state
!= qualification

IDS check PASS
!= engineering verification PASS
!= physical conformance

BCF issue Closed
!= physical discrepancy resolved
!= retest PASS

source-declared property
!= measured physical observation
```

## Raw source facts vs derived projection disposition

A critical review repair in this source freeze is that the corpus does **not** contain a generic `profile_supported=true` oracle input.

Instead it freezes raw/source-facing facts such as:

```text
kind
source_profile
artifact_valid
artifact_identity_bound
adapter_identity_bound
external_identity_preserved
canonical_mapping
units_frames
currentness
unsupported_semantics
loss_reported
IDS result
workflow state
value authority
evidence-promotion request
physical-authority request
```

A future independent oracle must derive support from the top-level `supported_profiles` registry and then derive the case disposition.

This prevents the qualifier from proving a result merely because the source already asserted `supported=true`.

## Identity layers

Preserve these independently:

```text
external artifact identity
!= projection adapter identity
!= external object identity
!= canonical owner mapping
```

The source artifact must be exact and immutable enough to identify the imported subject.

The adapter must have its own exact profile/version identity.

External object identity must survive import independently of friendly/display labels.

Mapping an external object onto a canonical Symthaea owner is a separate operation that may be:

```text
Resolved
Unresolved
Ambiguous
```

A preserved IFC GUID/object identity therefore does not imply that the correct canonical engineering subject has been resolved.

## Source-profile support

The corpus freezes a bounded supported-profile registry:

```text
IFC -> IFC4.3.2.0
IDS -> IDS1.0
BCF -> BCFWorkflowProjectionV1
```

These are adapter-contract identifiers, not claims that every construct in the external standards is implemented.

An unsupported source profile must fail closed as `SchemaProfileBlocked`.

An unknown extension may remain representable only through explicit unsupported/loss semantics. It cannot silently become supported.

## Projection result vocabulary

The frozen corpus uses non-ranked dispositions:

- `ProjectionAdmissible`
- `ProjectionAdmissibleWithLoss`
- `ArtifactValidityBlocked`
- `ArtifactIdentityBlocked`
- `AdapterIdentityBlocked`
- `SchemaProfileBlocked`
- `ExternalIdentityBlocked`
- `IdentityMappingBlocked`
- `UnitFrameBlocked`
- `CurrentnessBlocked`
- `ProjectionLossBlocked`
- `InformationRequirementSatisfied`
- `InformationRequirementSatisfiedPhysicalConflict`
- `InformationRequirementBlocked`
- `WorkflowStateOnly`
- `WorkflowStatePhysicalResolutionReferenced`
- `HistoryIntegrityBlocked`
- `EvidenceAuthorityBlocked`
- `AuthorityBoundaryBlocked`

These are source-contract dispositions, not maturity levels or certification states.

## Fail-closed precedence

A future independent validator should derive each known-answer result using precedence equivalent to:

```text
physical-authority request
-> evidence-authority promotion request
-> history integrity
-> external artifact validity
-> exact artifact identity
-> exact adapter identity
-> supported source profile
-> external-object identity preservation
-> canonical-owner mapping
-> units/frames
-> currentness
-> hidden projection loss
-> kind-specific IFC / IDS / BCF result
```

No later kind-specific result may bypass an earlier blocker.

## IFC projection

For an admitted IFC projection:

```text
valid artifact
+ exact artifact identity
+ exact adapter identity
+ supported source profile
+ external IDs preserved
+ canonical mapping resolved
+ units/frames resolved
+ current applicability
+ no hidden loss
-> ProjectionAdmissible
```

If unsupported semantics are retained and explicitly disclosed by the loss report:

```text
-> ProjectionAdmissibleWithLoss
```

Unreported semantic loss is blocked.

Duplicate display labels are not identity conflicts if exact external IDs remain distinct.

## IDS information requirements

IDS remains an information-delivery check.

```text
IDS Pass
-> InformationRequirementSatisfied
```

If separately owned physical evidence contradicts the model:

```text
IDS Pass
+ physical conflict
-> InformationRequirementSatisfiedPhysicalConflict
```

The information-delivery result remains historically true while the contradictory physical evidence remains independently visible.

```text
information field present
!= field physically true
```

## BCF / issue workflow

Issue closure remains workflow state.

```text
issue Closed
+ no physical-resolution evidence ref
-> WorkflowStateOnly
```

An exact external physical-resolution/retest reference may be retained:

```text
-> WorkflowStatePhysicalResolutionReferenced
```

but this projection does not establish that referenced physical proposition.

Historical issue/change evidence must not be deleted merely because a later repair or close event occurs.

## Source-declared values

External BIM/catalog/property values may be useful without being measured evidence.

The corpus therefore distinguishes:

```text
SourceDeclared
!= PhysicalObservation
```

A source-declared property may remain `ProjectionAdmissible` as descriptive external data.

Attempting to promote that same source-declared value into physical measurement authority is `EvidenceAuthorityBlocked`.

This distinction is separate from requesting physical execution authority, which is `AuthorityBoundaryBlocked`.

## Synthetic corpus

Schema:

`built-bim-001a-openbim-projection-reference-v1`

Canonical compact sorted-key JSON + final newline SHA-256:

`fcb2a5b33c284b8cb6f591180807d56bbcb231d3beb75edd0b76dae5aff99c01`

Exact cases: **25**

Disposition census:

- `ProjectionAdmissible`: 3
- `SchemaProfileBlocked`: 3
- `UnitFrameBlocked`: 2
- `IdentityMappingBlocked`: 2
- every remaining disposition: 1

The exact cases cover:

1. supported minimal IFC projection;
2. invalid external artifact;
3. missing external artifact identity;
4. missing projection-adapter identity;
5. unsupported IFC version;
6. unsupported IFC view/profile;
7. unknown property retained with explicit loss;
8. unit mismatch;
9. coordinate/frame ambiguity;
10. duplicate human labels with distinct external IDs;
11. lost/collapsed external object identity;
12. unresolved canonical-owner mapping;
13. ambiguous canonical-owner mapping;
14. stale model revision vs as-built evidence;
15. IDS information requirement PASS;
16. IDS PASS plus contradictory physical observation;
17. missing IDS-required field;
18. unsupported IDS profile/semantics;
19. BCF close without retest evidence;
20. BCF close with external physical-resolution reference;
21. deleted workflow history;
22. hidden projection loss claimed as lossless;
23. manufacturer/catalog property retained as source-declared;
24. source-declared property improperly promoted to measured evidence;
25. attempted physical-authority promotion.

## Read-only implementation gate

Do not add write/export/round-trip mutation until a read-only adapter can prove:

- exact source artifact identity;
- exact standard/profile/version identity;
- exact adapter/profile identity;
- deterministic projection identity;
- external object-ID preservation;
- separate canonical-owner mapping;
- explicit unsupported semantics;
- deterministic loss reporting;
- units/frame preservation or explicit unresolved state;
- currentness/applicability references;
- source-value authority preservation;
- zero evidence or execution-authority escalation.

If a mature parser/validator is used, bind its exact version/tool identity.

```text
parser success
!= engineering truth
```

## Relationship to BUILT-CX

BUILT-CX #6037 may consume:

- exact model/projection refs;
- IDS information-delivery results;
- issue/workflow refs.

It must still obtain physical installation, test, observation and commissioning evidence from their canonical owners.

```text
model complete
!= facility commissioned
```

## Relationship to ENG-DESIGN / SE-SEM

This source subject does not depend on unqualified ENG-DESIGN child code.

It is designed to later compose with:

- ENG-DESIGN typed design-thread refs;
- SE-SEM external source/projection provenance;
- SE-SEM quantity/interface/configuration semantics;
- SE-OBS/FIELD physical observations;
- ETK/currentness/qualification results.

No field in this corpus becomes a competing canonical implementation of those owners.

## Qualification plan

After source review, open a separate qualifier that:

1. hard-binds this exact repaired source head and both source blobs;
2. requires exactly one qualifier commit;
3. uses a stdlib-only independent oracle;
4. imports no Symthaea production code;
5. verifies canonical JSON bytes and exact case set;
6. derives source-profile support from `supported_profiles` rather than trusting a case boolean;
7. independently derives all 25 results;
8. hostile-tests source/profile drift, identity collapse, mapping ambiguity, unit/frame mismatch, hidden loss, IDS/BCF laundering, evidence promotion and physical-authority promotion;
9. requires clean postflight.

A qualifier PASS would establish only faithful representation of this frozen synthetic projection contract.

## Claim ceiling

This source contract establishes no:

- architectural correctness;
- physical as-built conformity;
- structural/geotechnical adequacy;
- code compliance;
- permit/occupancy approval;
- commissioning completion;
- physical defect resolution;
- external certification;
- procurement/resource allocation;
- physical construction or execution authority.
