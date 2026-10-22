# IND-COMP-001A — Functional-component qualification contract

Issue: #5989  
Parent: IND-COMP-000 #5965  
Frozen corpus: `docs/release/evidence/ind-comp-001a-functional-component-reference-v1.json`  
Canonical SHA-256: `11cd7e2eb13c22493341edfd29ea4e37905506637612880ffe21e7c8fb9ae799`

## Purpose

Freeze the first reference boundary for industrial components whose function cannot be inferred from nominal geometry or material identity alone.

The contract preserves:

`component design/fabrication != dimensional conformance != functional observation != requested duty-envelope support != wear/lifetime evidence != qualified substitute != installed-system qualification != productive closure`.

This tranche is documentation/data only.

## Ownership

IND-COMP composes existing owners:

- ENG-DEVICE #5670 owns reusable subsystem architecture.
- ENG-TOL #4883 owns tolerance/fit/metrology semantics.
- MFG-PROC #5686 owns generic manufacturing process/capability.
- ROB-REALIZE #4859 owns physical/as-built generations.
- FIELD/SE-OBS owns physical observations.
- MFG-LIFE #5705 owns repair/remanufacture/lifecycle projections.
- CIV-BOOT #5774/#5782 owns productive and multi-generation closure.
- Mycelix owns actual inventory, work, custody, maintenance and lifecycle events.

IND-COMP adds only function-specific component qualification profiles.

## Initial families

The frozen V1 corpus exercises semantic fixtures for:

- bearings;
- gears/transmission elements;
- seals/gaskets;
- shafts/couplings;
- springs/fasteners;
- low-energy pumps;
- valves;
- compressors/blowers as synthetic profiles only.

No fixture contains real operating ratings or hazardous process instructions.

## Evidence planes

Keep independently representable:

1. exact component/article/configuration identity;
2. material/surface/process state;
3. dimensional/fit evidence;
4. installation/assembly context;
5. functional observation;
6. exact duty/applicability profile;
7. canonical quantity/measurement references;
8. wear/lifetime/maintenance evidence;
9. repair/remanufacture generation;
10. metrology/calibration currentness;
11. imported dependencies;
12. productive closure.

## Governing examples

- bearing fit != acceptable runout/friction/lifetime;
- gear geometry != qualified surface/process/load-life behavior;
- seal material name != installed leakage/lifetime evidence;
- pump rotates != requested duty envelope;
- valve actuates != leakage/tightness/flow-control qualification;
- compressor runs != delivered flow/pressure/quality envelope.

## Reference dispositions

The V1 known-answer outcomes are:

- `DimensionalConformanceFunctionUnresolved`
- `SurfaceOrProcessStateUnresolved`
- `LeakageUnresolved`
- `DutyEnvelopeUnresolved`
- `ProfileBoundedSubstitute`
- `RepairObservedDurabilityUnresolved`
- `PartialProductiveClosure`
- `MetrologyBlocked`
- `NewConfigurationSubjectRequired`
- `SemanticIdentityUnchanged`
- `InvalidDuplicateReference`
- `HistoryPreserved`
- `PredictionOnlyNoPhysicalObservation`
- `NoComponentOrSystemExecutionAuthority`

These are reference-corpus outcomes, not another generic component or manufacturing state engine.

## Adversarial corpus

The exact 16 cases cover:

- bearing fit without runout/lifetime;
- gear geometry without surface/process evidence;
- seal material without leakage evidence;
- pump operation without duty-envelope evidence;
- valve actuation without tightness/flow evidence;
- compressor/blower operation without delivered-envelope evidence;
- a lower-demand-only local substitute;
- one repaired functional observation without durability;
- imported critical rolling/seal material;
- shared metrology/reference loss;
- changed installed article/configuration;
- friendly-label-only change;
- duplicate evidence refs;
- append-only negative history;
- prediction without fresh physical observation;
- zero operation/execution authority.

## Qualification path

Next child:

1. independent stdlib-only validator over these exact bytes;
2. dedicated hosted workflow;
3. only after source qualification, audit whether code should be a thin profile/view over existing owners or a small domain crate.

Passing the reference validator will not establish a real component.

## Claim ceiling

This tranche establishes no real bearing/gear/seal/pump/valve/compressor rating, pressure/load/speed/flow/efficiency capability, wear/lifetime, leakage safety, certification, economics, productive closure, installed-system qualification, or physical operation authority.
