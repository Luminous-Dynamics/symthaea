# Extensible Manufacturing Process Architecture

Status: architecture contract

Parent: #5686 (`MFG-PROC-000`)

Cross-repository operations bridge: `Luminous-Dynamics/mycelix#3085`

## Purpose

Symthaea already has strong fabrication geometry, slicing/toolpath, machine-session authority, artifact provenance, digital-twin and engineering evidence machinery. Mycelix manufacturing already has BOMs, work orders, routings, machines, scheduling and MRP.

The missing layer is a neutral representation of **what manufacturing process is intended, what transformation it performs, what capability is required, which exact recipe is admitted, and what evidence supports the claim that a resource can perform it**.

This document defines that layer without expanding the fabrication kernel into a universal MES/CAM controller and without turning Mycelix into a process-physics engine.

## Core distinction

```text
process label
!= process definition
!= process family/taxonomy
!= capability requirement
!= resource capability
!= recipe
!= process plan
!= prepared machine program
!= executed operation
!= observed result
!= accepted output
!= production-qualified capability
```

A manufacturing system becomes extensible by preserving these distinctions, not by adding a `Custom(String)` escape hatch to an otherwise closed enum.

## Ownership

### Symthaea engineering

Owns:

- canonical process semantics;
- process identity and extension schema;
- transformation/effect semantics;
- material/geometry/state applicability;
- capability requirements;
- process recipes;
- process models and prediction evidence;
- manufacturability reasoning;
- process-chain optimization;
- qualification/discrepancy semantics;
- standards/taxonomy mappings.

### Symthaea fabrication kernel

Owns:

- machine-program validation;
- machine/session capability negotiation;
- execution guards;
- operator/machine authority;
- realized-artifact provenance;
- machine telemetry and execution evidence for supported adapters.

The existing FDM/slicing/G-code path remains a concrete realization backend, not the universal manufacturing ontology.

### Mycelix manufacturing

Owns:

- distributed process/resource registry;
- resource/provider offers;
- work orders;
- process routings/plans at operations level;
- scheduling and MRP;
- supplier/site/machine availability;
- lot/batch/unit lineage;
- cross-organization production evidence.

### Other canonical owners

- SE-SEM-001: quantities, units, frames and dimensional semantics;
- ENG-CATALOG: equipment/tool/component identity and exact source documents;
- ENG-MAT/materials: material/property evidence and material-state identities;
- FIELD: physical observations;
- QIF adapters: inspection/quality exchange;
- HAL/fabrication authority layers: physical actuation.

## Design principle: stable kernel, extensible process profiles

Do not encode every manufacturing technology as a central Rust enum.

Use a small stable semantic kernel with namespaced and versioned process profiles.

Conceptual stable types:

```text
ProcessDefinitionId
ProcessFamilyRef
ProcessDefinition
TransformationEffect
ProcessCapabilityRequirement
ProcessCapabilityProfile
ProcessRecipeId
ProcessRecipe
ProcessPlanId
ProcessStep
ProcessPlan
ProcessModelProfile
ProcessExecutionSubjectRef
ProcessResultRef
ProcessQualificationProfile
```

Process-family references are opaque, canonical, resolvable identities such as:

```text
iso-astm-52900:material-extrusion
iso-astm-52900:powder-bed-fusion
luminous:cnc-milling/v1
luminous:investment-casting/v1
luminous:ald-thin-film/v1
org.example:novel-process/v2
```

Unknown extensions may be stored and transported. Strong engineering reasoning requires the referenced profile/schema to resolve and validate.

```text
extension ID present
!= extension schema resolved
!= extension trusted
!= resource qualified
```

## Process families and transformation effects are orthogonal

A named process may cause several effects. Therefore process-family identity and transformation semantics should not be the same enum.

Initial transformation/effect classes:

- MaterialAddition
- MaterialRemoval
- PlasticDeformation
- SolidificationOrMolding
- JoiningOrBonding
- SeparationOrDisassembly
- HeatTreatment
- DensificationOrSintering
- ChemicalTransformation
- ElectrochemicalTransformation
- SurfaceModification
- ThinFilmDeposition
- MicrostructureModification
- SemiconductorDopingOrActivation
- PatternTransfer
- CleaningOrPreparation
- Assembly
- InspectionOrMeasurement
- NonDestructiveTest
- ElectricalTest
- LeakOrVacuumTest
- ConditioningOrCure
- Transport
- Storage
- Packaging
- CustomExtension

A process may declare multiple effects, ordered when order matters.

Example:

```text
laser powder-bed fusion
= MaterialAddition
+ SolidificationOrMolding
+ HeatTreatment-like local thermal history
+ MicrostructureModification
```

The process profile describes the actual admitted semantics; these effect tags support reasoning and discovery rather than replacing physics.

## Process execution modes

The core must work for more than discrete CNC/3D-printing workflows.

Support at least:

- Discrete
- Batch
- Continuous
- Assembly
- DisassemblyRepairRework
- InspectionMeasurement
- MaterialHandlingStorage

Mode affects planning/execution identity but is not itself a process family.

## Initial extension packs

The bootstrap registry should cover broad industrial families while remaining open-ended.

### Additive manufacturing

Map the additive family taxonomy to ISO/ASTM 52900 where possible.

Profiles should cover:

- material extrusion;
- vat photopolymerization;
- powder-bed fusion;
- directed-energy deposition;
- binder jetting;
- material jetting;
- sheet lamination.

Existing fabrication-kernel FDM behavior becomes an adapter to the material-extrusion profile.

### Subtractive / cutting

Profiles should cover:

- 3-axis milling;
- 5-axis milling;
- turning;
- drilling;
- boring;
- tapping/threading;
- broaching;
- sawing;
- grinding;
- honing;
- lapping;
- polishing;
- EDM;
- waterjet;
- laser cutting.

### Forming / deformation

- forging;
- rolling;
- extrusion;
- wire/tube drawing;
- stamping;
- bending;
- spinning;
- hydroforming;
- stretch forming.

### Casting / molding

- sand casting;
- investment casting;
- die casting;
- permanent-mold casting;
- centrifugal casting;
- injection molding;
- compression molding;
- transfer molding;
- blow molding;
- rotational molding.

### Joining

- arc welding families;
- resistance welding;
- laser welding;
- electron-beam welding;
- friction / friction-stir joining;
- ultrasonic joining;
- diffusion/solid-state bonding;
- brazing;
- soldering;
- adhesive bonding;
- mechanical fastening.

### Thermal / densification / cure

- annealing;
- normalizing;
- tempering;
- solution/age treatments;
- quench profiles;
- sintering;
- hot-isostatic pressing;
- vacuum bakeout;
- polymer/composite cure and post-cure;
- ceramics firing.

### Surface engineering / finishing

- deburr;
- polishing;
- blasting;
- shot peening;
- electropolishing;
- anodizing;
- passivation;
- electroplating;
- electroless plating;
- thermal spray;
- PVD;
- CVD;
- ALD;
- painting;
- conformal coating.

### Electronics manufacturing

- PCB patterning/etch/plating;
- pick-and-place;
- solder paste print;
- reflow;
- wave/selective solder;
- THT assembly;
- cable/harness fabrication;
- potting/encapsulation;
- conformal coating;
- ICT/flying-probe/electrical test.

### Semiconductor / microfabrication

- crystal growth;
- wafer slicing/lapping/polishing;
- oxidation;
- diffusion;
- ion implantation;
- photolithography;
- e-beam lithography;
- wet etch;
- dry/plasma etch;
- PVD/CVD/ALD deposition;
- epitaxy;
- CMP;
- wafer bonding;
- dicing;
- die attach;
- wire/flip-chip bonding;
- packaging;
- wafer/die electrical test.

### Photonics / precision optics

- optical glass forming;
- precision grinding;
- conventional polishing;
- magnetorheological finishing;
- single-point diamond turning;
- optical coating;
- fiber preparation/splicing;
- precision alignment;
- optical bonding;
- interferometric/optical qualification.

### Composites / advanced materials

- hand layup;
- automated fiber placement;
- filament winding;
- resin transfer / infusion;
- pultrusion;
- prepreg/autoclave processing;
- oven cure;
- powder metallurgy;
- ceramic forming/sintering.

### Inspection / NDT / lifecycle

- dimensional inspection;
- CMM inspection;
- optical metrology;
- surface metrology;
- electrical test;
- leak/vacuum test;
- radiographic/CT inspection;
- ultrasonic NDT;
- dye penetrant;
- magnetic-particle inspection;
- cleaning;
- assembly;
- disassembly;
- repair/rework;
- packaging.

The bootstrap registry is not a claim that Symthaea can already simulate, plan, qualify or execute every listed process.

## ProcessDefinition

A process definition identifies a process independently of any machine, supplier or recipe.

Conceptually:

```text
ProcessDefinition {
    id,
    semantic_version,
    family_refs,
    operation_modes,
    transformation_effects,
    accepted_input_state_schema_refs,
    produced_output_state_schema_refs,
    geometry_feature_applicability_refs,
    required_capability_schema,
    recipe_parameter_schema_ref,
    tooling_role_refs,
    fixture_role_refs,
    consumable_role_refs,
    environment_requirement_refs,
    quality_characteristic_refs,
    required_inspection_profile_refs,
    hazard_profile_refs,
    model_profile_refs,
    provenance_refs,
}
```

V1 should keep these primarily as identity/schema references until SE-SEM and material-state semantics are qualified rather than inventing another quantity system.

## Capability requirements and capability profiles

Separate the process requirement from the offered/observed resource capability.

```text
ProcessCapabilityRequirement
!=
ProcessCapabilityProfile
```

A capability profile binds:

- exact resource/equipment identity;
- process definition/family;
- applicable process envelope/profile;
- supported materials/material states;
- geometry/feature envelope;
- tooling/fixture requirements;
- environment/site state;
- evidence class and exact source evidence;
- qualification profile;
- validity/revision/freshness.

Suggested evidence dispositions:

```text
Declared
ManufacturerSpecified
ObservedCapability
QualifiedUnderProfile
ProductionQualifiedUnderProfile
UnavailableOrUnknown
```

These are not necessarily linearly ordered. Qualification remains profile-relative.

```text
QualifiedUnderProfile(A)
!= universally qualified
!= qualified under profile B
```

## ProcessRecipe

A process definition describes the semantic process family. A recipe specifies exact admitted parameters for a particular application.

```text
ProcessDefinition
+ material/input-state profile
+ design/feature applicability
+ exact parameters
+ tooling/fixture configuration
+ environment profile
+ model/qualification refs
-> ProcessRecipe
```

Recipes must be immutable/versioned evidence subjects.

Changing a semantic recipe parameter creates a new recipe identity.

Changing only a display label does not.

Recipe identity does not grant execution authority.

## ProcessPlan

A manufacturing routing should become an evidence-bearing graph rather than only an ordered vector of operation strings.

Required node/edge semantics include:

- sequential transformation;
- parallel branch;
- assembly/join;
- split/batch subdivision;
- merge/batch aggregation;
- external/subcontract step;
- inspection/hold point;
- accepted/rejected disposition;
- bounded rework path;
- repair path;
- transport/storage step.

Intermediate article/material/batch state identities should be explicit.

### Rework

Do not represent arbitrary cycles as valid production flow.

A rework route must bind:

- triggering nonconformance/inspection disposition;
- maximum admitted attempts or another bounded termination policy;
- process/recipe refs;
- inspection required before re-entry;
- output disposition.

## Process models

Symthaea may attach models predicting process consequences such as:

- dimensional error/tolerance capability;
- surface finish;
- residual stress/distortion;
- thermal history;
- microstructure/property evolution;
- defect probability;
- tool wear;
- cycle/setup time;
- yield/scrap;
- energy/resource consumption;
- emissions/waste;
- cost.

Each model retains its own evidence/claim ceiling.

```text
process model predicts capability
!= machine demonstrated capability
!= process qualified on physical articles
```

The current `ManufacturingTwin`/HDC predictor may consume process observations as an advisory model, but its normalized state score must not become process qualification authority.

## Fabrication-kernel adapter

The existing FDM path should be the first concrete adapter.

Conceptually:

```text
ProcessDefinition(material-extrusion)
+ ProcessRecipe
+ exact design/geometry
+ exact MachineCapabilityProfile
        ↓
FDM adapter
        ↓
existing ProcessPreparedMesh
existing slicer/toolpath
existing MachineProfile / session negotiation
existing release/operator/execution authority
```

This preserves the current hardened machine-execution architecture while moving the process ontology out of FDM-specific structures.

Later adapters may support CNC, laser cutter, robot cell or other machines without weakening the execution boundary.

## Mycelix bridge

Mycelix should reference canonical process/profile/recipe identities instead of copying engineering semantics.

Routing evolution:

```text
legacy Operation {
  name,
  machine_type: String,
  tooling: Option<String>,
  ...
}
```

becomes migration-compatible with typed refs such as:

```text
process_definition_ref
recipe_ref
capability_requirement_ref
input_state_refs
output_state_refs
inspection_profile_refs
resource_constraints
```

Old string fields may remain as compatibility/navigation metadata during migration.

Unresolved legacy strings cannot satisfy stronger typed claims.

## Resource discovery and scheduling

A scheduler should match requirements to offers rather than compare machine-type labels.

```text
ProcessCapabilityRequirement
+ material/geometry/site constraints
        ↓
resource-offer resolution
        ↓
engineering-feasible candidate set
        ↓
operations/commercial scheduling
```

Commercial criteria such as price, lead time and availability remain distinct from engineering feasibility.

## Standards interoperability

Standards should map to the internal model rather than define internal identity.

### ISO/ASTM 52900

Use as an additive-manufacturing taxonomy mapping where applicable.

### ISA-95 / IEC 62264

Use as a reference for the enterprise/control/manufacturing-operations boundary and terminology. Do not force Symthaea engineering semantics into an ERP hierarchy.

### ISO 23247

Use its manufacturing digital-twin concepts for interoperability/alignment. Symthaea's evidence/digital-twin architecture remains the internal authority model.

### MTConnect

Map manufacturing-equipment observations/capability data into resource observations. MTConnect data does not by itself establish qualification.

### OPC UA companion specifications

Use technology-neutral machine/resource interfaces where available. OPC UA connectivity does not grant execution authority.

### QIF / ISO 23952

Use for inspection/measurement-plan/results interoperability. FIELD remains physical-observation authority.

## Quality and inspection

Preserve:

```text
inspection requirement
!= inspection plan
!= inspection executed
!= measurement result
!= disposition accepted
```

A process plan may require a QIF/FIELD-backed inspection node before downstream steps are admitted.

## Extensible schema registry

A process extension should carry at minimum:

```text
namespace
profile_id
semantic_version
schema_digest
schema_locator/audit_ref
publisher/authority identity
supersedes refs
compatibility declaration
```

Validation should distinguish:

```text
ResolvedExact
Unresolved
UnsupportedVersion
SchemaDigestMismatch
WrongPublisherOrNamespace
DeprecatedButReadable
Incompatible
```

Do not dynamically execute extension-provided code merely because a schema resolves.

Extensions are data/schema first. Solver/controller plugins require separate qualified adapter boundaries.

## Hazard and authority semantics

Manufacturing process representation must be safe to make broad; execution authority must remain narrow.

```text
CanRepresent
!= CanModel
!= CanPlan
!= CanGenerateProgram
!= CanRelease
!= CanExecute
```

High-energy laser processes, plasma processes, high-voltage equipment, pressure systems, hazardous chemistry, high-temperature processes and other hazardous operations remain behind explicit safety/operator/HAL/fabrication authority.

An optimizer may propose a process chain. It cannot energize equipment merely because the chain scores well.

## Evidence ladder

A useful generic manufacturing evidence ladder is:

```text
ProcessDefined
CapabilityDeclared
CapabilityCharacterized
RecipeModeled
PlanPrepared
MachineProgramValidated
ExecutionAuthorized
ExecutionRecorded
InspectionObserved
OutputDispositioned
QualifiedUnderProfile
```

No later label should be inferred solely from the presence of an earlier one.

## Manufacturing capability optimization

Once the semantic layer is stable, Symthaea can optimize not only part geometry but the production route itself.

Potential objective dimensions:

- process feasibility;
- tolerance/quality;
- yield/scrap risk;
- setup/cycle time;
- capital/tooling availability;
- energy/water/material consumption;
- emissions/waste;
- maintenance/tool wear;
- supply-chain resilience;
- repairability/recyclability;
- cost;
- qualification burden.

Use Pareto fronts rather than one universal `manufacturing_score`.

## Initial implementation sequence

1. **MFG-PROC-001** — process ID + extension-profile registry contract.
2. **MFG-PROC-002** — transformation/effect and material/article state refs.
3. **MFG-PROC-003** — capability requirement/profile + evidence classes.
4. **MFG-PROC-004** — immutable recipe identity.
5. **MFG-PROC-005** — process-plan graph + hold/rework semantics.
6. **MFG-PROC-006** — existing FDM fabrication-kernel adapter.
7. **MFG-PROC-007** — Mycelix typed routing/work-order bridge.
8. **MFG-PROC-008** — QIF/FIELD inspection profile.
9. **MFG-PROC-009** — MTConnect/OPC UA resource-observation adapters.
10. **MFG-PROC-010** — process model/discrepancy/digital-twin integration.
11. **MFG-PROC-011** — capability matching and process-chain optimization.
12. **MFG-PROC-012** — standardized/community extension packs.

## Exit criterion

The architecture is successful when:

- a new manufacturing technology can be represented without editing a giant central enum;
- process semantics remain independent from vendor/machine protocol;
- a resource can advertise a capability without that advertisement becoming qualification;
- Mycelix can schedule/reference exact process/profile/recipe identities;
- fabrication execution remains behind existing authority controls;
- process outcomes can be compared with physical inspection evidence;
- process-chain optimization can propose alternatives without rewriting evidence or directly controlling hardware.

## Nonclaims

This architecture does not establish any process as physically qualified, does not grant hazardous-process execution authority, does not define complete physics for every manufacturing technology, and does not make external standards or provider declarations authoritative merely because they are mapped into the system.
