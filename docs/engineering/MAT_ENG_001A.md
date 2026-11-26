# MAT-ENG-001A — Engineering Demand ↔ Materials Return Contract

Issue: #6186  
Parent: #6185  
Reference corpus: `docs/engineering/data/mat_eng_001a_reference_v1.json`  
Canonical corpus SHA-256 (exact compact UTF-8 bytes): `f252c72ebf89be0115738708423e9565de7be80b9680f2bf4128a59a8e35c134`

## Status

This is a source/data contract only. It defines deterministic synthetic semantics for translating an engineering material demand into a bounded research target and for projecting materials evidence back into engineering.

It does **not** establish any real material property, synthesizability, manufacturability, engineering adequacy, safety, lifetime, product qualification, certification, procurement/build authority, or deployment authority.

## Governing distinctions

```text
engineering requirement
!= justified materials-discovery demand
!= research target
!= candidate
!= scientific property evidence
!= process-conditioned engineering property evidence
!= article qualification
```

and:

```text
property improvement
!= system bottleneck removal
!= system improvement
```

## Contract responsibilities

A claim-bearing bridge binds, directly or by canonical reference:

- exact engineering subject + generation;
- exact requirement/objective/risk/discrepancy origin;
- exact operating/environment/lifecycle profile;
- hard vs target/desirable/exploratory constraint classes;
- exact material property/function demands;
- process/form/geometry restrictions;
- bottleneck/system-leverage evidence;
- competing non-material interventions;
- exact research target projection;
- materials candidate/evidence maturity;
- synthesis/process/form state;
- ENG-MAT projection inputs and reduction assumptions;
- uncertainty / robust-envelope semantics;
- currentness / applicability dependencies;
- exact claim ceiling.

Friendly labels cannot carry a claim by themselves.

## Required flow

```text
engineering problem
  -> bottleneck attribution
  -> engineering material demand
  -> research target
  -> candidate/search/evaluation
  -> synthesis/process/form evidence
  -> conditioned property evidence
  -> ENG-MAT projection
  -> engineering model/prototype
  -> physical discrepancy
  -> revised evidence/demand
```

Every arrow is explicit. No stage may silently mint authority belonging to a later stage.

## Demand semantics

The source recognizes that a correct outcome can be:

```text
material variable not bottlenecking
existing material sufficient
evidence insufficient
process innovation preferable
interface innovation preferable
architecture alternative competitive
discovery demand justified under exact profile
```

Therefore there is no universal `needs_new_material` flag.

## Engineering projection boundary

Before scientific materials evidence enters claim-bearing engineering analysis, the projection must preserve:

- source scientific subject/evidence identity;
- exact material/process state;
- condition/orientation/form applicability;
- property or constitutive-model reduction;
- uncertainty transformation;
- omitted/unsupported properties;
- transfer assumptions;
- currentness dependencies;
- evidence/source class;
- claim ceiling.

Examples:

```text
0 K DFT elastic tensor
!= room-temperature process-conditioned modulus

perfect-crystal conductivity
!= porous article conductivity

intrinsic magnet property
!= coercivity
!= motor performance

bulk ionic conductivity
!= interface conductivity
!= cell performance
```

## Process and scale boundary

Keep distinct:

```text
route proposed
!= process executed
!= target phase/form established
!= property measured
!= process-conditioned engineering evidence
!= article qualification
```

and:

```text
atomistic cell
!= simulated microstructure
!= specimen
!= coupon
!= subcomponent
!= article
!= qualified production population
```

## Reference corpus

The canonical corpus contains **32 ordered synthetic cases** spanning:

- conditioned engineering demands;
- bottleneck justification;
- existing-material sufficiency;
- non-material alternatives;
- requirement-generation changes;
- advisory/surrogate authority attacks;
- computational-to-engineering applicability failures;
- process/form and synthesis-state boundaries;
- matching and mismatching coupon projections;
- uncertainty-tail robustness;
- lot/process currentness;
- lifecycle/degradation;
- bottleneck migration;
- diagnostic ambiguity;
- Pareto incomparability;
- negative-result feedback;
- physical contradiction retention;
- synthetic-to-product authority attacks.

Positive controls include:

- a fully represented conditioned hard demand;
- an existing-material-sufficient case;
- a matching coupon/process/orientation engineering projection;
- negative synthesis evidence correctly entering search memory.

## Independent outputs

There is no universal `material_ok`, `discovery_success`, or `engineering_ready` flag.

The frozen cases instead derive independent dimensions such as:

```text
DemandRepresentationDisposition
BottleneckJustificationDisposition
ResearchTargetDisposition
ScientificEvidenceDisposition
ProcessFormDisposition
EngineeringProjectionDisposition
RobustnessDisposition
CurrentnessDisposition
FeedbackDisposition
AuthorityDisposition
```

## Qualification sequence

```text
#6186 source/data freeze
-> independent stdlib-only oracle
-> hosted exact-head qualification
-> typed MAT/ENG adapters
-> closed-loop TIM benchmark
-> later physical coupon bridge
```

The independent oracle must derive outcomes from raw inputs and compare with `expected` only after derivation.

## Claim ceiling

A future exact-head PASS may establish only faithful deterministic software semantics for the frozen synthetic engineering-demand/materials-return corpus.

It establishes no real scientific/material truth, synthesis success, manufacturing capability, engineering suitability, safety, durability, qualification, certification, procurement/build authority, or deployment authority.
