# ION-001A — Reusable ion-transport evidence architecture

Status: architecture-only design subject

Date: 2026-09-26

Related issues:

- ION-001 #4996
- SSE-001 #4993
- DLE-001 #4995
- MAT-MLIP-001 #4992
- MAT-CALC-001 #5114
- MAT-RD-001 #4997

## Purpose

Define one reusable ion-transport evidence plane for solid-state electrolytes, selective-ion membranes, and later ionic materials without allowing site topology, migration barriers, molecular-dynamics trajectories, derived conductivity, experimental impedance measurements, or selectivity to collapse into one generic `ion_transport` scalar.

This document creates no production Rust type, no solver execution, and no scientific result. It freezes ownership, identity, capability, calibration, and adversarial requirements only.

## Core theorem

```text
mobile-ion network identified
!= migration path proposed
!= NEB barrier calculated
!= finite-temperature diffusion observed
!= ionic conductivity derived
!= experimental conductivity measured
!= competitive-ion selectivity established
!= useful electrochemical device
```

Every stronger proposition must preserve the exact subject state and evidence ancestry that made it possible.

## Ownership boundary

ION-001 should own domain-specific transport semantics only.

Reuse qualified/common owners for:

- material/phase subject identity;
- conditions and units;
- evaluator identity and applicability;
- exact external execution/convergence;
- scientific evidence origin/provenance;
- negative-result/search memory;
- sample/process lineage;
- synthesis and physical-action authority.

ION-001 must not create a second provenance system, second process runner, second generic uncertainty ontology, or battery-specific device model.

Recommended composition:

```text
MAT material subject / state
+ MAT condition profile
+ MAT evaluator / calibration / applicability
+ MAT-CALC execution evidence where computational
+ ION transport-specific evidence
-> bounded ion-transport proposition
```

## Transport subject state

A transport claim must bind the exact state under which mobile species move.

Conceptually:

```text
IonTransportSubjectStateV1 {
  material_subject_ref,
  phase_or_structure_state_ref,
  mobile_species,
  charge_state_convention,
  defect_state,
  vacancy_interstitial_state,
  dopant_state,
  occupational_disorder_state,
  composition_state,
  pressure_state,
  field_state,
  temperature_state,
  strain_state,
  interface_or_bulk_scope,
  generation,
}
```

A friendly material name is insufficient identity.

Examples that require distinct transport subjects include:

- same nominal composition with different vacancy fraction;
- same crystal family with different dopant occupancy;
- ordered versus disordered/high-entropy configuration;
- same bulk phase at different temperature regimes when state changes matter;
- bulk versus grain-boundary/interface transport;
- same membrane composition at a different feed/contact state.

## Evidence classes

### 1. `IonSiteNetworkEvidence`

Represents a proposed or observed network of mobile-ion sites/path connectivity.

Bind where applicable:

- exact subject state;
- site identities and coordinate frame;
- crystallographic/structural source artifact;
- site-finding implementation/profile;
- occupancy assumptions;
- connectivity rule/profile;
- topology result;
- uncertainty/ambiguity;
- evaluator/origin.

This evidence may support path hypotheses. It does not establish a kinetic barrier or finite-temperature transport.

### 2. `MigrationPathEvidence`

Represents one exact migration path proposal.

Bind:

- initial/final site identities;
- subject state;
- path construction method;
- image count/initialization profile;
- endpoint artifact identities;
- supercell/cell state;
- charge/defect state;
- evaluator/request identity.

A path proposal is not a converged barrier.

### 3. `MigrationBarrierEvidence`

Represents a barrier derived from a qualified NEB-like execution.

Require where applicable:

- exact migration path;
- exact solver/runtime/input/potential identities;
- MAT-CALC execution/convergence witness;
- image-energy profile artifact;
- climbing-image/NEB method profile;
- force/convergence criteria;
- forward/reverse direction where asymmetric;
- barrier definition and units;
- finite-size/supercell state.

```text
one low barrier
!= percolating network
!= bulk diffusion coefficient
!= room-temperature conductivity
```

### 4. `TrajectoryTransportEvidence`

Represents a finite-temperature trajectory capable of supporting diffusion analysis.

Bind at minimum:

- exact subject state;
- trajectory artifact identity;
- force-field/DFT evaluator identity;
- model/weights/runtime identity where MLIP is used;
- applicability/calibration profile;
- ensemble;
- thermostat/barostat profiles;
- time step;
- equilibration policy;
- production duration;
- cell/supercell dimensions;
- mobile-ion count;
- total atom count;
- temperature/pressure/field;
- trajectory completeness/convergence state;
- random/initial condition identities where stochastic;
- replicate identity.

A trajectory receipt alone does not establish diffusion; analysis is separate.

### 5. `DiffusionEvidence`

Represents diffusion inferred from one or more exact trajectories.

Bind:

- trajectory evidence refs;
- exact analysis implementation/profile;
- displacement definition;
- unwrap/PBC handling;
- dimensionality/tensor convention;
- time-window selection;
- ballistic/transient exclusion policy;
- long-time linear-regime evidence;
- diffusion tensor/coefficient;
- uncertainty method;
- block/replicate statistics;
- anomalous-diffusion diagnostics where applicable.

Do not force normal diffusion when MSD does not support a valid long-time linear regime.

Suggested dispositions include:

```text
NormalDiffusionSupportedUnderProfile
SubdiffusiveOrNonlinear
InsufficientTimeScale
InsufficientCarrierEvents
TrajectoryNonstationary
AnalysisInapplicable
```

### 6. `IonicConductivityEvidence`

Keep conductivity derivation separate from diffusion.

Bind:

- diffusion evidence;
- charge carrier density;
- species charge convention;
- Nernst-Einstein or alternative conversion profile;
- Haven/correlation treatment if used;
- tensor/scalar averaging convention;
- temperature;
- units;
- uncertainty propagation;
- assumptions/limitations.

```text
self diffusion
!= collective conductivity
```

A Nernst-Einstein conversion must remain visibly assumption-bearing when correlated motion is not resolved.

### 7. `ExperimentalConductivityObservation`

Experimental impedance/EIS conductivity remains a physically observed evidence class, not a subtype of simulated conductivity.

Bind where applicable:

- exact physical sample/process lineage;
- specimen geometry/dimensions;
- electrode/contact configuration;
- temperature/environment;
- frequency range;
- excitation amplitude;
- instrument/calibration refs;
- raw impedance artifact;
- equivalent-circuit/fitting profile when used;
- geometry-factor conversion;
- bulk/grain-boundary assignment where claimed;
- reported conductivity and uncertainty.

```text
same numeric conductivity
from MD
!= same evidence
as measured EIS conductivity
```

## Bulk, interface, and grain-boundary scope

Do not expose a generic conductivity without transport scope.

At minimum support scope classes such as:

```text
BulkCrystal
BulkDisordered
GrainBoundary
ElectrodeElectrolyteInterface
MembraneSolutionInterface
CompositeEffective
OtherExplicit
```

A bulk high-conductivity electrolyte can still fail because interface transport dominates the device.

## Disorder and high-entropy materials

A single ordered representative structure may not characterize a disordered/high-entropy transport subject.

Bind where relevant:

- occupational configuration identity;
- configurational ensemble/profile;
- sampling method;
- number of configurations;
- weighting assumptions;
- configuration-to-configuration variance.

Preserve:

```text
one favorable disorder realization
!= ensemble transport property
```

and:

```text
average structure
!= average transport
```

## MLIP capability and calibration boundary

Current SSE literature reinforces an important separation:

```text
low force RMSE
!= reliable ion-transport dynamics
```

A model may be useful for transport only when an exact applicability/calibration artifact supports the relevant chemistry, structure/state, temperature, and dynamical task.

Transport qualification should therefore distinguish at least:

```text
StaticEnergyQualified
ForceQualified
RelaxationQualified
ShortTrajectoryQualified
TransportDynamicsQualified
```

Do not infer the latter from the former.

Recommended transport-calibration evidence includes where feasible:

- force/energy errors on transport-relevant configurations;
- barrier/path comparisons against higher-fidelity calculations;
- short AIMD/DFT trajectory comparison;
- diffusion-event/state coverage;
- stability against extrapolative local environments;
- temperature-range coverage;
- uncertainty/error-detection behavior;
- chemistry/structure/OOD slices;
- trajectory failure/pathology census.

MAT-MLIP owns the generic model/calibration artifact. ION consumes it and decides whether that calibration profile is sufficient for a transport capability.

## Temperature dependence

Transport properties are temperature-conditioned trajectories, not one timeless scalar.

When fitting Arrhenius-like behavior, bind:

- exact temperature points and their evidence identities;
- fit equation/profile;
- selected regime/window;
- weighting/error model;
- activation energy;
- prefactor;
- goodness/residual evidence;
- regime-change diagnostics;
- extrapolation range.

```text
high-temperature MD fit
!= room-temperature conductivity observation
```

unless extrapolation is explicitly represented and bounded.

If non-Arrhenius behavior is supported, do not force an Arrhenius fit merely for convenience.

## Replication and finite-size/time discipline

Report independently where possible:

- trajectory duration;
- supercell size;
- mobile-ion count;
- number of independent initializations/replicates;
- observed hops/transitions;
- block variance;
- replicate variance;
- finite-size checks;
- time-window sensitivity;
- fit sensitivity.

Preserve:

```text
long trajectory
!= independent replicate
```

and:

```text
large supercell
!= adequate trajectory time
```

## Multi-species transport and selectivity

The shared kernel should be able to represent multiple mobile species without pretending that this automatically establishes membrane selectivity.

For each species preserve separate:

- diffusion;
- mobility/conductivity contribution;
- concentration/state;
- transport tensor;
- uncertainty.

DLE/selective-membrane layers may later compose:

```text
species-resolved transport
+ exact feed/interface/process state
+ competitive-ion evidence
-> bounded selectivity proposition
```

ION alone must reject:

```text
fast Li transport
-> high Li/Mg selectivity
```

## Capability vocabulary

A provider/evaluator should declare narrowly scoped capabilities such as:

```text
IonSiteNetworkAnalysis
MigrationPathConstruction
NEBBarrier
AbInitioMolecularDynamics
MLIPMolecularDynamics
EnhancedSampling
TrajectoryDiffusionAnalysis
ConductivityFromDiffusion
ExperimentalEISNormalization
```

Capabilities are not interchangeable.

Examples:

```text
NEBBarrier
-X-> IonicConductivity

MLIPMolecularDynamics
-X-> ExperimentalConductivity

ExperimentalEISNormalization
-X-> MigrationBarrier
```

## MAT-CALC boundary

MAT-CALC may prove that a solver executed and converged under an exact profile.

It does not prove the ion-transport proposition by itself.

```text
MAT-CALC execution/convergence
+ ION solver-specific capability/profile
+ ION scientific interpretation
-> bounded computational transport evidence
```

For example, a converged NEB execution still needs path identity and barrier semantics before it can become `MigrationBarrierEvidence`.

## Negative results and refusal states

Preserve transport failures as evidence-bearing outcomes, including:

- no valid percolating site network under profile;
- NEB nonconvergence;
- path reconstruction failure;
- spontaneous structural transformation;
- MLIP applicability refusal;
- trajectory instability;
- no diffusion events observed;
- insufficient simulation duration;
- anomalous/nonlinear MSD;
- no stable fit regime;
- EIS fit ambiguity;
- electrode/grain-boundary contributions unresolved;
- incompatible condition comparison.

Do not replace failed candidates until the campaign protocol explicitly permits a new lineage.

## Required adversarial fixtures

The first synthetic/reference corpus should include at least:

1. same material + changed vacancy fraction -> distinct transport subject;
2. same nominal composition + changed disorder configuration -> distinct subject/evidence;
3. low NEB barrier on one isolated path -> no conductivity promotion;
4. NEB converged but endpoint identity changed -> reject;
5. MD completed but model OOD under exact state -> no ordinary transport promotion;
6. low force RMSE but failed transport calibration -> transport capability absent;
7. identical MSD slope + changed carrier density -> different derived conductivity;
8. self-diffusion result recast as collective conductivity without correlation profile -> reject;
9. high-temperature extrapolation reported as measured room-temperature conductivity -> reject;
10. insufficient long-time linear MSD regime -> diffusion unresolved;
11. short trajectory with zero hops -> no `D=0` truth claim, only censored/insufficient evidence;
12. same trajectory + changed analysis window -> distinct diffusion result identity;
13. same numeric D from incompatible temperature states -> not interchangeable;
14. one disorder realization promoted to high-entropy ensemble property -> reject;
15. average structure used as though it were ensemble-average transport -> reject;
16. one long trajectory presented as independent replication -> reject;
17. bulk conductivity promoted to electrode-interface compatibility -> reject;
18. simulated conductivity numerically equals EIS conductivity -> evidence classes remain distinct;
19. EIS fit changes equivalent-circuit profile -> new normalized observation identity;
20. geometry factor changes after measurement -> derived conductivity identity changes;
21. Li conductivity promoted to Li/Mg selectivity -> reject;
22. two species simulated independently promoted to competitive-ion membrane behavior -> reject;
23. MAT-CALC convergence alone promoted to ion-transport truth -> reject;
24. failed/nonconverged candidate omitted from prospective denominator -> reject campaign scoring;
25. transport evidence promoted to battery safety/cycle-life/manufacturability -> reject authority promotion.

## External design motivation

External work is architectural motivation only and does not become local Symthaea evidence.

- 2026 SSE MLFF perspective emphasizing data/reference quality, transport-specific physical reliability, and the insufficiency of force RMSE alone:
  https://www.nature.com/articles/s44456-026-00014-4
- 2026 high-entropy solid-electrolyte screening using ML + DFT + MD and reporting strong material-family dependence:
  https://www.nature.com/articles/s41524-026-02116-8
- curated experimental lithium-ion-conductivity database with explicit temperature conditions:
  https://www.nature.com/articles/s41524-022-00951-z

## First adopter sequence

Recommended order:

```text
ION-001A
  architecture + scientific non-equivalences

ION-001B
  frozen synthetic hostile corpus

ION-001C
  independent known-answer semantic validator

ION-002A
  production subject/evidence contracts over qualified MAT owners

ION-NEB-001A
  one exact NEB solver profile

ION-MD-001A
  one exact trajectory/analysis profile

SSE-DATA-001
  halide SSE benchmark custody

SSE-BENCH-001
  retrospective transport/stability benchmark

SSE-PROSPECT-001
  prospective deferred evaluation

DLE
  only after shared ion-transport semantics are proven reusable
```

## Production implementation gate

Architecture/corpus/reference work may proceed independently.

Do not open production ION contracts that claim integration authority until usable qualified/current owners exist for the exact common semantics they consume, including the intended MAT subject, conditions, evaluator/calibration, execution, and evidence layers.

Do not use SSE or DLE as an excuse to duplicate those owners locally.

## Claim ceiling

A qualified ION implementation may establish exact ion-transport subject identity and bounded site/path/barrier/trajectory/diffusion/conductivity evidence under declared computational or physical conditions. It does not establish battery safety, cycle life, interface compatibility unless separately evidenced, selective extraction, synthesis, manufacturability, commercial value, or physical experiment authority.