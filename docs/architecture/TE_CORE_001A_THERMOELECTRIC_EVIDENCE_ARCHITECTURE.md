# TE-CORE-001A — Thermoelectric transport evidence architecture

Status: architecture-only design subject

Date: 2026-09-26

Related issues:

- TE-001 #4994
- MAT-MLIP-001 #4992
- MAT-CALC-001 #5114
- MAT-RD-001 #4997
- MAT-HYPOTHESIS-001 #5254

## Purpose

Define the scientific evidence boundary for thermoelectric materials discovery so Symthaea can represent temperature- and carrier-conditioned electronic transport, phonon/lattice thermal transport, stability, and derived thermoelectric figures of merit without collapsing them into one context-free `zT` or model score.

This document creates no production Rust type, no solver execution, no candidate ranking, and no scientific result. It freezes architecture, ownership, compatibility rules, and adversarial requirements only.

## Core theorem

```text
candidate generated
!= phase stable
!= dynamically stable
!= useful electronic transport
!= low lattice thermal conductivity
!= compatible component transport state
!= high derived zT
!= experimentally optimized doping
!= efficient thermoelectric device
```

The central rule is:

```text
zT is derived evidence
not a primitive property
```

## Canonical thermoelectric relation

For an explicitly compatible state/profile:

```text
zT = S^2 * sigma * T / (kappa_e + kappa_l)
```

but this equation is admissible only when the referenced components are compatible in material state, temperature, carrier/doping state, transport direction/tensor convention, and required model assumptions.

A numerically valid arithmetic substitution is not enough.

## Ownership boundary

TE-CORE owns thermoelectric-specific state and evidence composition.

Reuse qualified/common owners for:

- material/phase subject identity;
- generic conditions and units;
- evaluator identity/applicability/calibration;
- exact external execution/convergence;
- scientific provenance;
- experimental sample/process lineage;
- negative-result/search memory;
- synthesis/manufacturing authority.

Do not create a thermoelectric-only process runner, uncertainty ontology, provenance graph, or candidate-authority system.

## Thermoelectric subject state

A transport observation must bind the exact scientific state.

Conceptually:

```text
ThermoelectricSubjectStateV1 {
  material_subject_ref,
  phase_structure_ref,
  composition_state,
  dopant_or_alloy_state,
  defect_state,
  carrier_type,
  carrier_concentration_or_chemical_potential,
  temperature,
  pressure,
  strain_state,
  transport_direction_or_tensor_scope,
  microstructure_scope,
  bulk_or_device_scope,
  generation,
}
```

Distinct examples include:

- same parent compound at two carrier concentrations;
- same chemistry with different dopant sites/fractions;
- ordered versus disordered alloy state;
- same material at 300 K versus 700 K;
- in-plane versus cross-plane transport;
- ideal crystal versus measured polycrystal;
- pristine bulk versus thermoelectric leg with contact/interface losses.

## Primitive/independently evidenced transport quantities

At minimum keep these independently addressable:

```text
SeebeckCoefficient S(T,n,...)
ElectricalConductivity sigma(T,n,...)
ElectronicThermalConductivity kappa_e(T,n,...)
LatticeThermalConductivity kappa_l(T,...)
TotalThermalConductivity kappa_total(T,...)
CarrierConcentration n or p
ChemicalPotential
PowerFactor S^2*sigma
```

Additional evidence may include:

- Hall mobility;
- carrier mobility;
- effective masses;
- band gap;
- density of states;
- valley degeneracy;
- deformation potentials;
- phonon group velocities;
- phonon lifetimes;
- heat capacity;
- scattering-rate components;
- anisotropy tensors.

These may support interpretation but must retain their own method and condition identity.

## Derived result classes

### `PowerFactorEvidence`

May be derived only from compatible Seebeck and electrical-conductivity evidence.

```text
best S from state A
+ best sigma from state B
-X-> power factor for either state
```

### `ZTDerivedEvidence`

Requires exact compatible references to:

- `S`;
- `sigma`;
- `kappa_e`;
- `kappa_l` or exact compatible `kappa_total` decomposition/profile;
- temperature;
- carrier/doping state;
- transport direction/tensor convention;
- derivation implementation/profile;
- propagated uncertainty/profile.

The derivation should fail closed on mixed states.

## Electronic-transport evidence

Separate the following evidence levels:

```text
ElectronicStructureEvidence
BandTransportKernelEvidence
ScatteringModelEvidence
CarrierStateEvidence
ElectricalTransportEvidence
```

### Electronic structure

Bind where applicable:

- exact relaxed structure/subject;
- DFT/evaluator profile;
- exchange-correlation functional;
- pseudopotential/basis artifacts;
- k mesh;
- spin/SOC treatment;
- band/DOS artifact identities;
- band-gap result and method limitations.

### Boltzmann transport

Bind:

- electronic-structure evidence;
- interpolation/profile;
- carrier concentration or chemical potential grid;
- temperature grid;
- transport direction/tensor convention;
- scattering/relaxation-time model;
- exact solver/profile;
- convergence/grid evidence.

Preserve the distinction:

```text
sigma / tau
!= sigma
```

and likewise for any relaxation-time-normalized transport quantity.

A constant-relaxation-time result cannot silently become an absolute conductivity without a separately justified relaxation time.

### Scattering models

Keep explicit model families such as:

- constant relaxation time;
- deformation-potential acoustic scattering;
- polar optical phonon scattering;
- ionized-impurity scattering;
- alloy/disorder scattering;
- experimentally fitted relaxation profile;
- other explicit qualified profile.

Changing scattering physics changes the evidence identity.

## Phonon and lattice-thermal evidence

Separate:

```text
harmonic dynamic stability
!= anharmonic force constants
!= phonon lifetime
!= lattice thermal conductivity
```

### Harmonic phonons

Bind:

- exact material state;
- force-constant/DFPT method;
- supercell/q mesh;
- non-analytic correction profile where relevant;
- imaginary-mode handling;
- dynamic-stability interpretation scope.

### Anharmonic transport

Bind where used:

- second-order and third-order or higher IFC artifacts;
- displacement/supercell profile;
- q mesh;
- isotope profile;
- scattering channels;
- boundary/grain-size treatment;
- iterative vs relaxation-time solution profile;
- temperature grid;
- convergence evidence;
- exact solver/runtime.

A reported `kappa_l` without those model assumptions remains bounded to its original source/profile.

## Electronic thermal conductivity

If Wiedemann-Franz or another relation is used to derive `kappa_e`, bind:

- exact electrical-conductivity evidence;
- Lorenz-number model/value;
- temperature/carrier state;
- derivation profile;
- uncertainty.

```text
fixed Lorenz number
!= universally valid kappa_e
```

## Bipolar/high-temperature effects

At elevated temperature or small band gap, bipolar conduction may materially alter Seebeck and thermal transport.

The architecture should support an applicability/refusal state when the chosen transport model neglects a mechanism that is material under the exact state.

Do not allow a low-temperature/single-band approximation to claim ordinary support at high temperature merely because execution converged.

## Stability compatibility

Keep independent:

```text
formation/hull stability
harmonic dynamic stability
finite-temperature phase stability
chemical/environmental stability
```

A candidate with impressive predicted transport but unsupported stability remains a transport prediction, not a viable thermoelectric material.

## Doping and alloy-state identity

Doping is not metadata.

Bind where applicable:

- dopant element/species;
- site/occupation model;
- concentration/fraction;
- compensation assumptions;
- defect chemistry;
- configurational ordering/disorder;
- whether carrier concentration is imposed computationally or arises from explicit defects/dopants.

Preserve:

```text
rigid-band carrier sweep
!= explicit doped material
!= experimentally realized carrier concentration
```

## Anisotropy

Transport may be tensorial.

A scalar average must identify its averaging convention and source tensor.

```text
high in-plane power factor
!= high isotropic/device performance
```

Do not mix directional S, sigma, or kappa components when deriving zT unless the profile explicitly supports that operation.

## Experimental observations

Experimental TE evidence remains distinct from calculations.

Bind where applicable:

- exact physical specimen/process lineage;
- composition/phase characterization;
- density/porosity;
- grain size/microstructure;
- dimensions/orientation;
- contact/electrode state;
- temperature profile;
- carrier/Hall measurement state;
- instrument/calibration refs;
- raw artifact refs;
- measurement-method profile;
- uncertainty.

Examples of separate physical observations:

```text
MeasuredSeebeck
MeasuredElectricalConductivity
MeasuredThermalDiffusivity
MeasuredHeatCapacity
MeasuredDensity
MeasuredThermalConductivity
MeasuredHallCarrierState
```

A measured `zT` assembled from separate instruments still requires exact compatibility and derivation custody.

## Material versus device boundary

Material zT does not establish device efficiency.

Keep separate:

- contact resistance;
- contact chemistry/diffusion;
- leg geometry;
- p/n pair compatibility;
- mechanical/thermal cycling;
- electrode/interconnect losses;
- module heat-exchanger conditions;
- device efficiency/power output.

```text
high material zT
!= high module efficiency
```

## Candidate generation boundary

Generative/inverse-design systems belong to the proposal/advisory plane.

```text
generated structure
-> candidate proposal
-> identity/dedup/constraint validation
-> lower-fidelity evaluation
-> higher-fidelity evaluation
```

not:

```text
generated structure
-> thermoelectric evidence
```

Candidate generation must preserve exact generator/model/corpus/exposure identity for later prospective scoring, but the generator does not own scientific truth.

## Hierarchical screening

A TE campaign may legitimately use multiple fidelities, but each stage keeps its own authority.

Example:

```text
proposal generator
-> geometric/chemical constraints
-> ML surrogate stability/transport screen
-> DFT relaxation / hull
-> harmonic phonons
-> electronic transport
-> expensive anharmonic kappa_l
-> experimental validation
```

Passing an early filter is not evidence that later properties are satisfied.

## MLIP/calibration boundary

A universal interatomic potential may assist relaxation, phonons, forces, or thermal workflows only under capabilities established by exact calibration/applicability artifacts.

Preserve:

```text
energy/force accuracy
!= phonon accuracy
!= anharmonic IFC accuracy
!= kappa_l accuracy
```

Task-specific qualification is required.

## Dataset split and leakage discipline

Thermoelectric datasets contain highly correlated rows across temperature, carrier concentration, dopant fraction, and derivative compositions.

Therefore random row splits can grossly overstate generalization.

Benchmark splitting should support:

- parent-compound holdout;
- chemistry-family holdout;
- structure/prototype holdout;
- dopant/alloy-family holdout;
- temporal/source holdout;
- full temperature/carrier trajectory grouping.

```text
same parent compound at 300 K in train
+ same parent at 700 K in test
!= strong unseen-material generalization
```

## Prospective campaign compatibility

Future TE-PROSPECT should reuse MAG-005C-style preregistration/scoring rather than invent another prospective framework.

Freeze before high-fidelity answers:

- candidate universe;
- generator/ranking policy;
- top-K;
- random/diversity/OOD controls;
- property/evaluator profiles;
- temperature/carrier targets;
- primary metrics;
- compute budget;
- terminal dispositions;
- statistical profile.

All nonconvergence/failures stay in the committed census.

## Metric vector — no universal thermoelectric score

Report independent quantities such as:

- stable-candidate precision/enrichment;
- dynamic-stability precision/enrichment;
- component-property error/calibration;
- power-factor performance under exact state;
- lattice-thermal-conductivity performance under exact state;
- valid compatible-zT derivation rate;
- Pareto position across transport/stability/resource axes;
- chemistry/structure diversity;
- high-fidelity compute per validated candidate;
- failure/nonconvergence census;
- OOD/calibration slices.

Do not collapse these into one hidden weighted `TE_score`.

## Required adversarial fixtures

The first synthetic/reference corpus should include at least:

1. S and sigma from different temperatures -> power factor reject;
2. S and sigma from different carrier concentrations -> reject;
3. kappa_l from another phase -> zT reject;
4. in-plane S combined with cross-plane sigma -> reject unless explicit tensor profile permits;
5. `sigma/tau` presented as absolute sigma -> reject;
6. changed relaxation-time model -> distinct transport identity;
7. same band structure + changed scattering model -> distinct sigma evidence;
8. constant Lorenz number silently used for kappa_e without profile -> reject;
9. harmonic phonon stability promoted to low kappa_l -> reject;
10. converged third-order calculation with unconverged q mesh -> no ordinary kappa_l promotion;
11. imaginary modes hidden before zT calculation -> stability remains unresolved/negative;
12. rigid-band carrier optimization promoted to synthesized doping state -> reject;
13. computationally imposed carrier concentration promoted to measured Hall concentration -> reject;
14. same parent compound split by temperature across train/test and called unseen-material holdout -> reject classification;
15. same composition with distinct dopant site/order collapsed into one subject -> reject;
16. model force accuracy promoted to thermal-transport capability -> reject;
17. generated candidate promoted to novelty/scientific evidence -> reject;
18. candidate filtered out at low fidelity removed from search-history denominator -> reject campaign history;
19. top-K high-zT candidate dynamically unstable -> retained failure, no replacement under one-shot protocol;
20. stored zT differs from independent recomputation -> reject stored summary;
21. highest individual component values cherry-picked across states to synthesize zT -> reject;
22. high material zT promoted to module efficiency -> reject;
23. one high-temperature point promoted to useful operating-range performance -> reject;
24. experimental S/sigma/kappa values from different specimens/process states combined without declared compatibility -> reject;
25. prospective computational PASS promoted to synthesis/manufacturability/economic value -> reject authority promotion.

## External design motivation

External sources motivate architecture only; they do not become local Symthaea evidence.

- 2026 thermoelectric inverse-design workflow using a generative model plus hierarchical ML/DFT/phonon/transport screening:
  https://www.nature.com/articles/s41524-026-02307-3
- 2026 Nature Reviews Materials research highlight emphasizing the coupled electronic/thermal nature of thermoelectric discovery:
  https://www.nature.com/articles/s41578-026-00965-9
- 2026 high-throughput first-principles thermoelectric screening example:
  https://www.nature.com/articles/s41699-026-00715-z

## First implementation train

Recommended order:

```text
TE-CORE-001A
  architecture + compatibility/authority rules

TE-CORE-001B
  frozen synthetic hostile corpus

TE-CORE-001C
  independent known-answer reference derivation

TE-CORE-002A
  production state/component/derived-evidence contracts

TE-SOLVER-001A
  exact electronic-transport capability profile

TE-SOLVER-001B
  exact harmonic-phonon profile

TE-SOLVER-001C
  exact anharmonic-kappa_l profile

TE-BENCH-001
  half-Heusler retrospective benchmark with family-safe splits

TE-PROSPECT-001
  preregistered prospective top-K/control campaign
```

The expensive solver lanes should not be bundled into one first production PR.

## Production implementation gate

Architecture/corpus/reference work may proceed independently.

Do not open production integration code claiming scientific authority until usable qualified/current owners exist for the exact MAT subject, evidence, evaluator/calibration and execution contracts consumed by the implementation.

TE-CORE must remain a typed domain extension rather than forcing every thermoelectric property into the common MAT kernel.

## Claim ceiling

A qualified TE implementation may establish bounded thermoelectric component transport evidence and compatible derived power-factor/zT results under exact material, carrier, temperature, method and direction states. It does not establish synthesized material performance, optimized doping, device/module efficiency, lifetime, manufacturability, commercial value, or physical experiment authority.