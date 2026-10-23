# Advanced Physical Device Engineering Architecture

Status: architecture-only integration note

Related roots:

- PHOT-ENG-000 #5668
- PLASMA-ENG-000 #5669
- ENG-DEVICE-000 #5670
- ENG-SEMI-000 #5671

This document defines how Symthaea should extend from general engineering and electro-acoustics into photonics/lasers, plasma systems, semiconductor devices, and the reusable hardware infrastructure these domains share.

It deliberately does **not** create physical actuation authority or claim that Symthaea can currently design a qualified laser, plasma generator, semiconductor device, or other advanced physical system.

## 1. Core direction

The target is not a collection of special-purpose generators such as a "laser designer" or "plasma designer".

The target is one evidence-oriented physical-device engineering loop:

```text
requirements / use profile
        ↓
exact engineering subject
        ↓
analytical/reduced-order models
        ↓
external numerical solvers
        ↓
multi-fidelity / robust design search
        ↓
materials + components + geometry + interfaces
        ↓
fabrication / purchased BOM / as-built configuration
        ↓
calibrated FIELD observations
        ↓
prediction ↔ observation residual
        ↓
digital twin / model discrepancy
        ↓
next design generation
```

The same loop should support photonics, plasma, semiconductors, electro-acoustics, robotics, sensors, energy hardware and scientific instruments.

## 2. Existing ownership remains authoritative

Do not duplicate these existing layers:

- `symthaea-optics`: geometric-optics analytical/reference models;
- `symthaea-core::physics::electromagnetism`: exploratory Maxwell/photonics HDC semantics;
- `symthaea-continuum-physics`: continuum EM, thermal, acoustics and plasma/MHD physics;
- `symthaea-circuits`: analytical circuit reference models;
- `symthaea-sim-bridge`: backend-neutral solver requests/results;
- `symthaea-solver-closure`: transitive external-solver input closure;
- ENG-EXEC-001: explicit process working-directory/environment policy;
- ENG-FEM-001: generic Elmer FEM/multiphysics adapter;
- FIELD-004: calibrated optical/laser observations;
- FIELD-005: calibrated plasma observations;
- FIELD-010: future HAL-mediated physical field actuation authority;
- materials/fabrication/as-built/metrology/digital-twin/formal-safety programs;
- SE-SEMANTICS for quantity/interface/configuration semantics;
- ENG-POWER-001 for low-voltage power topology/protection evidence.

New domain crates own **engineering semantics and model composition**, not duplicate numerical solvers or physical observation systems.

## 3. Photonics / laser engineering

A future `symthaea-photonics` domain should model photonic-device subjects and reduced-order physics while delegating high-fidelity work externally.

### 3.1 Initial analytical/model surface

- Gaussian beam propagation;
- paraxial resonator / ABCD matrices;
- cavity stability and resonant-mode descriptors;
- gain-medium and population-rate reduced-order models;
- source threshold/efficiency predictions with explicit validity envelopes;
- polarization, coherence and linewidth descriptors;
- waveguide/fiber/coupler/component subjects;
- detector/responsivity subjects;
- coatings/multilayers as engineered surfaces;
- thermo-optic and thermal-lensing models;
- alignment/tolerance/optomechanical sensitivity;
- nonlinear-optical model references where separately supported.

### 3.2 Numerical/tool hierarchy

Prefer mature external software:

```text
symthaea-optics
    ↓ analytical/geometric reference
Optiland
    ↓ lens/optical-system design, tolerancing, differentiable ray tracing
Meep
    ↓ full-wave FDTD, dispersive/nonlinear/gain/loss electromagnetic behavior
MPB
    ↓ eigenmodes / waveguides / periodic photonic structures
Elmer
    ↓ thermal + structural + electromagnetic coupling
ngspice / device models
    ↓ source drivers, detector/readout electronics
```

No one solver becomes universal optical authority.

### 3.3 Evidence distinctions

```text
GeometricOpticsPrediction
!= FullWavePrediction
!= ThermoOpticPrediction
!= OpticalBenchMeasurement
!= CalibratedFIELDObservation
```

A cavity mode does not prove lasing. A simulated source does not establish emitted optical power. An optical power prediction does not establish product safety classification.

### 3.4 Initial pilot direction

Start with low-risk systems whose physics can be independently measured:

1. lens/imaging train;
2. fiber/waveguide coupling;
3. passive/low-power resonator;
4. spectrometer/interferometer;
5. optical communications/sensing component;
6. thermal optical-characterization fixture.

Do not make maximum optical power the first optimization objective.

## 4. Plasma engineering

Plasma must be **regime-explicit**. The word `plasma` is not a sufficient model definition.

### 4.1 Required regime separation

Conceptually:

```text
ColdAtmospheric
LowTemperatureLowPressure
ThermalArcOrJet
MagnetizedFluidMHD
Kinetic
HighTemperatureFusion
LaserPlasmaInteraction
OtherQualifiedProfile
```

Models and solvers must declare their admitted regimes and assumptions.

```text
cold atmospheric plasma
!= thermal plasma
!= ideal MHD
!= kinetic PIC
!= high-temperature fusion plasma
```

### 4.2 Cold / low-temperature plasma

Required model hooks include:

- electron vs heavy-species temperatures;
- species/composition and ionization state;
- reaction-set identity;
- excitation/recombination/transport model identity;
- gas pressure/flow;
- electrode/dielectric/source geometry;
- surface/wall interaction model identity;
- optical-emission/reactive-species prediction;
- thermal and fluid coupling.

Ideal MHD is not an admissible substitute for this chemistry/non-equilibrium physics.

### 4.3 Hot / magnetized / fusion plasma

Preserve separately:

- fluid vs kinetic model;
- magnetic topology;
- equilibrium model;
- transport/closure model;
- radiation/loss model;
- instability/disruption model;
- plasma-facing materials;
- synthetic diagnostic configuration;
- controller simulation vs physical control authority.

Existing fusion prediction/HDC work should consume this architecture rather than become a separate numerical authority.

### 4.4 Solver hierarchy

```text
existing Symthaea plasma/MHD formulas
        ↓ analytical reference
PlasmaPy
        ↓ formulary, dispersion, dielectric, diagnostic/reference calculations
Gkeyll / BOUT++
        ↓ fluid/moment/magnetized/kinetic-capable models as applicable
PICMI engineering projection
        ↓
WarpX / Smilei / PIConGPU
        ↓ kinetic/PIC backends
Elmer / CFD / materials solvers
        ↓ coupled hardware/material/thermal behavior
```

PICMI should be treated as an interchange language/capability projection, not proof that different PIC backends are equivalent.

### 4.5 Initial plasma pilots

- cold-plasma simulation + diagnostics fixture with no human/animal exposure;
- low-pressure reference discharge simulation;
- optical-emission synthetic diagnostic;
- plasma-facing material/thermal coupon simulation;
- MHD analytical-vs-independent-solver benchmark;
- existing fusion-disruption benchmark as prediction/observation work only.

## 5. Semiconductor/device physics

Circuit simulation alone is insufficient for designing semiconductor devices.

A future `symthaea-semiconductor` / `symthaea-device-physics` layer should preserve:

- material/bandgap/affinity/permittivity evidence;
- carrier mobility/lifetime/model identities;
- doping/profile identities;
- contacts/interfaces/heterojunctions;
- geometry/mesh identity;
- electrostatic and drift-diffusion model identity;
- generation/recombination;
- thermal/self-heating coupling;
- optoelectronic coupling;
- compact-model extraction into ngspice;
- fabrication/as-built deviation evidence.

Preferred first TCAD backend: DEVSIM through the common solver-closure/execution contract.

Core distinction:

```text
TCAD prediction
!= fitted compact model
!= SPICE circuit prediction
!= fabricated device
!= measured device
```

This layer becomes foundational for laser diodes, photodiodes, LEDs, power devices, sensors and RF/analog components.

## 6. Shared critical-device infrastructure

Domain crates must not each invent their own pumps, coils, cooling systems, drivers, mounts or metrology identities.

ENG-DEVICE owns reusable engineering profiles for these common families.

### 6.1 Electrical / RF / source electronics

Compose ENG-POWER-001 and ngspice rather than replacing them.

Needed extensions include:

- regulated source profiles;
- conversion/isolation;
- analog front ends;
- sensor bias/readout;
- RF/microwave source and matching-network semantics;
- filters/EMI profiles;
- magnetic components;
- transient/pulse engineering profiles;
- thermal/fault derating.

This is engineering description/simulation only; high-energy/high-voltage physical actuation is outside the layer.

### 6.2 Magnetics

- permanent magnets;
- coils/solenoids/electromagnets;
- cores/magnetic circuits;
- saturation/hysteresis model references;
- field/gradient maps;
- inductance/resistance/thermal state;
- structural force coupling;
- cooling and material dependencies.

### 6.3 Vacuum and gas systems

- chambers;
- pumps;
- valves/conductance paths;
- gauges;
- regulators/flow controllers;
- species/composition evidence;
- seals/feedthroughs;
- leaks/outgassing/permeation model references;
- vent/purge/process states;
- contamination compatibility.

```text
pressure command
!= measured pressure
```

### 6.4 Thermal management

- conduction/convection/radiation boundaries;
- sinks/cold plates/heat pipes;
- liquid/gas cooling;
- thermoelectrics;
- thermal interfaces;
- temperature/derating envelopes;
- expansion/stress coupling;
- cryogenic extensions only when separately qualified.

### 6.5 Precision mechanics / optomechanics

- mounts and kinematic constraints;
- translation/rotation stages;
- alignment datums;
- tolerance stacks;
- vibration/isolation;
- flexures/bearings;
- backlash/runout/repeatability;
- position metrology.

### 6.6 Engineered surfaces/coatings

- optical coatings;
- electrodes/conductive surfaces;
- dielectric/insulating surfaces;
- plasma-facing surfaces;
- vacuum-compatible surfaces;
- thermal-interface materials;
- protective coatings;
- roughness/finish/cleanliness;
- degradation/aging evidence.

### 6.7 Instrumentation

Physical observation remains FIELD-owned. ENG-DEVICE should only bind exact instruments/probes/components to FIELD quantity/calibration identities.

## 7. Solver infrastructure dependency graph

The advanced-device program should depend on the generic execution work in this order:

```text
ENG-SIM-INPUT-001
  complete transitive input closure
        ↓
ENG-EXEC-001
  explicit cwd/environment execution context
        ↓
generic external adapters
  Elmer / ngspice / Optiland / Meep / MPB / DEVSIM / plasma solvers
        ↓
domain model composition
  PHOT / PLASMA / SEMI / EAC / robotics
```

No adapter should grow a private subprocess/evidence implementation where the common bridge can be improved instead.

## 8. Measurement and digital-twin loop

Every domain follows:

```text
Prediction
    ↓
exact prototype/as-built identity
    ↓
FIELD calibrated observations
    ↓
Residual / discrepancy
    ↓
parameter/model applicability update
    ↓
next design
```

A residual may indicate model error, wrong parameters, manufacturing variation, calibration error, environmental drift or device degradation. The architecture should preserve those possibilities rather than immediately fitting the simulation to the observation.

## 9. Authority and physical safety

Physical source control remains separate from engineering cognition.

```text
CanModel
!= CanPropose
!= CanExecute
```

Any future optical/plasma/high-energy physical action must cross FIELD-010 and `symthaea-hal` with:

- exact device/deployment identity;
- exact reviewed configuration;
- bounded operating envelope;
- current calibration where required;
- operator/authority grant;
- independent hardware interlocks/output gating;
- watchdog/fault handling;
- safety-case evidence;
- independently observed resulting physical state.

Engineering simulation is never permission to energize hardware.

This program is intended for scientific, industrial, sensing, communications, manufacturing, materials, energy and other beneficial engineering applications. Weapon-system optimization is outside the program.

## 10. Priority sequence

Do not expand all fronts simultaneously.

### Priority 0 — qualify the generic foundation

1. ENG-SIM-INPUT-001 exact-head qualification;
2. ENG-EXEC-001 implementation/qualification;
3. ENG-FEM-001 Elmer execution path;
4. ngspice closure integration.

### Priority 1 — reusable device substrate

1. ENG-VAC-001;
2. ENG-MAG-001;
3. ENG-THERM-001;
4. ENG-PWR-002;
5. ENG-MECH-001;
6. ENG-SURF-001.

### Priority 2 — photonics first analytical pilot

1. PHOT-001 typed component/material subjects;
2. PHOT-002 Gaussian/resonator analytical reference;
3. Optiland adapter;
4. FIELD-004 comparison on a low-risk instrument.

### Priority 3 — semiconductor pilot

1. ENG-SEMI-001 material/device subject;
2. analytical diode/MOS references;
3. DEVSIM adapter;
4. compact-model projection to ngspice;
5. measured residual.

### Priority 4 — cold plasma

1. PLASMA-001 regime/state contract;
2. PlasmaPy reference bridge;
3. explicit cold-plasma chemistry/model references;
4. FIELD-005 synthetic/physical diagnostic mapping;
5. low-risk characterization fixture.

### Priority 5 — high-fidelity plasma and active photonics

Only after the previous evidence chain is working:

- Meep/MPB;
- Gkeyll/BOUT++;
- PICMI + one PIC backend;
- thermo-optic/optomechanical design;
- custom optical source/device programs;
- advanced plasma multiphysics.

## 11. Critical non-equivalences

Freeze these across all programs:

```text
simulation converged != physical model validated
solver agreement != physical validation
component selected != as-built component installed
CAD geometry != manufactured geometry
command issued != physical effect established
measurement exists != calibration established
calibration established != fit for every decision
prototype success != manufacturing capability
numerical optimum != globally best physical design
```

## 12. Program outcome

The desired result is that Symthaea can design and improve complete physical systems across multiple fidelity levels while every claim remains attributable to its actual authority source.

A laser, plasma source, semiconductor device, loudspeaker, robot actuator or scientific instrument then becomes a **consumer of one common engineering intelligence**, not a separate silo.