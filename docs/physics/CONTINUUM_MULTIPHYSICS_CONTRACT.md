<!--
Status: proposed contract / design note, not an implementation-completeness claim.
Scope: Symthaea reference physics models and cross-project integration boundaries.
-->
# Continuum Multiphysics Contract (v0.1)

## Purpose and maturity

This document defines the smallest credible contract for coupling hydrodynamics, electromagnetics, and thermodynamics across the Luminous Dynamics stack. It is a proposed engineering target, not a claim that the present solvers satisfy it.

The stack should share **physical quantities, coupling semantics, run provenance, and validation criteria** while keeping specialist solvers modular. Do not turn this into one monolithic solver, and do not treat a compiling model as a validated engineering model.

The current `symthaea-continuum-physics` crate already contains starting points for Maxwell FDTD, incompressible Navier–Stokes, continuum electrodynamics, acoustics, and plasma MHD. The next step is to establish the contracts and verification harness needed to trust and couple these models.

## 1. Ownership boundaries

| Project | Physical-systems responsibility | Explicit boundary |
|---|---|---|
| **Symthaea** | Reference equations, material/property models, coupled solver experiments, uncertainty estimates, and reproducible validation cases | A model's output is evidence only within its declared assumptions and validated operating envelope |
| **Symtropy** | Deterministic rigid-body, actuator, and robot dynamics; reduced-order energy/thermal state for bodies and workcells; import of forces/loads from validated models | Do not duplicate CFD or Maxwell solvers inside the rigid-body hot path |
| **Mycelix** | Versioned provenance, resource/material/energy records, approvals, maintenance evidence, and cross-party audit trails | Governance, reputation, or consensus cannot turn an unverified physical result into a verified one |
| **Sol Atlas** | Spatial boundary conditions and context: terrain, catchments, weather/solar data, site constraints, and geospatial presentation of run outputs | Preserve model version and uncertainty with each derived layer; map visualization is not a substitute for solver validation |

These boundaries are architectural proposals; each project's implementation status must be checked independently before claiming an integration exists.

## 2. Minimum physical state and unit contract

Every physical field or scalar crossing a module boundary must declare:
- quantity name and physical dimension;
- canonical SI unit (or a typed conversion to SI);
- reference frame and sign convention, where applicable;
- material/domain and boundary-condition identity;
- valid range, initialization, and missing-data behavior;
- model source/version and uncertainty or known limitations.

Use typed quantities at API boundaries where practical. Inside numerical kernels, conversions may be optimized away, but undocumented bare `f64` values must not silently mix kelvin, degrees Celsius, pressure, energy, and dimensionless metrics.

Reject or explicitly classify non-finite state, negative absolute temperature, non-positive density/heat capacity/permittivity where the selected model requires positivity, invalid dimensions, and incompatible units. Never silently clamp invalid input into a plausible-looking result.

A consciousness, information, trust, or harmony metric is not a physical energy, temperature, entropy, or material parameter unless there is a separately documented and empirically calibrated mapping. Keep information-theoretic analogies separate from conservation-law bookkeeping.

## 3. Coupling contract

Use conservative coupling terms and account for energy only once.

### Fluid mechanics

For the initial incompressible model:

```text
∇ · u = 0
ρ(∂u/∂t + u · ∇u) = -∇p + ∇ · τ + f
```

Here `u` is velocity, `ρ` density, `p` pressure, `τ` viscous stress, and `f` body-force density. State clearly whether a run assumes constant density/viscosity and which boundary conditions are applied. Use compressible or low-Mach models when the incompressible assumption is not defensible.

### Electromagnetics

The electromagnetic model must state its regime: electrostatic, magnetostatic, quasi-static, or full Maxwell. For the conducting-medium case, declare how current is computed and whether displacement current is retained. When appropriate:

```text
J = σ E
f_EM = ρ_e E + J × B
Q_Joule = J · E = σ |E|²  (linear, passive Ohmic conductor)
```

`f_EM` is force per volume, and `Q_Joule` is a non-negative volumetric heat source for a passive Ohmic material. More general dispersive/nonlinear materials require a matching energy-transfer model rather than blindly reusing this expression.

### Thermal transport

For a simple single-phase continuum:

```text
ρ c_p (∂T/∂t + u · ∇T) = ∇ · (k ∇T) + Q
```

where `T` is kelvin, `c_p` is specific heat capacity, `k` is thermal conductivity, and `Q` is declared volumetric heating. Model-specific terms (phase changes, radiation, reaction heat, mechanical dissipation) must be explicit rather than hidden inside a tuning constant.

### Required coupling paths

1. **Electromagnetic → thermal:** deposit Ohmic loss in the same cells/material regions where the corresponding current and electric field are computed.
2. **Thermal → electromagnetic:** allow temperature-dependent conductivity/resistivity only when a declared material law supplies it.
3. **Fluid → thermal:** advect temperature/enthalpy using the fluid state; do not confuse fluid velocity with heat flux.
4. **Electromagnetic → fluid:** apply Lorentz body force only for models that resolve the relevant charge/current and field regime.
5. **Thermal → fluid/material:** enable buoyancy, viscosity, density, and heat-capacity changes only when the selected constitutive laws support them.
6. **Across solver boundaries:** exchange time-stamped, unit-labelled fields; report interpolation and coupling lag. Coupled stability must be assessed for the full loop, not each solver in isolation.

## 4. Conservation and admissibility invariants

Every coupled test case must report the applicable invariants and residuals, not just whether the program ran.

- **Mass:** integrated mass change equals boundary mass flux plus declared sources/sinks.
- **Momentum:** changes reconcile with boundary stresses/fluxes and declared body forces.
- **Charge:** charge continuity is satisfied to a declared discrete tolerance; for Maxwell solvers, monitor Gauss-law residuals as appropriate.
- **Electromagnetic energy:** input work reconciles with field-energy change, Poynting flux through boundaries, and material losses.
- **Thermal/total energy:** stored energy change equals injected work/heat minus boundary outflow and declared losses. Do not add Joule heat twice if field energy has already been transferred to the material ledger.
- **Entropy:** for the chosen passive irreversible constitutive model, computed entropy production must not be negative beyond the documented discretization tolerance.
- **Admissibility:** finite states; positive material properties where required; bounded concentrations/fractions; and solver-specific divergence or constraint residuals.

For an invariant `I`, publish the normalization used for its residual (absolute, relative, or scale-aware). A relative residual with a near-zero denominator is not meaningful; report the absolute residual or a justified reference scale in that case. Tolerances must be stated by model and tested at the exact source revision.

## 5. Numerical stability and boundary conditions

- Derive time-step limits from the selected schemes and the actual spatial resolution. For explicit coupled runs, obey the most restrictive applicable advection, diffusion, wave-propagation, and material-response limit.
- Do not present a generic CFL constant as valid for every discretization or mesh.
- Make domain size and resolution valid by construction; reject zero-sized grids, non-finite/non-positive spacing, and invalid time steps.
- Specify boundary behavior explicitly: walls, inflow/outflow, periodic, symmetry, fixed-temperature/heat-flux, driven EM sources, absorbing layers, and/or radiative boundaries as applicable.
- For iterative solvers, expose convergence tolerance, iteration cap, final residual, and a non-convergence status. A fixed iteration count alone is not evidence of convergence.
- Add grid/time refinement studies to estimate discretization error. For coupled runs, test energy and constraint residuals across coupling intervals as well.
- Preserve deterministic replay where practical, but do not prioritize bitwise determinism over physical correctness or silently use nondeterministic parallel reductions in a test that claims exact replay.

## 6. Verification ladder

Start with small reference cases before optimizing for factory-scale or planetary-scale problems.

| Gate | Minimal cases | Evidence required |
|---|---|---|
| **0 — input and dimensions** | Invalid sizes, NaN/∞ parameters, zero/negative material constants, unit conversions | Expected rejection/error; no panics or silently accepted invalid state |
| **1 — single physics** | Manufactured/analytical diffusion; quiescent and lid-driven cavity flow; vacuum EM pulse and dielectric reflection; 1D heat conduction | Expected order of accuracy, finite state, documented boundary behavior, reference comparison |
| **2 — conservation** | Closed/periodic fluid box where applicable; EM source/absorber energy balance; insulated thermal body | Mass/charge/energy residual traces and normalized error bounds |
| **3 — two-way coupling** | Resistor or conductor Joule heating; temperature-dependent conductivity fixture; buoyancy or heated-channel case; simple MHD wave where justified | Analytic/referenced solution, energy transfer ledger, sensitivity to time step and mesh |
| **4 — systems cases** | Solar plus battery/inverter thermal budget; pump-and-pipe hydraulic/thermal network; motor/actuator load and heat; site water/energy model | Reproducible run manifest, input provenance, uncertainty envelope, failure-mode report |
| **5 — optimization/design** | Parameter sweep or design search over a validated model | Candidate ranking robust to model uncertainty; hard engineering constraints kept separate from objective scores |

For CFD practice, use an explicit verification/validation process: code verification, solution verification, comparison to physical/reference data, uncertainty assessment, and a declared applicability range. See NASA's [Overview of CFD Verification & Validation](https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html).

## 7. Reproducible run manifest

The versioned JSON contract is defined in [`physics-run-manifest.schema.json`](physics-run-manifest.schema.json). Consumers should reject unsupported schema versions, preserve the exact producer commit, and derive their readiness labels from recorded gate outcomes rather than trusting a producer-supplied summary status. The schema validates structure; the Rust implementation must additionally validate cross-field energy residual arithmetic, uniqueness constraints, and exact-head gate semantics.

Each saved result used outside a local experiment must carry a manifest containing at least:

- repository, immutable source commit, solver/model version, build features, and platform;
- model equations/regime and constitutive-property sources;
- geometry/mesh dimensions, grid spacing, time step, total simulated time, and coupling cadence;
- initial state, boundary conditions, forcing, material properties, and canonical units;
- random seed if any; input-data hashes; external dataset versions and license/usage constraints;
- final solver residuals, conservation traces, refinement evidence, warnings/failures, and uncertainty statement;
- output field hashes and explicit statement of whether the result passed each verification gate.

CI status is supporting evidence, not a substitute for the manifest. Queued, skipped, cancelled, or stale-head tests are not passes.

## 8. Property data and interoperability

Use evaluated, provenance-bearing data instead of silently embedding guessed material constants. The NIST [Thermophysical Properties of Fluid Systems](https://webbook.nist.gov/chemistry/fluid/) exposes properties such as density, heat capacity, enthalpy, viscosity, and thermal conductivity for supported fluids. Record the fluid, state range, unit system, reference convention, and data-source date for any imported value.

OpenModelica documents the [Functional Mock-up Interface (FMI)](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/fmitlm.html) for model exchange and co-simulation. FMI can be a later adapter boundary for specialist models; introducing it does not replace the need for local invariants, validation, and a run manifest. The [MOOSE physics modules](https://mooseframework.inl.gov/modules/) are another useful reference for modular composition of fluid, electromagnetic, and heat-transfer models.

## 9. First implementation sequence

The first code slice is proposed in [Symthaea PR #7448](https://github.com/Luminous-Dynamics/symthaea/pull/7448): an adiabatic, lumped-conductor Joule-heating reference model with explicit energy accounting. Its test cases are authored but have **not yet been run at the exact PR head**; this is a starting oracle, not a validation pass or a spatial multiphysics solver.

At the energy evidence-ingestion boundary, [Mycelix PR #5025](https://github.com/Luminous-Dynamics/mycelix/pull/5025) separately proposes fail-closed checks for non-finite kWh quantities, reporting periods, meter readings, and trade windows; its tests also remain unrun.

For model-side reservoir and ledger integrity, [Symtropy PR #1622](https://github.com/Luminous-Dynamics/symtropy/pull/1622) hardens energy and heat updates, corrects constant-heat-capacity entropy change, and fixes the 310 K Landauer reference. It does not claim to be a spatial thermal solver or a calibrated neural-energy model.

For geospatial consumers, [Sol Atlas PR #110](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/110) adds typed run-manifest/evidence metadata, including exact-head gate semantics. This adapter makes provenance and readiness inspectable; it does not validate the underlying physics model.

1. **Inventory and label maturity:** enumerate existing solver assumptions, boundary conditions, material models, and existing tests. Distinguish reference examples from validated models.
2. **Harden constructors and stepping contracts:** prevent invalid grids/parameters, add explicit statuses for unsupported/non-converged runs, and test safe failure paths.
3. **Strengthen single-solver tests:** add analytic/manufactured cases, residual checks, and grid/time refinement evidence before coupling.
4. **Add the shared manifest and invariant reporting:** versioned schema, SI quantity metadata, residual normalization, and input/output hashes.
5. **Implement one narrow coupled vertical slice:** a resistive conductor with electrical input, field/current bookkeeping, and transient Joule heating; compare against a lumped analytical energy balance before extending to spatial coupling.
6. **Carry verified results outward:** expose the run manifest and reduced-order model to Symtropy; let Mycelix record provenance/approvals; let Sol Atlas associate results with spatial/site inputs and uncertainty.
7. **Only then scale up:** multi-cell thermal networks, fluid/heat coupling, magnetohydrodynamics, and GPU acceleration should each have independent acceptance gates.

## Non-goals

- No claim that the present code is production-grade CFD, EM, MHD, or thermal engineering software.
- No use of a consciousness/information score as a substitute for joules, kelvin, pascals, or empirically measured material properties.
- No claim of physical validation from unit tests alone, and no claim of successful CI until the exact intended commit has a completed, attributable result.
- No automatic design or deployment action based only on an unvalidated simulation.

The success criterion is not the number of equations implemented. It is whether a user can determine **which assumptions produced a result, how accurately it conserves the relevant quantities, where it agrees with reference evidence, and where it must not be trusted**.
