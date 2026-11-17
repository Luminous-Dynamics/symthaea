# Optical ISRU 1-D finite-volume thermal column v0.1

**Status:** standalone Rust numerical reference authored; exact-head formatting/Clippy/tests pending  
**Implementation:** `research/asteroid-optical-isru-v0.2/thermal_column.rs`  
**Tracking:** [ISRU-THERMAL-001 #12616](https://github.com/Luminous-Dynamics/symthaea/issues/12616)  
**Scope:** depth-resolved thermal surrogate for algorithm/verification development; not a calibrated asteroid regolith model

## 1. Why replace the lumped response

The B1 material model uses a lumped active thermal depth. That is adequate for testing causal interfaces and coarse pulse/cooling behavior, but it cannot represent a surface hot spot while deeper material remains cold, a moving thermal penetration front, or the fact that a measured surface temperature can hide a different subsurface history.

This matters for optical ISRU because beam absorption, thermal conduction, volatile release, surface refresh and melt formation do not occur at one temperature or one depth. It also matters for transfer: an asteroid material cannot simply inherit a lunar-simulant thermal parameter set.

A 2026 USGS-indexed thermophysical study of asteroid (3) Juno reports a derived thermal inertia of (13 pm 10 mathrm{J,m^{-2},K^{-1},s^{-1/2}}), alongside large uncertainties in emissivity and dielectric properties. This is remote-sensing/model-inferred information for Juno, not a coupon measurement or a universal asteroid value. It reinforces the need to represent material-specific uncertainty rather than hard-code lunar-regolith assumptions.

Reference: https://www.usgs.gov/publications/thermal-and-dielectric-properties-junos-regolith-one-millimeter-wavelength (DOI: https://doi.org/10.3847/PSJ/ae468f)

## 2. Numerical formulation

The column is divided into (N) equal-depth finite volumes of thickness (Delta x), each holding a cell-average temperature (T_i). The first cell is the illuminated surface; the final cell has an adiabatic lower boundary in this initial reference.

For interior cells, the one-dimensional energy balance is:

[
ho c_p Delta x,rac{dT_i}{dt}
=
rac{k}{Delta x}(T_{i-1}-T_i)
+
rac{k}{Delta x}(T_{i+1}-T_i)
+
q''_{mathrm{abs},i}
]

The optical source uses an exponential penetration depth (delta), measured process-path transmission (	au), and surface absorptivity (alpha):

[
q''_{mathrm{abs},i}
=
F,	au,alpha
left[
e^{-z_i/delta}-e^{-z_{i+1}/delta}
ight]
]

where (F) is incident irradiance and (z_i=iDelta x). This represents volumetric deposition integrated over each cell. It is an assumed Beer–Lambert-like surrogate; scattering, multiple reflections, pore-scale transport and wavelength-dependent optical constants are not solved.

The top-cell boundary includes radiative exchange and a caller-supplied latent-heat sink:

[
q''_{mathrm{surface}}
=
epsilonsigma(T_s^4-T_{mathrm{env}}^4)
+
dot m''_v L_v
]

The latent mass flux is supplied by a separate material/inventory model. This thermal module does not predict evaporation kinetics, vapor transport or captured product.

The explicit step is conservatively bounded using the cell areal heat capacity and the largest conduction/radiation row coupling:

[
Delta t le
0.4,
rac{ho c_pDelta x}
{2k/Delta x+4epsilonsigma T_s^3}
]

The bound is recalculated as surface temperature changes. Excessive substep count or a nonphysical intermediate temperature aborts the transaction; state is committed only after the complete pulse passes checks.

## 3. Energy accounting

The receipt separates:

- incident beam energy;
- energy delivered after process-optic transmission;
- energy rejected/lost in the optical path;
- energy absorbed within the modeled column;
- energy not absorbed within the modeled depth;
- net radiative exchange;
- latent heat removed;
- change in column internal energy; and
- energy-balance residual and normalized residual.

The internal balance is:

[
E_{mathrm{abs}} -
E_{mathrm{rad,net}} -
E_{mathrm{latent}} -
Delta U = E_{mathrm{residual}}
]

Internal conductive fluxes cancel between adjacent finite volumes. The implementation rejects a result whose normalized residual exceeds the declared tolerance. The code's `simulation_only` marker is always true.

## 4. Initial software tests

The first Rust test slice covers:

1. A surface-to-depth temperature gradient after illumination.
2. A 50% process-optic transmission reduction giving a 50% reduction in energy absorbed by the column under the same input.
3. Energy-balance closure with an explicit latent-heat sink.
4. Radiative cooling toward a colder environment.
5. Rejection and rollback of an invalid optical input.
6. Exact replay for identical initial state and pulse.
7. Separation of incident, post-optic and in-column energy.
8. Failure and rollback when a supplied latent sink would drive the state nonphysical.

These are numerical software tests only. They do not validate thermal conductivity, specific heat, absorption depth, emissivity or any physical process rate.

## 5. Why this is not yet a regolith physics model

Current deliberate simplifications:

- constant thermal conductivity, heat capacity and bulk density;
- one-dimensional depth only; no lateral heat flow or beam-shape geometry;
- constant optical absorption depth and absorptivity for a pulse;
- no temperature-dependent optical properties or wavelength-resolved absorption;
- a simple radiative environment boundary, without view-factor-resolved radiation between particles;
- adiabatic lower boundary;
- latent heat is an externally specified sink, not an evolved volatile mass balance;
- no phase-change enthalpy, melt-front tracking, spallation, fracture, grain rearrangement, sintering or surface recession;
- no reactive chemistry, gas diffusion, recondensation, plume transport or optical coating deposition;
- no parameter inference or uncertainty propagation yet.

Consequently, the model can investigate whether a spatial temperature gradient changes policy behavior in a synthetic campaign; it cannot yet predict the volatile production rate or excavation rate of an asteroid.

## 6. Verification and physical calibration roadmap

The issue #12616 acceptance sequence is:

### A. Code verification
- Derive and solve a manufactured 1-D heat-equation solution with matching boundary conditions.
- Run mesh refinement and time-step refinement independently; report convergence order.
- Compare the numerical column against an analytical semi-infinite-solid solution for a simple surface heat-flux case.
- Cross-check a matched case with an independently implemented finite-element reference.
- Test energy balance, stability, and extreme parameter bounds.

### B. Coupon identification
For each material family and geometry, measure density/porosity, heat capacity, thermal conductivity or thermal inertia, absorptance versus wavelength, emissivity and optical penetration. Record instrument calibration, sample identity, atmosphere/pressure, surface preparation and uncertainty.

Use independent temperature observations at the surface and multiple depths where practical. Do not fit all thermal parameters to a single surface pyrometer trace: absorption depth, heat capacity and conductivity can be partially confounded.

### C. Blinded validation
Fit on a designated training set, then freeze parameters before a held-out batch/geometry campaign. Report surface and depth temperature RMSE, peak-temperature error, time-to-threshold error, energy-balance residual and predictive-interval coverage. Include calibration and sensor-placement uncertainty.

Only after these steps should a calibrated parameter set be attached to a physical rig manifest. Even then, the model's scope must be limited to the measured envelope.

## 7. How this changes the competitive strategy

Nectar A/B optical-mining tests show that irradiance, mineral morphology, brittle–ductile transition, surface refresh and organic-vapor deposition materially affect the process. The finite-volume column addresses one missing explanatory layer—spatial thermal history—but is not a replacement for the material-regime, optical-health or gas/capture models.

Reference: https://doi.org/10.1016/j.actaastro.2025.04.033

NASA's LR-1 molten-regolith electrolysis demonstration reports a 25 kg simulant campaign and a sustained average oxygen production rate around 0.07 kg/h over 9 hours under vacuum. NASA's CaRD work separately emphasizes integrated concentrator, optical protection, reactor, fluid and gas-analysis interfaces and reports CO detection for the solar-carbothermal reaction. These are valuable system-integration and throughput references for distinct process families; they are not direct benchmarks for asteroid optical water release.

References:
- LR-1 MRE / ICES 2026: https://ntrs.nasa.gov/citations/20260006167
- CaRD / AIAA ASCEND 2026: https://ntrs.nasa.gov/citations/20260002864

The credible near-term advantage is not a claimed higher product rate. It is the ability to distinguish material behavior from optical-path degradation, source drift and depth-dependent thermal response, and then choose the lowest-cost evidence-gathering action before wasting a campaign on the wrong diagnosis.

## 8. Next algorithmic integration

The thermal column should provide a depth-temperature profile and uncertainty descriptor to the material-regime layer. The controller can then choose whether a new pulse, cooling interval, deeper temperature measurement, optical reference, QCM/witness inspection or independent assay most reduces the expected loss of the final engineering decision.

Do not expose the solver's synthetic temperatures as physical evidence. Do not let a fitted simulation parameter satisfy a calibration gate. The physical evidence path remains independent.
