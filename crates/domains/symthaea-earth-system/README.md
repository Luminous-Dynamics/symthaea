# symthaea-earth-system

A self-contained **reduced-order climate-baseline** layer for Symthaea. It
provides compact, inspectable equations, transient protocols, uncertainty
ranges, and tipping-point diagnostics for use as scientific reference models.

Pure `std`, zero dependencies, no `symthaea-core` link.

## Current capabilities

| Area | API |
|------|-----|
| Radiative equilibrium | `EnergyBalanceModel`, `effective_temperature`, `effective_emissivity_surface_temperature` |
| Forcing / sensitivity | named CO₂ forcing, constant/ramp/pulse/sinusoidal protocols, event-aligned schedules |
| Ice-albedo feedback | `IceAlbedoModel::equilibria`, `transition_saddle_node`, `equilibrium_sweep` |
| Cumulative emissions | central TCRE functions, `TcreEstimate`, `WarmingRange`, `BudgetHeadroomRange` |
| Transient response | one-box and two-box models, exact two-box thermal modes, convergence diagnostics |
| Latitudinal climate | equal-area diffusive EBM with conservative spherical heat transport and smooth temperature-dependent albedo |
| Calibration | effective-emissivity inversion, `OneBoxCalibration`, observation-error summaries |
| Driver export | strictly ordered climate, equal-area zonal, hydrology, and soil-carbon records |
| Dynamic carbon | exact two-box constant-emissions, atmospheric-pulse, and piecewise mitigation-path oracles; configurable three-reservoir exchange; coupled carbon–climate chain |
| Sea level | transparent fast/slow linear response components with exact constant-warming trajectories |
| Land water | conserved single-bucket storage, storage-limited evapotranspiration, exact runoff timing, and water-budget evidence |
| Soil carbon | exact fast/slow pool turnover with Q10 temperature scaling, transfer, respiration, and carbon-budget evidence |
| Nutrient cycling | exact organic/mineral turnover, deposition, uptake, leaching, and cumulative nutrient-budget evidence |
| Productivity ledger | explicit environmental and nutrient ceilings with carbon allocation and dual budget closure |
| Ensembles | bounded deterministic one-box Cartesian sweeps and summaries |
| Scientific contract | [`MODEL_CARDS.md`](MODEL_CARDS.md), [`COUPLING.md`](COUPLING.md) |

The effective-emissivity surface model is a calibrated net outgoing-longwave
parameterization. It is not a literal one-layer atmospheric emissivity model.
TCRE headroom is transparent sensitivity arithmetic, not a full assessed
remaining-carbon-budget calculation.

## Example

```rust
use symthaea_earth_system::{
    ForcingProtocol, IceAlbedoModel, OneBoxClimateModel, SECONDS_PER_YEAR,
};

let warm = IceAlbedoModel::earth().warm_stable_temperature().unwrap();
assert!(warm > 273.15);

let protocol = ForcingProtocol::linear_ramp(0.0, 3.7, 70.0 * SECONDS_PER_YEAR).unwrap();
let samples = OneBoxClimateModel::earthlike()
    .simulate_protocol(288.0, &protocol, SECONDS_PER_YEAR, 70)
    .unwrap();
assert_eq!(samples.len(), 70);
```

## Scope boundary

Not yet: a calibrated multi-reservoir carbon cycle (the three-box model is illustrative),
radiative-convective column, resolved atmosphere-ocean circulation, mechanistic sea ice or
ice sheets, groundwater or routed catchment hydrology, calibrated soil or nutrient biogeochemistry, mechanistic vegetation,
assessed sea-level projections, formal Bayesian calibration, or an observational data registry.

## Explicit trajectory contracts

`SimulationGrid` distinguishes integration intervals from returned samples. The
new `simulate_protocol_including_initial` methods return exactly `steps + 1`
timestamped samples, beginning at `t = 0`; legacy post-step-only methods remain
available. `OneBoxClimateModel::constant_forcing_convergence` records numerical
error against the exact one-box solution.

## Analytic calibration

The calibration module inverts the effective-emissivity and one-box equations
without an optimizer, and reports residuals against exact constant-forcing
trajectories. These are transparent parameter identifications, not complete
uncertainty-aware observational assessments.

## Dynamic carbon-to-climate chain

`TwoBoxCarbonCycle` is a reversible, mass-conserving atmospheric/reservoir
anomaly model with explicit GtC/year and ppm units. It now exposes analytic
cumulative-emissions integrals, an exact constant-emissions partition solution,
and integrated budget residuals as numerical oracles. `CarbonClimateModel`
jointly integrates emissions, atmospheric concentration, Myhre-1998 forcing,
and the one-box temperature response. Its illustrative exchange rates are not
an observational carbon-cycle fit; the value is the transparent conservation
and coupling contract.

## Deterministic ensembles

The ensemble API performs bounded Cartesian sweeps over one-box heat capacity,
feedback, and forcing. It preserves input order and returns every member; its
min/median/mean/max summary carries no probabilistic interpretation unless the
caller supplies one.

## Numerical domain guards

One-box, two-box, carbon-cycle, and carbon-climate RK4 steps validate every
intermediate stage, not only the final state. Oversized steps that produce
non-finite states, non-positive absolute temperatures, or non-positive CO₂
concentrations fail closed with a `ModelError`.


## Event-aware and periodic protocols

Pulse and ramp breakpoints can now be converted into bounded event-aligned
integration intervals. Pulse endpoints use explicit left-hand values for the
closing RK4 stage, preventing a discontinuity from being numerically smeared
inside one step. Smooth sinusoidal forcing is available for deterministic
seasonal experiments.

## Exact two-box thermal modes

The surface/deep model exposes its two negative eigenvalues, fast and slow
response timescales, and an exact constant-forcing solution. This provides an
independent oracle for both state trajectories and empirical RK4 order.

## Three-reservoir carbon baseline

`ThreeBoxCarbonCycle` adds fast and slow reversible storage, analytic
equilibrium fractions, derivative-level mass conservation, guarded RK4, and
event-aligned emissions protocols. Reservoir names are structural labels; the
default rates are illustrative and must not be presented as an observational
ocean-land calibration.


## Atmospheric pulse-response oracle

`TwoBoxCarbonCycle::exact_atmospheric_pulse_response` gives the closed-form
partition of an instantaneous atmospheric carbon anomaly under the declared
reversible exchange rates. `pulse_airborne_fraction` is therefore an exact
oracle for this reduced model, not an observational airborne-fraction curve.

## Equal-area latitudinal energy balance

`LatitudinalEnergyBalanceModel` uses `x = sin(latitude)` cells, a conservative
finite-volume form of `d/dx[(1-x²)dT/dx]`, fixed zonal albedo, quadrupole annual-
mean insolation, and linear outgoing longwave radiation. Internal transport
sums to zero exactly on the discrete grid. The Earth-like constructor is an
illustrative annual-mean baseline, not a calibrated climatology.

`latitude_temperature_drivers` exports band index, latitude, equal-area weight,
absolute temperature, and anomaly while deliberately leaving productivity and
disturbance assumptions to the receiving ecology layer.

## Recovery and resource diagnostics

Ice-albedo equilibria expose local net-radiation slopes and heat-capacity-
dependent e-folding recovery times. Recovery time becomes undefined at the
constructed saddle node, making critical slowing down executable without
claiming that a real-world early-warning signal has been observed. Fixed-step
trajectory builders reject more than `MAX_TRAJECTORY_STEPS` samples and reject
non-finite total durations before allocation.


## Exact mitigation pathways

`PiecewiseConstantEmissions` composes exact constant-emissions solutions at
caller-declared stage boundaries. It supports positive emissions, zero-emission
holds, and net removal while preserving the integrated carbon budget. It is an
exact oracle for the reversible two-box model, not a claim that real mitigation
pathways are piecewise constant or that the model has calibrated sink dynamics.

## Temperature-dependent zonal albedo

`LatitudinalIceAlbedoModel` layers a smooth local warm-to-cold albedo transition
on the conservative equal-area EBM. The global heat-transport closure remains
exact, while colder zones absorb less solar energy. The cold fraction is a
phenomenological feedback coordinate, not resolved snow, sea ice, glaciers, or
ice-sheet geometry.

## Multi-timescale sea-level response

`SeaLevelResponseModel` exposes independent fast and slow first-order response
components, their exact constant-warming solutions, and a guarded RK4 oracle.
Component labels are structural unless the caller supplies a documented
calibration; the illustrative constructor is not an assessed projection.


## Conserved land-water bucket

`HydrologyBucket` tracks finite storage, precipitation, storage-limited
evapotranspiration, and saturation runoff. Constant precipitation has an exact
solution, including the time at which the bucket first reaches capacity and
exact cumulative precipitation, evapotranspiration, and runoff budgets. It does
not resolve snow, infiltration fronts, groundwater, vegetation, or routing.

## Exact two-pool soil carbon

`TwoPoolSoilCarbon` tracks fast and slow carbon pools under constant litter
input, transfer, respiration, and a Q10 temperature multiplier. The triangular
linear system has an exact constant-environment solution, including the
equal-rate limit and a cumulative carbon-budget residual. Default parameters
are illustrative and must not be described as an observational soil model.

`hydrology_drivers` and `soil_carbon_drivers` export ordered, dependency-neutral
records while deliberately leaving water-stress, productivity, and atmospheric
feedback assumptions to higher-level adapters.


## Exact two-pool nutrient turnover

`TwoPoolNutrientCycle` tracks organic and mineral nutrient stocks under constant
organic input, deposition, mineralization, biological uptake, and leaching. The
triangular system has an exact state solution, a stable equal-rate limit, and
cumulative loss integrals derived from total nutrient conservation. Nutrient
identity and units remain caller-declared; the model is not a calibrated
nitrogen or phosphorus cycle.

## Finite-interval productivity accounting

`EcosystemProductivityModel` limits potential gross primary production by an
explicit environmental multiplier and finite mineral-nutrient availability,
then partitions assimilated carbon into autotrophic respiration, retained
biomass, and litter. The result is a closed ledger, not a vegetation dynamics
model. It does not infer leaf area, allocation strategy, nutrient recycling,
community composition, or carbon-to-population conversion.

`nutrient_drivers` and `productivity_driver` export these quantities without
silently creating ecological growth rates or carrying capacities.
