# Earth-system model cards

These cards define the scientific contract of each reduced-order model. They
are not validation certificates for a particular observational application.
Any calibrated use should record parameter source, uncertainty, time span, and
validity domain outside the model itself.

## Radiative equilibrium

**Purpose:** global-mean blackbody and effective-emissivity reference values.

**Inputs:** solar constant (W/m²), Bond albedo (fraction), effective outgoing-
longwave emissivity (fraction).

**Outputs:** effective or calibrated surface temperature (K).

**Oracle:** Stefan-Boltzmann round trip and the `emissivity = 1` identity.

**Boundary:** no atmospheric layers, lapse rate, clouds, spatial gradients, or
spectral transfer. The effective emissivity is a fitted net parameter.

## CO₂ forcing: Myhre 1998

**Purpose:** canonical logarithmic forcing baseline.

**Inputs:** positive concentration and baseline concentration in the same units.

**Output:** forcing anomaly (W/m²).

**Oracle:** doubling from 280 to 560 ppm is approximately 3.71 W/m².

**Boundary:** no overlap correction, state dependence, or non-CO₂ forcing.

## TCRE arithmetic

**Purpose:** transparent conversion between cumulative CO₂ emissions and
warming, including an explicit ordered sensitivity envelope.

**Inputs:** cumulative GtC or GtCO₂; low/central/high TCRE in °C/GtC.

**Outputs:** ordered warming or emissions-headroom ranges.

**Oracle:** 1000 GtC gives 1.65 °C under the central default.

**Boundary:** not an assessed remaining carbon budget. Probability level,
non-CO₂ forcing, zero-emissions commitment, and additional Earth-system
feedback adjustments are absent.

## Ice-albedo energy balance

**Purpose:** minimal nonlinear climate feedback with multiple equilibria,
branch stability, and fold detection.

**Inputs:** solar constant, effective emissivity, frozen/warm albedos, and the
linear transition-temperature interval.

**Outputs:** equilibria, local restoring classification, analytic transition-
branch saddle node, and sampled equilibrium sweeps.

**Oracles:** every reported root zeroes net radiation; a constructed tangent
simultaneously zeroes net radiation and its derivative.

**Boundary:** piecewise-linear albedo and zero spatial dimensions. Sweep samples
are not a continuation algorithm and do not infer path-dependent branch
selection by themselves.

## One-box transient climate

**Purpose:** first-order thermal inertia under constant, ramp, or pulse forcing.

**Inputs:** heat capacity (J m⁻² K⁻¹), feedback (W m⁻² K⁻¹), baseline
 temperature (K), and forcing (W/m²).

**Outputs:** temperature trajectory and top-of-atmosphere imbalance.

**Oracle:** exact constant-forcing solution and e-folding time `C/λ`; every RK4 stage remains in the finite positive-temperature domain.

**Boundary:** one response timescale. Discontinuous pulse boundaries should be
aligned with integration steps when event timing matters.

## Two-box transient climate

**Purpose:** surface/deep heat exchange and delayed surface response.

**Inputs:** two heat capacities, feedback, exchange coefficient, baseline
 temperature, and forcing.

**Outputs:** surface/deep temperatures, ocean heat flux, total heat anomaly, and
top-of-atmosphere imbalance.

**Oracle:** heat-content tendency equals top-of-atmosphere imbalance; internal
exchange conserves total heat when feedback and forcing are zero.

**Boundary:** not resolved ocean circulation. Parameters are illustrative unless
a caller supplies and records a calibration.

## Reversible two-box carbon cycle

- **State:** atmospheric and aggregate-reservoir carbon anomalies in GtC.
- **Input:** emissions rate in GtC/year.
- **Invariant:** total carbon-anomaly tendency equals the emissions rate.
- **Oracles:** analytic protocol integrals, exact constant-emissions partitioning, and integrated mass-budget closure.
- **Coupling:** atmospheric anomaly maps to ppm, then named Myhre-1998 forcing.
- **Not represented:** nonlinear ocean chemistry, land feedbacks, airborne
  fraction calibration, permanent sinks, permafrost, or assessed scenario data.
- **Numerics:** every RK4 stage must preserve finite carbon anomalies and positive atmospheric concentration.
- **Status:** deterministic coupling oracle, not an Earth-system carbon emulator.


## Event-aligned forcing schedule

**Purpose:** integrate piecewise forcing without straddling known pulse or ramp
breakpoints.

**Oracle:** every generated interval is positive, bounded by the nominal step,
and terminates exactly at in-domain protocol events. Pulse-closing RK4 stages
use the left-hand endpoint value.

**Boundary:** this is deterministic breakpoint handling, not adaptive local-error
control. Smooth dynamics can still require a smaller nominal step.

## Exact two-box thermal modes

**Purpose:** expose the fast mixed-layer and slow deep-reservoir decay modes of
the linear two-box climate model.

**Oracle:** both eigenvalues are negative for validated parameters; the exact
constant-forcing state independently tests RK4 convergence.

**Boundary:** linear feedback, constant coefficients, and constant forcing for
the closed-form solution.

## Three-reservoir carbon exchange

**Purpose:** represent two reversible storage timescales while retaining an
inspectable mass-conservation invariant.

**Oracle:** all equilibrium fractions sum to one, the equilibrium partition has
zero exchange tendency, and total carbon tendency equals emissions.

**Boundary:** linear boxes are not identified with calibrated ocean, land, or
carbonate-chemistry reservoirs.


## Atmospheric carbon pulse response

**Purpose:** exact impulse-response oracle for the reversible two-box carbon
anomaly model.

**Input:** an instantaneous atmospheric anomaly in GtC and non-negative elapsed
time.

**Oracle:** total pulse mass is conserved; atmospheric retention relaxes
exponentially from one to the model's equilibrium atmospheric partition.

**Boundary:** not an observational airborne-fraction impulse response, Bern
model, ocean-chemistry fit, or permanent-sink representation.

## Diffusive latitudinal EBM

**Purpose:** smallest spatial climate baseline with equal-area zonal cells and
conservative meridional heat transport.

**Inputs:** heat capacity, linear OLR feedback/intercept, diffusion, quadrupole
annual-mean insolation, and fixed equator-to-pole albedo profile.

**Oracles:** zonal insolation averages to `S0/4`; discrete transport convergence
sums to zero; the area-mean heat tendency equals the global top-of-atmosphere
imbalance.

**Boundary:** annual mean, no seasons, dynamic ice edge, water cycle, lapse rate,
clouds, ocean circulation, land contrast, or observational parameter fit.

## Linear recovery diagnostics

**Purpose:** expose local restoring slope and e-folding time near an ice-albedo
equilibrium.

**Oracle:** stable roots have negative net-radiation slope and positive recovery
time; the analytic saddle has zero slope and no finite linear recovery time.

**Boundary:** local linearization only. It is not empirical evidence of critical
slowing in observed climate records.


## Piecewise-constant emissions pathway

**Purpose:** exact propagation of mitigation, hold, overshoot, and removal stages
through the reversible two-box carbon model.

**Oracle:** every stage uses the closed-form constant-emissions solution; final
total carbon differs from the initial total by exactly cumulative stage
emissions in real arithmetic.

**Boundary:** stagewise-constant rates, linear exchange, and no calibrated
carbon-climate feedback or policy uncertainty.

## Temperature-dependent zonal albedo

**Purpose:** couple local temperature to absorbed solar radiation in the
equal-area latitudinal model.

**Oracle:** albedo remains bounded between each band's warm profile and the
declared cold value; conservative transport still sums to zero, so the mean
heat tendency closes against the top-of-atmosphere imbalance.

**Boundary:** smooth phenomenological transition, no ice thickness, motion,
snow aging, ocean heat transport changes, elevation, or hysteretic ice-sheet
dynamics.

## Fast/slow sea-level response

**Purpose:** transparent commitment and lag baseline under a declared
temperature anomaly.

**Oracle:** each component has an exact exponential constant-warming solution
and an independently testable equilibrium sensitivity and e-folding time.

**Boundary:** linear superposition with caller-supplied parameters. Structural
components are not automatically thermal expansion, glaciers, Greenland, or
Antarctica, and the illustrative defaults are not projections.


## Conserved hydrology bucket

**Purpose:** smallest finite-storage land-water and runoff oracle.

**Inputs:** bucket capacity, full-storage evapotranspiration, initial storage,
and constant precipitation.

**Oracles:** exact exponential drydown, exact saturation time, bounded storage,
and cumulative water-budget closure.

**Boundary:** no snow, interception, infiltration physics, groundwater,
vegetation feedback, spatial routing, or calibrated catchment parameters.

## Two-pool soil carbon

**Purpose:** exact temperature-sensitive turnover and respiration baseline.

**Inputs:** fast/slow decay rates, transfer fraction, Q10 parameters, initial
pools, constant litter input, and temperature.

**Oracles:** closed-form fast and slow pools, equilibrium flux closure,
equal-rate limit, and cumulative carbon-budget residual.

**Boundary:** no microbial states, mineral protection, moisture response, priming,
vertical transport, nutrient limitation, or observational calibration.

## Hydrology and soil-carbon driver records

**Purpose:** preserve ordered state and diagnostic evidence across crate
boundaries without inventing ecological responses.

**Boundary:** exported soil moisture, runoff, carbon stocks, and respiration are
physical model outputs, not automatically habitat quality, productivity, or
atmospheric forcing. Those transformations require a named adapter.

## Organic-mineral nutrient cycle

**Purpose:** smallest conserved nutrient-turnover oracle with biologically
available and organic pools.

**Inputs:** organic input, external deposition, mineralization, uptake, and
leaching rates in one caller-declared nutrient unit system.

**Oracles:** exact constant-input state, exact equilibrium, stable coincident-rate
limit, and cumulative stock-plus-loss budget closure.

**Boundary:** no explicit microbes, plant biomass, stoichiometric feedback,
weathering, fixation, denitrification, sorption, vertical profile, or calibrated
nitrogen/phosphorus interpretation.

## Ecosystem productivity ledger

**Purpose:** auditable finite-interval carbon allocation under environmental and
nutrient ceilings.

**Inputs:** potential gross production, environmental multiplier, finite mineral
nutrient, carbon-per-nutrient ratio, respiration fraction, and litter fraction.

**Oracles:** explicit limiting ceiling and simultaneous carbon and nutrient
budget closure.

**Boundary:** accounting only. No vegetation state, acclimation, canopy physics,
turnover dynamics, species composition, or universal carbon-to-demography law.
