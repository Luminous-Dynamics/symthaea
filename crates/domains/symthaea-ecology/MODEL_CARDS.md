# Ecology model cards

These models are analytic baselines and comparison oracles. They do not imply
that stochastic, spatial, evolutionary, or agent-based ecosystems should match
them exactly. Departures should be measured and interpreted against the
assumptions below.

## Logistic growth

**Purpose:** one-species density-limited growth.

**Inputs:** intrinsic growth rate, carrying capacity, initial population, time.

**Outputs:** instantaneous tendency, closed-form population, future target time.

**Oracles:** initial state, carrying-capacity asymptote, and target inversion.

**Boundary:** homogeneous environment, no age structure, delays, migration, or
stochasticity.


## Strong Allee effect

**Purpose:** critical-population and restoration-threshold baseline.

**Inputs:** positive growth rate, carrying capacity, and threshold with
`0 < A < K`.

**Outputs:** tendency, extinction/threshold/persistence basin, three equilibria,
and a guarded non-negative trajectory.

**Oracle:** extinction and carrying capacity are stable; the Allee threshold is
unstable and separates the two attraction basins.

**Boundary:** the cubic density law is phenomenological. It does not identify
the demographic, genetic, mating, or cooperative mechanism causing the Allee
effect.

## Harvested logistic growth

**Purpose:** constant-yield fold/tipping baseline.

**Inputs:** logistic parameters and non-negative constant harvest.

**Outputs:** harvest regime and non-negative equilibria with local stability.

**Oracle:** maximum sustainable yield is `rK/4`; the two equilibria merge at
`K/2` and disappear above that control value.

**Boundary:** constant yield is not effort-based harvesting and can drive the
model through zero if integrated beyond collapse.

## Classical Lotka-Volterra

**Purpose:** analytic predator-prey equilibrium, oscillation, and invariant
baseline.

**Inputs:** four positive interaction parameters and positive initial states.

**Outputs:** derivatives, equilibrium, Jacobian, angular frequency, first
integral, and RK4 trajectory.

**Oracles:** equilibrium zeroes both derivatives; numerical trajectories should
show bounded first-integral drift at resolved time steps.

**Boundary:** prey has no carrying capacity and predation does not saturate.
Closed orbits are structurally fragile under more realistic terms.

## Competitive Lotka-Volterra

**Purpose:** two-species mutual-invasion and exclusion classification.

**Inputs:** two positive carrying capacities and non-negative competition
coefficients.

**Outputs:** coexistence equilibrium, qualitative phase-portrait outcome, explicit derivatives, Jacobian stability, and guarded trajectories.

**Oracle:** weak symmetric competition coexists; strong symmetric competition
is bistable.

**Boundary:** no spatial refuges, dispersal, trait evolution, higher-order interactions, or environmental variation. Deterministic intrinsic growth rates are supplied explicitly by `CompetitionDynamics`.

## Rosenzweig-MacArthur predator-prey

**Purpose:** bounded prey growth plus saturating Holling type-II predation.

**Inputs:** prey growth and capacity, attack rate, handling time, conversion
efficiency, predator mortality, and positive initial states.

**Outputs:** functional response, coexistence equilibrium, Jacobian, local
stability, and RK4 trajectory.

**Oracle:** coexistence zeroes both derivatives. Increasing carrying capacity
can change the local equilibrium from stable to unstable under the same other
parameters.

**Boundary:** no refuge, predator interference, stage structure, adaptation, or
spatial movement. Positivity requires a sufficiently resolved numerical step.

## Environmental logistic coupling

**Purpose:** explicit and replaceable bridge from temperature, productivity,
and disturbance to logistic parameters.

**Outputs:** effective model, each multiplier, and local temperature
sensitivities.

**Oracle:** the named baseline environment recovers the baseline model; analytic
thermal derivatives match finite differences.

**Boundary:** the Gaussian thermal curve and multiplicative composition are
assumptions, not universal ecological laws. Calibration and provenance remain
caller responsibilities.

## Oracle summaries

`logistic_error_summary` reports MAE, RMSE, and maximum error against the closed
form. `lotka_volterra_invariant_drift` reports RMS and maximum drift from the
first integral. These metrics quantify disagreement; they do not decide whether
a richer simulation is scientifically wrong.

### Enrichment continuation

The Rosenzweig–MacArthur model exposes the analytic carrying-capacity Hopf
threshold and a deterministic continuation sweep containing coexistence state,
Jacobian trace/determinant, and local stability. This identifies the reduced
model's bifurcation; it is not by itself evidence that a real ecosystem will
follow the same enrichment pathway.


## Levins metapopulation

**Purpose:** smallest analytic spatial persistence baseline.

**State:** occupied fraction of suitable habitat patches in `[0,1]`.

**Oracle:** persistence exists only when colonization exceeds local extinction;
the exact trajectory relaxes to `1-e/c` or zero.

**Boundary:** homogeneous patches, global dispersal, no rescue-effect detail,
patch quality, occupancy observation error, or finite-patch stochasticity.

## Leslie stage projection

**Purpose:** deterministic stage-structured demography.

**Inputs:** non-negative fecundities and adjacent-stage survivals in `[0,1]`.

**Oracle:** Euler–Lotka dominant growth factor, net reproductive rate, stable
stage distribution, and `L v = lambda v` residual.

**Boundary:** time-invariant rates, no density dependence, no environmental
stochasticity, and no uncertainty-aware vital-rate inference.

## Periodic environment

**Purpose:** reproducible seasonal forcing of the existing ecological coupling.

**Oracle:** every full cycle repeats exactly and all driver minima remain inside
the validated physical domain.

**Boundary:** sinusoidal protocols are not observational climatologies and do
not represent extreme events or autocorrelated variability.


## Hutchinson delayed logistic

**Purpose:** explicit delayed density-dependence and oscillatory-instability
baseline.

**Inputs:** positive growth rate, carrying capacity, response delay, constant
prehistory, and a fixed integration step no larger than the delay.

**Oracle:** `N = K` changes local stability at `rτ = π/2`; constant history at
`K` remains exactly constant.

**Boundary:** phenomenological delay, constant prehistory, deterministic rates,
and linearly interpolated method-of-steps history. No age mechanism or noise is
implied.

## Directional two-patch occupancy

**Purpose:** smallest heterogeneous connectivity baseline.

**Inputs:** directional colonization rates and patch-specific extinction rates.

**Oracle:** extinction loses stability when `c12 c21 > e1 e2`; the positive
coexistence occupancy zeroes both tendencies.

**Boundary:** two patches, occupancy probabilities, no explicit distance, rescue
mechanism, finite-patch stochasticity, or within-patch abundance.

## Scalar recovery diagnostics

**Purpose:** executable local critical-slowing arithmetic.

**Oracle:** stable scalar equilibria have `f'(X*) < 0` and recovery time
`-1/f'(X*)`; fold or transcritical thresholds have zero derivative and no finite
linear recovery time.

**Boundary:** local deterministic linearization, not a statistical early-warning
test on noisy observations.


## Poisson branching process

**Purpose:** finite-founder extinction-risk baseline that deterministic mean
population equations cannot express.

**Oracle:** ultimate extinction is one for mean offspring at or below one; for
a supercritical Poisson process it is the smallest root of
`q = exp(mean * (q - 1))`.

**Boundary:** independent identical reproduction, no density dependence, age
structure, environmental variation, migration, or generated random paths.

## Monod chemostat

**Purpose:** explicit one-resource/one-consumer limitation and washout baseline.

**Oracle:** persistence requires growth at inflow substrate to exceed dilution;
the positive equilibrium zeroes substrate and biomass tendencies.

**Boundary:** perfectly mixed constant volume, fixed inflow and dilution, one
limiting substrate, constant yield, no maintenance, inhibition, or multiple
consumer strains.

## Ricker map

**Purpose:** discrete-generation density-dependence and fixed-point instability
baseline.

**Oracle:** carrying capacity is an exact fixed point with multiplier `1-r`; the
first period-doubling threshold is `r = 2`.

**Boundary:** deterministic non-overlapping generations. Instability above the
threshold is not automatically labelled chaos, and no observation model or
process noise is implied.

## Directed patch-network occupancy

**Purpose:** generalize local extinction and colonization from homogeneous and
two-patch cases to a bounded directed network.

**Oracle:** extinction loses stability when the spectral radius of
`diag(1/e) C` exceeds one; identity-shifted power iteration reports an eigenpair
residual.

**Boundary:** deterministic occupancy fractions, fixed network and rates, no
finite-patch stochasticity, habitat quality dynamics, within-patch abundance,
or explicit dispersal travel time.


## Beverton-Holt generation map

**Purpose:** monotone discrete-generation density-dependence baseline.

**Oracle:** exact finite-generation solution; extinction multiplier is `R` and
the carrying-capacity multiplier is `1/R`.

**Boundary:** deterministic non-overlapping generations, fixed capacity, no
process noise, age structure, harvesting, or oscillatory overcompensation.

## Two-patch source-sink demography

**Purpose:** distinguish local habitat quality from network-level low-density
persistence and demographic rescue.

**Oracle:** exact two-state matrix exponential, migration cancellation in total
abundance, and dominant-eigenvalue persistence classification.

**Boundary:** linear low-density dynamics with no carrying capacity, demographic
stochasticity, travel delay, or state-dependent dispersal. Positive-growth
networks therefore grow without bound.

## Biodiversity summaries

**Purpose:** compare abundance distributions on entropy, concentration,
effective-species, evenness, and dominance scales.

**Oracle:** equal abundance across `S` observed species gives Hill numbers `S`
and Pielou evenness one; common abundance scaling leaves all relative metrics
unchanged.

**Boundary:** taxonomic identities, phylogeny, traits, detectability, sampling
effort, and causal interaction structure are absent.

## Soil-moisture logistic coupling

**Purpose:** explicit bridge from relative bucket storage to population growth
and carrying capacity.

**Oracle:** optimum moisture recovers the climate-only model; analytic local
moisture sensitivities match finite differences.

**Boundary:** piecewise-linear water response with caller-declared exponents,
not a universal drought law. Soil moisture must be paired with provenance,
spatial scale, rooting depth, and time-unit conversion in production use.

## SIR epidemic oracle

**Purpose:** closed-population transmission threshold and final-size baseline.

**Inputs:** positive transmission and removal rates plus non-negative
susceptible, infectious, and removed compartments.

**Oracles:** total-population conservation, phase-plane invariant, effective
reproduction threshold, final susceptible root, and predicted infectious peak.

**Boundary:** homogeneous frequency-dependent mixing, immediate infectiousness,
permanent removal, deterministic compartments, and no demography, latency,
network, spatial, observation, or stochastic process model.

## Mineral-nutrient logistic coupling

**Purpose:** explicit nutrient limitation layered onto climate and soil moisture.

**Oracle:** bounded monotone half-saturation response and analytic local
sensitivities matching finite differences.

**Boundary:** caller-declared mineral stock and units, no nutrient uptake feedback,
stoichiometry, competition among species, or universal nutrient-growth law.

## Community succession chain

**Purpose:** conservative discrete transitions among a bounded set of community
or habitat states.

**Oracle:** row-stochastic total conservation, known stationary distributions,
and explicit stationary residual.

**Boundary:** fixed supplied transition probabilities, no within-state dynamics,
state inference, environmental feedback, uncertainty, or guaranteed unique
stationary distribution for reducible or periodic chains.

## Trophic transfer ledger

**Purpose:** account for transfer, detrital routing, and dissipative loss across
finite trophic steps.

**Oracle:** every level and the complete chain close exactly to the initial
caller-supplied input.

**Boundary:** static accounting, caller-declared efficiencies, no predation
rates, population feedback, nutrient quality, omnivory, recycling dynamics, or
claim of a universal trophic-transfer percentage.
