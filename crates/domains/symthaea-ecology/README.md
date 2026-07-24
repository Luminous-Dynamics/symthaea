# symthaea-ecology

Small analytic population-ecology models for Symthaea. The crate is pure
`std` with zero dependencies and provides inspectable equilibria, invariants,
stability classifications, tipping-point baselines, executable trajectories,
and quantitative oracle comparisons for richer agent-based models.

## Current models

- validated logistic growth and future-target inversion;
- Hutchinson delayed logistic growth with the analytic `rτ = π/2` stability threshold;
- constant-yield harvested logistic growth and its `rK/4` fold;
- strong Allee-effect dynamics with explicit extinction and persistence basins;
- Levins metapopulation occupancy with a colonization–extinction threshold;
- directional two-patch occupancy with an analytic connectivity/extinction threshold;
- directed patch-network occupancy with a next-generation spectral threshold;
- exact two-patch source-sink abundance dynamics and a dominant-growth rescue criterion;
- stage-structured Leslie projection with dominant growth and stable composition;
- Poisson branching-process extinction risk for finite founder populations;
- Monod chemostat resource-consumer dynamics and washout thresholds;
- discrete-generation Ricker dynamics and the first period-doubling threshold;
- exact monotone Beverton-Holt generation dynamics as a non-oscillatory comparator;
- classical predator-prey Lotka-Volterra equilibrium, Jacobian, invariant, and RK4 trajectory;
- competitive Lotka-Volterra invasion classification, Jacobians, and guarded dynamics;
- bounded Rosenzweig-MacArthur predator-prey dynamics and local stability;
- explicit temperature, productivity, disturbance, and soil-moisture multipliers with local sensitivity diagnostics;
- scale-invariant Shannon, Simpson, Hill-number, evenness, and dominance summaries;
- piecewise-linear and smooth periodic environmental drivers with generic non-autonomous logistic replay;
- closed-form logistic calibration for known carrying capacity;
- analytic enrichment Hopf thresholds and deterministic continuation slices;
- guarded timestamped integration, bounded trajectory allocation, local recovery times, and analytic-versus-simulation oracle metrics.

The environmental bridge is intentionally assumption-transparent. It does not
claim that temperature, productivity, and disturbance alone constitute a full
ecological forecast. Higher-level integrations should preserve provenance,
units, calibration source, uncertainty, and validity range.

See [`MODEL_CARDS.md`](MODEL_CARDS.md) for the scientific contract and failure
boundaries of every model.

## Guarded trajectory contracts

Timestamped predator-prey and competition APIs include the initial state and
fail closed if an RK4 stage leaves the positive finite population domain. The
Allee runner permits the extinction equilibrium while rejecting negative or
non-finite stages. Legacy permissive methods remain available, but now use the
crate's allocation-free internal RK4 implementation rather than a sibling
dependency.

## Analytic calibration

`fit_logistic_known_capacity` uses the exact logistic logit linearization to
recover growth rate and initial population for a caller-supplied carrying
capacity. It exposes transformed-space RMSE and R² rather than hiding the fit
assumption behind a generic optimizer.

## Time-varying environmental replay

`EnvironmentalTimeline` linearly interpolates validated driver waypoints and
holds endpoint conditions outside the supplied interval. The non-autonomous
logistic runner evaluates the environment at every RK4 substep and includes the
initial state. Timeline time must use the same unit as the ecological rate
parameters; climate seconds are never silently reinterpreted.

## Threshold and competition experiments

`StrongAlleeModel` makes the critical population threshold executable: initial
states below the threshold approach extinction, while states above it approach
carrying capacity. `CompetitionDynamics` adds explicit intrinsic growth rates,
Jacobian diagnostics, and guarded trajectories to the existing invasion-based
outcome classifier, allowing coexistence, exclusion, and bistability claims to
be checked dynamically.


## Spatial and stage-structured baselines

`LevinsMetapopulation` treats occupancy as the fraction of suitable patches and
exposes the exact colonization–extinction persistence threshold.
`LeslieMatrix` adds finite-interval stage projection, net reproductive rate, a
robust Euler–Lotka dominant growth factor, stable stage distribution, and an
explicit eigen-residual. Neither model represents density within patches,
dispersal geometry, individual heterogeneity, or stochastic demography.

## Periodic environmental replay

`EnvironmentalDriverSource` separates the non-autonomous population integrator
from a particular driver representation. Both piecewise-linear timelines and
validated periodic temperature/productivity/disturbance signals can drive the
same RK4 path, including all intermediate stages.


## Delayed density dependence

`HutchinsonDelayLogistic` implements `dN/dt = rN(t)[1-N(t-τ)/K]`, classifies the
positive equilibrium using the analytic `rτ = π/2` threshold, and integrates by
a fixed-step method of steps with interpolated constant prehistory. The solver
requires `dt <= τ`; it is a deterministic delay oracle, not a demographic
mechanism model.

## Directional two-patch persistence

`TwoPatchMetapopulation` permits asymmetric colonization and extinction. The
extinction equilibrium changes stability when `c12 c21 = e1 e2`, and the
positive equilibrium is available in closed form above that threshold. This is
still patch occupancy, not abundance, explicit dispersal geometry, or finite-
patch stochasticity.

## Critical slowing and allocation bounds

Scalar recovery diagnostics convert the local derivative `f'(X*)` into
stability and, for attracting equilibria, an e-folding time. Harvest folds and
colonization/extinction thresholds therefore expose diverging local recovery
times without treating them as observed early-warning evidence. Checked
trajectory APIs reject more than `MAX_TRAJECTORY_STEPS` steps and non-finite
total durations before allocation.


## Finite-population extinction oracle

`PoissonBranchingProcess` separates expected growth from extinction risk. A
supercritical lineage can have an increasing expected population while still
having a non-zero probability of ultimate extinction. The model computes
probabilities analytically; it does not provide a random-number generator or a
real demographic likelihood.

## Explicit resource limitation

`ChemostatModel` tracks substrate and consumer biomass under Monod growth,
dilution, and a declared yield coefficient. It exposes washout, break-even, and
persistence regimes, coexistence equilibria, Jacobians, and guarded trajectories.
It assumes a perfectly mixed constant-volume environment with one limiting
resource.

## Discrete generations and network persistence

`RickerModel` provides a discrete-generation density-dependent map and labels
the positive fixed point as stable, at its first period-doubling threshold, or
unstable without equating every unstable trajectory with chaos.
`PatchNetworkMetapopulation` generalizes occupancy to a bounded directed network
and derives persistence from the spectral radius of the next-generation matrix.


## Monotone discrete generations

`BevertonHoltModel` provides a discrete life-cycle comparator whose positive
carrying-capacity fixed point is approached monotonically rather than through
Ricker-style oscillation. Its finite-generation solution is exact and can be
used to distinguish mechanism-driven overshoot from numerical error.

## Source-sink rescue

`TwoPatchSourceSink` tracks low-density abundance under local growth or decline
and directional dispersal. Its exact matrix-exponential trajectory and dominant
eigenvalue separate local sink status from network persistence. Because the
model is linear, persistent trajectories are not density bounded.

## Biodiversity accounting

`biodiversity_summary` reports richness, Shannon entropy, Simpson concentration
and diversity, Hill numbers of orders zero through two, Pielou evenness, and
Berger-Parker dominance. These are descriptive summaries, not causal ecosystem
models.

## Moisture-aware environmental coupling

`HydroLogisticEnvironmentCoupling` composes the existing climate coupling with a
bounded soil-moisture response. The input fraction matches the Earth-system
hydrology driver contract, while the wilting point, optimum, floor, and growth
and capacity exponents remain explicit and replaceable.


## Closed-population epidemic threshold

`SirModel` implements the homogeneous susceptible-infectious-removed baseline
with frequency-dependent transmission. It exposes the basic and effective
reproduction numbers, epidemic growth threshold, conserved phase-plane
quantity, final-size root, predicted infectious peak, and guarded trajectories.
It does not model latency, births, waning immunity, contact networks, spatial
structure, pathogen evolution, or stochastic transmission.

## Mineral-nutrient environmental coupling

`NutrientLogisticEnvironmentCoupling` layers a bounded Monod-type mineral
response onto the existing climate-and-moisture logistic bridge. The
half-saturation stock, floor, and growth/capacity exponents remain explicit. The
adapter does not consume nutrient from the Earth-system pool; feedback ordering
belongs in a higher-level coupled integrator.

## Community succession and trophic ledgers

`CommunitySuccession` projects abundance or area through a bounded row-stochastic
transition matrix and reports total conservation and stationary residuals.
Transition probabilities are supplied, not learned, and reducible or periodic
chains may not have a unique attracting stationary composition.

`TrophicTransferModel` closes a finite transfer ledger across caller-declared
efficiencies, detrital routing, and dissipative losses. It is ecosystem
accounting rather than a food-web population model.
