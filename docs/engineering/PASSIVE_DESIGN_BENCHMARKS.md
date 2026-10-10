# Passive Design Benchmark Suite

## Goal

Test whether Symthaea can discover useful fixed-geometry / no-moving-mechanical-part designs more effectively than ordinary parameter search.

The benchmark is deliberately cross-domain because a result that only works for one physical regime would be weak evidence for a general engineering capability.

## Benchmark A — Fluidic rectification

Task: maximize forward-to-reverse flow resistance asymmetry using fixed geometry.

Primary metric:

    D = DeltaP_reverse / DeltaP_forward

Additional metrics:

- pressure loss
- material volume
- minimum feature size
- orientation sensitivity
- Reynolds-number robustness

The 2026 fixed-geometry thermofluidic-diode literature is a particularly useful reference because it treats diodicity, thermal behaviour, and gravity orientation as coupled objectives rather than independent parameters.

A secondary reference line is topology-optimized passive fluidic diodes, which emphasize 3D CFD confirmation after simplified optimization models.

## Benchmark B — Passive thermal regulation

Task: maintain a target temperature band using geometry/material response without active control.

Primary metrics:

- peak temperature error
- settling time
- thermal resistance
- active power = 0
- mass / volume

Candidate families may include heat-spreader geometry, phase-change structures, thermal rectifiers, or radiative structures.

The benchmark must distinguish zero active power from zero thermodynamic activity.

## Benchmark C — Acoustic filtering

Task: realize a target transmission / attenuation spectrum with a static geometry.

Primary metrics:

- pass-band error
- stop-band attenuation
- bandwidth
- mass / volume
- manufacturing feature feasibility

Resonance and wave propagation are permitted because the structure itself supplies the physical response.

## Benchmark D — Structural load distribution

Task: minimize mass while satisfying a defined structural safety margin.

Primary metrics:

- safety factor
- compliance / displacement
- mass
- fatigue proxy
- manufacturability

This benchmark overlaps with the existing structural fitness path and is the easiest starting point for validating integration of the passive objective.

## Benchmark E — Electromagnetic field shaping

Task: shape a static-field or wave-field response using passive geometry/material arrangement.

Primary metrics:

- field error
- bandwidth
- loss
- material volume
- active power = 0

Use direct field solvers as the truth boundary for final ranking.

## Search arms

Every benchmark should compare at least three arms:

### Arm 1 — Parameter baseline

Human-selected topology family with ordinary parameter optimization.

### Arm 2 — Generative search

Symthaea generative / evolutionary geometry search with physics-grounded fitness.

### Arm 3 — HDC-guided search

HDC representation of:

- function
- input/output domain
- mechanism family
- topology descriptors
- material descriptors
- failure lineage

The HDC representation should guide retrieval, novelty preservation, and candidate selection, but direct physics remains responsible for the score.

## Hard constraints

The following are not soft preferences:

- numerical validity of the candidate
- manufacturability envelope
- declared material availability
- solver convergence
- safety constraints
- strict passive-function policy when selected

A candidate that violates a hard constraint should not be allowed to dominate a compliant candidate merely by having superior performance elsewhere.

## Novelty and diversity

Do not judge the system only by the best score.

Record:

- number of feasible candidates
- Pareto-front cardinality
- geometric diversity
- topology diversity
- mechanism-family diversity
- failure-family diversity
- unique physical behaviours

The research target is specifically whether HDC-guided search maintains useful diversity while converging toward strong candidates.

## Evidence ladder

Each candidate receives an explicit lifecycle state:

    Generated
    Simulated
    Verified
    Fabricated
    Measured
    Validated

The states are monotonic in the evidence graph only when the required evidence for the next transition exists.

A surrogate model may assist candidate selection, but its prediction is not direct-solver evidence.

## Reproducibility

Freeze for each benchmark run:

- commit SHA
- generator seed set
- material database version
- solver/backend versions
- boundary conditions
- mesh/tessellation policy
- passivity policy
- optimizer configuration
- stopping criteria

Record failed candidates as well as successes.

## Primary statistical comparison

For each search arm, report:

- best feasible performance
- median feasible performance
- feasible-candidate yield
- unique feasible topology count
- Pareto-front size
- compute cost per feasible candidate
- rate of independently reproduced candidates

Avoid claiming superiority from a single best candidate or a single random seed.

## Why this benchmark is technically realistic

Computational metamaterial research already reports that optimization and computational physics can expand the reachable design space beyond unaided human intuition. Recent work also explores graph-space representations, latent generative models, and multi-scale topology/material optimization.

The novel piece here is not "use AI to design metamaterials."

It is:

> combine compositional HDC guidance with hard passive-function contracts and an explicit evidence/provenance boundary.

That can be tested without making any claim that Symthaea is already capable of autonomous physical invention.

## Current implementation

The fabrication kernel now exposes:

- passive-function contracts
- conservative passive evidence extraction
- explicit contradiction detection
- passive multi-objective scoring
- integration into generative design evaluation

The engineering facade now exposes a separate passive fabrication gate.

## Immediate experimental target

Start with Benchmark D.

It already has an analytical structural fitness function and mesh validation. That lets us test the architecture before introducing a CFD dependency.

Then move to Benchmark A because fixed-geometry fluidic rectification is a particularly clear demonstration of useful physical dynamics without moving solid parts.

Finally add coupled thermofluidic and multiphysics cases.

## References

- Bonfanti et al., "Computational design of mechanical metamaterials", Nature Computational Science (2024).
- Dudek et al., "Shape-morphing metamaterials", Nature Reviews Materials (2025).
- Maurizi et al., "Designing metamaterials with programmable nonlinear responses and geometric constraints in graph space", Nature Machine Intelligence (2025).
- Sattari & Karimi, "Multi-objective design of a fixed-geometry thermofluidic diode for pulsating heat pipes" (2026).
- "Design of SFR fluidic diode axial port using topology optimization" (fixed-geometry passive fluidic diode with 3D CFD validation).
