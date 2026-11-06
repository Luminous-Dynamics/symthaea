# Symthaea Causal–Temporal–Geometric Physics Programme

Date: 2026-11-06  
Status: research programme / hypotheses and executable work packages  
Repository base: `c6d3d0d6242f9d26ef01c8b5d6363d4f00a8ecd6`  
Branch: `research/causal-temporal-physics-v0.1`

## Executive thesis

Build a reproducible scientific-discovery stack that connects quantum-information
models, causal order, non-equilibrium temporal dynamics, spacetime geometry and
instrumental anomaly analysis without treating analogy as physical evidence.

The programme has four rules:

1. Mathematical identity, numerical agreement, physical interpretation and
   experimental confirmation are separate evidence levels.
2. Every physical hypothesis must produce an observable prediction and a
   falsification condition.
3. Every benchmark needs explicit nulls, adversarial controls, finite-size
   analysis, uncertainty and exact-head provenance.
4. An unresolved observation remains unresolved; the system must not infer
   exotic propulsion, non-human origin or new physics from missing metadata.

## Current repository audit

### Causal-set foundation

Existing components:
- `symthaea-order-theory`: finite posets, meets/joins, lattices and fixed points.
- `symthaea-graph-theory`: graph traversal, topological sorting, cycle checks,
  centrality, minimum spanning tree, max flow and PageRank.
- `symthaea-core` causal/temporal encodings and world-model code.

Gap: no dedicated causal-set implementation with a sprinkling generator,
Myrheim–Meyer estimator, longest-chain observable and Lorentz-order invariance
suite was found in the default branch.

New work: `symthaea-causal-set-lab`, std-only and isolated from the cognitive
core. The first 1+1D implementation uses a Fenwick dominance counter for
related-pair counts and a reconstructed longest-nondecreasing-subsequence
algorithm for longest chains; both are O(N log N) time / O(N) memory. The
optimized implementations have brute-force differential tests on small
quantized sets, including repeated coordinates. These Rust tests still need
to be run on the pinned toolchain.

### Time-crystal foundation

Existing component:
- `symthaea-fractal-time-lab::floquet_time_crystal` is a classical, scalar,
  period-two surrogate with an FFT subharmonic score and an early/late
  persistence ratio.

Known limitation:
- It is not a quantum many-body Hamiltonian simulator.
- Its seeded alternating state and near-deterministic flip make period-two
  output easy to obtain by construction.
- Its score does not yet establish spontaneous discrete time-translation
  symmetry breaking, stability over a phase diagram, prethermal lifetime, or
  resilience to generic perturbations.
- Its current semantic-stream adapter uses a single zero null score and one
  dimension at a time; this is not a suitable inferential design.

### Quantum many-body / tensor-network foundation

An additional source audit changes the implementation plan materially:

- `symthaea-frontier-physics::dmrg::HeisenbergChain` returns the two-site
  singlet energy exactly. For larger chains, `ground_state_energy_exact()`
  uses the infinite-chain Bethe-ansatz energy density plus a fixed boundary
  correction; it does not run DMRG sweeps or solve a finite-size effective
  Hamiltonian.
- Its Schmidt spectrum is a synthetic exponential model derived from an
  entropy ansatz. It is not computed from a wavefunction or reduced density
  matrix. Consequently, its truncation-error routine is a property of that
  synthetic spectrum, not a measured DMRG truncation error.
- `symthaea-frontier-physics::tensor_networks::MatrixProductState` provides
  basic product/GHZ state construction, a transfer-contraction norm, and
  bookkeeping metrics. It does not currently implement canonicalization,
  two-site gate evolution (TEBD), variational optimization, DMRG sweeps,
  environment contraction, or a PEPS solver.

These are useful pedagogical/reference primitives, but they must not be
presented as competitive DMRG or production tensor-network capabilities.
The new time-crystal implementation must therefore begin with an exact
small-system Hamiltonian/Floquet oracle. Only then should an MPS/TEBD path be
added and compared against external reference implementations such as
TeNPy, ITensor and quimb. This is a more demanding but scientifically honest
route to competing with the current state of the art.

### Broader physics-module audit

A read-through of the existing physics modules found a consistent boundary that
must be reflected in all external claims:

- `symthaea-core::physics::general_relativity` explicitly describes itself as
  HDC encodings of GR concepts. Its metric/curvature/geodesic types are not a
  numerical relativity solver.
- `symthaea-core::physics::quantum_information` and `qft` represent concepts,
  gate types, particles, propagator labels and vertices as HDC structures.
  They do not currently implement complete complex-amplitude circuit
  evolution, momentum-dependent QFT propagators, renormalized amplitudes or
  scattering calculations.
- `symthaea-frontier-physics::path_integrals` includes a pedagogical
  Euclidean-action / Metropolis path sampler. Its analytic shortcuts and
  conventions (including the free-particle propagator helper) require
  normalization, units and exact-reference audits before scientific claims.
- `holographic.rs` implements formula-level quantities. In particular,
  mapping mutual information directly to a wormhole/bridge "width" is an
  illustrative proxy, not a derived bulk metric relation.
- `nonequilibrium_thermo.rs` contains formula-level helpers; the current
  `minimum_entropy_production` helper sets selected forces to zero rather
  than solving the general constrained quadratic minimization, and the
  Jarzynski estimator needs log-sum-exp stabilization and sampling diagnostics.

These modules are useful vocabulary, interfaces and teaching prototypes.
They must not be counted as production quantum-information, QFT, holography,
thermodynamics or GR solvers. Each needs an equation-to-code audit, explicit
units/conventions, analytic reference cases, dimensional checks and
uncertainty/error analysis before it can support physical inference.

### Time-crystal implementation audit

A further examination of `floquet_time_crystal.rs` shows a structural
limitation beyond the already documented "classical toy" caveat:

- For the alternating initial spin pattern, the two nearest neighbours cancel
  in the local field, so the implemented `j_interaction` term contributes
  exactly zero in the intended even periodic chain.
- The update uses `spin.signum()` and a near-deterministic flip/retain
  interpolation. It therefore preserves a period-two sign pattern by
  construction; this does not demonstrate interaction-stabilized spontaneous
  discrete time-translation symmetry breaking.
- The detector is a period-two/Nyquist detector for a sequence sampled once
  per drive cycle, not a general time-crystal detector.
- The current semantic-stream adapter uses one component at a time and a
  single zero-valued null score. That cannot support calibrated significance
  or broad claims about language-model activation dynamics.

Immediate response: retain this crate as a toy regression fixture, but do not
promote its "Persistent 2T response" score as physical DTC evidence. Replace
the scientific path with (i) exact complex-state Floquet evolution for small
spin systems, (ii) independent Hamiltonian and gate-order references, (iii)
interaction/noninteraction controls, (iv) phase-rigidity and perturbation
sweeps, and (v) finite-size/lifetime analysis. Only then add a validated
MPS/TEBD or PEPS route for larger 1D/2D systems.

### GravCraft foundation

Existing component:
- `symthaea-gravcraft` has a shaping function, three-amplifier shift ansatz,
  simplified simulator, HDC command projection, Phi gating, equivalence
  principle regression checks and an energetics estimate.

Critical model gap:
- The simulator's acceleration function returns the shift vector as a
  coordinate acceleration. It does not construct (g_{\mu\nu}), calculate
  (g^{\mu\nu}), derive Christoffel symbols from metric derivatives, or
  integrate the full geodesic system.
- The energetics module explicitly reuses the single-bubble Alcubierre density
  formula as an order-of-magnitude check, not as the stress-energy solution
  for the three-amplifier metric.
- Phi-dependent actuation is a design choice, not a known physical coupling.

Reframe: a metric-ansatz exploration tool until a constraint-consistent GR
solver and physical source model exist.

### Chain Theory

The October 23–24, 2026 proposal begins from finite-dimensional maximally
entangled qudit pairs, some shared, with knowledge specified through
distillation. Its first published result gives a qutrit–qubit construction,
a 172-record weighted count, a candidate inverse fine-structure constant
(137.035999189566\ldots), and a derivation of a gauge-group structure
under the paper's definitions.

The paper's own scope matters:
- The internal calculation is formally checked in a Lean 4 file.
- The identification of the qutrit–qubit link with the photon is a conjecture.
- The construction does not provide a momentum-dependent propagator,
  charged loop, vertex factor, loop momentum or scattering amplitude.
- The paper notes that experimental determinations of the fine-structure
  constant remain in tension, making future independent measurement decisive.

Work package: independently reproduce the finite-dimensional construction,
formalization and algebraic value before testing its physical interpretation.
Do not market the number as a derivation of QED.

## Paradigm shifts to pursue

### 1. From object-first to relation-first physics

Use one explicit event/relationship representation for causal order, while
keeping Hilbert-space entanglement and spacetime causal order as different
typed relations. Do not conflate entanglement adjacency with light-cone order.

### 2. From model fitting to prediction tournaments

Each candidate theory competes against:
- Standard model / GR baseline where applicable.
- Simpler phenomenological alternatives.
- Randomized and adversarial controls.
- Held-out regimes and datasets.
- A complexity penalty and preregistered prediction set.

The winner is the model with the best calibrated out-of-sample predictions,
not the most compelling narrative.

### 3. From single observable to cross-observable consistency

A proposed spacetime model should simultaneously face causal-order,
dimension, interval-volume, longest-chain, curvature and propagator-related
constraints. A proposed DTC should simultaneously face subharmonic response,
phase rigidity, interaction dependence, lifetime scaling and perturbation tests.

### 4. From simulation scores to proof-carrying experiments

Every result should carry:
- Source commit and clean-tree state.
- Exact model equations and assumptions.
- Seed, input data digest and configuration.
- Independent baseline implementation.
- Null and negative controls.
- Confidence intervals / uncertainty budget.
- Claim and non-claim statements.

## Work packages and gates

| ID | Work package | Deliverable | Gate |
|---|---|---|---|
| CTGP-01 | Causal-set kinematics | 1+1D sprinkling, order, chain, dimension estimator | Minkowski reference + Lorentz invariance |
| CTGP-02 | Causal-set scaling | Multi-N, multi-seed finite-size study | Error bars, convergence and null controls |
| CTGP-03 | Higher-dimensional sprinkling | 2+1D and 3+1D Alexandrov intervals | Independent sampler, volume-law validation |
| CTGP-04 | Chain Theory reproduction | Quantum-state primitives, record enumeration, exact alpha algebra | Independent reproduction + formal proof audit |
| CTGP-05 | Chain Theory stress tests | Alternative choices, uniqueness and sensitivity analysis | All assumptions enumerated; blind alternatives |
| CTGP-06 | Quantum DTC simulator | Floquet Ising and anisotropic Heisenberg reference systems | Exact diagonalization agreement on small systems |
| CTGP-07 | DTC phase diagram | Finite-size scaling, lifetime, rigidity, heating | Reproduce published qualitative phase boundaries |
| CTGP-08 | DTC sensing | Frequency response and linewidth benchmark | Compare against matched conventional sensor baselines |
| CTGP-09 | GR metric engine | ADM 3+1 metric, constraints, geodesics and curvature | Minkowski, Schwarzschild and weak-field regression |
| CTGP-10 | GravCraft physical audit | Stress-energy tensor and energy-condition ledger | No trajectory claim without source/constraint solution |
| CTGP-11 | UAP evidence pipeline | Calibrated multi-sensor reconstruction and provenance | Blinded known-object challenge before anomaly claims |
| CTGP-12 | Cross-domain discovery | Typed causal/quantum/temporal/geometric hypothesis graph | Out-of-sample prediction beats null models |

## Benchmarks

### Causal sets

1. Conditional Poisson sprinkling in 1+1D Minkowski intervals.
2. Ordering fraction (r=2R/[N(N-1)]); expected (r(2)=0.5).
3. Myrheim–Meyer dimension recovery across multiple seeds and event counts.
4. Longest-chain scaling with (N^{1/d}), using independent calibration.
5. Lorentz-boost invariance of causal relations and order-derived observables.
6. Thinning, boundary and finite-size sensitivity.
7. Higher-dimensional controls: 2+1D and 3+1D, with expected ordering
   fractions from the gamma-function relation.

### Time crystals

1. Small-system exact Floquet evolution as the reference oracle.
2. 1D Ising and 2D anisotropic Heisenberg interactions.
3. Subharmonic response (f_mathrm{drive}/2), phase rigidity and lifetime.
4. Phase diagram versus drive error, interaction anisotropy, disorder and size.
5. Finite-size scaling and prethermal heating rate.
6. Noise, missing-data, aliasing, spectral leakage and adversarial periodic
   controls.
7. Sensing: sensitivity, linewidth, dynamic range, stability and energy per
   measurement, matched against conventional sensors.

### GravCraft / GR

1. Minkowski metric: zero curvature, zero source, inertial geodesics.
2. Schwarzschild exterior: analytic metric, Christoffel and curvature checks.
3. Weak-field limit: Newtonian acceleration and post-Newtonian observables.
4. ADM Hamiltonian and momentum constraint residuals.
5. Full stress-energy reconstruction from the metric ansatz.
6. NEC/WEC/SEC/DEC tests over observers and spacetime regions.
7. Quantum-energy-inequality / quantum-interest feasibility analysis where
   the applicable theorem and sampling assumptions are explicit.
8. Numerical convergence, coordinate-gauge tests and independent solver
   comparison.

### UAP observatory

1. Known-object blind set: aircraft, balloons, birds, satellites, flares,
   drones and astronomical/atmospheric phenomena.
2. Sensor-motion/parallax reconstruction with ground-truth trajectories.
3. Timestamp synchronization and calibrated uncertainty propagation.
4. Range/altitude observability and multi-sensor triangulation.
5. False-positive rate at a fixed recall, with confidence intervals.
6. Calibration error, missing metadata and adversarial video transformations.
7. A separate unresolved category; no forced classification.
8. **1+1D algorithmic scaling:** compare O(N log N) observables with an
   O(N²) brute-force oracle through at least N=2,048; then target N=10⁵ and
   N=10⁶ for throughput and peak-memory measurements. Performance targets
   are not claims until measured on a named machine, exact commit and input
   seed.

## State-of-the-art comparison and strategy

| Area | Mature reference | Symthaea strategy |
|---|---|---|
| Causal sets | Analytic estimators and research implementations | Independent Rust kernels, exact invariance tests, reproducible finite-size suite |
| Quantum many-body | Exact diagonalization, ITensor, TeNPy, quimb, IBM quantum hardware | Small exact oracle first; then validated MPS/tensor-network backend |
| Time crystals | IBM quantum devices + tensor networks; diamond nuclear-spin sensing | Reproduce published observables before exploring new protocols |
| GR | Einstein Toolkit, SpECTRE, established numerical-relativity methods | Verify analytic benchmarks and constraints before novel metric ansatz |
| UAP | NASA/AARO multi-source assessment and calibrated sensor analysis | Stronger public provenance, uncertainty and blinded benchmark harness |

Do not attempt to replace mature tensor-network or numerical-relativity stacks immediately. First match their reference cases, then seek a measurable advantage in reproducibility, cross-model consistency, compute efficiency or data provenance.

## Immediate next actions

1. Complete and test the new std-only 1+1D causal-set crate.
2. Publish its exact outputs with multiple seeds and independent reference
   calculations.
3. Add higher-dimensional sprinkling only after the 1+1D invariance suite is
   stable.
4. Reproduce Chain Theory's published algebra and audit the formalization.
5. Upgrade the time-crystal lab with an exact small-system quantum oracle.
6. Replace GravCraft's shift-as-acceleration shortcut with an explicitly
   derived metric and constraint-consistent geodesic solver.
7. Define UAP data contracts before importing imagery or case narratives.

## Scientific non-claims

This programme does not claim that:
- Chain Theory is established physics.
- Time crystals explain UAP or consciousness.
- Causal-set kinematics proves spacetime discreteness.
- HDC similarity represents quantum entanglement.
- IIT Phi generates gravitational fields.
- GravCraft is physically realizable propulsion.
- An unresolved UAP is evidence of non-human technology.

The research objective is to make these questions more testable and to make
negative results as visible as positive ones.
