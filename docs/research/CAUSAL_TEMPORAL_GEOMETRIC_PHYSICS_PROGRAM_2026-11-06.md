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
quantized sets, including repeated coordinates. A dimension-generic
module now also implements fixed-cardinality sprinkling, analytic interval
volumes, causal order and Lorentz boosts for 2+1D and 3+1D. Independent
Python reference sweeps over 12 seeds per dimension (N=1,500) recover the
expected Myrheim–Meyer dimensions within about 0.02 on average. The higher-D
reference pair-count and chain observables remain O(N²).
Open-interval abundance histograms count exact numbers of interior
elements for every related endpoint pair. The 1+1D path now uses persistent
prefix segment trees over compressed null coordinates: an interval rectangle
query is O(log N), giving O(N^2 log N) total time and O(N log N) memory. The
2+1D/3+1D reference path retains successor/predecessor bitsets, with O(N^2)
bits of storage and O(R ceil(N/64)) interval-intersection work. Both paths are
differentially checked against brute-force interval enumeration and
Lorentz-boost invariance.

Two finite-set BDG action brackets are now exposed:
- 1+1D: 2N - 4N0 + 8N1 - 4N2.
- 3+1D: N - N0 + 9N1 - 16N2 + 8N3, with physical prefactor
  4 sqrt(6) (ell/ell_p)^2.

The action methods return integer combinatorial brackets only. Tests include
hand-counted total chains and antichains, unsupported-dimension rejection,
and exact interval-histogram controls. This remains a finite-set action
prototype: no discrete d'Alembertian, path-integral measure, Einstein
equations, or continuum-limit demonstration is implemented.

### Time-crystal foundation

Existing components:
- The original `symthaea-fractal-time-lab::floquet_time_crystal` remains a
  classical scalar period-two surrogate with an FFT subharmonic score.
- A new `symthaea-fractal-time-lab::exact_floquet` module now implements exact
  complex state-vector evolution for finite kicked-Ising interaction graphs,
  including 1D chains and 2D open square lattices, with ZZ couplings,
  longitudinal fields and a global X pulse.
- It includes finite-window subharmonic response statistics, zero-interaction
  controls, 1D and 2D coupling/drive sweeps, and independent Python reference
  results for finite systems up to 3x3 spins.
- The graph oracle is exact only for its small finite Hilbert space (currently
  capped at 20 spins); it is not a thermodynamic-limit solver or a DTC phase
  proof.

Known limitation of the original scalar surrogate:
- It is not a quantum many-body Hamiltonian simulator.
- Its seeded alternating state and near-deterministic flip make period-two
  output easy to obtain by construction.
- Its score does not establish spontaneous discrete time-translation symmetry
  breaking, stability over a phase diagram, prethermal lifetime or resilience
  to generic perturbations.
- The new exact state-vector oracle is a small-system reference, not a
  many-body thermodynamic-limit solver or a DTC phase demonstration.
- Both exact kicked-Ising and Trotterized anisotropic-Heisenberg engines now
  accept arbitrary finite complex initial-state vectors, normalize them to
  unit norm, and fail closed on wrong-sized, non-finite or zero-norm inputs.
  Tests check unitary norm preservation from non-basis superpositions. This
  enables initial-state ensembles but does not yet establish phase rigidity
  across those ensembles.
- Independent finite-window sweeps show strong subharmonic response for a
  symmetry-broken all-up/all-down product initial state at moderate ZZ
  coupling, while the zero-coupling control loses phase coherence under pulse
  errors. Mixed initial states do not show the same uniform magnetization
  order. Stronger coupling is not monotonically better in small graphs.
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

### General-relativity reference foundation

A new std-only crate, `symthaea-metric-geometry-lab`, now implements
symmetric 4x4 metric tensors, pivoted inversion, a Jacobi metric-inertia
check, metric derivatives, Levi-Civita Christoffel symbols, finite-difference
Riemann/Ricci curvature tensors and invariants, and RK4 affine geodesics for
Minkowski Cartesian and Schwarzschild exterior coordinate charts.

Independent Python references at M=1, r=10, theta=pi/2 matched five analytic
Schwarzschild Christoffel components to a maximum absolute error of 1.9e-11.
The Ricci residual was about 1.6e-7 and the Kretschmann invariant agreed with
48 M^2/r^6 to an absolute error of about 1.9e-11. Radial and off-radial
timelike trajectories preserved normalization, Killing energy and angular
momentum at the reported finite-step tolerances. These are independent
reference numbers, not Rust execution evidence. The crate does not solve
Einstein's equations, ADM constraints, matter evolution or a metric source
problem.

### Cross-domain scientific evidence contract

A new std-only crate, `symthaea-science-evidence`, is being added as the
common manifest/receipt layer across these work packages. It binds a full
source commit, clean-tree declaration, toolchain/platform, coordinate and
unit conventions, equations/assumptions, configuration, seeds, baselines,
primary metrics, uncertainty method, input/output BLAKE3 digests, claims,
non-claims and falsification conditions.

Verification states remain distinct: `NotRun`, `Queued`, `Running`,
`Passed`, `Failed`, `Skipped`. A passing run requires a clean source
tree, exact-head checks, log artifact digests, matching runner environment,
and outputs. This is a local integrity contract, not an authenticated
signature or proof of physical truth. The included demonstration explicitly
emits a `NotRun` receipt and cannot qualify as a pass.

### UAP evidence foundation

A new std-only crate, `symthaea-uap-observatory`, defines the initial
measurement-quality contract:

- Typed EO, IR, radar, RF, magnetometer and IMU/GNSS modalities.
- Explicit measurement units, coordinate frames and one-sigma uncertainty.
- TAI timestamps, sensor clock offsets and timestamp uncertainty.
- Calibration validity windows, modality/frame matching and calibration
  artefact digests.
- Raw observation digests, duplicate-record rejection and a data-quality tier.
- Uncalibrated records are retained and marked, never silently repaired.

The crate now also includes a bearing-only triangulation prototype:
ECEF sensor origins, body-to-ECEF quaternions, azimuth/elevation uncertainties,
line-intersection initialization, weighted Gauss-Newton refinement, a local
linearized covariance, residual chi-square and geometry-conditioning gates.
An independent Python experiment over 1,000 synthetic noisy four-sensor
trials recovered the target with 3.1 cm 3D RMSE and 94.4% coverage of the
nominal 95% covariance ellipsoid. These are synthetic Python reference values,
not Rust execution or real sensor performance. Pose/clock covariance,
asynchronous fusion, atmospheric effects, multi-target tracking and blind
known-object tests remain open. No object classification is performed.

Frame-level multi-target evaluation now includes a dependency-free GOSPA
reference with global exact assignment, separate localization/missed/false
components, and brute-force differential tests. A separate trajectory-metrics
module implements fixed-window OSPA(2): it time-aggregates each truth/estimated
track pair and solves one global track-set assignment for the whole window.
It therefore exposes identity swaps and fragmentation that frame-only
unordered set metrics can miss. This is metric code, not a target tracker or
data-association algorithm. The implementation caps the smaller assignment
side at 20 tracks and supports uniform or caller-supplied time weights.

The current implementation is OSPA(2), not T-GOSPA. The 2026 T-GOSPA family
adds an explicit track-switch cost on top of localization and cardinality
penalties. It should be implemented or integrated as an independent comparator
before any superiority claim about track continuity.

### UAP sensor-placement design

The triangulation primitive exposes a local Fisher-information matrix,
point-target D/A-optimal placement, and a finite-target-region robust ranking.
For every target hypothesis, existing sensor bearings are re-predicted from
their sensor poses and declared angular noise; the implementation does not
reuse the nominal measured bearing at a different target location. The robust
score reports prior-weighted expected log(det(FIM)) and covariance trace plus
worst-case information, covariance and condition-number metrics. Its
lexicographic objective maximizes worst-case log(det(FIM)), then expected
log(det(FIM)), then minimizes worst-case covariance trace.

An independent NumPy reference on three target hypotheses and three candidate
placements ranked the elevated candidate first: expected log(det(FIM)) 28.820,
worst-case log(det(FIM)) 27.507, expected CRLB trace 5.94e-4 m^2, and worst-case
trace 1.13e-3 m^2. These are finite-scenario synthetic results, not Rust
execution or field performance. The criterion does not provide a continuous
target-region guarantee and still omits pose/boresight covariance, clock drift,
correlated errors, visibility, terrain, atmosphere and deployment cost. Full
protocol: docs/research/uap-bearing-sensor-placement-reference-v0.2.md.

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

Existing components:
- The legacy `symthaea-gravcraft::simulator` remains a kinematic toy. Its
  acceleration method directly uses the shift vector as coordinate
  acceleration; it is not a geodesic solver.
- A new `adm_reference` path builds a full covariant 4D metric from the
  three-amplifier shift using the explicit ADM choice lapse=1 and
  spatial metric=identity. It uses a fixed command direction per bubble and
  a radial Alcubierre shaping profile around the focal point; this is smooth
  at the bubble centre. It checks Lorentzian inertia, derives Christoffel
  symbols from metric derivatives, normalizes timelike four-velocities, and
  integrates the full affine geodesic equation through the metric-geometry
  kernel.
- The same path reconstructs the Einstein-implied stress-energy
  (T_{\mu\nu}=G_{\mu\nu}/(8\pi)) from the metric's curvature, reports
  Eulerian energy density, and samples null-energy contractions.
- A separate defect audit found that the older radial-away-from-focal-point
  shift is discontinuous at the exact focus: it is manually set to zero at
  the focus, while its directional limit is nonzero and depends on approach
  direction. The ADM path no longer uses that legacy radial vector field.

This is an important mathematical improvement: the amplifier field is now
tested as a metric, not treated as a free acceleration. But it remains a
prescribed static metric ansatz with fixed amplifier focal points. The
implied stress-energy is not a source model or evidence of realizable matter.
The finite-difference curvature and finite-direction energy-condition sample
require step-size/angular convergence and independent symbolic checks. The
legacy simulator remains explicitly non-physical; no propulsion claim follows.

Next: ADM Hamiltonian/momentum projection consistency, source/constraint
residuals, coordinate transformations, curvature convergence, integrated
energy requirements and independent Einstein Toolkit/SpECTRE comparisons.

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

The 2022 CODATA recommended inverse fine-structure constant is
137.035999177(21). The proposal's candidate value 137.035999189566... differs
from that central value by about 1.26e-8, roughly 0.60 of the quoted CODATA
standard uncertainty (the candidate currently supplies no independently
derived uncertainty of its own). Numerical proximity at this level is a
reproduction target, not evidence of identification with the QED coupling.
A physical claim would still need a parameter-free derivation, a defined
uncertainty budget, independent predictions such as the scale dependence of
the coupling, and tests against alternative formula selection and
post-selection/complexity penalties.

### Independent physical-consistency audit

A separate public audit by Cody Mitchell, pinned to
`SproutSeeds/chain-theory-audit@95cd02da5955450397a848c25f8287c1a037d9f2`,
reports several checks beyond the 172-row enumeration:

- Under the Letter's stated diagonal assignments, the unbroken generator
  decomposes exactly as
  (U=\frac13(6Y)-\frac13\operatorname{diag}(0,0;1,1,-2)).
  It therefore has no weak-isospin component and has a nonzero colour
  component. Under those assignments it is not (Q=T_3+Y).
- The two terms in the proposed (2,3) link carry (+2/3) and (-1/3)
  under the audit's stated electric-charge convention. Their superposition
  is not an eigenstate of that charge generator. This is a consistency
  tension if the same pair state is identified with a photon state.
- With the compendium's conditional \(\cos^2\theta_W=1/2\) and
  \(\alpha_{em}=\alpha_Y\cos^2\theta_W\), common Y, 2Y, 6Y and
  GUT-normalized conventions imply very different inverse electromagnetic
  couplings; the audit says the photon identification would require a
  \(\sqrt2\) hypercharge rescaling.
- The compendium's \(319^k\) running ladder is arithmetically distinct
  from the Letter's 31,104 readout dimension.
- The external audit reports a census of 1,492,992 combinations of the
  alternatives explicitly listed in the Letter, with 9,275 distinct
  values and one within its stated 1-sigma CODATA 2022 window. It explicitly
  leaves the broader look-elsewhere question open because undocumented
  development-time choices are not recoverable from the published list.

Symthaea now has a separate exact-integer `physical_consistency` module
covering the diagonal generator decomposition, pair-term charge values,
weak-angle normalization arithmetic, readout/ladder mismatch and the
candidate's CODATA offset. These are small, conditional checks under the
published assignments; they do not rederive the complete gauge model or
independently reproduce the 1.49-million-case census. The exact source and
scope of the external findings are recorded in the crate README.

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

### 5. From monolithic simulation to a physics compiler

Represent each hypothesis as a typed, executable object containing:
- state space and units;
- equations, gauge/sign conventions and boundary conditions;
- conserved quantities and constraints;
- measurement map and observable predictions;
- numerical method, truncation and uncertainty budget;
- validation domain and explicit non-claims.

The compiler rejects dimensionally inconsistent expressions, unconstrained
metrics, invalid probability objects and missing calibration inputs before
running expensive simulations. A hypothesis cannot be promoted merely because
its trajectory looks plausible.

### 6. From one best theory to an adaptive experimental portfolio

Use multi-fidelity evaluation: analytic limits first, exact small systems
second, reduced-order or tensor-network models third, high-fidelity solvers
last, and physical experiments only where observable and instrumentation
budgets justify them. Allocate effort to experiments that best discriminate
between competing hypotheses, not simply to the most complex simulation.

## Work packages and gates

| ID | Work package | Deliverable | Gate |
|---|---|---|---|
| CTGP-01 | Causal-set kinematics | 1+1D sprinkling, order, chain, dimension estimator | Minkowski reference + Lorentz invariance |
| CTGP-02 | Causal-set scaling | Multi-N, multi-seed finite-size study | Error bars, convergence and null controls |
| CTGP-03 | Higher-dimensional sprinkling | 2+1D and 3+1D Alexandrov intervals (prototype added) | Rust tests + independent volume, distribution and boost checks |
| CTGP-04 | Chain Theory arithmetic reproduction | Independent std-only enumerator and 172 row bands; exact charge/generator identities added as a separate physical-consistency module | Run full CSV byte-diff, pinned Lean build, and independent charge/normalization comparison |
| CTGP-05 | Chain Theory physical/selection stress tests | Exact charge and weak-angle consistency checks added; external audit census documented | Reproduce full listed-alternative census, compare every convention and Ward identity, and preregister blinded independent-measurement tests |
| CTGP-06 | Quantum DTC simulator | Exact kicked-Ising graph oracle plus second-order Trotter anisotropic-XYZ graph solver (prototype added) | Dense exact small-system agreement, second-order convergence and published-protocol comparison |
| CTGP-07 | DTC phase diagram | Finite-size scaling, lifetime, rigidity, heating | Reproduce published qualitative phase boundaries |
| CTGP-08 | DTC sensing | Frequency response and linewidth benchmark | Compare against matched conventional sensor baselines |
| CTGP-09 | GR metric engine | Minkowski/Schwarzschild metric, signature, Christoffel, Riemann/Ricci and geodesic reference kernel (prototype added) | Rust tests + independent curvature, conserved quantities and weak-field regression; ADM constraints remain future work |
| CTGP-10 | GravCraft physical audit | ADM metric + true 4D geodesic reference and Einstein-implied stress-energy prototype added | Curvature convergence, ADM constraint projection checks, energy-condition sampling and no propulsion claim without self-consistent source solution |
| CTGP-11 | UAP evidence pipeline | Calibration/provenance, bearing triangulation, local covariance, point FIM and finite-target-region robust placement prototypes added | Pinned Rust tests + exact Rust/Python agreement, pose/time covariance, Monte Carlo coverage and blind known-object challenge |
| CTGP-12 | Cross-domain discovery | Typed causal/quantum/temporal/geometric hypothesis graph | Out-of-sample prediction beats null models |
| CTGP-13 | Scientific evidence plane | Shared exact-head BLAKE3 manifest/receipt contract (prototype added) | Unit tests; reject dirty trees, stale heads, queued/skipped checks and missing log/output digests |
| CTGP-14 | ADM/Einstein bridge | Amplifier-derived ADM metric, geodesics and implied source tensor (prototype added) | Independent symbolic reference, step-size convergence and ADM projection residuals |

## Benchmarks

### Causal sets

1. Conditional Poisson sprinkling in 1+1D Minkowski intervals.
2. Ordering fraction (r=2R/[N(N-1)]); expected (r(2)=0.5).
3. Myrheim–Meyer dimension recovery across multiple seeds and event counts.
4. Longest-chain scaling with (N^{1/d}), using independent calibration.
5. Lorentz-boost invariance of causal relations and order-derived observables.
6. Thinning, boundary and finite-size sensitivity.
7. Higher-dimensional controls: 2+1D and 3+1D, with expected ordering
   fractions 8/35 and 0.1 and interval volumes pi*T^3/12 and pi*T^4/24.
8. Compare sampled time-coordinate and spatial-radius distributions with their
   analytic conditional laws; check interval-volume scaling with T.
9. Boost along every spatial axis and verify pair counts, ordering fraction
   and longest-chain length; include extreme but representable rapidities.
10. 1+1D algorithmic scaling: compare O(N log N) observables with an
    O(N²) brute-force oracle through at least N=2,048; then target N=10^5 and
    N=10^6 for throughput and peak-memory measurements. Performance targets
    are not claims until measured on a named machine, exact commit and seed.
11. Open-interval cardinality histogram: compare bitset counts against a
    cubic brute-force oracle on all small posets; verify invariance under
    Lorentz boosts in D=2,3,4 and test empty/chain/antichain/duplicate-coordinate
    controls.
12. 4D BDG action: compare the exact integer bracket against an
    independent pairwise oracle and published coefficient set; then test
    ensemble means/variance on Poisson sprinklings against continuum
    Einstein-Hilbert curvature for geometries with known scalar curvature.
    Keep discreteness scale, Planck scale, boundary terms and sign convention
    explicit in every physical action comparison.
13. Complexity comparison: separate classical algorithms from quantum
    counting results. A 2026 PR Research paper gives a ~O(N²) quantum algorithm
    for BDG action evaluation; compare the correct oracle/query/hardware model
    rather than treating that quantum complexity as a classical Rust target.

### Chain Theory physical interpretation and selection

1. Reproduce all 172 rows and the exact radical sum from a pinned author
   commit, then byte-diff the full CSV.
2. Build the author's Lean file with its declared Lean/Mathlib versions;
   record assumptions, axioms and externally imported theorems explicitly.
3. Re-derive the unbroken generator decomposition, ray stabilizer,
   charge assignments and Ward-identity results from the primary definitions.
4. Compare all standard hypercharge normalizations and energy-scale
   conventions; no rescaling may be chosen after seeing the numerical answer.
5. Independently reproduce the 1,492,992-case census of alternatives
   explicitly listed in the Letter; report duplicated/identical values,
   grammar complexity and all boundaries.
6. Keep two different null families separate: (a) enumerated alternatives
   actually listed by the Letter and (b) broader plausible formula/model
   grammars. A uniqueness result in (a) is not uniqueness in (b).
7. Preregister future alpha predictions with closed form, scale, uncertainty
   and decision rule before measurement release. Compare g-2 and atom-recoil
   values individually as well as any later CODATA adjustment; do not rely
   solely on an adjusted mean when inputs are in tension.
8. A photon/QED identification requires a propagating pole, correct
   gauge/charge representation, momentum-dependent propagator, interaction
   vertex, Ward identities and at least one parameter-free scattering or
   running prediction. A match to one dimensionless constant is insufficient.

### Causal-set action and interval-abundance benchmarks

1. Match all open-interval abundance bins against an independent brute-force
   endpoint/interior enumerator on exact posets and seeded Poisson sprinklings.
2. Check 1+1D and 3+1D BDG brackets against independently transcribed
   hand-computed references; verify endpoint convention, layer indexing and
   orientation (past versus future layers).
3. Check Lorentz-boost invariance of the full histogram and action bracket,
   not only the related-pair count.
4. For Poisson sprinklings in 1+1D and 3+1D, measure abundance-bin means,
   covariance, variance scaling and finite-size effects over many seeds.
   Compare with analytic Poisson expectations for the declared bounded
   Alexandrov interval, including boundary terms; do not assume a zero mean
   for a finite region.
5. In 3+1D, compare the classical exact bracket with the 2026 quantum-counting
   construction's mathematical output on shared small causal sets. Compare
   resource models separately: CPU wall time/peak memory/word operations versus
   quantum query complexity, oracle depth, state preparation and readout
   assumptions.
6. Benchmark 1+1D persistent range counting against the retained
   bitset-reference path and brute force on the same seeded event sets.
   Report N, related-pair count R, interval-bin checksum, runtime, peak
   memory and machine/compiler details.
7. Extend to the 3D action only after the dimension-dependent coefficients
   and interval-layer convention are independently derived and reviewed.
8. Add BDG scalar-curvature and nonlocal d'Alembertian tests on scalar fields:
   flat-space plane waves, constant fields, convergence with sprinkling
   density, nonlocality scale sweeps and curved analytic backgrounds.
9. Do not interpret a correct finite-set action bracket as a quantum measure,
   a continuum GR solution or evidence that spacetime is discrete.

### Time crystals

1. Small-system exact Floquet evolution as the reference oracle.
2. 1D Ising and 2D anisotropic Heisenberg interactions.
3. Subharmonic response at f_drive/2, phase rigidity and lifetime; compare
   against the exact pi-pulse noninteracting control.
4. Phase diagram versus drive error, interaction anisotropy, disorder and size.
5. Finite-size scaling and prethermal heating rate.
6. Initial-state ensemble study: all-up/down, single- and multi-spin defects,
   random product states with sampled Bloch vectors, and seeded Haar-like
   complex vectors. Report full response distributions, not just the best
   initial state; separately distinguish symmetry-broken and symmetric
   ensembles.
7. Noise, missing-data, aliasing, spectral leakage and adversarial periodic
   controls.
7. Sensing: sensitivity, linewidth, dynamic range, stability and energy per
   measurement, matched against conventional sensors.

### GravCraft / GR

1. Minkowski metric: Lorentzian inertia, zero Christoffels/curvature and
   inertial geodesics.
2. Schwarzschild exterior: analytic metric, Christoffels, vacuum Ricci tensor
   and Kretschmann invariant 48 M^2/r^6.
3. Timelike geodesic normalization, Killing-energy and angular-momentum
   conservation; step-size convergence.
4. Weak-field limit: Newtonian acceleration and post-Newtonian observables.
5. For the amplifier ADM ansatz, compare numeric Christoffels and curvature
   against an independent symbolic/autodiff reference over a frozen corpus.
6. ADM Hamiltonian/momentum projections: independently compute lapse, shift,
   spatial metric and extrinsic curvature; compare projected Einstein tensor
   with the energy/momentum densities reconstructed from T_mu_nu.
7. Repeat curvature and source calculations over a derivative-step ladder;
   report convergence order and reject unstable digits.
8. Sample NEC/WEC/SEC/DEC over observer and null-direction ensembles; perform
   angular refinement and label finite samples as diagnostics, not proofs.
9. Integrate positive/negative energy density over explicit domains; test
   boundary falloff, finite-domain dependence and energy conditions.
10. Coordinate/gauge transformations: transform the same metric and
    geodesic to another chart and compare invariant observables.
11. Compare matching initial data with Einstein Toolkit Hypatia's gauge-wave,
    linear-wave and other applicable test problems; do not equate a prescribed
    metric geodesic with BSSNOK/Z4c evolution.
12. Quantum-energy-inequality / quantum-interest feasibility only where the
    theorem, sampling function, duration and field assumptions apply.

### UAP observatory

1. Known-object blind set: aircraft, balloons, birds, satellites, flares,
   drones and astronomical/atmospheric phenomena.
2. Sensor-motion/parallax reconstruction with ground-truth trajectories.
3. Timestamp synchronization and calibrated uncertainty propagation.
4. Range/altitude observability and multi-sensor triangulation.
5. False-positive rate at a fixed recall, with confidence intervals.
6. Calibration error, missing metadata and adversarial video transformations.
7. A separate unresolved category; no forced classification.
8. Calibration perturbation, clock drift, sensor motion, multipath and
   parallax controls with known ground truth.
9. Track-level false-positive rate and calibration error at fixed recall,
   with confidence intervals and a sealed test set.
10. Sensor-placement comparison on shared candidate sets: point-target D/A/E
    optimality, expected-FIM/Bayesian design, finite-region minimax and
    cost-constrained Pareto fronts.
11. Target-region refinement: increase hypothesis density and compare robust
    scores/rankings; reject any claim that a finite point set bounds the
    continuous region without a separate interpolation/error argument.
12. Local CRLB coverage audit: Monte Carlo empirical covariance, bias,
    NEES/NIS and 50/90/95/99% ellipsoid coverage under correctly specified
    and misspecified noise, pose and clock models.
13. Frame-set GOSPA and fixed-window OSPA(2) on identical sealed truth/track
    sets. Report localization, missed/false cardinality and unmatched-track
    terms separately; freeze p, q, cutoff, weights, window length and units.
14. Track-identity challenge: construct crossing tracks whose unordered point
    sets match at each frame but whose estimate IDs switch. Verify frame GOSPA
    can be zero while OSPA(2) remains nonzero under its fixed global assignment.
15. Fragmentation/coalescence challenge: split a true trajectory into two
    estimate IDs, merge two truths into one track, add short-lived false
    tracks, and include prolonged missed segments.
16. Compare OSPA(2) with the 2026 T-GOSPA reference, which adds an explicit
    track-switch penalty, and with standard tracking metrics such as ID
    switches and track continuity. Do not label a custom switch term as
    T-GOSPA without reproducing its published association/cost formulation.
17. Evaluate metric sensitivity to window length, custom time weights,
    asynchronous/missing frames and track birth/death boundaries; provide
    confidence intervals across independent scenes, not only frame averages.

### New 2026 frontier comparator: anisotropic-Heisenberg DTCs

The 28 January 2026 Nature Communications paper by Switzer et al. reports
a two-dimensional DTC with anisotropic Heisenberg interactions, combining IBM
quantum processors and tensor-network methods. A separate 24 February 2026
npj Quantum Information paper reports clean two-dimensional kicked-Ising DTC
dynamics on a 133-qubit IBM Heron processor for up to 100 Floquet cycles.
These are distinct Hamiltonians, hardware layouts and protocols. Symthaea's
anisotropic-XYZ graph is a small numerical prototype, not a replication of
either experiment. The immediate benchmark is protocol-matched small-system
exact evolution and convergence against tensor-network reference data before
any scale claim. It does not yet reproduce either paper's full phase diagram
or hardware/tensor-network scale.

## State-of-the-art comparison and strategy

The goal is not to claim that Symthaea already outperforms mature specialist
software. It does not. The present advantage to pursue is a rigorous
cross-domain evidence layer: explicit assumptions, typed units, independent
oracles, calibration provenance, falsification gates and reproducibility.

| Area | Mature reference / current frontier | Symthaea current position | Credible path to compete |
|---|---|---|---|
| Causal sets | BDG action/d'Alembertian literature; 2026 quantum-counting method for 4D action with ~O(n^2) quantum runtime and ~O(n) oracle depth | D=2,3,4 flat kinematics, 2D/4D BDG brackets, 1+1D persistent interval range counting; no quantum measure or dynamics | Exact abundance-bin agreement, Poisson finite-size/variance study, measured classical resource curves, then discrete wave-operator and continuum tests |
| Quantum many-body | TeNPy and quimb implement real MPS/PEPS, DMRG and TEBD; IBM hardware has demonstrated 2D DTCs | Exact small 1D/2D kicked-Ising oracle plus controlled second-order Trotter XYZ graph solver; existing MPS/DMRG helpers remain pedagogical | Match dense exact diagonalization for N<=9; verify second-order convergence; then validate MPS/TEBD against TeNPy/quimb/ITensor and published hardware observables |
| Time crystals | 2026 demonstrations include anisotropic 2D Heisenberg DTCs and clean 2D DTCs on a 133-qubit IBM processor, measured up to 100 Floquet cycles | Small exact kicked-Ising oracle and second-order Trotter anisotropic-Heisenberg prototype; arbitrary complex initial-state input added; no large-scale phase demonstration | Match exact small-system and published protocol observables; then establish initial-state-ensemble robustness, finite-size phase rigidity, heating/lifetime and tensor-network agreement |
| Numerical GR | Einstein Toolkit provides mature BSSN/Z4c infrastructure; Hypatia adds GPU-accelerated CarpetX evolution thorns; SpECTRE targets high-order spectral elements and large-scale parallelism | HDC concept encodings plus a new std-only analytic metric / geodesic reference kernel; still no Einstein evolution or constraints | First match Minkowski, Schwarzschild, gauge-wave, constraint and convergence tests; interoperate with specialist solvers rather than replace them |
| UAP / sensor analysis | Calibrated multi-sensor measurement, triangulation, CRLB design and track metrics (OSPA/OSPA(2)/T-GOSPA) | Typed evidence provenance, local bearing triangulation/covariance, finite-region FIM placement, frame GOSPA and fixed-window OSPA(2); no tracker | Exact synthetic Rust/Python and covariance calibration, sealed known-object tracks, OSPA(2)/T-GOSPA parity, false-track rates and cost-aware placement |

### Where Symthaea can realistically do better

1. **Reproducibility as a first-class product:** every result becomes a signed,
   content-addressed experiment package with source revision, immutable inputs,
   calibration, configuration, oracle, uncertainty and claim scope.
2. **Cross-domain consistency:** one hypothesis must survive several
   independently measured observables. A fit to one number or one trajectory
   cannot override failure on charge, symmetry, causality, energy or
   instrumental controls.
3. **Adaptive discovery:** use active experimental design to choose the next
   measurement with the highest expected discrimination between models.
4. **Independent implementation:** maintain a small Rust reference kernel
   and compare it against a separate implementation or established library.
5. **Evidence-weighted automation:** an autonomous researcher may propose a
   theory, but cannot self-certify it. Separate verifiers own numerical,
   formal, physical and empirical claims.

The competitive benchmark should therefore be a multi-axis frontier:
accuracy, calibration, numerical stability, runtime, memory, energy cost,
reproducibility, robustness to misspecification and scientific claim quality.
Symthaea should only claim SOTA on axes where an exact-head, matched-baseline
benchmark demonstrates it.

## Competitive scorecard and improvement targets

The objective is not to replace mature specialist software before matching
its reference results. The first competitive advantage should be
cross-domain reproducibility and explicit claim control; numerical speed or
physical fidelity only count after matched benchmarks.

| Workstream | Current Symthaea prototype | Strong external comparator | Near-term parity gate | Longer-term differentiator |
|---|---|---|---|---|
| Causal sets | 1+1D O(N log N) pair/chain observables; 1+1D interval histogram O(N² log N); 2+1D/3+1D bitset reference O(R ceil(N/64)); 2D/4D BDG brackets | Causal-set BDG action/d'Alembertian literature; 2026 quantum-counting algorithm | Brute-force abundance-bin oracle, Poisson variance/boundary study, boost-invariant action, CPU scaling and independent 2026-formula comparison | Discrete d'Alembertian/curvature, matter propagation, curved backgrounds and continuum convergence |
| Chain Theory | Independent 172-row enumerator plus separate exact charge/generator arithmetic checks | Author's Lean/Python source; independent SproutSeeds consistency and alternatives audit | Full CSV byte diff, Lean axiom/read-cap audit, exact charge and weak-angle tests, reproducible census of listed alternatives | Parameter-free propagator/vertex/scattering predictions; preregistered blind comparisons and explicit look-elsewhere accounting |
| Time crystals | Exact 1D/2D small-system kicked-Ising state vector | Exact diagonalization, TeNPy/quimb/ITensor, IBM experiments | Match exact small-system unitary and published observables | Validated MPS/TEBD, phase rigidity, heating/lifetime and sensing metrics |
| GR / GravCraft | Analytic Minkowski/Schwarzschild kernel plus new amplifier-derived ADM metric, full 4D geodesics and Einstein-implied source diagnostics | Einstein Toolkit Hypatia (GPU BSSNOK/Z4c), SpECTRE, Apples-with-Apples and analytic GR references | Exact Minkowski/Schwarzschild references; ADM projection consistency; curvature step-size and gauge tests | Self-consistent 3+1 evolution, matter coupling, global energy-condition and source/boundary analysis |
| UAP | Calibration/provenance, bearing triangulation, local covariance, point/finite-region FIM placement, frame GOSPA and OSPA(2) metric references; no tracker | EKF/UKF/JPDA/MHT and random-finite-set tracking; MathWorks OSPA(2); 2026 T-GOSPA with explicit switch cost | Synthetic 93–97% ellipsoid coverage, independent geometry agreement, sealed known-object tracks and OSPA(2)/T-GOSPA comparison | Calibrated low false-track rates, robust identity continuity, nuisance covariance, active sensor placement under visibility/cost constraints |
| Evidence plane | Exact-head BLAKE3 manifest and receipt prototype | Reproducible-build, SBOM and scientific workflow provenance systems | Tamper/stale-head/skip/dirty-tree negative controls | Reusable proof-carrying experiments across Symthaea, Mycelix and Sol-Atlas |

### Measurable targets

- **Correctness first:** zero mismatches against analytic or independent exact
  oracles on the declared validation domain.
- **Calibration:** nominal 95% uncertainty intervals achieve 93–97% empirical
  coverage on held-out synthetic trials before real sensor claims.
- **Robustness:** report false-positive and false-negative rates with
  confidence intervals under missingness, drift, noise and adversarial inputs.
- **Scaling:** publish runtime, peak memory and error together; no speedup
  claim without same machine, compiler, input, baseline and output digest.
- **Replication:** every headline figure can be rebuilt from one manifest,
  source SHA, frozen inputs, toolchain and raw result artifacts.
- **Scientific scope:** no extrapolation beyond the tested dimension,
  Hamiltonian, coordinate chart, sensor calibration domain or claim boundary.

## Research expansion — 8 November 2026

The programme now has three explicit extensions beyond the initial causal-set / time-crystal / GR / UAP integration:

### Measurement observability before trajectory estimation

- [UAP-ASYNC-001 #11029](https://github.com/Luminous-Dynamics/symthaea/issues/11029) specifies gauge-aware joint timing, pose/boresight calibration and trajectory observability.
- `symthaea-uap-observatory::observability::analyze_jacobian` reports singular values, numerical rank, nullity and weak parameter directions from a scaled **data-only** Jacobian. It accepts no priors, so posterior regularization cannot mask a data null direction.
- Its present implementation is a small-block Jacobi eigensolver over the Gram matrix (J^T J), capped at 64 parameters. This is a reference diagnostic, not yet a production SVD or an end-to-end asynchronous tracker. Squaring the condition number in the Gram construction is a known numerical limitation; compare against rank-revealing QR/SVD before using it for high-dynamic-range calibration problems.
- The current bearing triangulator remains a common-epoch, known-pose local geometry primitive. Clock offset/drift, platform pose, boresight covariance and correlated calibration uncertainty remain unimplemented in the estimator.

### Scientific experiment compiler: three distinct objectives

- [SCIENCE-DESIGN-001 #11132](https://github.com/Luminous-Dynamics/symthaea/issues/11132) now treats experiment design as an end-to-end compilation loop: prediction ledger, comparability, local/global observability, information/discrepancy objectives, resource-bounded selection, frozen execution contract and evidence ingestion.
- `gaussian_predictive_kl_nats` adds an analytic directed scalar-Gaussian KL reference for expected predictive discrepancy, distinct from the existing discrete model-identity EIG.
- `model_recovery::evaluate_model_recovery` adds deterministic planted-model Monte Carlo recovery under a frozen MAP/abstention rule, with a confusion matrix, coverage/selective accuracy and an optional unlisted generator. The unlisted generator is only a data source, never an inference candidate.
- The experiment-compiler specification separates EIG, expected predictive KL, model-recovery accuracy, parameter information and downstream decision utility. A high prior-weighted recovery score can still be misleading when the experiment contains no information; an explicit regression test demonstrates this with identical model likelihoods.

### Time-crystal frequency locking and sensing

- [DTC-PHASE-001 #11033](https://github.com/Luminous-Dynamics/symthaea/issues/11033) separates a period-two signal from phase rigidity, finite-size lifetime scaling and metastability.
- `symthaea-fractal-time-lab::frequency_locking` adds a finite-sweep subharmonic peak and frequency-locking diagnostic. It reports FFT grid spacing separately from observation-limited resolution. It is a signal-level diagnostic, never a DTC phase classifier.
- [DTC-SENSE-001 #11128](https://github.com/Luminous-Dynamics/symthaea/issues/11128) now targets the 2026 interaction-assisted DTC magnetic-sensing literature. The relevant diamond experiment reports 0.5–50 kHz operation and sub-70 mHz linewidths, with an AC-resonant lifetime-extension mechanism. Symthaea has not reproduced that experiment; its exact-state-vector model is capped at 20 spins and its anisotropic-Heisenberg dynamics are second-order Trotterized.
- The next physics step is to report classical readout Fisher information and total experiment-time-normalized sensitivity, separately from quantum Fisher information and finite-window subharmonic amplitude.

### GravCraft: numerical metric consistency before source claims

- The separate `adm_reference` path constructs a static ADM metric with alpha=1 and gamma_ij=delta_ij from the amplifier shift, computes true Christoffels/geodesics through the metric-geometry kernel, and reconstructs Einstein-implied T_mu_nu=G_mu_nu/(8 pi).
- A new `adm_derivative_step_sweep` example varies the nested finite-difference step across multiple scales and reports Ricci/Kretschmann invariants, Eulerian energy density, sampled null-energy minimum and adjacent-step differences.
- [GR-SOURCE-001 #11135](https://github.com/Luminous-Dynamics/symthaea/issues/11135) defines the next stronger gate: independent 3+1 Hamiltonian/momentum constraint checks against 4D Einstein-tensor projections, Bianchi/source-conservation diagnostics, derivative validation with analytic or automatic differentiation, and convergent energy-condition/source integration.
- The legacy shift-as-acceleration simulator remains a kinematic toy. The ADM path remains a static metric ansatz and an Einstein-implied source calculation; neither supplies matter capable of generating that metric or demonstrates propulsion.

### Causal interval abundance and finite-geometry comparison

- [CAUSAL-GEOMETRY-001 #11129](https://github.com/Luminous-Dynamics/symthaea/issues/11129) adds a full interval-abundance profile and continuum-closeness research target, motivated by Surya, Phys. Rev. D 113, 024034 (2026).
- `symthaea-causal-set-lab::IntervalAbundanceProfile` retains both the open-interval cardinality histogram and the fraction of incomparable pairs, and provides fixed-width total-variation, Hellinger and Jensen–Shannon statistical distances.
- These generic statistical distances are **not** the published Lorentzian closeness function and are not Lorentzian Gromov–Hausdorff distances. The paper-faithful construction must be implemented separately from its exact definition.
- [CAUSAL-INTERVAL-PERF-001 #11035](https://github.com/Luminous-Dynamics/symthaea/issues/11035) tracks the observed algorithm crossover between the persistent-tree and bitset interval histogram implementations.

## Industry and SOTA comparison — honest baseline

| Area | External benchmark frontier | Current Symthaea level | What would constitute a genuine advance |
|---|---|---|---|
| DTC phase identification | 2D anisotropic-Heisenberg and clean 2D DTC experiments on IBM hardware; small-system ED, tensor networks and finite-size scaling | Exact kicked-Ising state vector up to 20 spins; finite-graph anisotropic Heisenberg with second-order Suzuki–Trotter | Correct phase-vs-signal discrimination, held-out parameter-region rigidity, initial-state and metastability controls, exact/Trotter/tensor-network convergence |
| DTC sensing | 2026 13C diamond sensor: 0.5–50 kHz and <70 mHz linewidth using AC-resonant order-parameter lifetime extension | No faithful sensor reproduction yet; frequency-locking signal diagnostic | Matched total-time sensitivity, calibrated classical FI, open-system noise model, independent data reproduction |
| Asynchronous sensor fusion | Joint timing/geometry/target calibration, factor graphs, delayed-measurement tracking and real-world collaborative localization are current literature/industry baselines | Known-pose/common-epoch bearing triangulation, local FIM placement and data-only small-block rank diagnostic | Joint clock/pose/target estimator with gauge-aware identifiability, calibrated uncertainty and field dataset performance under matched information |
| Causal-set continuum tests | Ordering fraction, interval abundance, BDG actions and coarse-graining/continuum-likeness studies | 1+1D to 3+1D flat sprinkling, boost invariants, interval histograms, BDG references and new normalized abundance profiles | Held-out geometry discrimination and density/coarse-graining stability beyond scalar dimension recovery, with exact independent reference |
| GR / metric engineering | Einstein Toolkit / SpECTRE class infrastructure: nonlinear evolution, matter coupling, gauge control and standardized analytic/benchmark solutions | Metric kernel with analytic Minkowski/Schwarzschild references and geodesic invariants; GravCraft remains speculative | Full metric-derived connection/curvature, constraint propagation, convergence and Einstein-implied stress-energy/boundary consistency before any engineered-metric claim |
| Scientific discovery | Bayesian experimental design, model selection, factor graphs, independent reproducible numerical pipelines | Science-evidence receipts and a scalar discrete-model discrimination kernel | Active design that values expected discrimination, open-world alternatives, nuisance parameters, cost, falsifiability and sealed holdouts |

The frontier is not to claim a larger simulator than an industry or academic platform. The near-term competitive proposition is **epistemic integrity as a measurable capability**: expose unobservable directions, calibration dependence, phase-vs-signal ambiguity, false confidence, numerical error and resource cost. A claim of performance superiority still requires a pinned baseline, matched information and compute, held-out data and independent reproduction.

## Immediate next actions

1. Run `cargo fmt --check` and targeted `cargo test -p ...` for all new
   crates with the pinned Rust toolchain. Prior exact-head CI confirmed the
   causal-set and Chain Theory packages before the latest interval-histogram,
   BDG and arbitrary-state additions; the new exact-head run must validate
   those additions. No skipped or queued check is treated as a pass.
2. Run the Chain Theory audit example, extract the emitted CSV section with
   `sed -n '/^record,construction,slot_kinds/,$p'`, and diff all 172 rows
   against the author's commit-pinned CSV.
3. Run the causal-set 1+1D and N-dimensional examples; add sampled-coordinate
   distribution tests and scale the N log N implementation against brute force.
4. Compare the exact Floquet graph engine with an independent dense matrix
   Floquet operator for N<=9. Run
   `cargo run -p symthaea-fractal-time-lab --example initial_state_ensemble`
   to preserve the seeded all-up/down, defect, random-product and Haar-like
   state sweep under J=0 and interacting controls. Then extend to disorder,
   heating and finite-size scaling; report state-class distributions rather
   than selecting a best initial state.
5. Run the new GravCraft ADM reference: verify zero-shift Minkowski,
   Lorentzian inertia, full 4D geodesic norm conservation and Einstein-implied
   source diagnostics. Then add ADM projection checks, derivative-step
   convergence and independent symbolic comparison. Keep the legacy
   shift-as-acceleration simulator labelled as a heuristic toy.
6. Extend UAP triangulation into clock synchronization, pose/boresight
   covariance, platform-motion/parallax estimation, asynchronous multi-sensor
   track association and sealed known-object challenges.
7. Bind each run to the shared science-evidence manifest, exact-head logs,
   input/output digests and explicit claim boundary.
8. Preserve independent baselines, negative controls and failed results;
   no theory, benchmark or physical claim self-certifies.

## External research references checked 2026-11-07

- NASA UAP Independent Study Team (2023), Final Report: calibration, sensor
  metadata, synchronized multi-sensor observations and explicit evidence
  thresholds are prerequisites for reliable UAP analysis.
  https://science.nasa.gov/uap/
- NIST, Metrological Traceability: traceability requires an unbroken
  calibration chain with stated measurement uncertainty at every link.
  https://www.nist.gov/metrology/metrological-traceability
- Xu et al. (2025), Optimal Sensor Placement for Target Localization in IoT
  Systems: a CRLB-based overview of placement criteria and methods.
  https://doi.org/10.1109/MIOT.2025.3575317
- MathWorks Sensor Fusion and Tracking Toolbox, trackOSPAMetric (R2026b):
  documents frame OSPA and cumulative OSPA(2), including its windowed
  time-aggregation formulation, custom weights and fixed track-history
  comparison.
  https://www.mathworks.com/help/fusion/ref/trackospametric-system-object.html
- García-Fernández et al. (2026), GOSPA and T-GOSPA Quasi-Metrics for
  Evaluation of Multiobject Tracking Algorithms, IEEE Transactions on
  Aerospace and Electronic Systems 62, 9631-9644. T-GOSPA includes an
  explicit track-switch cost alongside localization and cardinality terms.
  https://doi.org/10.1109/TAES.2026.3686336
- Switzer et al. (2026), Realization of two-dimensional discrete time crystals
  with anisotropic Heisenberg coupling, Nature Communications 17, 605.
  https://doi.org/10.1038/s41467-025-67787-1
- Shinjo et al. (2026), Unveiling clean two-dimensional discrete time crystals
  on a digital quantum computer, npj Quantum Information 12, 41.
  https://www.nature.com/articles/s41534-026-01193-3
- Adamson and Wallden (2026), Benincasa-Dowker-Glaser causal set actions by
  quantum counting, Physical Review Research 8, 023188.
  https://doi.org/10.1103/qdst-yrh6
- Mitchell (25 October 2026), independent Chain Theory audit:
  arithmetic reproduction, census of the Letter's listed alternatives,
  photon/charge-assignment consistency and preregistration methodology.
  https://github.com/SproutSeeds/chain-theory-audit/tree/95cd02da5955450397a848c25f8287c1a037d9f2
- Wissner-Gross (23 October 2026), *A Parameter-Free Candidate for the
  Fine-Structure Constant from Entangled Qudits*: the arithmetic result and
  separate photon conjecture.
  https://github.com/physical-superintelligence/chain-theory

- NIST CODATA (2022 adjustment; published 2025), Recommended Values of the
  Fundamental Physical Constants.
  https://physics.nist.gov/constants

- Einstein Toolkit Hypatia release (10 July 2026): GPU-accelerated
  BSSNOK/Z4c evolution with 4th-order stencils, matter support, and gauge/linear
  wave initial-data thorns; also includes exact Kerr and Teukolsky initial data.
  https://www.einsteintoolkit.org/about/releases/ET_2026_05_announcement.html
- Einstein Toolkit Hypatia gallery tests include binary black holes, scalar
  waves, stable neutron stars and binary neutron stars.
  https://docs.einsteintoolkit.org/et-docs/Release_Details

## 2026-11-08 research and execution update

### New Observatory measurement primitives

The UAP Observatory now contains two separate measurement layers:

- `observability::analyze_jacobian`: column-scaled, data-only numerical-rank diagnostics. The implementation now uses a direct one-sided Jacobi SVD instead of forming `J^T J`, avoiding condition-number squaring. It returns singular values, nullity, condition number and weak right-singular directions. It accepts no prior matrix, so a prior cannot hide a data-only null direction. It remains a small dense kernel capped at 64 parameter columns; it is not a factor-graph solver or a global-identifiability proof.
- `frequency_locking::assess_frequency_locking`: finite-sweep spectral peak and contiguous detuning-lock diagnostics. It reports both FFT grid spacing and observation-limited resolution. A trace cannot be marked locked when the requested tolerance is narrower than the observation duration supports. This remains a signal-level test, not a DTC phase classifier.

### Robust sensor placement: make the test discriminating

The sensor-placement example now includes a four-hypothesis target region and a deliberately chosen rank-reversal fixture. The nominal single-point D-optimal winner is `candidate-high`, whereas the prior-weighted finite-region expected-logdet winner is `candidate-up-far`. The expected-logdet ranking is not directly numerically comparable to the single-point logdet because it aggregates a different objective; the meaningful result is the changed winner.

This is still a finite collection of local Fisher-information calculations. It does not include continuous-region guarantees, platform pose/calibration covariance, occlusion, cost, correlated sensor errors or time-varying target dynamics. Those omissions motivate the active-design workstream rather than a stronger claim.

### New active experimental-design workstream

[UAP-OED-001 #11050](https://github.com/Luminous-Dynamics/symthaea/issues/11050) defines budgeted multifidelity experiment design under observability constraints. It is grounded in active prior art:
- Ramon et al., *Multifidelity Sensor Placement in Bayesian State Estimation Problems*, Applied Mathematics for Modern Challenges (2026-07-15), DOI: 10.3934/ammc.2026006 / arXiv:2602.07269. It uses budget-constrained D-optimal design, fidelity/cost tradeoffs, marginal-gain-per-cost greedy selection and rank-one updates.
- Maio et al., *Multi-type Sensor Placement for PDE-based Bayesian Inverse Problems*, arXiv:2607.08074 (2026-07-09), uses expected information gain with multiple sensor types and knapsack constraints.
- Wu et al., *Sensor calibration for autonomous driving: A survey*, Measurement (2026), DOI: 10.1016/j.measurement.2026.122492, reviews spatiotemporal calibration, online adaptation and V2X cooperative calibration.
- Wang et al., *Bio-Inspired Observability Enhancement Method for UAV Target Localization and Sensor Bias Estimation with Bearing-Only Measurement*, Biomimetics (2025), DOI: 10.3390/biomimetics10050336, and Pérez & Jauffret, *Observability in Target Motion Analysis with Biased Bearings*, JAIF (2025), make observer excitation and measurement-bias identifiability central.

The differentiating Symthaea hypothesis is not that greedy sensor placement is new. It is whether finite-region robustness, calibration nuisance variables, data-only rank/abstention, multifidelity cost, active platform maneuvers and reproducible evidence can be evaluated in one matched benchmark. Small cases must be checked against exhaustive subset enumeration; no general approximation guarantee is assumed for worst-case or nonlinear scenario objectives.

The next objective correction is specified by [UAP-OED-002 #11235](https://github.com/Luminous-Dynamics/symthaea/issues/11235) and implemented in `symthaea-uap-observatory::budgeted_design`: partition the state into target parameters of interest and calibration nuisance parameters, optimize the target marginal information gain through a Schur complement, and report full joint-state information gain separately. The Cholesky solve must fail closed on singular nuisance precision; no pseudo-inverse may conceal a gauge or unobservable calibration parameter.

This distinction is operationally important. A clock-calibration observation can have positive joint-state information but zero target information when it is uncoupled. It can legitimately improve target precision when target and clock are correlated through the prior or measurement model. The design example now declares target XYZ as parameters of interest and a shared clock offset as a nuisance parameter. The synthetic tests include the no-nuisance limit, nuisance-only control, coupled calibration benefit, correlated-prior reference, shared-clock versus fictitiously independent clocks, and a determinant-identity check.

A shared latent clock/pose state is **not** the same as correlated residual noise. The current additive `J^T J` interface is valid only after correct joint whitening, or where shared errors are represented explicitly as latent nuisance parameters. Arbitrary cross-sensor residual covariance requires selected-set joint whitening and remains a separate implementation gate. See [the target-marginal design specification](uap-target-marginal-experiment-design-v0.1.md).

The next layer, [UAP-OED-003 #11237](https://github.com/Luminous-Dynamics/symthaea/issues/11237), keeps shared latent nuisance states distinct from correlated residual noise. The new `correlated_design` module accepts a fixed observation pool, full raw Jacobian and full residual covariance. Each selected action set extracts its covariance principal submatrix and jointly whitens the selected rows before computing data-only rank, target Schur EIG and joint EIG. The diagonal limit must match the independent-row API exactly; common-mode test cases must show that repeated correlated observations do not receive fictitious independent information. This dense reference is capped at 512 pool observations and is not yet a high-performance sparse selector. See [the correlated residual design specification](uap-correlated-residual-design-v0.1.md).

### Chain Theory differential reproduction

The Chain Theory audit now vendors the author's record CSV from the author's repository commit `a28db8998bce5bd5ef11195c05f13c86cd9dac5c`, including Apache-2.0 provenance, and the CTGP workflow runs the independent Rust example and diffs all 172 record rows against the pinned source. This moves the work from testing counts and internal classification bands to byte-for-byte reproduction of the published record census. It still reproduces the stated calculation only; it does not validate the photon conjecture or a physical derivation of QED.

### Time-crystal comparison target

The current research target should be the 2D anisotropic-Heisenberg experimental benchmark, not just 1D kicked-Ising period doubling:
- Switzer et al., *Realization of two-dimensional discrete time crystals with anisotropic Heisenberg coupling*, Nature Communications 17, 605 (2026-01-28), DOI: 10.1038/s41467-025-67787-1. The work combines IBM quantum hardware and tensor-network methods, explores a 2D anisotropic model, maps ergodic/localized/time-crystalline regimes, and studies initial-state, interaction-anisotropy and drive dependence.
- *Discrete time crystals enabled by Floquet strong Hilbert space fragmentation*, Physical Review B 113, 214303 (2026-06-01), DOI: 10.1103/338m-mvsg, studies disorder-free fragmentation, multiple-period beating in small systems, lifetime scaling and finite-size diagnostics.

The frequency-locking primitive deliberately cannot promote a spectral peak to a phase claim. The next DTC increment should add an independent exact small-system Heisenberg comparator, more initial-state families, a predeclared frequency-locking phase map, finite-size lifetime predictions and convergence-controlled MPS/TEBD comparison. Hardware-sized fidelity claims remain out of scope for exact state vectors capped at 20 spins.

### Exact-head validation boundary

The full CTGP matrix and lockfile checks passed on `03dce1aab380b7493bf91d0a46cd5da076c84d92`: all seven focused packages passed formatting, library tests and example compilation; the UAP placement example and anisotropic-Heisenberg sweep also ran; the committed lockfile passed `cargo metadata --locked` and all six focused library tests. Later commits add the direct SVD, frequency-resolution guard, rank-reversal fixture, author CSV differential check and active-design documentation. Those later changes require a new exact-head matrix; older success does not qualify them.

## 2026-11-08 extension: finite-volume causal-set link sampling

New workstream: [CAUSAL-LINK-BIAS-001 #11167](https://github.com/Luminous-Dynamics/symthaea/issues/11167), motivated by Boguñá & Krioukov, *Sampling biases in sprinkled causal sets* (arXiv:2610.03247, 2026-10-02).

The paper shows that a root-centred finite constant-time cone can bias the normalized proper-time distribution of links in spatial dimension d>=2, because the available hyperbolic rapidity volume depends on proper time. This is a regulator/exhaustion bias, not simply a finite-event-count error. The sigma-finite Lorentz-invariant intensity per unit hyperbolic volume and the probability distribution of a link chosen from a finite sampling window are distinct mathematical objects.

The causal-set crate now has a first analytical reference module, `link_sampling_bias`, with:
- dimensionless link-intensity profile `q_d(tau_hat)=(d+1) tau_hat^d exp(-tau_hat^(d+1))`;
- hyperbolic-ball volume and log-volume for spatial dimensions 1, 2 and 3;
- finite-cone normalized proper-time profile;
- product-window normalized profile;
- the paper's double-scaling cutoffs and a declared finite root-admissibility diagnostic.

An executable report is registered as `link_sampling_bias_reference` and run by the CTGP causal-set workflow. It is an analytic formula reproduction, not yet a finite sprinkling experiment or a causal-set quantum-dynamics implementation.

**Independent reproduction discrepancy retained:** the first independent quadrature for the printed finite-cone equations does not match the finite-cone means transcribed from Table I. The product-window means are close to the printed values, but that does not validate the finite-cone branch. The report deliberately prints both values and their deltas and carries the status `FINITE_CONE_TABLE_I_DISCREPANCY_REQUIRES_CONVENTION_AUDIT`. The root location, time cutoff convention, hyperbolic angular multiplicity, density normalization and table-column interpretation must be reconciled before these values can become golden regression fixtures. No coefficient has been adjusted to force agreement.

Next gates:
1. reproduce the finite-cone normalization and mean using two independent quadrature paths and the source paper's exact convention;
2. reconcile the Table I finite-cone mean link degrees and proper-time means, including the root placement and cutoff definition;
3. implement an independent event-centred future-cone generator and exact open-interval link test, keeping fixed-cardinality and Poisson semantics separate;
4. compare finite-cone and product-window histograms under the paper's double-scaling conditions with confidence intervals;
5. only then connect these corrected link statistics to the existing link-matrix/path-sum propagator workstream (#11054).

The claim ceiling remains finite-order-statistic and sampling-estimator reproduction. This does not establish a continuum spacetime limit, quantum dynamics, or quantum gravity.

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
