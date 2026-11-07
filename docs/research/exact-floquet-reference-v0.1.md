# Exact Floquet reference v0.1

Date: 2026-11-06  
Status: executable model added; Rust compilation and tests remain unverified in this environment.

## Model

The new module `symthaea-fractal-time-lab::exact_floquet` implements exact
finite-dimensional state-vector evolution for a finite, open-boundary kicked
Ising chain:

[
U_F =
\exp\left[-i\left(\sum_{i=1}^{L-1}J_i Z_i Z_{i+1}
+\sum_{i=1}^{L} h_i Z_i\right)\right]
\exp\left[-i\frac{\theta}{2}\sum_{i=1}^{L}X_i\right].
]

Units use \(\hbar=1\) and one Floquet period per step. The model applies a
factorized global x-rotation and an exactly diagonal Ising/field phase to a
complex state vector of dimension \(2^L\). It is exact up to f64 floating
point arithmetic; it is not an MPS approximation.

The current size cap is 20 spins, making this a small-system oracle rather
than a large-system production solver.

## Validation included

- Norm conservation over repeated unitary steps.
- A four-spin ideal-pi-pulse reference trace.
- A two-spin, one-period amplitude comparison against an explicitly written
  closed-form result.
- A one-spin imperfect-pulse negative control that exhibits phase drift
  rather than interaction-mediated locking.
- Rejection of invalid size, coupling count and non-finite parameters.

The example compares interacting and noninteracting finite chains and prints
the complete stroboscopic magnetization and norm.

## Essential scientific boundary

An ideal pi pulse produces period-two magnetization even with zero
interactions. This is a driven two-cycle, not evidence of a discrete time
crystal. The new module provides a quantum reference simulator so future
experiments can ask whether interactions, disorder and the drive protocol
stabilize a subharmonic response against pulse errors and perturbations.

Before any DTC phase claim, the benchmark must include:
- Interaction-off and disorder-off controls.
- Drive-angle error sweeps and phase rigidity.
- Multiple initial states and observables.
- Finite-size scaling across spin counts.
- Lifetime / heating-rate characterization.
- Noise, decoherence, sampling and readout models.
- Comparison to exact diagonalization for small systems and to an
  independently validated tensor-network implementation at larger sizes.

## Independent reference checks

A Python state-vector implementation of the same factorized unitary was
checked against the following invariants:
- A 6-spin interacting chain retained norm squared 1 to approximately
  1e-14 after 32 Floquet periods.
- A 4-spin zero-interaction, ideal-pi-pulse chain produced the expected
  alternating \(+1,-1\) magnetization trace.
- A 1-spin pulse-angle error of 0.12 radians produced a demodulated response
  drift of approximately 0.862 over 12 periods, as expected without
  interaction-mediated locking.

These are independent mathematical reference calculations, not Rust test
results. Rust compile and test status is pending verification on the pinned
toolchain.


## v0.2 graph extension and finite-window diagnostics

The exact state-vector engine has been generalized from a 1D open chain to
an arbitrary simple undirected ZZ interaction graph, with duplicate/self
edges and out-of-range endpoints rejected. A square-lattice constructor
creates open-boundary 2D grids with independently configurable horizontal
and vertical couplings.

The module now exposes a finite-window subharmonic response summary:
mean demodulated order, mean absolute order and sign coherence. Those are
signal statistics, not a thermodynamic phase certificate. The zero-coupling
pi-pulse control is mandatory because a perfect pi pulse gives period-two
magnetization without interactions.

Two executable examples are available:
- exact_floquet_reference: compare an interacting and a noninteracting 1D chain.
- exact_floquet_phase_sweep: sweep chain size, coupling and pulse error.
- exact_floquet_2d_sweep: sweep 2x2, 2x3 and 3x3 square lattices.

Independent Python finite-size reference results are recorded in
docs/research/dtc-exact-floquet-reference-results-v0.2.md. They show a
strong interaction-dependent finite-time subharmonic response for ordered
product initial states, but also strong dependence on initial state and
coupling. The report does not claim an asymptotic DTC phase.

All Rust compilation and unit-test gates remain open until run on the
repository-pinned toolchain.
