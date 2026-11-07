# Exact Floquet kicked-Ising reference results v0.2

Date: 2026-11-07
Status: independent Python state-vector reference; Rust compilation and tests pending.

## Model

The exact Floquet module now supports arbitrary simple interaction graphs
with unitary step

U_F = exp[-i(sum_(i,j) J_ij Z_i Z_j + sum_i h_i Z_i)]
      exp[-i(theta/2) sum_i X_i].

The 1D chain constructor is retained, and a rectangular open-boundary square
lattice constructor provides horizontal and vertical couplings. The full
complex state vector is evolved exactly up to f64 arithmetic, with a size cap
of 20 spins. This is an exact small-system oracle, not an MPS or thermodynamic
limit solver.

The finite-window response metric computes q_n=(-1)^n m_z(n), then reports:
- mean demodulated order: mean(q_n);
- mean absolute order: mean(abs(q_n));
- sign coherence: abs(mean(q_n))/mean(abs(q_n)).

These are signal statistics. They are not a DTC order parameter proof,
thermodynamic-limit phase detector or substitute for phase-rigidity and
finite-size scaling studies.

## Independent 1D finite-size pulse-error sweep

A separate Python complex-state-vector implementation independently evaluated
the same factorized Floquet unitary. The initial state is all-up, longitudinal
fields are zero, the pulse is theta=pi+epsilon, and the table reports
mean demodulated order over periods 20 through 100 (81 samples).

| Spins | J | epsilon=0.05 | epsilon=0.10 | epsilon=0.20 |
|---:|---:|---:|---:|---:|
| 4 | 0.0 | -0.4394 | -0.1870 | 0.1012 |
| 4 | 0.5 | 0.9971 | 0.9895 | 0.9561 |
| 4 | 1.5 | 0.9806 | 0.9226 | 0.7634 |
| 6 | 0.0 | -0.4394 | -0.1870 | 0.1012 |
| 6 | 0.5 | 0.9972 | 0.9872 | 0.9537 |
| 6 | 1.5 | 0.9600 | 0.8488 | 0.5215 |
| 8 | 0.0 | -0.4394 | -0.1870 | 0.1012 |
| 8 | 0.5 | 0.9972 | 0.9864 | 0.9492 |
| 8 | 1.5 | 0.9460 | 0.7991 | 0.3357 |
| 10 | 0.0 | -0.4394 | -0.1870 | 0.1012 |
| 10 | 0.5 | 0.9972 | 0.9855 | 0.9404 |
| 10 | 1.5 | 0.9377 | 0.7277 | 0.2666 |
| 12 | 0.0 | -0.4394 | -0.1870 | 0.1012 |
| 12 | 0.5 | 0.9972 | 0.9846 | 0.9340 |
| 12 | 1.5 | 0.9323 | 0.6695 | 0.2097 |

This shows that the finite-time response is strongly coupling dependent for
this initial state and parameter family. It also shows that larger coupling
is not monotonically better: the J=1.5 response degrades with size and pulse
error. The zero-coupling control is essential because an ideal pi pulse gives
a two-cycle even without interactions.

## Independent 2D square-lattice sweep

The graph implementation was independently checked on open-boundary square
lattices of 2x2, 2x3 and 3x3 spins. For each lattice, all-up initial state,
zero longitudinal fields, equal horizontal/vertical coupling J, 300 periods,
and the response window 20..300 were used.

Mean demodulated order for each theta=pi+epsilon:

| Lattice | J | epsilon=0.05 | epsilon=0.10 | epsilon=0.20 |
|---|---:|---:|---:|---:|
| 2x2 | 0.0 | -0.0140 | -0.0679 | 0.0052 |
| 2x2 | 0.5 | 0.9987 | 0.9947 | 0.9771 |
| 2x2 | 1.5 | 0.9442 | 0.0910 | 0.0011 |
| 2x3 | 0.0 | -0.0140 | -0.0679 | 0.0052 |
| 2x3 | 0.5 | 0.9988 | 0.9948 | 0.9733 |
| 2x3 | 1.5 | 0.9737 | 0.8968 | 0.6697 |
| 3x3 | 0.0 | -0.0140 | -0.0679 | 0.0052 |
| 3x3 | 0.5 | 0.9991 | 0.9964 | 0.9827 |
| 3x3 | 1.5 | 0.9851 | 0.9458 | 0.8299 |

For J=0.5, the sign-coherence statistic was 1.0 across these windows because
the demodulated signal retained a consistent sign. For J=0, coherence was
0.023, 0.106 and 0.008 at the three respective pulse errors. At J=1.5, the
2x2 case deteriorated sharply while the larger graphs remained more robust,
which is a useful finite-size / connectivity warning.

## Initial-state and lifetime controls

For an 8-spin open chain, J=0.5, epsilon=0.1 and 500 Floquet periods:

- all-up initial basis state: mean demodulated order over periods 20..500
  was 0.9873;
- all-down initial basis state: the same magnitude with opposite sign;
- a single-spin-flipped basis state: mean demodulated order was -0.0324;
- zero-interaction control: mean demodulated order was -0.0238.

For the all-up state, the minimum 20-period rolling mean absolute
demodulated response was 0.9852 over periods 20..500; it never crossed 0.5.
The zero-interaction response crossed below 0.5 early. This confirms that
the response depends strongly on the initially ordered state and interaction
regime. A full study must include ensembles of initial states and observables,
not only a favorable symmetry-broken product state.

The exact graph engine now also accepts explicit ZZ edges and creates
open-boundary square lattices. For the independent 2D sweep, the interaction
term preserves a high finite-window response across 2x2, 2x3 and 3x3 grids at
J=0.5 and pulse errors 0.05, 0.10 and 0.20. The zero-coupling controls have
near-zero mean demodulated order and low sign coherence. The smallest 2x2
grid at J=1.5 loses its response quickly under pulse error, demonstrating
that stronger coupling is not automatically better.

## Relation to the state of the art

Recent 2026 work demonstrated:
- two-dimensional DTC dynamics with anisotropic Heisenberg interactions using
  quantum hardware together with tensor-network simulation; and
- a clean 2D kicked-Ising DTC on a 133-qubit IBM Heron processor, tracking
  magnetization for up to 100 Floquet cycles.

Symthaea's exact graph engine now allows independent small-system checks for
a 2D kicked-Ising family. It does not compete with those hardware-scale
demonstrations in system size, noise modelling or experimental validation.
Its near-term value is exact finite-size reference evolution, auditable
unitarity, explicit zero-interaction controls, graph generality and
reproducible parameter sweeps.

## Next gates

1. Run the Rust unit tests and both exact Floquet examples with the pinned
   toolchain; current Python values are not Rust test evidence.
2. Cross-check the exact 2D graph Floquet unitary against an independent
   matrix/exact-diagonalization implementation for N<=9.
3. Sweep more lattice sizes, couplings, drive errors, longitudinal fields,
   coupling disorder and initial-state ensembles.
4. Report heating / lifetime envelopes and confidence intervals over disorder
   realizations; separate prethermal finite-time order from asymptotic DTC
   claims.
5. Compare 2D kicked-Ising observables with the published IBM / tensor-network
   benchmark before implementing anisotropic Heisenberg dynamics.
6. Only after exact references, add an MPS/TEBD backend and compare it against
   TeNPy, quimb or ITensor with convergence and truncation error recorded.

## Scientific non-claims

These calculations do not demonstrate a thermodynamic DTC phase, a quantum
hardware implementation, sensor advantage, or new fundamental physics.
They are exact finite-graph state-vector experiments that expose which
initial states and interaction regimes generate robust finite-time
subharmonic responses under the chosen drive.
