# GEOM-ODE-001 — Phase-Space Geometry and Reversibility Protocol v0.1

**Status:** proposed numerical-verification increment. Source tests exercise Symthaea's production integrator through the isolated ODE harness; no local Rust toolchain is available in the authoring environment, so compilation and runtime results are not claimed here.

## Motivation

The ODE verification ladder already covers invalid request contracts, analytical accuracy, adaptive-step acceptance, and long-horizon energy/momentum/angular-momentum drift. Those gates are necessary but do not directly test the defining geometric properties of a Hamiltonian time integrator.

For conservative dynamics, a plausible trajectory and bounded energy error are not enough to establish that a method preserves the relevant phase-space structure. Symplectic integration is motivated by geometric preservation over long times; non-symplectic methods may still be excellent choices at a matched accuracy or compute budget. The benchmark must measure these properties separately rather than rank all methods by a single score.

## Bounded test model

Use the autonomous unit harmonic oscillator in canonical state order `[q, v]`:

`q' = v`, `v' = -q`.

The equations and exact linear flow are analytic. The first slice tests the production one-step maps at `dt = 1` and does not invoke HDC trajectory encoding, scene rendering, or a physical measurement pipeline.

## Acceptance checks

### 1. One-step phase-space area

For this linear two-dimensional state map, evaluate the integrator at the origin and the two basis vectors. These three outputs recover the affine map matrix (up to floating-point rounding) without choosing a finite-difference epsilon.

For a one-degree-of-freedom canonical linear map, symplecticity is equivalent to the matrix determinant being one.

- **Velocity-Verlet:** require `abs(det(M) - 1) <= 1e-12` for the frozen oscillator and step.
- **RK4 negative control:** require `abs(det(M) - 1) > 1e-3` at `dt = 1`. The analytic RK4 map has determinant `569/576`, so this test confirms the harness would not mistake a generic high-order method for a symplectic map.

This is a narrow linear, one-degree-of-freedom test. It is not a proof of symplecticity for all states, arbitrary Hamiltonians, constrained systems, or higher dimensions.

### 2. Time-reversal round trip

Apply the fixed-step map with `+dt`, then apply it to the resulting state with `-dt`. On the autonomous oscillator:

- **Velocity-Verlet:** maximum absolute state error after the round trip must be `<= 1e-12`.
- **RK4 negative control:** with `dt = 1` from `[1, 0]`, round-trip error must exceed `1e-3`.

The test directly probes the one-step algebra. It intentionally does not route negative time through `try_simulate`, whose checked public contract requires a forward, non-reversed interval.

### 3. Keep properties distinct

Do not interpret one passed property as another:

- Phase-space area preservation is not exact physical-energy conservation.
- Time reversibility is not sufficient for accuracy.
- Energy drift is not model validity.
- Numerical verification is not experimental validation or engineering qualification.
- A negative control passing proves only that this fixture separates these particular discrete maps.

The existing long-horizon orbital tests remain responsible for energy, linear-momentum, angular-momentum, and timestep-refinement diagnostics.

## Next benchmark stages

Only extend after the first stage passes on the exact source head:

1. Test a nonlinear Hamiltonian (for example, a declared Hénon–Heiles case) against a separately implemented Jacobian-based `M^T J M = J` diagnostic over a frozen grid of initial states and step sizes.
2. Sweep timestep and initial-condition families; retain maximum/median/P95 defects, not only a single aggregate.
3. Compare Verlet and RK methods at matched achieved state error and RHS-evaluation budgets, reporting the work–precision frontier and regressions alongside wins.
4. Add an independent implementation or reference calculator before treating results as cross-implementation corroboration.
5. Admit experimental datasets only as a separate physical-validation stage with provenance, measurement uncertainty, and an explicit applicability envelope.

## Preregistered claim ceiling

This slice can support only: *the tested one-step implementation preserves the oscillator's canonical area and reverses this frozen step within the stated floating-point tolerances, while the RK4 negative controls are detected.*

It cannot support a universal solver ranking, physical-law discovery, measured physical fidelity, real-world safety, or an SOTA claim. Runtime, allocation, and work–precision improvements require repeated measurements on a declared machine and matched-error cases. No performance gain has been measured by this protocol.

## Literature and practice

- Hairer, Lubich, and Wanner, [Geometric Numerical Integration illustrated by the Störmer–Verlet method](https://doi.org/10.1017/S0962492902000144), Acta Numerica (2003): geometric structure and long-time behavior.
- NASA Glenn Research Center, [Verification Assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html): analytical/reference solutions, consistency, and convergence studies.
- Roache, [Code Verification by the Method of Manufactured Solutions](https://doi.org/10.1115/1.1436090), Journal of Fluids Engineering (2002): use sufficiently rich known solutions to expose implementation errors.

These references motivate the design; they do not certify this repository or establish a head-to-head performance result.
