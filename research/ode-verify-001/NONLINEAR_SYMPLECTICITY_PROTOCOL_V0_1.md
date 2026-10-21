# GEOM-ODE-002 — Nonlinear Hamiltonian Symplecticity v0.1

**Status:** test and protocol authored; no Rust compilation/test result is claimed by the authoring environment. The tests are intended to run on the existing ODE Verification 001 focused workflow.

## Research question

Does the implemented fixed-step Verlet map preserve the canonical symplectic form beyond the one-degree-of-freedom linear oscillator?

The preceding GEOM-ODE-001 slice checks the determinant criterion for a two-dimensional linear oscillator. This follow-on tests a nonlinear, two-degree-of-freedom separable Hamiltonian. It is deliberately distinct from Symthaea PR #8177, which formally checks the encoded polynomial identity for conservation of the Hénon–Heiles Hamiltonian; here we test the *numerical time-step map*, not the equation's symbolic invariant.

## Frozen model and state layout

Use the canonical Hénon–Heiles Hamiltonian with unit masses:

`H = 1/2 (p_x^2 + p_y^2 + x^2 + y^2) + x^2 y - y^3/3`.

Canonical state order for the numerical map is `[x, y, p_x, p_y]`. The equations are:

`x' = p_x`, `y' = p_y`,
`p_x' = -x - 2xy`,
`p_y' = -y - x^2 + y^2`.

This matches the position/velocity layout assumed by the existing Verlet implementation for unit masses. It is a controlled numerical test, not a claim that this model describes a particular experimental system.

## Measured map property

At a fixed initial state and timestep, approximate the Jacobian `M = d Phi_h / dz` of the actual one-step map using centered finite differences with frozen `epsilon = 1e-5`. Measure the max-entry defect:

`D = max_ij |(M^T J M - J)_ij|`,

where the canonical symplectic matrix in `[q_1, q_2, p_1, p_2]` order is

`J = [[0,0,1,0], [0,0,0,1], [-1,0,0,0], [0,-1,0,0]]`.

The numerical Jacobian is a diagnostic, not a formal proof. Its finite-difference epsilon and norm are part of the benchmark contract and must not be silently changed to turn a failing implementation green.

## Acceptance and negative controls

The executable fixture freezes three initial states:

- `[0.12, 0.20, 0.05, -0.10]`
- `[0.20, -0.10, 0.12, 0.03]`
- `[-0.15, 0.08, -0.04, 0.11]`

and timesteps `0.2, 0.4, 0.6`.

- **Verlet:** every one of the nine state/timestep pairs must have `D <= 2e-7`.
- **RK4 negative control:** on the first state with `dt = 0.5`, require `D > 1e-4`.

The RK4 negative control is essential: a test harness that only checks the candidate solver's output could accidentally measure the wrong matrix convention or a defect insensitive to this fixture. The threshold separation was checked in an independent small arithmetic prototype; that prototype is not Rust execution or repository validation.

## Interpretation boundaries

A passing test supports only the narrow statement that this implementation's map is numerically consistent with the canonical symplectic condition over this frozen nonlinear fixture and finite-difference diagnostic. It does not prove:

- exact symplecticity for all states, timesteps, or Hamiltonian systems;
- long-horizon bounded energy error (that remains a separate test);
- correct symbolic derivation of the physical model;
- empirical model validity, uncertainty calibration, engineering safety, or SOTA superiority.

The existing Hénon–Heiles symbolic conservation proof and these numerical-map tests should be reported as different evidence classes, never collapsed into one generic "physics verified" badge.

## Next research gates

1. Sweep a preregistered initial-condition and timestep grid, including near-boundary states where the Hamiltonian remains within a declared bounded region.
2. Replace finite-difference Jacobians with an independently derived tangent-map/Jacobian oracle for a subset of cases; compare both methods.
3. Add phase error, Hamiltonian drift, reversibility and wall-clock/RHS-evaluation cost to a matched-error benchmark for Verlet, RK4 and appropriate symplectic/non-symplectic comparators.
4. Keep failures and unresolved cases; report median, P95 and maximum structural defect.
5. Do not compare physical fidelity or solver speed until both implementations are bound to identical model semantics and independently qualified references.

## Research anchors

- Hairer, Lubich, and Wanner, [Geometric Numerical Integration illustrated by the Störmer–Verlet method](https://doi.org/10.1017/S0962492902000144), Acta Numerica (2003).
- Symthaea [PR #8177](https://github.com/Luminous-Dynamics/symthaea/pull/8177), the separate exact-rational Hénon–Heiles polynomial conservation obligation.
- NASA [NASA-STD-7009B](https://standards.nasa.gov/standard/nasa/nasa-std-7009) and its [2026 handbook](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009), for the broader distinction between verification evidence, model validation, intended use, and simulation credibility.

These anchors inform methodology. They do not independently validate the code or establish a performance claim.
