# Full-Newton Backward Euler Qualification Protocol v0.1

**Status:** implementation proposal, not solver qualification. The added Rust tests are authored but have not been compiled or executed here. The candidate must remain unqualified until exact-head build/test evidence and the benchmark ladder below are recorded.

## Motivation: coupled stiff systems require coupled corrections

For backward Euler, each time step solves the nonlinear system

\[
G(z) = z-y_n-h f(t_{n+1}, z) = 0.
\]

Its Newton matrix is

\[
J_G(z) = I - h\,J_f(t_{n+1},z).
\]

The previous compatibility implementation approximates only diagonal derivatives and applies component-wise corrections. That loses off-diagonal coupling. The new fallible API estimates the full dense Jacobian by finite differences, solves the Newton linear system with partial pivoting, and commits a time step only after a normalized residual check passes.

This is not a claim that backward Euler is accurate at arbitrary step sizes. L-stability damps stiff modes; it does not remove discretization error, fix a bad model, or guarantee Newton convergence.

## Failure and resource contract

The API `try_solve_implicit_euler` returns a trajectory plus diagnostics only when every step satisfies its nonlinear residual threshold. Otherwise it returns an `ImplicitEulerError` containing the failure class, last accepted state/time, completed-step count, RHS evaluation count, Newton work, and residual information. A failed trial step is never appended to the output trajectory.

Current explicit limits:
- 1–100,000 requested steps.
- 1–100 Newton corrections per step.
- At most 64 state variables for this dense implementation.
- At most 1,000,000 RHS evaluations.
- At most 100,000,000 estimated dense work units, counted as dimension cubed per Newton correction.

These are containment limits for a small dense solver, not performance claims. Larger systems need a separate sparse Jacobian / sparse linear solve interface rather than silently exceeding the intended envelope.

The normalized residual is the infinity norm of component residuals scaled by (1 + \max(|z_i|,|y_{n,i}|)). The acceptance threshold is (10^{-10}). This threshold is a nonlinear-solve criterion, not a global integration error bound.

## Research references

- Boost.Odeint's implicit-system concept states that implicit routines require both the ODE right-hand side and a Jacobian and must solve a linear problem: https://www.boost.org/doc/libs/latest/libs/numeric/odeint/doc/html/boost_numeric_odeint/concepts/implicit_system.html
- SUNDIALS CVODE documents Newton systems of the form (I - \gamma J), Jacobian construction, and solver setup/convergence handling: https://sundials.readthedocs.io/en/v7.4.0/cvode/Usage/index.html
- SciPy recommends implicit Radau/BDF methods for stiff problems and notes that explicit RK methods are generally appropriate for non-stiff systems: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

Symthaea currently has a dense backward-Euler path, not a CVODE-, BDF-, or Radau-equivalent solver. No equivalence or performance parity is claimed.

## Qualification ladder

1. **Analytical linear algebra.** Verify full-Jacobian Newton on coupled linear systems with nonzero off-diagonal terms. Compare each result against the exact backward-Euler linear system, not merely against another run of the same solver.
2. **Stiff scalar reference.** For (y'=-\lambda(y-1)), test the exact backward-Euler recurrence and exact continuous solution separately. Report recurrence residual and continuous-time discretization error as distinct quantities.
3. **Nonlinear failure.** Force insufficient Newton iterations and assert typed non-convergence, unchanged last-accepted state, no committed failed step, and an accurately identified failed step.
4. **Jacobian quality.** Compare finite-difference Jacobians with analytical Jacobians for seeded linear and nonlinear systems. Record component-wise relative error and the effect of state scaling.
5. **Convergence order.** On smooth non-stiff problems, halve the time step and measure global error. Backward Euler should exhibit first-order global convergence in its asymptotic regime.
6. **Stiffness ladder.** Sweep decay rates over at least (10^0) to (10^8), step sizes beyond the explicit Euler stability range, coupled matrices with complex eigenvalues, and systems with disparate state magnitudes. Report failures; do not equate finite output with accuracy.
7. **Adversarial contracts.** Exercise bad time spans, zero/oversized step budgets, zero/oversized Newton budgets, dimension mismatch, non-finite initial states, malformed/non-finite RHS values, singular/ill-conditioned Newton matrices, and work/evaluation budget exhaustion.
8. **Cost reporting.** Report RHS evaluations, Newton corrections, estimated dense work, residual per accepted step, wall-time distribution, continuous reference error, and all typed failure counts.

## Proposed release gates

- 100% of invalid-contract cases produce the expected error class.
- 0 non-converged Newton steps are committed to a successful trajectory.
- 0 non-finite states escape through a successful report.
- 100% of successful steps satisfy the declared normalized residual threshold.
- Coupled linear reference systems agree with their exact backward-Euler solutions within problem-specific floating-point tolerances.
- Smooth-problem refinement exhibits the expected first-order trend before a formal order claim.
- No runtime or accuracy claims without paired measurements on the same toolchain and hardware.

## Evidence ledger

| Exact candidate SHA | Rust build/tests | Analytic references | Stiffness ladder | Status |
| --- | --- | --- | --- | --- |
| Pending | Not run here | Not run here | Not run here | Unqualified |
