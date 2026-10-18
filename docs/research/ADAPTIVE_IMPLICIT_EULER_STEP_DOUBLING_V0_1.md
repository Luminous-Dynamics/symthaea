# Adaptive Implicit Euler by Step Doubling — Qualification Protocol v0.1

**Status:** proposed implementation and authored regressions. This is not a solver qualification. The Rust source has not been compiled or executed in the current environment. Exact-head evidence must be recorded before this implementation is described as validated.

## Research question

The full-Newton backward-Euler path distinguishes nonlinear convergence from a successful trajectory, but its fixed step size leaves users responsible for selecting a grid. A small Newton residual is not a measurement of time-discretization error.

This API introduces a second control loop: compare one full backward-Euler step with two half steps. The fine trajectory is accepted only when the normalized discrepancy falls within the caller's component-wise tolerance contract. If the error is too large, the macro-step is rejected and retried at a smaller time step. A retryable Newton-convergence or singular-Jacobian failure also reduces the step size. Malformed RHS shape, non-finite values and other structural failures stop immediately.

## Method and estimator

Backward Euler is first order. For each candidate interval `[t, t+h]`, compute:

- a single full step (Y_h);
- a half step to (Y_{h/2});
- a second half step to (Y_{h/2,h/2}).

Use the step-doubling discrepancy as a first-order local-error estimate:

\[
e_i \approx Y_{h/2,h/2,i} - Y_{h,i}.
\]

Its normalized RMS error is

\[
E = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left(
\frac{e_i}{\mathrm{atol}_i + \mathrm{rtol}\,\max(|y_i|,|Y_{h/2,h/2,i}|)}
\right)^2}.
\]

Accept the two-half-step endpoint when (E \leq 1). For a first-order method the controller uses an error exponent of (-1/2) with bounded growth/shrink factors. This estimator is an error-control signal, not a rigorous global-error bound; discontinuities, nonsmooth right-hand sides, ill-conditioned systems and inaccurate nonlinear solves require separate qualification.

## Distinct acceptance criteria

1. **Nonlinear-solve convergence:** each backward-Euler substep must satisfy the full-Newton residual contract.
2. **Local time-discretization control:** the normalized step-doubling discrepancy must not exceed one.
3. **Global accuracy:** the completed trajectory must be compared to analytic or independently computed references.

These criteria must never be collapsed into a single confidence score. In particular, a small residual for the implicit equation does not imply a small integration error.

## Resource and failure contract

- The macro-step attempt budget counts accepted and rejected attempts.
- RHS evaluations, Newton corrections and estimated dense work are accumulated across the full step and both half-step solves, including rejected attempts.
- The inner dense Newton solver remains limited to 64 state variables.
- Failed or rejected macro-steps do not advance the published trajectory.
- Failure receipts identify the last accepted time/state, accepted/rejected counts, attempt count, work counters and typed failure cause.
- The method remains first-order; high-accuracy or very stiff workloads should be compared with qualified variable-order BDF or Radau implementations.

The current global limits reuse the fixed solver's 1,000,000 RHS-evaluation and 100,000,000 estimated dense-work-unit limits. Work estimates count dimension-cubed units per completed Newton correction and are a proxy, not hardware-independent FLOP counts.

## Reference implementations

- SciPy's `solve_ivp` guidance recommends explicit Runge–Kutta methods for non-stiff problems and implicit Radau/BDF methods for stiff problems, and documents tolerance-based integration controls: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- SUNDIALS CVODE distinguishes nonlinear convergence failures, error-test failures, and excess-work failures. This separation motivates keeping the step-doubling error controller distinct from Newton residual validation: https://sundials.readthedocs.io/en/v7.0.0/cvode/Usage/index.html

The Symthaea implementation should not be described as reproducing SciPy, CVODE, BDF or Radau.

## Qualification ladder

1. **Analytic scalar systems:** (y'=y), (y'=-\lambda y), constant derivatives, decays near zero, and sign changes. Compare the adaptive trajectory with the exact continuous solution and with the exact backward-Euler recurrence separately.
2. **Tolerance response:** sweep `atol` and `rtol` over at least six orders of magnitude. Report global error, accepted/rejected steps, RHS evaluations and dense work. Tighter tolerance should generally improve error at higher cost on smooth reference problems, but do not require strictly monotone work for every adaptive case.
3. **Mixed scales:** combine state magnitudes spanning at least 24 orders of magnitude; rescale one state and its absolute tolerance by a unit conversion and compare normalized trajectories.
4. **Coupled systems:** non-diagonal linear systems with known backward-Euler solutions, plus nonlinear coupled problems. Compare full trajectory error and every accepted Newton residual.
5. **Newton failure recovery:** confirm singular or non-convergent full-step trials shrink the step without mutating the last accepted state. Confirm non-finite or malformed RHS values fail immediately.
6. **Attempt/work budgets:** exercise one-attempt exhaustion, high rejection fractions, RHS-evaluation and dense-work exhaustion. Every case must terminate via success or a typed error.
7. **Global convergence:** halving the target tolerance and measuring global error does not replace a time-step refinement study. For a first-order method, halve an imposed maximum step and measure the asymptotic global convergence trend.
8. **Independent references:** compare benchmark trajectories to analytic results and a separate stiff integrator (e.g. SciPy BDF/Radau). Pin solver settings, versions, tolerances, and exact candidate SHA.
9. **Reproducible evidence:** record compiler/toolchain, host hardware, seeds, accepted/rejected trials, RHS evaluations, Newton corrections, max residual, max accepted local estimator, global error and all typed failure counts.

## Proposed release gates

- 100% of invalid input tests yield the correct typed failure.
- 0 failed or rejected time steps enter a successful trajectory.
- 0 successful results with non-finite state values.
- 100% of accepted macro-steps meet (E \leq 1).
- All successful substeps satisfy the declared Newton residual contract.
- Tightening tolerance materially improves global error on the prescribed smooth benchmark ladder before any precision claim is made.
- Step refinement demonstrates the expected first-order trend before an order claim is made.
- No accuracy or performance claim without same-machine baseline runs and independent-reference evidence.

## Evidence ledger

| Exact candidate SHA | Rust build/tests | Analytic errors | Independent reference | Status |
| --- | --- | --- | --- | --- |
| Pending | Not run here | Not run here | Not run here | Unqualified |
