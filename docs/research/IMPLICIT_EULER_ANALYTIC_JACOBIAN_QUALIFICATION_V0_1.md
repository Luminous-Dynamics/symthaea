# Analytical Jacobians for Backward Euler — Qualification Protocol v0.1

**Status:** proposal and authored tests only. This candidate has not been compiled or executed here. No test or performance result is claimed until the exact candidate SHA is built and benchmarked.

## Why provide a Jacobian contract?

For an ODE \(y' = f(t,y)\), backward Euler solves
\[
G(z) = z-y_n-h f(t_{n+1},z)=0,
\]
whose Newton matrix is
\[
J_G(z) = I-hJ_f(t_{n+1},z),\qquad J_f=\frac{\partial f}{\partial y}.
\]

The finite-difference implementation estimates each Jacobian column by perturbing one state component and evaluating the RHS again. For an N-dimensional dense system, this adds approximately N RHS evaluations per Newton correction, besides the predictor and residual evaluations. A known analytical Jacobian avoids those probes and provides a direct place for a model to expose exact derivative structure.

This is an opt-in API; existing callers retain the finite-difference fallback. A new caller-supplied Jacobian is not automatically trusted: its length and every entry are validated, and callback evaluations are explicitly counted.

## API contract

- `ODEJacobian` has the signature `fn(time, state) -> Vec<f64>`.
- The returned vector is a row-major flattened \(N\times N\) matrix where element `i * N + j` is \(\partial f_i/\partial y_j\).
- Exactly \(N^2\) finite entries are required.
- The backward-Euler Newton matrix is assembled as \(I-hJ_f\), separately checking overflow/non-finite entries.
- Jacobian callback evaluations and RHS evaluations have separate counters, including error receipts. Adaptive integration aggregates work across full and half-step solves and enforces a Jacobian-evaluation budget as well.
- A malformed or non-finite Jacobian fails closed and preserves the last accepted state. It is not silently replaced with finite differences, because such fallback would hide a broken callback contract and make work accounting ambiguous.

The API remains dense, bounded to the current maximum dimension of 64. A later sparse callback should carry sparsity structure/values and use a sparse linear solver; returning a dense vector must not be presented as a scalable sparse interface.

## Reference implementations

- SciPy recommends providing a Jacobian rather than relying on finite-difference approximation for stiff Radau/BDF/LSODA methods. It also supports sparsity structure for finite-difference Jacobians: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- Boost.Odeint's implicit-system concept explicitly requires both the RHS and \(df/dy\), then solves the induced linear system: https://www.boost.org/latest/libs/numeric/odeint/doc/html/boost_numeric_odeint/concepts/implicit_system.html
- SUNDIALS exposes user-supplied Jacobian functions for matrix-based linear solvers and distinguishes that path from its internal difference-quotient approximation: https://sundials.readthedocs.io/en/v7.5.0/cvode/Usage/

This work adds an API capability present in established solvers. It does not claim performance parity, correctness superiority, or solver equivalence with SciPy, Boost.Odeint, or SUNDIALS.

## Qualification ladder

1. **Linear coupled systems.** Use exact off-diagonal Jacobians and compare each fixed-step result with the exact backward-Euler linear solve \((I-hA)y_{n+1}=y_n\). Include a non-diagonal matrix so diagonal-only methods cannot pass by accident.
2. **Finite-difference parity.** On deterministic smooth systems, compare trajectories using analytical and finite-difference Jacobians within problem-specific tolerances. Both must independently satisfy the residual contract.
3. **Work reduction.** Compare same-problem, same-tolerance runs; report RHS calls and Jacobian calls separately. Require fewer RHS calls in cases where finite-difference Jacobian probes dominate. Report total wall-time too; a user Jacobian may itself be expensive.
4. **Malformed callback controls.** Test N²-1, N²+1, NaN and infinity entries. Every case must return the expected typed failure, preserve the last accepted state and provide accurate callback counts.
5. **Adaptive parity.** Compare step-doubling accepted/rejected decisions, endpoint accuracy and work for analytical and finite-difference Jacobians. The local-error estimator remains separate from Newton residual convergence.
6. **Jacobian accuracy.** For analytic test systems, compare each supplied derivative to a separately coded exact matrix and separately finite-difference reference. Include nonsymmetric matrices, coupling, sign changes and nonlinear systems.
7. **Scale behavior.** Test state variables spanning at least 24 orders of magnitude, state-unit conversions, small/large Jacobian entries, and near-singular Newton matrices. Declare scale conversions and acceptance thresholds in advance.
8. **Stiffness ladder.** Sweep decay rates from \(10^0\) to \(10^8\), coupled stable matrices and rejection-heavy cases. Compare with an independent stiff solver for representative systems.
9. **Bounded execution.** Exercise Newton work, RHS evaluation and Jacobian callback budgets, and ensure each case terminates in success or a typed failure.
10. **Reproducible reporting.** Record exact candidate SHA, Rust toolchain, target, hardware, test seed, continuous-reference error, backward-Euler residual, RHS/Jacobian counts, Newton corrections, rejected trials and runtime distributions.

## Proposed release gates

- 100% of malformed callback cases fail with the expected typed class.
- 0 successful states contain non-finite values.
- 100% of accepted implicit steps pass the nonlinear residual contract.
- Analytical and finite-difference results agree within predeclared tolerances on the reference ladder.
- The analytical path consumes fewer RHS calls on the coupled reference case; total runtime is reported rather than assumed improved.
- The adaptive path preserves local-error acceptance criteria and improves or matches global reference error at matched tolerances.
- No broad performance claim until paired runs on the same machine/toolchain and multiple system dimensions are recorded.

## Evidence ledger

| Exact candidate SHA | Rust build/tests | Linear reference | Adaptive parity | Work comparison | Status |
| --- | --- | --- | --- | --- | --- |
| Pending | Not run here | Not run here | Not run here | Not run here | Unqualified |
