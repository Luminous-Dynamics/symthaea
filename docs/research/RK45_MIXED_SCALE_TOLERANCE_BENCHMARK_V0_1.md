# RK45 Mixed-Scale Tolerance Qualification Protocol v0.1

**Status:** proposed qualification protocol. The accompanying source changes and test cases are authored but have not been compiled or executed in this environment. This document does not claim solver qualification or benchmark passes.

## Why this change

A single scalar tolerance couples variables with different units and magnitudes. A trajectory might combine position, angular rate, concentration, temperature, or a small probability-like quantity; one scalar can over-constrain some components while effectively ignoring others. The new API allows per-component absolute tolerances and one relative tolerance while preserving the existing scalar-tolerance entry point.

## Error-control contract

For component i, the solver scales its embedded local error estimate using

\[
s_i = \mathrm{atol}_i + \mathrm{rtol}\,
\max(|y_{n,i}|, |y_{n+1,i}|).
\]

The aggregate controller signal is the root-mean-square of component errors normalized by these scales:

\[
E = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left(\frac{e_i}{s_i}\right)^2}.
\]

The candidate step is accepted when E ≤ 1. Each absolute tolerance must be finite and strictly positive, with exactly one value for every state component. The relative tolerance must be finite and non-negative. Setting relative tolerance to zero gives absolute-error-only control. Invalid shape/value contracts are reported with typed errors.

The existing scalar API maps its tolerance to both atol and rtol. Algebraically, this preserves its previous scale \(\mathrm{tol}(1+\max(|y_n|,|y_{n+1}|))\), avoiding an API-breaking switch in solver behavior.

**Important limitation:** this is a local embedded-error controller, not a proof of global solution accuracy. Global error must be measured against known solutions or an independently qualified reference.

## Reference implementations and distinction

- SciPy's `solve_ivp` documents separate absolute and relative tolerances, including vector-valued absolute tolerances, with a scale based on `atol + rtol * abs(y)`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- Boost.Odeint documents controlled Runge–Kutta error scaling and the use of absolute and relative tolerance terms: https://www.boost.org/doc/libs/latest/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html

The Symthaea scale above uses the maximum magnitude across old and candidate states. It is conceptually related to these references but should not be described as a bit-for-bit reproduction of either library.

## Qualification ladder

1. **Input contract.** Exercise empty systems, state dimension mismatch, time-span errors, absolute-tolerance vector length mismatch, zero/negative/NaN/infinite absolute tolerances, negative/NaN/infinite relative tolerances, non-finite initial state, and malformed/non-finite RHS output. Every invalid case must return a typed error and never claim convergence.
2. **Analytic mixed-scale systems.** Include independent exponential growth/decay, sine/cosine oscillators, constant derivatives, near-zero components, sign crossings, and state components spanning at least 24 orders of magnitude. Test scalar and vector tolerance contracts separately.
3. **Unit-rescaling invariance.** Rescale a state component and its absolute tolerance by the same positive unit conversion. The normalized controller decisions and result in rescaled units should agree within declared floating-point allowances.
4. **Tolerance response.** Tightening one component's absolute tolerance while leaving all other settings unchanged should generally spend more work and reduce that component's measured global error on a smooth deterministic reference problem. Do not assert monotonic step counts for every adaptive problem; step-controller branch changes can make work non-monotonic.
5. **Observed order.** On smooth, non-stiff analytic problems, measure global error under step/tolerance refinement. Report observed convergence order; use fifth order as the method's theoretical target, but set a practical acceptance band only after establishing a reproducible baseline.
6. **Independent references.** Compare representative cases against analytic solutions and an independent solver (for example, SciPy DOP853 for smooth non-stiff cases, plus an appropriate stiff solver when relevant). Record exact solver settings; do not use agreement between two runs of the same implementation as independent validation.
7. **Adversarial determinism and resource bounds.** Run at least 1,000 seeded cases covering scale disparity, near-zero values, rapidly changing derivatives, and rejection-heavy cases. Every case must terminate through success or an explicit failure receipt under a shared attempt budget; no NaN/Inf may escape as a successful final state.
8. **Cost accounting.** Report accepted and rejected attempts, RHS evaluations, wall-clock distribution, maximum component-wise absolute and relative global errors, maximum normalized local error, and typed failure counts. Record the source commit, compiler/toolchain, CPU/GPU when relevant, and random seed.

## Proposed qualification gates (targets, not results)

- 100% of invalid-input cases return the expected typed failure class.
- 100% of benchmark cases terminate within the configured attempt budget.
- 0 successful results containing non-finite states.
- 0 tolerance-vector length mismatches accepted.
- Analytic global-error thresholds are specified per problem and units before running a case; local error estimates are not substituted for global-error measurements.
- Smooth non-stiff refinement results show the expected convergence trend before a release claims fifth-order behavior.
- Mixed-scale unit-rescaling cases agree within an explicitly reported numerical threshold.
- No performance regression is accepted without a paired baseline on the same machine and compiler configuration.

## Evidence record

Fill one row per exact candidate SHA. Queued, skipped, absent, or non-triggered workflows are not passes.

| Candidate SHA | Rust build/tests | Deterministic benchmark | Independent reference | Status |
| --- | --- | --- | --- | --- |
| Pending | Not run here | Not run here | Not run here | Unqualified |

