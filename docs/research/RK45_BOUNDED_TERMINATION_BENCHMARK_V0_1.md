# Adaptive RK45 bounded-termination and error-contract benchmark v0.1

**Status:** proposed numerical-verification protocol and authored Rust regression tests. Exact-head compilation/test execution is required before qualification. A typed result is a necessary contract, not evidence that every numerical output is globally accurate.

## Paradigm shift: make termination and failure first-class outputs

An adaptive solver is not just a function from initial state to final state. It is a controlled process that accepts/rejects trial steps while adapting its resolution. Its public result must answer independently:

1. Did it reach the requested end time?
2. How much computation was attempted, including rejected steps?
3. Did each RHS/stage preserve dimensions and finite arithmetic?
4. Did the local controller accept the last step under the stated tolerance rule?
5. How much global error remains relative to an independently known solution?

A result that returns the initial state after an exhausted loop, or whose only signal is a boolean the caller fails to inspect, must not be treated as a successful solve.

## Failure mode addressed

The former RK45 loop guarded termination with `steps_taken < max_steps`, but incremented `steps_taken` only when a step was accepted. Rejected attempts did not consume the limit. The rejection branch also raised the next step back to `h_min` and then checked whether it was below `h_min`; that check could never detect the clamp. With a persistent rejection, the step size could stop shrinking while simulation time remained unchanged.

The new `try_solve_rk45` API returns `Result<RK45Result, RK45Error>`. It uses a shared attempt budget for accepted and rejected trials; distinguishes invalid inputs, malformed/non-finite RHS values, error-estimation failure, underflow, non-progress, and budget exhaustion; and includes the last accepted time/state plus accepted/rejected counts in the error receipt. The legacy `solve_rk45` method remains compatible but maps failures to `converged = false`; new callers should prefer the fallible API.

## Implementation contract

- **Time domain:** finite forward-only `t0 <= t_end`; zero duration is the identity map after input validation.
- **Initial state:** non-empty system, exact declared dimension, all finite values.
- **Tolerance:** finite and strictly positive; no silent floor to another tolerance.
- **RHS:** every stage must return exactly `system.dim` derivatives and all must be finite. Intermediate stage state must be finite before invoking the RHS.
- **Error estimator:** all component scales and the aggregate normalized error must be representable and finite. A non-finite estimate is failure, not a reason to accept.
- **Attempt budget:** at most 100,000 trials, including rejected steps.
- **Progress:** time must advance on an accepted step. A non-positive, unchanged, or non-representable step produces an explicit error.
- **Termination:** `converged = true` only when the integration reaches `t_end`; every other path is failure.
- **Diagnostics:** the typed failure records the last accepted time and state, accepted/rejected counts, and last available error estimate.

The current public API accepts one scalar tolerance and uses the existing `tol * (1 + max(|y|, |y_new|))` scale. This is kept for compatibility in v0.1, but it should not be mistaken for the separate absolute/relative tolerance contract common in mature integrators. A next API version should provide explicit absolute and relative tolerances, optionally per component.

## Regression cases added

- Zero-duration solve returns the exact initial state with zero steps.
- Reversed/non-finite time interval, invalid tolerance, dimension mismatch, empty system and non-finite initial state return typed errors.
- An RHS with the wrong output dimension is rejected at the first stage.
- A non-finite derivative is rejected at the first stage.
- A deliberately discontinuous RHS forces rejection with a private one-attempt budget; the test verifies that a rejected attempt consumes the budget and cannot loop forever.
- The compatibility wrapper must mark a failed call as `converged = false`.

These regression tests are authored and committed, but remain unexecuted by the authoring environment.

## Benchmark ladder

### R0 — exact known-answer cases

Use independent analytical references for:
- exponential growth and decay: `y' = λy`, `y(t) = y0 exp(λ(t-t0))`;
- harmonic oscillator: `y'' + ω²y = 0`, checking both position and velocity;
- constant-derivative systems and linear multi-component systems;
- a conserved quantity such as oscillator energy, measured separately from trajectory error.

Record absolute error, relative error with explicit near-zero handling, accepted steps, rejected steps, function evaluations, final time reached, and runtime. Do not reduce each case to a single pass bit.

### R1 — adversarial termination corpus

Freeze at least 1,000 deterministic cases spanning:
- stiffness/time-scale ratios;
- RHS magnitudes and state component scale differences;
- sharp but finite transients and high-frequency smooth RHS functions;
- zero-duration and near-representable time intervals;
- non-finite parameters, input state and RHS outputs;
- malformed RHS dimensions;
- tolerances near floating-point limits;
- workloads engineered to repeatedly reject steps.

Targets:
- zero cases claim convergence without reaching the requested end time;
- zero accepted trial states or derivatives contain NaN/Inf;
- every case terminates within the declared attempt budget;
- the exact refusal cause and denominator are reported for the negative-control set.

### R2 — tolerance response and global-error calibration

For smooth known-answer cases, sweep tolerances such as `1e-3` through `1e-10`, then report actual global error versus tolerance, accepted/rejected counts, and RHS evaluations. The tighter tolerance should generally reduce error on the frozen smooth corpus, but do not require strict monotonicity on every isolated floating-point case. Calibrate the relationship rather than equating a local embedded error estimate with a global error bound.

### R3 — scale robustness

Use systems where state components differ by many orders of magnitude. The next API should compare:
- the current single-tolerance scale;
- scalar absolute/relative tolerance;
- per-component absolute tolerances.

Report per-component errors; one large component must not hide a failed small but important component in a pooled RMS norm.

### R4 — independent differential comparison

Compare the same frozen initial-value problems with an independent implementation such as SciPy `solve_ivp` or another independently maintained solver, pinning version, method and tolerance settings. For analytical problems, the mathematical solution remains the primary reference. Cross-solver agreement alone is not proof because both solvers can share failure modes.

### R5 — stiffness and algorithm selection

Benchmark non-stiff RK45 separately from stiff problems. Under equal error targets, measure function evaluations, rejected-step fraction and wall-clock cost against an independently implemented stiff solver. A solver that exhausts its budget must be reported as incomplete, not slower-but-valid or successful.

## Required qualification receipt

Pin exact source SHA, Rust toolchain/target/features, initial data, RHS identity or a versioned RHS fixture, timespan, tolerance settings, corpus digest, all raw outputs, accepted/rejected counts, evaluations, final time, global analytical error, termination cause, and test logs. For a function-pointer RHS that lacks source identity, record the source revision/configuration externally; a user-supplied label alone is not identity.

A missing receipt, skipped/queued/stale workflow, or absent test execution is not a pass.

## Research basis

- SciPy, [`solve_ivp` reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): distinguishes successful endpoint reach from failed integration, and exposes function-evaluation counts and termination status/message.
- MathWorks, [Troubleshoot Common ODE Problems](https://www.mathworks.com/help/matlab/math/troubleshoot-common-ode-problems.html): describes relative/absolute tolerances and the need to detect too-small steps and stiffness-related difficulty.
- Boost.Odeint, [Controlled steppers](https://www.stage.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/concepts/controlled_stepper.html): accepted/rejected steps explicitly update the step size; rejected attempts leave state/time unchanged.
- Dormand and Prince, “A family of embedded Runge-Kutta formulae,” *Journal of Computational and Applied Mathematics* 6 (1980), pp. 19–26, as cited by the SciPy solver reference.

The references inform software contracts and benchmark design; the target corpus scores and performance thresholds below are project proposals, not established standards.
