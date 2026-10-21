# ODE-WP-DOPRI5-COMPARISON-001

**Status:** draft numerical benchmark; not a performance claim or physical validation.

## Purpose

Compare Symthaea's checked Cash–Karp RK45 implementation against the independently implemented Dormand–Prince 5(4) method in `ode_solvers = 0.6.2` on a frozen analytical harmonic oscillator.

The comparison reuses the source-included numerical module from `research/ode-verify-001`; the external solver lives only in this isolated research crate and is not a production dependency. The direct/transitive Rust dependency graph is committed in `research/ode-dopri5-reference/Cargo.lock`; every qualification command uses `--locked`. The CI gate validates the tracked lockfile and records its SHA-256 in the comparison receipt, so the runner cannot silently resolve a newer graph during qualification.

## Frozen problem

The two methods solve the same model:

- `dx/dt = v`
- `dv/dt = -x`
- initial state `[scale, 0]`
- terminal time `t = 10`
- scales `1e-6, 1, 1e6`
- relative tolerances `1e-3, 1e-5, 1e-7`
- absolute tolerance `scale * rtol * 1e-3`

The exact reference is `x(t) = scale*cos(t)`, `v(t) = -scale*sin(t)`.

## What is recorded

For each of the nine matched cases, the JSON receipt reports:
- endpoint L2 error normalized by the state scale;
- relative energy drift (accepted states for Symthaea; declared dense-output samples for Dopri5);
- accepted/rejected steps and right-hand-side evaluations;
- completion, finiteness, and the frozen global numerical-contract disposition;
- source PR head, actual checked-out Git commit, and SHA-256 of the dependency lockfile.

The acceptance envelope for this smooth regression family is `10 * rtol` for endpoint error and sampled/accepted-state energy drift. This is a local benchmark target, not a universal bound or a published standard. Both solvers must independently satisfy the declared envelope. The comparison does not require either solver to win.

Wall-clock timing is intentionally excluded from the byte-deterministic receipt. A later performance experiment should use repeated runs, fixed hardware/toolchain settings, and matched achieved error; function-evaluation counts alone do not establish runtime superiority.

## Comparison semantics and limits

The methods use the same equations, initial state, final time and nominal scalar absolute/relative tolerance values, but their embedded methods, error estimators, and controller norms differ. This is a controlled cross-method comparison, not an implementation-identical test. Dopri5 dense output samples energy at `0.1` time increments, so its sampled maximum is not identical to the custom integrator's maximum over each accepted step.

This initial comparison does **not** establish SOTA performance, stiff-system superiority, physical validity, or universal solver correctness. Follow-up coverage should add a stiff problem such as Robertson chemistry and a qualified mature solver such as SUNDIALS/CVODE, preserving exact equations, scale-aware tolerances, solver versions, lock identity, raw receipts, failed runs, and benchmark-specific acceptance rules.

## Second family: ODE-WP-DECAY-DOPRI5-001

The harmonic oscillator is a smooth, energy-conserving periodic case. It does not cover a decaying transient or probe whether the trajectory briefly increases or crosses below zero. The separate `decay_work_precision` example adds the analytical family

\[
y'(t)=-\lambda y(t),\qquad z'(t)=0,\qquad
y(t)=s\,e^{-\lambda t},\quad z(t)=0.
\]

It crosses three rates `lambda = 1, 10, 50`, three initial scales `s = 1e-6, 1, 1e6`, and three relative tolerances `1e-3, 1e-5, 1e-7` for **27 cases**. Both solvers use the same initial state, terminal time `t=1`, and `atol = scale * rtol * 1e-3`. Dopri5's dense output is sampled at `dt=0.01`; Symthaea is checked at each accepted state.

The machine-readable receipt records maximum normalized trajectory error against the exact exponential, terminal error, RHS evaluations, accepted/rejected steps, maximum monotonicity increase, monotonicity violations beyond a declared `10*atol` slack, non-negativity within that absolute floor, finiteness, and both per-solver dispositions. The proposed global error gate is `max_scaled_absolute_trajectory_error <= 10*rtol`, alongside the custom method's local weighted-RMS acceptance gate. Neither method is required to win.

The benchmark gate deliberately exposes structural properties separately from the numerical error score. For this positive scalar decay family it requires no material increase beyond the `10*atol` slack and no state below `-10*atol`; a failure is preserved in the JSON record and blocks qualification. These criteria apply to this specific family and sampling procedure—not to all ODE systems or all continuous extensions.

This expands the comparison from one smooth periodic family to a second analytic transient family. It still does **not** qualify stiff-system performance in the usual solver-benchmarking sense. The next independent family should be Robertson chemical kinetics compared against an established stiff reference (e.g. SUNDIALS/CVODE) with a carefully chosen high-accuracy reference trajectory, positivity/conservation checks, matched achieved error, and repeated timing distributions. That work should follow only after the present lockfile-bound oscillator and decay gates pass on exact heads.

## Upstream references

- [ode_solvers 0.6.2](https://docs.rs/crate/ode_solvers/0.6.2) provides adaptive Dormand–Prince 5(4), RK4, and Dormand–Prince 8(5,3) methods.
- [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/) provides a broader work–precision comparison methodology.
- [SciPy RK45 documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html) documents per-component absolute tolerance with relative error control.
