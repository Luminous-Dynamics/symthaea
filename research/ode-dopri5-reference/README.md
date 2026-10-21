# ODE-WP-DOPRI5-COMPARISON-001

**Status:** draft numerical benchmark; not a performance claim or physical validation.

## Purpose

Compare Symthaea's checked Cash–Karp RK45 implementation against the independently implemented Dormand–Prince 5(4) method in `ode_solvers = 0.6.2` on a frozen analytical harmonic oscillator.

The comparison reuses the source-included numerical module from `research/ode-verify-001`; the external solver lives only in this isolated research crate and is not a production dependency.

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

## Upstream references

- [ode_solvers 0.6.2](https://docs.rs/crate/ode_solvers/0.6.2) provides adaptive Dormand–Prince 5(4), RK4, and Dormand–Prince 8(5,3) methods.
- [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/) provides a broader work–precision comparison methodology.
- [SciPy RK45 documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html) documents per-component absolute tolerance with relative error control.
