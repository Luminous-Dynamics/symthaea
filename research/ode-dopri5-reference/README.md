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
- endpoint error normalized by the state scale;
- maximum **sampled** trajectory error against the exact solution;
- relative energy drift sampled at the same times for both solvers;
- accepted/rejected integration steps and integration right-hand-side evaluations;
- whether both methods produced the same 101 output times, and the maximum pointwise solver disagreement on that grid;
- completion, finiteness, source PR head, tested checkout commit, and dependency lock SHA-256.

The frozen output grid is (t_i=i/10), for (i=0,ldots,100), over [0,10]. Dopri5 uses its documented dense-output increment of 0.1. Both integrators run with their native adaptive step selection and emit sparse accepted-step states. Dopri5 is explicitly configured with `OutputType::Sparse`, which disables its native dense-output sampling. The harness reconstructs both accepted-step trajectories using the same benchmark-only quintic-Hermite interpolant at 101 fixed output times, using the oscillator's known first and second derivatives. This is not a production dense-output implementation. Interpolation error contributes to the sampled metric, which is not a rigorous continuous-time error bound. Unit tests check the common grid, interpolation accuracy on an exact oscillator trajectory, and fail-closed handling of malformed trajectories.

The accepted/rejected/RHS counters measure integration, not the later interpolation work. They are useful algorithm-work diagnostics, not direct runtime costs. Interpolation work is not timed separately yet. The acceptance envelope is `10 * rtol` for endpoint error, sampled trajectory error, and sampled energy drift. Both solvers must meet it; in addition the common-grid maximum solver disagreement must be at most `20 * rtol`, consistent with the two independently declared `10 * rtol` error envelopes. This is a benchmark-specific gate, not a universal bound or a published standard.

Wall-clock timing is intentionally excluded from the deterministic receipt. A later performance experiment should use repeated runs, fixed hardware/toolchain settings, and matched achieved error; function-evaluation counts alone do not establish runtime superiority.

## Comparison semantics and limits

The methods use the same equations, initial state, terminal time, nominal component tolerances, frozen output grid, and reconstruction procedure, but their embedded methods, error estimators, and controller norms differ. This remains a controlled cross-method comparison, not an implementation-identical test. The sampled-trajectory metric is only a common-grid diagnostic, not the true maximum error over continuous time.

This initial comparison does **not** establish SOTA performance, stiff-system superiority, physical validity, or universal solver correctness. Follow-up coverage should add a stiff problem such as Robertson chemistry and a qualified mature solver such as SUNDIALS/CVODE, preserving exact equations, scale-aware tolerances, solver versions, lock identity, raw receipts, failed runs, and benchmark-specific acceptance rules.

## Upstream references

- [ode_solvers 0.6.2](https://docs.rs/crate/ode_solvers/0.6.2) provides adaptive Dormand–Prince 5(4), RK4, and Dormand–Prince 8(5,3) methods.
- [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/) provides a broader work–precision comparison methodology.
- [SciPy RK45 documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html) documents per-component absolute tolerance with relative error control.
