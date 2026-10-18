# Differential-equations benchmark: non-degenerate reference qualification v0.1

**Status:** proposed, deterministic software verification contract. The thresholds below are project targets, not published standards or claimed measurements.

## Paradigm shift

Do not count a numerical solver as correct merely because it converges, keeps an invariant, or returns plausible-looking output. A benchmark must compare its complete output against an independently defined reference and include cases engineered to fool the old acceptance rule.

This increment makes two existing checks harder to fool:

1. The harmonic-oscillator BVP solves `y'' + y = 0` on `[0, π/2]`, with `y(0) = 0` and `y(π/2) = 1`. Its unique solution is `y(t) = sin(t)`. The former (y(0)=y(\pi)=0) conditions also admit the zero solution and do not determine the amplitude.
2. The heat-equation check uses `u(x,0) = sin(πx/L)` with zero Dirichlet boundaries. Its analytical solution is
   `u(x,t) = exp(-α(π/L)^2 * t) * sin(πx/L)`.
   Energy decay is retained as a qualitative invariant, but it no longer stands in for accuracy.

Verification against a known solution tests whether the implementation solves the stated mathematical problem. It does **not** establish that the mathematical model describes a real physical system.

## Metrics and v0.1 gates

| Case | Required evidence | Provisional gate |
|---|---|---:|
| IVP `y' = y, y(0) = 1` | Relative error at `t = 1` versus `e` | Existing project target `< 10^-2`; exact error is reported |
| Harmonic BVP | Boundary residual, recovered initial slope, maximum absolute trajectory error versus `sin(t)` | Each absolute residual/error `< 10^-4`; slope error `|s - 1| < 10^-4` |
| Heat PDE | Relative discrete `L2` error versus the analytical eigenmode, plus energy-decay invariant | Relative `L2` error `< 10^-2`, and energy must decay |

The fixed-case thresholds are conservative initial engineering gates. Calibrate them with exact-head execution and numerical error analysis before tightening or relaxing them. Never adjust thresholds after looking at a held-out corpus and then call that corpus independent.

## Falsification and negative controls

The next benchmark increments should add frozen cases for:

- **Degenerate BVP:** endpoint constraints that permit the all-zero trajectory. The benchmark must not infer a uniquely identified nonzero solution from boundary residual alone.
- **Wrong trajectory, correct endpoint:** perturb an interior state while preserving the endpoint, and require the full-trajectory score to fail.
- **Energy-decay impostor:** a solution that decays but has the wrong amplitude, spatial mode, sign, or decay rate must fail the analytical-error gate.
- **Wrong equation/sign:** reverse the diffusion sign or oscillator restoring term in a controlled test; the known-answer comparison must reject the result.
- **Input defects:** non-finite values, invalid domains, mismatched vector lengths, and invalid coefficients must fail closed or be explicitly marked non-qualified.
- **Metric tampering:** mutate the reported error without changing the returned solution and ensure the evaluator recomputes error from raw output.

Keep each negative-control ID, expected disposition, exact source/config digest, and independent reference implementation in the frozen corpus. A refusal or invalid input is not a successful numerical solution; report validity, convergence, and accuracy separately.

## Benchmark ladder

### Q0 — exact-reference correctness

Run the current IVP, non-degenerate BVP and analytical heat-mode tests. Emit each raw residual/error, not only a pass fraction. The test suite must reject a zero BVP trajectory and an energy-decaying but analytically wrong PDE trajectory.

### Q1 — refinement and observed order

- Heat: use nested spatial grids (initial proposal: 10, 20, 40, 80 interior points) and a declared timestep-refinement schedule. Record (L_2) error, (\Delta x), (\Delta t), time-step count, stability ratio, and runtime. The existing solver couples its explicit timestep to grid spacing, so document that coupling and confirm that the maximum-step cap is not active.
- RK4 IVP: refine the number of steps (initial proposal: 25, 50, 100, 200) and compare the observed order with the method's expected fourth-order behavior before the roundoff/error floor.
- BVP: vary integration resolution separately from root-finder tolerance. Do not infer fourth-order convergence while a fixed root tolerance dominates the error.

For observed order, report the actual value `p = log2(E_h / E_(h/2))` and all component errors. The provisional heat target is (p\ge1.7) for at least two consecutive refinement intervals before roundoff dominates. This target must be validated on the exact solver configuration before it becomes a release gate.

### Q2 — adversarial differential corpus

Aim for at least 1,000 deterministic seeded cases, including analytically solvable parameter variations and the negative controls above. The target is **zero false-positive accuracy passes** in the frozen invalid/adversarial set. Report the denominator, false rejects, invalid-input dispositions, and all per-family results. A perfect score on a small corpus must not be described as broad proof of correctness.

### Q3 — independent implementations

Compare against at least one separately implemented reference path using a different numerical strategy where practical. Agreement between two implementations that share the same helper or discretization bug is weak evidence. Retain exact analytical cases as anchors.

### Q4 — physical validation and uncertainty

Only after Q0–Q3 are stable should physical claims be assessed against independent measured data. Freeze the intended-use envelope, reference-data provenance, parameter calibration split, validation split, measurement uncertainty, model discrepancy, and acceptance criteria. Report bias and uncertainty by regime. Numerical verification is not empirical validation.

## Required result receipt

A qualification record should bind:

- exact source revision, compiler/toolchain, target architecture and feature flags;
- exact benchmark and reference-corpus bytes/digests;
- numerical parameters, grid, timestep schedule and solver settings;
- per-case raw output, analytical reference, residual, absolute and relative errors;
- negative-control census, invalid-input outcomes and explicit failure dispositions;
- observed convergence orders and the refinement range used to compute them;
- runtime and memory, reported only after correctness gates;
- limitations and a claim ceiling.

Missing provenance is a non-qualified run. Queued, skipped, stale-head or absent CI is not a pass.

## Research basis

- NASA, [NASA-STD-7009B: Standard for Models and Simulations](https://standards.nasa.gov/standard/NASA/NASA-STD-7009), active model-and-simulation credibility standard. It treats verification, validation and uncertainty as distinct evidence activities.
- NASA, [NASA-HDBK-7009B implementation guide](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009), published 2026-02-03, with practical implementation guidance.
- Salari and Knupp, [Code Verification by the Method of Manufactured Solutions](https://doi.org/10.1115/1.1436090), *Journal of Fluids Engineering* (2001). Known analytical solutions and systematic refinement are used to evaluate code accuracy.
- NASA Glenn, [Verification Assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html), outlines error estimation and spatial/temporal convergence checks.
- Srinivasan, [On Methods of Uncertainty Quantification in Computational Modeling and Simulation](https://doi.org/10.6028/NIST.AMS.100-85), NIST AMS 100-85 (2026-10-05), connects credibility assessment with risk-based uncertainty quantification.

These sources support the verification/refinement approach; they do not endorse the project's numerical threshold values.
