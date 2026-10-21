# Robertson tolerance-calibration probe (exploratory only)

**Date:** 2026-10-21  
**Purpose:** check that the proposed Robertson time horizon, sample grid, and reference-agreement gate are numerically plausible before implementing the Rust/CVODE benchmark harness.  
**Environment:** local exploratory SciPy 1.17.0; scipy.integrate.solve_ivp with the BDF and Radau methods; IEEE-754 binary64.  
**Qualification:** **not** a frozen reference trajectory, not CVODE, not production code, and not a solver ranking.

## Exact probe setup

The model is the Robertson system in PROTOCOL.md. The initial state is (1, 0, 0), interval [0, 1e11], and t_eval is this 17-point grid:

\[
0,\ 10^{-8},10^{-6},10^{-4},10^{-2},1,10,10^2,\ldots,10^{11}.
\]

Each run used vector absolute tolerances and the same requested relative tolerance for BDF and Radau. The common output grid compares returned states only; solver-internal accepted states were not captured, so these numbers cannot qualify the positivity gate over the entire trajectory.

## Results

| Method | Requested rtol | Actual rtol behavior | nfev | njev | nlu | Max output-grid invariant defect |
|---|---:|---|---:|---:|---:|---:|
| BDF | 1e-12 | honored | 13,382 | 15 | 786 | 9.99e-16 |
| Radau | 1e-12 | honored | 61,881 | 125 | 582 | 3.55e-15 |
| BDF | 1e-13 | honored | 15,675 | 12 | 1,083 | 6.44e-15 |
| Radau | 1e-13 | honored | 108,895 | 52 | 516 | 7.77e-15 |
| BDF | 1e-14 | clamped to 2.220446049250313e-14 by SciPy's tolerance floor | 17,489 | 11 | 1,413 | 3.77e-15 |
| Radau | 1e-14 | clamped to 2.220446049250313e-14 by SciPy's tolerance floor | 167,756 | 22 | 474 | 5.55e-15 |

All six solves returned success at t=1e11. The invariant defect was computed from the 17 reported samples as max(abs(y1+y2+y3-1)). These conservation figures are output-grid diagnostics, not an independent proof of solver correctness.

For the two floor-clamped runs, the maximum component-wise absolute BDF/Radau disagreement on the common grid was:

- y1: 3.0509e-13 at t=1e3;
- y2: 1.4064e-17 at t=10;
- y3: 3.0431e-13 at t=1e3.

Using the strictest proposed candidate profile atol=(1e-12,1e-16,1e-12), rtol=1e-8 as the denominator for this cross-reference comparison, the largest normalized disagreement over the 17 samples was 3.6775e-3, at t=1e8 in y1. This supports using a reference-agreement cap of 1e-2 as a candidate protocol threshold, instead of 1e-3, for the first qualification iteration. The CVODE plus independent-reference experiment must still meet this gate before the protocol can mark any reference qualified.

## Interpretation and limits

1. This proves only that two stiff algorithms available through this local SciPy installation completed this sampled fixture under these settings.
2. The 1e-14 request was not honored as written: SciPy warned that it raised rtol to its binary64 floor. It must not be described as a 1e-14-accuracy solve.
3. BDF and Radau are different algorithms but share the SciPy integration framework; this is weaker independence than the planned pinned CVODE plus separately implemented stiff method.
4. The max normalized difference is computed only at the public sample grid; no internal-step positivity report, reference digest, independently verified build, or raw accepted-state ledger was produced.
5. nfev, njev, and nlu are algorithm work counters, not runtime measurements. Different solvers' counters are not interchangeable units of computational cost.
6. The result motivates a tolerance and reference ladder; it does not qualify the Robertson reference, validate chemical kinetics experimentally, compare Symthaea's integrator, or establish SOTA.

The next executable harness should reproduce the exact fixture in the protocol, add exact source/build/input/output digests, capture every failed run, and generate reference candidates with SUNDIALS CVODE 7.9.0 BDF. Reference bytes must remain marked unresolved unless tighter-reference convergence and the independent cross-check both pass.
