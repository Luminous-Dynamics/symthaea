# PC-01B — condition-aware matrix-equation solver benchmark v1

**Owner:** PHYS-COMPUTE-ADAPT-001 #10933; composed with QARCH-019 #9236 and RELIABILITY-COMPUTE-001 #10993.  
**Status:** benchmark contract and software numerical oracle draft; no analog device result or SOTA claim.  
**Purpose:** compare computational primitives and solver architectures for linear equations without collapsing numerical correctness to one residual or operation-count metric.

## 1. Research hypothesis

The important question may not be which substrate performs matrix-vector multiplication fastest. It may be which numerical primitive and algorithm best match a workload's algebraic structure, conditioning, temporal dynamics, correctness contract and available physical resources.

For a linear system:

`A x = b`

candidate architectures include a digital direct solver, digital iterative solver, analog MVM inside digital iteration, direct analog dynamical/matrix-equation solver, analog preconditioner plus digital refinement, and a fault-aware hybrid with independent verification.

This is a workload-to-primitive-and-substrate selection experiment, not a claim that analog solving is universally faster or more accurate.

## 2. Software reference artifact

`research/physical_compute/equation_solver_contract_v1.rs` is a dependency-free numerical oracle draft providing:

- Dense finite-valued matrix representation with a 16,777,216-element ceiling.
- A shared `LinearOperator` interface for dense and matrix-free structured systems.
- Matrix-free 1-D Poisson operator with O(n) storage, O(n) MVM and exact spectral condition number.
- Stable diagonal-decay ODE with an analytic trajectory oracle and a classical RK4 digital baseline.
- Reproducible symmetric positive-definite family `A = λI + α11ᵀ`, with a known exact spectrum and condition number.
- Known-answer RHS generation from a frozen solution vector.
- Direct Cholesky solve.
- A matrix-free linear-operator interface shared by dense and structured workloads.
- Matrix-free 1-D Poisson operator (second-difference stencil) with O(n) storage and O(n) MVM; its exact condition number is available from the discrete spectrum.
- Unpreconditioned conjugate-gradient solve with explicit tolerance and iteration budget; it recomputes the true residual before acceptance and restarts the recurrence if residual replacement is needed.
- Jacobi-preconditioned conjugate gradient, with a fixed diagonal preconditioner, positive-diagonal checks and independent true-residual acceptance.
- Compensated dot-product accumulation in the digital reference to reduce avoidable summation error.
- Compensated digital dot-product accumulation.
- Separate residual norm, relative residual, relative backward error, absolute forward error and relative forward error.
- ODE maximum/final absolute and relative trajectory error against the analytic path.
- Negative controls for singular/nonsymmetric input, invalid dimensions/tolerances, non-finite values and ill-conditioned residual/forward-error divergence.

The exact condition number of the generated SPD family is:

`κ₂(A) = (λ + nα) / λ`

The reference solver is not a physical model and does not predict analog noise, conductance, power, settling time, yield or device reliability. It supplies frozen numerical workloads and independent reference values for later substrate comparisons.

Run its dependency-free tests with:

```sh
rustc --edition=2021 --test research/physical_compute/equation_solver_contract_v1.rs -o /tmp/equation_solver_tests
/tmp/equation_solver_tests
```

The 19 Rust unit tests are authored but remain uncompiled/unexecuted in the current authoring environment because `rustc` is unavailable.

## 3. Workload portfolio

### M1 — known-spectrum SPD

Use `A = λI + α11ᵀ` with a known exact solution and exact condition number. Sweep dimensions, `λ`, `α`, vector distributions and precision. This is a controlled correctness baseline, not a substitute for realistic matrix families.

### M2 — diagonally dominant SPD

Generate symmetric strictly diagonally dominant matrices with known reference solutions. Freeze the generator and seed lists before policy tuning.

### M3 — prescribed-spectrum SPD

Generate matrices with declared eigenvalue spectra and orthogonal basis. Record the exact spectrum and condition number. Separate generator/training, validation and held-out confirmatory seeds.

### M4 — sparse and structured systems

The first executable M4 case is the matrix-free 1-D Poisson operator:

`(Ax)_i = 2x_i - x_{i-1} - x_{i+1}`

with zero-boundary convention. Its eigenvalues are `4 sin²(kπ/(2(n+1)))`; therefore the exact 2-norm condition number is `cot²(π/(2(n+1)))`. It requires O(n) storage and O(n) work per MVM, unlike a dense matrix representation. This makes it a better baseline for sparse scientific workloads and a bridge toward discretized PDEs.

Next include banded, Toeplitz, circulant, graph-Laplacian-derived and block-structured systems. Count sparsity, storage, data movement and preconditioner cost. Do not compare a matrix-free operator to a dense solver without counting the difference in representation and memory traffic.

### D1 — stable diagonal linear ODE

The first executable differential-equation oracle is:

`dx_i/dt = -r_i x_i`, with `r_i ≥ 0` and known solution `x_i(t)=x_i(0)e^{-r_i t}`.

The software reference provides the analytic trajectory and a classical RK4 baseline. Compare maximum and final trajectory error over the complete horizon, not only the final state. Sweep multiple decay timescales and step sizes, and use the RK4 step-refinement ratio as a numerical regression. This is a controlled software fixture; it is not a model of a specific analog circuit or a complete stiff/nonlinear ODE benchmark.

Next extensions: coupled linear systems, oscillatory systems, stiff systems and nonlinear ODEs, then PDE spatial discretization. Keep analog-native direct dynamics, digital time integration, and analog MVM inside digital time stepping as distinct architecture tracks.

### M5 — negative controls

- Singular systems.
- Indefinite systems.
- Nonsymmetric and non-normal systems.
- Ill-conditioned systems with known solutions.
- Random dense full-rank systems.
- A perturbation that passes a residual threshold but violates forward-error tolerance.

The first Rust oracle currently implements M1, a matrix-free 1-D Poisson M4 workload and selected M5 negative controls. M2, M3, broader M4 families and the nonsymmetric solver lane remain protocol targets, not completed generators.

## 4. Numerical correctness contract

Never report only `||Ax-b||`. Report at least:

1. Absolute residual: `||Ax-b||₂`.
2. Relative residual: `||Ax-b||₂ / ||b||₂` (with the zero-`b` case explicitly defined).
3. Normwise relative backward error:

`||Ax-b||₂ / (||A||_F ||x||₂ + ||b||₂)`

4. Forward error against a hidden high-precision reference:

`||x - x*||₂ / ||x*||₂`

5. Condition number or a declared condition estimate and its uncertainty.
6. Convergence/stagnation status, iterations, stopping rule and residual history.

A small residual is not a guarantee of a small forward error for ill-conditioned systems. The benchmark includes a constructed regression case where relative residual is below `1e-9` while relative forward error exceeds `0.7`. It is a logical numerical counterexample, not a measured analog device result.

For analog or mixed-precision paths, the verifier must use an independent numerical path and report its own precision, calibration, noise and fault dependencies. The expected solution is evaluator-only and must never be disclosed to the candidate or online verifier.

## 5. Architecture comparison matrix

A. Best tuned digital direct solver (Cholesky for SPD; LU where appropriate).  
B. Best tuned digital iterative solver (CG / Jacobi-PCG for SPD; GMRES/BiCGSTAB only in a separately implemented nonsymmetric lane).  
C. Analog MVM inside a digital iterative solver.  
D. Direct analog dynamical/matrix-equation solve.  
E. Analog preconditioning or approximate initial solution plus digital mixed-precision refinement; compare against digital Jacobi-PCG as a baseline.  
F. Adaptive mathematical representation and fault-aware mapping.  
G. Independent residual/backward-error verification plus fail-closed rejection.  
H. Combined topology-aware mapping, verification and fallback/recovery.

Keep two attribution tracks separate:
- **Same algorithm across substrates:** isolates implementation effects.
- **Best competent co-design:** compares each architecture at its best under equal workload, correctness, deadline and resource budgets.

## 6. Fault and precision interaction

Cross matrix family and condition number with:
- independent stuck-off / stuck-on faults;
- spatially clustered and row/column faults;
- nested array/die/board/power/clock/controller failures;
- programming/write error;
- per-read noise;
- device-to-device and cycle-to-cycle variation;
- drift, temperature, retention and recalibration.

Do not merge these mechanisms into one generic “fault rate”. A 10% device fault count and a 10% read-noise level are not interchangeable.

For each architecture, report equal-physical-resource and equal-task-quality frontiers. Count factorization depth, physical cells, differential pairs, DAC/ADC, sensing, feedback, controller, host transfers, calibration, verifier, refinement and recovery.

## 7. Measurement contract

For hardware and full-system measurements, report separately:

- solution quality and confidence intervals;
- convergence probability and convergence time;
- p50/p95/p99 latency, including settling and control;
- joules per correct accepted solution at an explicitly declared board/system boundary;
- conversion, sensing, control, calibration, reprogramming and refinement energy;
- storage, data movement and memory footprint;
- device/crossbar count, usable cells, area and operating envelope;
- fault detection, false acceptance/rejection, abstention and recovery;
- exact device, package, firmware, calibration and instrument identity.

A device/core operation-energy figure cannot be compared directly with joules per accepted system solution. Do not infer hardware efficiency from algorithmic operation counts.

## 8. Suggested screening and confirmatory plan

Proposed condition-number bins for software screening: `κ₂ ≈ 1, 10, 25, 100, 1e3, 1e6`. These are planned stress bins, not universal analog limits. The first pass should use known-spectrum SPD cases, followed by sparse/structured families and negative controls. Freeze algorithm parameters after the screening design. Run held-out confirmatory matrices and fault seeds after freezing.

A candidate advances only if:
- the independent numerical oracle and exact problem identity are preserved;
- residual, backward-error **and** forward-error contracts pass;
- ill-conditioned and non-normal controls are reported honestly;
- the full system meets a declared latency/energy/resource constraint;
- the advantage survives held-out matrices, fault geometries and calibration conditions;
- any physical claim has raw instrumented receipts and independent reproduction.

Do not use a composite scalar to hide a reliability/availability trade-off. Report the Pareto vector.

## 9. External research anchors

- Sun et al., *Modern analog computing for solving differential and matrix equations*, npj Unconventional Computing (published 2026-11-03): https://www.nature.com/articles/s44335-026-00108-x. The review connects MVM, matrix-equation and differential-equation solving as related analog-computing primitives. It gives a first-order illustration: with ~6 effective bits (about 0.4–0.5% RMS input error) and a 10% target solution error, uncompensated direct solving suggests a condition-number range around 20–25. This is an assumption-dependent order-of-magnitude estimate, not a universal analog-compute limit; actual stability, error spectrum, scaling, nonlinearity and iterative correction matter.
- Sun et al., *Precise and scalable analogue matrix equation solving using resistive random-access memory chips*, Nature Electronics (2025): https://www.nature.com/articles/s41928-025-01477-0. Demonstrated 16×16 RRAM-chip inversion at 24-bit fixed-point precision using 3-bit devices and block/iterative methods. The paper separately reports low-precision behavior varying with conditioning (e.g. about 2.4 bits at κ=7.7 and 4–5 bits at κ=1.6), and studies high-precision solving at κ=2.5, 5, 33 and 34. Keep the fixed-point output precision, effective low-precision primitive accuracy and tested condition numbers separate. This is a paper-matched reproduction target, not evidence for arbitrary large or ill-conditioned systems.
- Mannocci et al., *Achieving high precision in analog in-memory computing systems*, npj Unconventional Computing (2026): https://www.nature.com/articles/s44335-025-00044-2. Relevant methods include bit slicing, RNS, ECC and mixed-precision iterative refinement.

## 10. Claim ceiling

A software solver passing known-answer tests does not prove analog hardware correctness. A small residual does not prove a small forward error. A paper-specific 16×16 experiment does not establish arbitrary-size scaling. A condition-number stress bin is not a universal substrate capability threshold. Algorithmic complexity, analog-core energy and end-to-end joules per correct accepted task are separate evidence classes. The error sources should also be recorded using the cross-layer mechanism taxonomy in `CROSS_LAYER_ERROR_BUDGET_V1.md`; never collapse device, circuit, conversion and system errors into a single rate.
