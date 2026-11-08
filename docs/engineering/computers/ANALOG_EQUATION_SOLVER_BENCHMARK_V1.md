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
- Matrix-free 1-D and 2-D Dirichlet Poisson operators with O(N) vector storage, O(N) MVM and exact spectral condition numbers.
- Stable diagonal-decay ODE with an analytic trajectory oracle and a classical RK4 digital baseline.
- Reproducible symmetric positive-definite family `A = λI + α11ᵀ`, with a known exact spectrum and condition number.
- Known-answer RHS generation from a frozen solution vector.
- Direct Cholesky solve.
- A matrix-free linear-operator interface shared by dense and structured workloads.
- Matrix-free 1-D and 2-D Poisson operators (three-/five-point stencils) with O(N) storage and O(N) MVM; their exact condition numbers are available from the discrete spectra.
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

The 23 Rust unit tests are authored but remain uncompiled/unexecuted in the current authoring environment because `rustc` is unavailable.

## 3. Workload portfolio

### M1 — known-spectrum SPD

Use `A = λI + α11ᵀ` with a known exact solution and exact condition number. Sweep dimensions, `λ`, `α`, vector distributions and precision. This is a controlled correctness baseline, not a substitute for realistic matrix families.

### M2 — diagonally dominant SPD

Generate symmetric strictly diagonally dominant matrices with known reference solutions. Freeze the generator and seed lists before policy tuning.

### M3 — prescribed-spectrum SPD

Generate matrices with declared eigenvalue spectra and orthogonal basis. Record the exact spectrum and condition number. Separate generator/training, validation and held-out confirmatory seeds.

### M4 — sparse and structured systems

The first executable M4 cases are matrix-free 1-D and 2-D Poisson operators. The 1-D operator is:

`(Ax)_i = 2x_i - x_{i-1} - x_{i+1}`

with homogeneous Dirichlet boundary conditions. Its eigenvalues are `4 sin²(kπ/(2(n+1)))`; therefore its exact 2-norm condition number is `cot²(π/(2(n+1)))`.

The 2-D five-point operator on an `nx × ny` interior grid applies `4x[i,j] - x[i-1,j] - x[i+1,j] - x[i,j-1] - x[i,j+1]`, omitting out-of-domain neighbors under the same zero-boundary convention. Its exact eigenvalues are `4 sin²(pπ/(2(nx+1))) + 4 sin²(qπ/(2(ny+1)))`; the 2-norm condition number is the largest such eigenvalue divided by the smallest. Both operators require O(N) vector storage and O(N) work per MVM, unlike dense matrix representation. The 2-D workload is still a controlled scalar-coefficient stencil, not a complete CFD, elasticity, heterogeneous-media or production PDE benchmark.

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

The first Rust oracle currently implements M1, matrix-free 1-D and 2-D Poisson M4 workloads and selected M5 negative controls. M2, M3, broader M4 families and the nonsymmetric solver lane remain protocol targets, not completed generators.

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


## 10. Industry comparison and next performance frontier

Use the official [HPCG benchmark](https://www.hpcg-benchmark.org/) as a workload-family and implementation-quality reference, not as a score Symthaea has already attained. HPCG exercises sparse matrix-vector multiplication, vector updates, global dot products, a local symmetric Gauss-Seidel smoother, sparse triangular solves and a multigrid-preconditioned conjugate-gradient algorithm over nested coarse grids. The current Rust oracle is **not** an HPCG implementation: it currently has unpreconditioned CG and Jacobi-PCG, but no multigrid hierarchy, distributed reductions, halo exchange or HPCG-conformant input/verification path.

The near-term apples-to-apples comparison should include:
- matrix-free 1-D and 2-D Poisson with exact spectrum and frozen known solutions;
- a tuned digital CG baseline;
- a tuned Jacobi-PCG baseline, counting preconditioner construction/application;
- a geometric multigrid-preconditioned CG baseline when implemented;
- an analog-MVM-assisted CG/PCG implementation, counting ADC/DAC, conversion, host movement and all refinement;
- direct analog or native-dynamics results only where the same problem and output-quality contract are supported.

Keep both algorithm-matched and best-co-designed tracks. Record grid shape, coefficient field, stencil, boundary condition, RHS spectrum/smoothness, initial guess, condition number, tolerance, iteration budget, preconditioner setup, operator applications and all verification costs. Include 32×32, 64×64, 128×128 and 256×256 screening grids, then larger sizes only when memory and hardware constraints are explicitly declared. Use anisotropic coefficients and heterogeneous coefficients as held-out follow-on cases; do not infer that a homogeneous Poisson win transfers to them.

The next algorithmic frontier is not adding more unpreconditioned CG iterations. It is testing whether a physically implemented approximate inverse or coarse-grid correction acts as a useful preconditioner, followed by independent high-precision residual/forward-error verification. A valid win must survive full-system accounting and held-out operators. Do not compare an analog core's MVM latency to an entire digital linear solve.

The 2026 fully integrated SRAM analogue closed-loop accelerator provides a relevant industry reference: its demonstrated device uses two 64×64 arrays and includes operational amplifiers, DACs and ADCs; its paper also provides system-level energy/latency/area comparisons and an explicitly labelled 14-nm projection. Treat measured 90-nm test-chip results and projected 14-nm figures as separate evidence classes; do not present the projection as measured silicon. See https://www.nature.com/articles/s41928-025-01549-1.

A second frontier is **adaptive representation under non-ideal hardware**. The 2026 fault-tolerant memristor work represents a target matrix as a product of adjustable submatrices to bypass defective devices; it reports >99.999% DFT-matrix cosine similarity despite 39% device faults, along with other cross-domain demonstrations. This motivates a Symthaea experiment on workload-dependent matrix factorization, device-aware mapping, and exact task-level acceptance under faults. Matrix similarity is a mapping metric, not a guarantee of solution accuracy: test the resulting `Ax=b` solution and forward error directly.

Proposed evaluation outputs:
1. **Correctness:** residual, backward error, forward error, condition number, convergence and false-accept bound.
2. **Whole-system performance:** p50/p95/p99 latency and joules per correct accepted solution, including setup, conversion, calibration, refinement, verifier and recovery.
3. **Resource efficiency:** area, usable cells, arrays, peripheral circuitry, memory footprint, data movement and energy boundary.
4. **Robustness:** yield, stuck-at faults, spatial clustering, drift, temperature, common-cause outages, abstention and recovery.
5. **Generalization:** held-out sizes, spectra, coefficient fields, RHS classes and fault geometries.

No numeric energy or speedup threshold is asserted before a common physical measurement boundary and a qualified hardware implementation exist.

## 11. Claim ceiling

A software solver passing known-answer tests does not prove analog hardware correctness. A small residual does not prove a small forward error. A paper-specific 16×16 experiment does not establish arbitrary-size scaling. A condition-number stress bin is not a universal substrate capability threshold. Algorithmic complexity, analog-core energy and end-to-end joules per correct accepted task are separate evidence classes. The error sources should also be recorded using the cross-layer mechanism taxonomy in `CROSS_LAYER_ERROR_BUDGET_V1.md`; never collapse device, circuit, conversion and system errors into a single rate.
