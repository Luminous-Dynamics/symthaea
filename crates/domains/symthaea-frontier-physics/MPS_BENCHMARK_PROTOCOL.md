# U(1) MPS and TEBD benchmark protocol

**Protocol:** 1.0  
**Snapshot:** 2026-11-13  
**Status:** benchmark specification; no cross-framework performance result has yet been measured.

## 1. Objective

Compare the Luminous U(1)-MPS/TEBD path with mature tensor-network implementations at **matched state error**, not by comparing default cutoffs or raw wall-clock times. The benchmark has four separate outcomes:

1. **Mathematical correctness:** charge conservation, gate unitarity, dense-state agreement and canonical-gauge receipts.
2. **Approximation quality:** MPS compression error, Trotter error and floating-point/gauge error reported separately.
3. **Resource efficiency:** peak bond dimension, block sizes, stored scalar count, peak memory, FLOPs where available and runtime.
4. **Evidence quality:** whether every reported error is independently checked and whether the reported bound covers the measured error.

A result is not SOTA merely because it beats a toy dense implementation. It must be matched against established MPS engines with the same circuit, precision, hardware and error target.

## 2. Current field baseline

This is a capability comparison from the projects' current public documentation, not a measured speed ranking.

| Framework | Established capability to match | Important comparison constraint |
|---|---|---|
| Quimb CircuitMPS | Maintains a circuit in MPS form; maximum bond, cutoff and per-gate options; auto-mps uses swap-and-split for two-qubit gates and nonlocal contraction for 3+ qubits; exposes norm-based fidelity/error estimates. | Benchmark eager CircuitMPS and CircuitMPSLazy. Lazy mode uses sub-MPO gates and periodic compression (compress_every and configurable method). |
| ITensorMPS.jl | Abelian QN-conserving MPS; explicit orthogonality center; orthogonalize! moves the center without truncating; truncate! supports cutoff, maxdim and per-bond callbacks. | Compare the same fixed total-QN sector and record callback truncation errors per bond. |
| TeNPy | TEBD with Suzuki-Trotter orders; Schmidt truncation error from discarded squared Schmidt values; canonical MPS machinery. | Keep Trotter discretization error separate from Schmidt truncation error. TeNPy explicitly warns that its truncation metric does not include Trotter error. |
| Qiskit Aer MPS | matrix_product_state simulator; configurable maximum bond dimension and a truncation threshold based on the sum of discarded squared coefficients; can run without a bond cap. | Map threshold semantics by measured discarded weight, not by copying parameter values from another package. |
| NVIDIA CUDA-Q tensornet-mps | GPU MPS backend; configurable maximum bond and absolute/relative cutoffs; single-GPU backend with single/double precision. | GPU claims require the same device, precision and gate workload; distinguish algorithmic wins from hardware acceleration. |

Public references:

- Quimb MPS circuit API: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- Quimb lazy MPS circuit API: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- ITensorMPS MPS/MPO operations and orthogonality center: https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html
- ITensorMPS QN-conserving DMRG: https://docs.itensor.org/ITensorMPS/stable/tutorials/QN_DMRG.html
- TeNPy TEBD: https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.tebd.html
- TeNPy truncation semantics: https://tenpy.readthedocs.io/en/latest/reference/tenpy.linalg.truncation.html
- Qiskit Aer MPS options: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
- CUDA-Q tensor-network MPS backend: https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/tnsims.html

### 2.1 Competitive honesty: current implementation gap

The current U1MpsTensor representation stores its data in a dense Vec<Complex64> of shape \(D_L\times2\times D_R\). It validates U(1) selection rules and performs SVDs inside admissible charge blocks, but **forbidden dense slots are still physically allocated**. It is therefore a charge-labelled, charge-block-computing implementation—not yet a true block-sparse tensor store.

ITensor's QN indices already provide block-sparse storage. Luminous must not claim a memory or throughput advantage until a sparse backing representation is implemented and measured against the dense one. The next storage experiment should preserve the public tensor semantics while storing only admissible \((q_L,s,q_R)\) blocks, and should measure actual scalar count, allocation bytes and contraction time.

## 3. Correctness gates before performance

All correctness gates are mandatory. No timing result can compensate for a failed correctness gate.

### 3.1 Exact small-system oracle

Use dense state vectors as independent oracles for \(N\le12\). Sweep all total-charge sectors \(K=0,\dots,N\), not just half filling.

Required checks:

- input charge leakage and output charge leakage;
- gate unitarity residual and inter-sector gate leakage;
- MPS selection-rule residual and virtual-bond charge identity;
- dense reconstruction max and L2 residual;
- transfer-contracted norm versus dense norm;
- MPS overlap versus dense overlap;
- canonical left/right Gram-isometry residuals;
- measured state-vector error versus the reported global error upper bound.

Initial acceptance targets (double precision, deterministic fixtures):

| Metric | Target |
|---|---:|
| Gate unitarity residual | \(\le10^{-12}\) |
| Inter-sector gate leakage | \(\le10^{-12}\) |
| MPS selection-rule residual | \(\le10^{-12}\) |
| Dense reconstruction max residual | \(\le10^{-10}\) |
| Dense reconstruction L2 residual | \(\le10^{-10}\) |
| Canonical left/right Gram residual | \(\le10^{-10}\) |
| Norm drift for exact gauge moves | \(\le10^{-10}\) |
| Bound coverage | measured error \(\le\) reported upper bound \(+10^{-10}\) |

These are engineering acceptance thresholds, not claims that current code has passed every case.

### 3.2 Strang/Trotter convergence

Use the three-site, one-excitation nearest-neighbour hopping chain with its analytic continuous-time solution. Hold total simulated time fixed and refine the step size:

\[
E(\Delta t)=\|\psi_{\mathrm{MPS}}(\Delta t)-\psi_{\mathrm{analytic}}\|_2.
\]

For a second-order Strang method, halving \(\Delta t\) should asymptotically reduce the global Trotter error by approximately four. Require a ratio in \([3.5,4.5]\) on the deterministic three-site fixture, then expand to noncommuting inhomogeneous chains. Also test forward/backward time-step recovery.

Do not fold this Trotter error into the MPS compression ledger. They are different error sources.

### 3.3 Error-ledger coverage

For each local truncation, preserve:

- discarded squared Schmidt weight;
- retained and discarded rank by charge sector;
- canonical-gauge state-reconstruction residual;
- left/right exterior operator-norm upper bounds;
- weighted local projection contribution;
- accumulated local and global error bounds;
- whether the assumptions needed for a global bound passed.

For a deterministic suite, target 100% bound coverage: every dense-oracle state error must be no larger than its reported upper bound plus the stated floating-point slack. A failed gauge receipt must invalidate global qualification for the remainder of that layer/step, while retaining local diagnostics.

The global bound assumptions must be checked, not inferred from a function name: unitary gates, orthogonal Schmidt projection in a mixed-canonical gauge, and no unreported nonlinear renormalization between truncations. Include finite-precision gauge-rebasing residuals in the ledger.

## 4. Matched framework experiment

For every compared run, freeze and publish:

- initial state vector and its digest;
- site/qubit ordering and bit convention;
- total U(1) sector \(K\);
- exact complex gate matrices or Hamiltonian parameters;
- gate ordering, circuit depth and total simulated time;
- dtype and numerical tolerance;
- bond cap and the **measured** truncation semantics;
- compression cadence (per gate, per layer, periodic/lazy);
- normalization policy;
- CPU/GPU model, memory, thread count, software versions and seeds;
- observables and checkpoint locations.

Cutoff parameters across packages are not semantically interchangeable. Calibrate each implementation by measured discarded weight and dense-reference error at \(N\le12\), then compare at equal achieved error. Include a no-truncation run where feasible to establish the numerical baseline.

For Quimb, include CircuitMPS and CircuitMPSLazy with more than one compression cadence. For ITensor, use QN-conserving indices, move the orthogonality center to the active bond, and record per-bond truncation callback results. For TeNPy, match the same Suzuki-Trotter order and separate its returned Schmidt truncation error from the time-step convergence error. For Aer and CUDA-Q, report simulator method, hardware and precision explicitly.

## 5. Workload ladder

| Workload | Size / sector | Purpose |
|---|---|---|
| Product states and isolated gates | \(N=4\ldots12\), all \(K\) | charge leakage, basis ordering, gate sign and exact factorization |
| Entangled fixed-charge states | \(N=4\ldots12\), all \(K\) | dense-to-MPS, exact split, canonical gauge and overlap |
| Nearest-neighbour hopping | \(N=3,4,6,8,10,12\), several \(K\) | analytic small-chain and dense-oracle correctness |
| Gapped low-entanglement chains | \(N=16,32,64,128\) | scaling where MPS should be effective |
| Critical chains | \(N=16,32,64,128\) | increasing bond growth and truncation sensitivity |
| High-entanglement / chaotic local circuits | \(N=16,32,64\) | expose cases where MPS loses its advantage |
| Peierls-phase / inhomogeneous hopping | \(N=8\ldots64\) | complex phase correctness and symmetry-respecting dynamics |
| Cutoff/cap stress | fixed circuits, sweep max bond and cutoff | characterize accuracy-resource Pareto fronts |

For larger \(N\), no dense oracle exists. Use at least two independent MPS engines and cross-check conserved quantities, observables, overlaps where possible, and time-step refinement. Label this as cross-framework validation, not exact ground truth.

## 6. Metrics and reporting

Report both per-run raw values and aggregate distributions.

**Accuracy:** dense L2 error, infidelity/overlap, observable error, charge leakage, norm drift, Trotter convergence order, local discarded weight, global bound coverage.

**Structure:** peak bond dimension, rank by charge sector, number of nonempty blocks, total admissible scalar count, total allocated scalar count, block occupancy, canonical isometry residual.

**Resources:** peak resident memory, allocated tensor bytes, SVD time, contraction time, total wall time, CPU/GPU utilization and energy where measured.

**Quality of evidence:** source SHA, fixture digest, toolchain version, runner identity, command line, seed, and qualification status. Do not treat queued/skipped CI as a pass.

Use an accuracy-resource Pareto frontier. A faster run that has worse error is not a win unless the point is compared at a common target accuracy. Publish error-vs-runtime, error-vs-memory and error-vs-peak-bond plots; include failed/nonconverged points rather than filtering them out.

## 7. Targets

### Required baseline parity

Before claiming superiority, demonstrate:

1. exact dense agreement on all small fixed-charge fixtures;
2. correct Strang order and second-order convergence;
3. measured global bound coverage on every deterministic dense-oracle case;
4. matched functionality against Quimb CircuitMPS, ITensorMPS QN-MPS and TeNPy TEBD;
5. matched accuracy/resource curves, including a no-truncation baseline.

### Research differentiator

The defensible differentiator is an **evidence-carrying symmetry-respecting evolution trace**: every gate, charge-block split, gauge change and truncation has a receipt; the local and global error channels are separate; a global claim fails closed when its assumptions are not proved.

### Stretch goals (not current results)

- True block-sparse storage with measured memory reduction against current dense-backed U(1) tensors.
- At least 2× reduction in allocated tensor bytes on selected low-entropy fixed-charge workloads, while matching dense-oracle error.
- At least 1.5× throughput improvement on a clearly identified workload/hardware pair at equal achieved error.
- 100% global-bound coverage for deterministic qualification fixtures; unexplained failures are release blockers.
- A reproducible Pareto improvement over at least two independent engines on a nontrivial subset of workload families.

These stretch targets must only be promoted to achieved results after benchmark artifacts, source hashes and exact commands exist.

## 8. Next engineering increments

1. Replace dense-backed tensors with an actual charge-block storage representation; maintain a dense adapter only for small-oracle tests.
2. Use block-aware contractions and block-aware QR/SVD without reconstructing dense forbidden slots.
3. Re-run canonical-gauge and global-bound qualification on the sparse representation.
4. Add higher-order Suzuki compositions and adaptive time-step controllers, reporting their discretization error independently.
5. Add nonlocal MPO application and compare Quimb lazy MPS paths on long-range gates.
6. Run the matched framework matrix and publish raw receipts.

The immediate competitive milestone is not “beat every MPS engine”. It is a measured Pareto win in fixed-charge, low-entanglement workloads while maintaining stronger, independently verified error accounting.
