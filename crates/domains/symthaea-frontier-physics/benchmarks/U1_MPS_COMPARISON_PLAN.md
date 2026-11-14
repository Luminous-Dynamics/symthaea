# U(1) MPS comparison and qualification plan

**Status:** benchmark contract and workload definitions authored; no external baseline measurements are claimed yet.

## 1. The field and the actual gap

The project should not claim novelty for MPS, charge conservation, two-site gates, SVD compression, or even/odd TEBD. Established systems already provide these:

| Reference | Existing capability to match |
|---|---|
| Quimb CircuitMPS / CircuitPermMPS | MPS circuit simulation, maximum bond dimension, singular-value cutoff, automatic swap-and-split for two-qubit gates, nonlocal methods for larger gates, eager/lazy/permutation-tracked execution, and fidelity/error estimates |
| ITensorMPS | Abelian quantum-number-conserving MPS, block-sparse tensors, fixed total-QN sectors, and symmetry-aware DMRG/MPS operations |
| Qiskit Aer MPS | Optional maximum retained Schmidt rank, discarded-squared-weight threshold, MPS bond/discard logging, plus CPU MPS and GPU tensor-network methods |
| NVIDIA cuTensorNet | Gate-split and tensor-SVD truncation, path optimization/slicing, distributed decomposition paths, and a preview MPS factorization interface |
| MPS TDVP references | One-/two-site variational evolution and a separate projection/Krylov/time-discretization error taxonomy for long-range MPO Hamiltonians |

External references:
- Quimb CircuitMPS API: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- Quimb MPS guide: https://quimb.readthedocs.io/en/latest/circuit/circuit-mps.html
- ITensor quantum-number conserving DMRG: https://docs.itensor.org/ITensorMPS/stable/tutorials/QN_DMRG.html
- Qiskit Aer MPS options: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
- NVIDIA cuTensorNet 26.11 overview: https://docs.nvidia.com/cuda/cuquantum/26.11.0/cutensornet/overview.html
- NVIDIA cuTensorNet 26.11 functions: https://docs.nvidia.com/cuda/cuquantum/26.11.0/cutensornet/api/functions.html
- MPS TDVP overview: https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html
- MPS TEBD overview: https://tensornetwork.org/mps/algorithms/timeevo/tebd.html

The current U(1) implementation performs charge-block SVD and validates sector labels, but its site-tensor payload is still a dense Vec with structurally forbidden entries represented as zeros. This is an important gap versus genuinely block-sparse storage such as ITensor. Do not claim memory superiority until packed charge-block storage is implemented and measured.

## 2. Proposed paradigm shift: certified cost-to-accuracy

The primary comparison metric is not raw gates per second. It is:

    C(N, circuit, epsilon) =
      minimum measured wall time subject to
      independently measured global state L2 error <= epsilon,
      charge conservation passing,
      and every applicable error certificate passing.

Report peak memory and retained charge-block ranks beside C. This makes accuracy, resource use, and trustworthiness one operating point rather than three disconnected charts.

The current code separates:
- canonicalization / representation error;
- gate unitarity and charge-sector defects;
- Hamiltonian-model projection error;
- Schmidt truncation / compression error;
- Trotter / time-discretization error.

The global certificate intentionally excludes Trotter error until step-refinement or a commutator bound is supplied. A local two-site residual budget is not automatically a global state-vector budget; the mixed-canonical receipt is required.

The new absolute error-budget compressor selects the smallest global Schmidt rank meeting a requested local L2 budget, checks the measured reconstruction residual, and fails closed if a hard maximum bond cannot meet the request. This complements, rather than replaces, fixed cutoff and bond-cap controls.

## 3. Workload ladder

| ID | Workload | Purpose | Oracle |
|---|---|---|---|
| U1-RABI-2 | Two-site single-excitation Rabi evolution | Exact gate, complex phase, Hamiltonian exponential | Analytic 2x2 exponential |
| U1-W-3 | Three-site one-excitation W state, all two-site centers | Dense-to-MPS, gauge sweeps, local compression | Full dense vector |
| U1-XX-3 | Three-site nearest-neighbour XX chain | Strang order, layer sequencing, energy and norm | Dense statevector / analytic one-particle sector |
| U1-XXZ-8 | Eight-site half-filled Néel quench | Moderate entanglement, fixed-Q sector | Dense statevector |
| U1-XXZ-12 | Twelve-site half-filled chain | Dense crossover and error-certificate coverage | Dense statevector |
| U1-BRICKWALL-16 | Seeded random U(1)-preserving brick-wall circuit | Error-budget rank adaptation and entanglement growth | Dense statevector |
| U1-BRICKWALL-32 | Same generator, larger chain | Throughput and memory scaling | Cross-simulator convergence |
| U1-BRICKWALL-64 | Same generator, larger chain | Large-N scaling and cap saturation | Cross-simulator convergence |
| U1-FILLING-SWEEP | K=1, K=N/4, K=N/2, K=N-1 | Low-/mid-/high-filling sector sparsity | Dense for small N; conserved observables thereafter |

Use fixed published seeds and emit the generated gate/Hamiltonian matrices as fixture data. A test should never regenerate a supposedly fixed benchmark from a mutable library RNG without recording its version and seed.

## 4. Fairness contract

Every comparison run must freeze and emit:
1. Exact initial state amplitudes or a canonical digest plus a reconstruction source.
2. Exact gate/Hamiltonian matrices, bond order, qubit/site ordering, and total charge.
3. Scalar precision and backend (complex64/complex128; CPU/GPU).
4. Bond cap and truncation semantics, including normalization behavior.
5. Evolution time, time step, splitting order, and measurement checkpoints.
6. Thread count, GPU model/driver/library version, compiler/build mode, and hardware.
7. Warm-up count, measured repetitions, median, p95, peak resident memory, and allocation/temporary-buffer policy.

Run all CPU implementations with the same pinned thread count. Run GPU implementations in a separate table. Never compare a warmed GPU result to a cold CPU result or a dense exact run to a compressed approximate run without reporting the achieved error.

Quimb, ITensor, Aer and cuTensorNet use different cutoff conventions. Translate each run to independently measured dense-reference L2/infidelity at N<=12. Do not equate a discarded-weight parameter with state L2 error without matching state norm and normalization semantics.

## 5. Acceptance gates and research targets

### Correctness / qualification gates

- No-truncation dense-reference L2 residual <= 1e-10 for supported N<=12 fixtures.
- U(1) total charge is unchanged; selection-rule residual <= 1e-12 for exact controls.
- Canonical left/right Gram residual <= 1e-10 on every declared exterior site.
- For every qualified global error receipt, actual dense-reference L2 error must not exceed the recorded bound (allow only the declared floating comparison tolerance).
- Zero false-success cases over at least 1,000 seeded feasible/infeasible error-budget requests.
- Every infeasible hard-cap request must fail closed with the best achievable measured residual.
- Strang global error ratio should approach 4 under step halving in the asymptotic regime; Trotter error is reported separately.

### Performance targets (not yet measured)

- First establish an honest baseline on the same CPU and precision. Do not optimize against unmatched settings.
- Near-term target: reach within 2x of Quimb's median runtime on matched nearest-neighbour CPU workloads at equal measured error for N<=32.
- Research target: reduce storage and SVD work in low-/moderate-filling sectors using packed charge blocks; compare bytes actually stored, bytes allocated, and peak RSS.
- Research target: for each fixed error budget, choose a smaller or equal bond dimension than a fixed relative-cutoff policy while meeting the same measured L2 target.
- Do not claim a performance win until benchmark receipts include the exact source SHA, baseline versions, hardware, and matched error.

### Scale gates

- N=8,10,12: dense oracle and certificate coverage.
- N=16,24,32: performance and memory crossover.
- N=48,64: bounded-bond scaling and cap saturation.
- N=128: only after the N=64 memory/runtime curve is stable and a representative workload remains tractable.

## 6. Immediate algorithmic roadmap

1. Validate absolute error-budget compression and canonical/global receipts on exact heads.
2. Implement a true packed charge-block storage representation; measure structural-zero elimination before changing the production tensor type.
3. Add an adaptive circuit-level budget allocator that distributes a requested global error budget across upcoming gates using measured exterior operator-norm bounds and remaining budget.
4. Add independent observables: energy, local density, two-point correlators, and current, with bias envelopes.
5. Add checkpoint/replay from gate receipts so any failed bound or divergence is reproducible from an exact source state and transition.
6. Compare Quimb eager/lazy/permutation MPS, ITensor QN MPS, Aer MPS, and cuTensorNet only on supported equivalent modes.

The desired differentiator is an auditable error/resource envelope, not a claim that the underlying tensor-network primitives are new.

## Current baseline parameter semantics (verified 2026-11-14)

- **Quimb CircuitMPS** exposes maximum bond dimension and singular-value cutoff controls, with automatic two-qubit swap-and-split and nonlocal handling for larger gates. Quimb also supplies norm-based fidelity/error estimates. Record those estimates, but independently compare dense statevectors for N<=12; do not use the Quimb estimate as the oracle. CircuitMPSLazy and CircuitPermMPS need separate rows because delayed compression and lazy qubit reordering change the work performed.
- **Qiskit Aer MPS** exposes maximum retained bond dimension, a truncation threshold defined through the sum of squared discarded Schmidt coefficients, and optional MPS log data including bond dimensions and discarded values. Translate this threshold to measured L2/infidelity on the same dense fixtures; do not identify the threshold itself with global state L2.
- **NVIDIA cuTensorNet 26.11** exposes absolute, relative and discarded-weight cutoff modes, an output-bond extent, and a singular-value normalization option in its ProjectionMPS insertion path. Its MPS factorization example also sets L2 singular-value normalization. Run both normalization-preserving and no-renormalization variants only when each is explicitly recorded. The docs distinguish distributed tensor QR/SVD support from gate-split support: distributed decomposition is available in supported SVD paths, while gateSplit currently does not accept distributed tensors.
- **ITensorMPS** is the strongest storage baseline for this programme: QN-labelled indices yield block-sparse tensors and preserve total Abelian QN. The documentation says the current QN feature set is Abelian, which is an appropriate match for U(1). Separate CPU and GPU comparisons: ITensor's GPU documentation presently warns that QN block-sparse GPU operations are still less mature than dense operations and may not work or may be slower.
- **Our packed prototype** currently only round-trips one site tensor. It is not yet integrated into MPS contraction, canonicalization, gate splitting or TEBD; do not compare it as a complete simulator.

The relevant references are the Quimb CircuitMPS API and its fidelity/error methods, Qiskit Aer 0.18.2 MPS options, cuTensorNet 26.11 ProjectionMPS/gate-split documentation and ITensorMPS QN/GPU documentation. Links are enumerated in the external references above and in the comparison fixture.
