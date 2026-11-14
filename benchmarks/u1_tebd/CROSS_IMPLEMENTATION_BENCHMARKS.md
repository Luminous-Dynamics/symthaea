# U(1)-MPS cross-implementation benchmark contract

## Purpose

This programme compares Symthaea's charge-labelled MPS/TEBD implementation against Quimb CircuitMPS and ITensorMPS on identical finite-chain problems. It is not a speed-only bake-off and it does not assume that library cutoff numbers have equivalent meanings.

The first release pins a common problem description, basis convention, Strang layer schedule, and receipt schema. Each implementation must emit the same JSON record shape. Small systems use a dense state-vector oracle; larger systems use matched observables, conserved-charge checks, norm drift, and step-refinement studies.

## Current reference implementations

- **Quimb 1.16.0**: CircuitMPS supports a maximum bond dimension, cutoff, gate options, and automatic MPS gate handling. Its `auto-mps` method uses swap-and-split for two-qubit gates and supports nonlocal MPO routes for larger gates. The API exposes norm-based fidelity and error estimates. Reference: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/mps/index.html
- **ITensorMPS 0.4.1 / ITensors 0.9.x**: supports QN-conserving MPS and TEBD by constructing two-site gates and applying the gate vector with cutoff/max-dimension controls. Abelian quantum numbers are supported; the conserved total QN is inherited from the initial state. References: https://docs.itensor.org/ITensorMPS/stable/tutorials/QN_DMRG.html and https://docs.itensor.org/ITensorMPS/stable/tutorials/MPSTimeEvolution.html
- **TeNPy 1.1.1** is an additional scientific reference for canonical-form diagnostics and Schmidt truncation accounting, not part of the first automated runner. Its documentation explicitly distinguishes discarded Schmidt weight from Trotter discretization error: https://tenpy.readthedocs.io/en/latest/reference/tenpy.linalg.truncation.html

The Quimb release is pinned to 1.16.0 (PyPI release 10 November 2026). ITensorMPS is pinned to 0.4.1. Record exact resolved dependency versions and the Julia/Python versions in every output file.

## Shared physical convention

- Computational basis order: `|00>, |01>, |10>, |11>`.
- Chain state-vector order: site 0 is the most-significant bit.
- Bit 1 is one unit of U(1) excitation; bit 0 is vacuum.
- Total charge: (K=\sum_j n_j).
- Local Hamiltonian on bond ((j,j+1)):
  [
  H_j=\mu_L n_j+\mu_R n_{j+1}+V n_jn_{j+1}
       +h\,|01\rangle\langle10|+h^*\,|10\rangle\langle01|.
  ]
- The two-site gate is (G_j(\tau)=\exp(-i\tau H_j)).
- One Strang step is even bonds at (dt/2), odd bonds at (dt), then even bonds at (dt/2). Gates in a parity layer are disjoint and execute in ascending bond order.
- No implementation may silently renormalize the state. If the library necessarily normalizes, record the exact scale factor and mark the no-renormalization comparison as not matched.

## Two fair comparison tracks

### Track A — fixed bond cap

Set the same global maximum bond dimension in every implementation and disable singular-value cutoffs. Symthaea uses `relative_singular_value_cutoff=0`; Quimb uses `cutoff=0, cutoff_mode="rel"`; ITensorMPS uses `cutoff=0` with the same `maxdim`. Report any library-specific behavior that still removes exact/near-zero directions.

This track compares the implementation cost at a controlled storage cap. Do not interpret the same numerical cutoff value as equivalent when cutoff modes differ.

### Track B — matched measured error

Choose a target measured state error (initial targets: (10^{-4},10^{-6},10^{-8}) for small dense-oracle cases). Tune each library's *native* cutoff and maximum bond independently to meet the same target, then compare runtime, peak memory and actual maximum bond. Record the full cutoff mode and parameters.

Quimb supports relative-singular-value (`rel`) and relative-discarded-squared-weight (`rsum2`) modes, among others. These are different contracts. ITensor's cutoff is a truncation-error threshold on its SVD/TEBD path. Do not compare raw cutoff scalars across these libraries without translating them through the measured discarded weight and dense-state error.

## Oracle and error decomposition

For (N\le12), compute two independent references:

1. **Discrete-circuit oracle**: apply the identical ordered sequence of two-site matrices to a dense state vector without compression. This isolates MPS representation, gauge and compression error from Trotter error.
2. **Continuous-time oracle**: diagonalize/exponentiate the full dense Hamiltonian (H=\sum_j H_j) and compare at the same physical time. This measures Strang discretization error separately.

For larger N, do not construct a dense state. Compare conserved-charge expectation/variance, norm drift, local density profiles, energy, current, entanglement/bond profiles, and time-step refinement.

A passing global error certificate must bind the exact initial-state fingerprint, gate sequence digest, Hamiltonian digest, canonical-gauge receipts, gate-unitarity/charge residuals, projection residuals, and the cumulative bound. The certificate is for the discrete Strang circuit; it is not a bound on Trotter error.

## Required output fields

Each result record must include:

- source commit SHA, implementation name/version, language/runtime, BLAS backend/thread count, CPU model, OS, and benchmark fixture digest;
- case id, N, K, initial bit string, Hamiltonian parameters, dt, step count, Strang order and deterministic gate order;
- cutoff mode/threshold, maxdim, normalization policy, actual peak bond and per-charge ranks where available;
- runtime median/p95 over at least 20 post-warmup runs, peak RSS, allocation/peak memory estimate, and gate throughput;
- input/output norm squared and drift, total-charge leakage probability or a structural QN guarantee, maximum selection-rule residual;
- per-gate truncation/discarded weights, canonical left/right isometry residuals, gate-unitarity residual, and global error-bound qualification;
- dense-discrete-oracle L2 error and fidelity for N<=12;
- continuous-time error and time-step refinement ratio where an exact dense evolution is supplied.

Do not substitute library-estimated fidelity for the dense oracle when the dense oracle is tractable. Library estimates are reported as a separate metric.

## Acceptance ladder and targets

| Tier | Case size | Acceptance |
|---|---:|---|
| Correctness smoke | N=2–3 | Discrete-oracle L2 <= 1e-12 without truncation; charge leakage <= 1e-12; norm drift <= 1e-12 |
| Dense qualification | N=4–12 | Dense-discrete fidelity >= 1-1e-10; state L2 <= 1e-8 at the declared target; gauge residual <= 1e-10 |
| Trotter order | N=3–8 | Strang refinement ratio 3.5–4.5 for dt -> dt/2 in the asymptotic second-order regime |
| Mid-scale | N=16–64 | Same-observable error envelope, max bond <= declared cap, charge/norm receipts complete |
| Performance | N=32–128 | At matched error and hardware, report throughput/memory frontier; no superiority claim until repeated medians and confidence intervals support it |

**Aspirational industry-beating target (not a current result):** at the same measured dense-oracle error, reach at least 2x gate throughput or 30% lower peak memory than the best tested CPU baseline on a representative low-entanglement U(1) workload, while preserving a machine-checkable per-gate receipt. If this target is not met, publish the Pareto frontier and identify which resource dimension is competitive.

## Why this could be a real contribution

MPS storage, U(1) conservation, canonical forms, TEBD and cutoff-based SVD are established capabilities in Quimb, ITensor and TeNPy. We do not claim those as novel. The differentiator to test is **proof-carrying numerical evolution**: a state transition has a replayable input, exact gate identity, symmetry-sector receipt, canonical-gauge witness, approximation decomposition, and a fail-closed statement of which error bounds are and are not qualified.

The paradigm shift is from “the simulation returned a plausible state” to “the simulation returned a state plus a machine-checkable numerical argument for what was preserved, what was approximated, and what remains unbounded.”

## Reproducibility rules

- Freeze fixtures and gate ordering before a run; do not let each library choose a different Trotter decomposition.
- Separate compile/JIT warm-up from steady-state time. Report both cold-start and warm runtime where meaningful.
- Run at least 20 timed repeats after 5 warm-ups; report median and p95, not the single fastest run.
- Fix thread counts and record CPU/BLAS/runtime versions.
- Never mix dense oracle time into MPS runtime.
- Publish raw per-case JSON, not only aggregate charts.
- CI correctness smoke tests are gates; performance tests are informational and must not use unstable absolute thresholds on shared runners.
