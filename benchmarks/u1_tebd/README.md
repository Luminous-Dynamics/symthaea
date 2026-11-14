# U(1)-MPS cross-implementation baseline runners

These scripts implement the same fixed-charge nearest-neighbour Hamiltonian and the same second-order even/odd/even Strang gate ordering in two independent MPS libraries.

## Quimb baseline

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r benchmarks/u1_tebd/external/quimb/requirements.txt
python benchmarks/u1_tebd/external/quimb/run_quimb_baseline.py \
  --case u1-two-site-hopping-known-answer \
  --track fixed_bond --warmups 5 --repeats 20 \
  --output benchmarks/u1_tebd/results/quimb-two-site.json
```

Quimb's dense `to_dense()` state is compared with an independently applied dense gate sequence for N<=12. For cases with N<=10, a separate sparse-Hamiltonian `expm_multiply` oracle measures continuous-time error. Quimb's norm-based `fidelity_estimate()` and `error_estimate()` are retained as separate library diagnostics, never substituted for the dense oracle.

## ITensorMPS baseline

From the repository root:

```bash
julia --project=benchmarks/u1_tebd/external/itensor -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks/u1_tebd/external/itensor \
  benchmarks/u1_tebd/external/itensor/run_itensor_baseline.jl \
  --case=u1-two-site-hopping-known-answer --track=fixed_bond \
  --warmups=5 --repeats=20 \
  --output=benchmarks/u1_tebd/results/itensor-two-site.json
```

ITensorMPS creates `S=1/2` sites with `conserve_qns=true`, encodes excitation 0 as `Up` and excitation 1 as `Dn`, and applies the same local gate matrices through `apply(gates, psi; cutoff, maxdim)`. Its cutoff is the library's native SVD truncation-error threshold and is not numerically interchangeable with Quimb's `rel`/ `rsum2` modes. The runner does not call `normalize!`; it records input/output norm drift.

The ITensor runner exports a dense state for N<=12. The full dense continuous-time Hamiltonian oracle is restricted to N<=10 to bound memory. Larger cases emit QN structural-conservation, norm, max-link-dimension and expectation-value measurements without allocating a dense vector.

## Comparison rules

- **Fixed-bond track:** same max bond, cutoff disabled, same exact local gates and order. Record native cutoff semantics.
- **Matched-error track:** tune each native cutoff independently until the measured dense-oracle state L2 target is met; compare elapsed time and actual maximum bond only after matching the error.
- Runtime excludes oracle construction and dense expansion. Cold-start/JIT and warm-state costs should be reported separately when profiling package startup.
- The first CI smoke test runs only the two-site known-answer case with short timings. Performance conclusions require at least 20 post-warm-up measurements on controlled hardware.
- These runners are benchmark baselines, not a claim that Symthaea currently outperforms either library.

