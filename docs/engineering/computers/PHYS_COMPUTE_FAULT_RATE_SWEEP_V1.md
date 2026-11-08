# Fault-rate sensitivity pilot v1 — fault-adaptive matrix factorization

**Status:** independent exploratory CPU-side mathematical sensitivity probe; not a paper reproduction, circuit/device simulation, or physical measurement.  
**Owner:** PHYS-COMPUTE-ADAPT-001 #10933  
**Date:** 2026-11-07

## Question

Does the relative benefit of a two-factor representation depend on both target rank/structure and independent stuck-off fault rate?

## Protocol

- Target matrices: 32×32 real DFT component (numerical rank 17), normalized Hadamard (rank 32), known synthetic rank-8, random orthogonal (rank 32).
- Fault rates: 10%, 25%, 50%.
- Independent Bernoulli stuck-off masks in each factor matrix.
- Factorized model: `M_hat=(A⊙S_A)(B⊙S_B)`; unconstrained real-valued factors; Adam, 800 steps, learning rate 0.03.
- Latent widths: k=8,16,24,32.
- Two factor seeds and three direct-mask seeds per condition.
- Direct baseline: one masked 32×32 matrix; each surviving cell is set to the target value (elementwise least-squares optimum in this simplified unconstrained model).
- Metric: cosine similarity. Raw parameter ratio is `2nk/n²=2k/n`; it is not physical cell count.

## Results — cosine similarity

Each cell is factorized mean cosine at the stated k. Direct baseline is shown separately.

| Fault rate | Target (rank) | Direct | k=8 | k=16 | k=24 | k=32 |
|---:|---|---:|---:|---:|---:|---:|
| 10% | DFT (17) | 0.9502 | 0.6860 | 0.9701 | 1.0000 | 1.0000 |
| 10% | Hadamard (32) | 0.9468 | 0.5000 | 0.7071 | 0.8660 | 1.0000 |
| 10% | Rank-8 (8) | 0.9514 | 0.9988 | 1.0000 | 1.0000 | 1.0000 |
| 10% | Random orthogonal (32) | 0.9430 | 0.5000 | 0.7071 | 0.8660 | 1.0000 |
| 25% | DFT (17) | 0.8619 | 0.6858 | 0.9695 | 1.0000 | 1.0000 |
| 25% | Hadamard (32) | 0.8605 | 0.5000 | 0.7071 | 0.8660 | 1.0000 |
| 25% | Rank-8 (8) | 0.8545 | 0.9587 | 1.0000 | 1.0000 | 1.0000 |
| 25% | Random orthogonal (32) | 0.8503 | 0.5000 | 0.7071 | 0.8660 | 1.0000 |
| 50% | DFT (17) | 0.7091 | 0.6357 | 0.8896 | 0.9883 | 0.9999 |
| 50% | Hadamard (32) | 0.7078 | 0.4988 | 0.7065 | 0.8648 | 0.9983 |
| 50% | Rank-8 (8) | 0.6976 | 0.7671 | 0.9589 | 0.9968 | 1.0000 |
| 50% | Random orthogonal (32) | 0.6987 | 0.4989 | 0.7066 | 0.8652 | 0.9983 |

## Interpretation

1. **Rank-8 target:** factorization is best even at k=8 (half the direct raw parameter count), including at 50% independent stuck-off faults. This is a favorable structured case.
2. **DFT rank-17:** k=16 (equal raw parameter count) is competitive at 10–25% fault rates; k=24 (1.5× raw parameter count) is needed for ~0.988 cosine at 50% faults.
3. **Full-rank controls:** at k=16, Hadamard/random orthogonal remain near 0.707, below the direct baseline at all tested rates. At k=24 they reach ~0.865; near-exact reconstruction requires k=32 (2× raw parameter count).
4. **Fault rate changes the operating point:** as fault rate rises, the direct masked baseline degrades; factorization can preserve structured transforms, but the resource/latent-width cost is target-dependent.

## Limitations

- Only independent stuck-off faults; no stuck-on, spatially clustered, row/column, correlated or time-varying failures.
- Unconstrained real-valued factors; no conductance levels, differential pairs, compensation layers or signed mapping.
- No read noise per inference, write offsets, drift, temperature, retention, endurance, ADC/DAC, peripheral circuit, calibration or routing.
- 2 factor seeds / 3 direct seeds are exploratory; no statistical SOTA or device robustness claim.
- Raw parameter ratio is not a physical-cell/area/energy/yield comparison.
- This is not Xu et al.'s algorithm; it is a clean-room mathematical probe. The author's AGPLv3 + Commons Clause repository was not copied.

## Next falsification tests

1. Spatially clustered, row/column and stuck-on faults.
2. Persistent write/programming offsets versus independent per-read noise.
3. Conductance clipping/quantization, signed differential-pair mapping and compensation layers.
4. Larger matrix sizes and held-out target families.
5. Compare equal physical-cell budgets and equal task quality.
6. Port reference semantics to Rust before any production integration; use physical device data only after independent device/array access.
