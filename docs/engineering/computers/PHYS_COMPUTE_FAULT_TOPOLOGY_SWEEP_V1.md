# Fault-topology sensitivity pilot v1

**Status:** independent exploratory CPU-only toy model; not a paper reproduction, circuit/device simulation, or physical-device measurement.  
**Owner:** PHYS-COMPUTE-ADAPT-001 #10933  
**Date:** 2026-11-07

## Question

At a fixed nominal 39% failure setting, does matrix factorization behave differently under independent cell faults, output-row loss, output-column loss, and shared latent-channel loss?

## Protocol

- Targets: 32×32 real DFT component (numerical rank 17), normalized Hadamard (rank 32), synthetic rank-8, random orthogonal (rank 32).
- Nominal fault setting: 39%.
- Factorized model: M̂ = (A ⊙ S_A)(B ⊙ S_B), unconstrained real-valued A/B, Adam, 800 steps, learning rate 0.03.
- Latent widths: k=8,16,24,32.
- 2 factor seeds and 3 direct baseline seeds.
- Four distinct masks:
  1. Independent cell faults in A and B.
  2. Output-row loss: rows of A are disabled (entire corresponding output rows disappear).
  3. Output-column loss: columns of B are disabled (entire corresponding output columns disappear).
  4. Shared latent-channel loss: same latent channel is disabled in the corresponding A column and B row.

These fault topologies do **not** share an identical physical denominator. For row/column losses, a whole output dimension is lost; for latent-channel loss, an entire rank-one path is lost. Observed mask survival fractions are included in the JSON.

## Results — cosine similarity

| Target (rank) | Topology | Direct | k=16 | k=24 | k=32 |
|---|---|---:|---:|---:|---:|
| DFT (17) | Independent cells | 0.7835 | 0.9523 | 0.9998 | 1.0000 |
| DFT (17) | Output-row loss | 0.8102 | 0.8130 | 0.8130 | 0.8130 |
| DFT (17) | Output-column loss | 0.8102 | 0.8130 | 0.8130 | 0.8130 |
| DFT (17) | Shared latent-channel loss | 0.7835 | 0.7230 | 0.9388 | 1.0000 |
| Hadamard (32) | Independent cells | 0.7808 | 0.7071 | 0.8660 | 1.0000 |
| Hadamard (32) | Output-row loss | 0.8034 | 0.7071 | 0.8099 | 0.8099 |
| Hadamard (32) | Output-column loss | 0.8034 | 0.7071 | 0.8099 | 0.8099 |
| Hadamard (32) | Shared latent-channel loss | 0.7808 | 0.5270 | 0.6843 | 0.8099 |
| Rank-8 (8) | Independent cells | 0.7692 | 0.9967 | 1.0000 | 1.0000 |
| Rank-8 (8) | Output-row loss | 0.8032 | 0.7940 | 0.7940 | 0.7940 |
| Rank-8 (8) | Output-column loss | 0.8111 | 0.7965 | 0.7965 | 0.7965 |
| Rank-8 (8) | Shared latent-channel loss | 0.7692 | 0.9904 | 1.0000 | 1.0000 |
| Random orthogonal (32) | Independent cells | 0.7678 | 0.7071 | 0.8660 | 1.0000 |
| Random orthogonal (32) | Output-row loss | 0.8034 | 0.7071 | 0.8099 | 0.8099 |
| Random orthogonal (32) | Output-column loss | 0.8034 | 0.7071 | 0.8099 | 0.8099 |
| Random orthogonal (32) | Shared latent-channel loss | 0.7678 | 0.5270 | 0.6843 | 0.8099 |

Direct baseline uses the output-row or output-column mask for those topologies, and an independent 39% cell mask for the independent-cell and latent-channel conditions. The raw parameter ratio for factorized parameters is 2k/n and is not physical device count.

## Interpretation

1. **Independent cell faults are the favorable case for low-rank redundancy.** The rank-8 target reaches 0.9967 at k=16; DFT rank 17 reaches ~0.952.
2. **Output-row and output-column loss are not solved by more latent width.** If the physical output row/column is disconnected, that output is unavailable regardless of factorization rank. At k=16→32, the DFT cosine remains ~0.813; full-rank targets reach only ~0.810 because a fixed share of outputs is lost.
3. **Shared latent-channel loss reduces effective rank.** At k=32 and nominal 39% channel loss, only roughly 61% of latent paths survive on average. The full-rank controls reach ~0.81 rather than near 1.0. A simple redundancy budget to retain 32 active channels at 39% independent channel loss is approximately 32/(1−0.39)=52.5 installed channels (about k=53), before any other hardware overhead. This is a first-order expected-value calculation, not an optimized design or a measured device result.
4. **Fault topology deserves its own benchmark dimension.** A headline “39% fault tolerance” is incomplete unless the fault mechanism, spatial correlation, affected layer, denominator and surviving topology are specified.

## Next design hypotheses

- Use row/column redundancy and spare output lanes for output-path failures; more latent width alone cannot recover a dead output.
- Use overcomplete latent channel banks, error detection and remapping for inner-channel loss.
- Use structured/fault-aware factor placement for clustered defects.
- Compare redundancy against calibration/remapping overhead and total physical resources.
- Keep a digital exact verifier for identity-sensitive tasks.

## Limitations

This is a simplified real-valued binary-mask model, not the author's method. It has no conductance levels, differential pairs, compensation circuits, stuck-on faults, measured correlation, read/write noise, drift, temperature, ADC/DAC, calibration, physical cells, energy, area or yield. Seed counts are small and exploratory. It does not support a SOTA or hardware robustness claim.
