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


## Reliability sizing — expected value is not availability

For the shared latent-channel model, if each installed channel independently survives with probability 0.61, then the probability of retaining at least 32 active channels is:

```
P(X >= 32),  X ~ Binomial(n, 0.61)
```

The exact binomial-tail calculation gives:

| Required availability | Minimum installed channels | P(X ≥ 32) at this n | P(X ≥ 32) at n−1 | Raw parameter ratio 2n/32 |
|---:|---:|---:|---:|---:|
| 95% | 63 | 96.2047% | 94.8822% | 3.9375× |
| 99% | 68 | 99.2878% | 98.9821% | 4.2500× |
| 99.9% | 74 | 99.9322% | 99.8973% | 4.6250× |
| 99.99% | 79 | 99.9925% | 99.9882% | 4.9375× |

At n=53, the expected active count is 32.33, but the chance of having at least 32 active channels is only 59.61%. Therefore 53 is an expectation-only estimate and must not be used as a reliability design.

These calculations assume independent identical Bernoulli failures. Spatially correlated/clustered defects, common-mode failures, nonuniform channel quality, repair/remapping and non-binomial aging violate that assumption. A physical design needs empirical fault distributions, confidence intervals and an explicit outage/availability target. Parameter ratios exclude all differential pairs, conductance, compensation, peripheral, routing, calibration and controller overhead.


## 3. Correlated channel-survival stress model — beta-binomial

The preceding availability table assumes independent, identically distributed channel survival. To expose sensitivity to shared array/die/environment conditions, a second analytical model uses an exchangeable beta-binomial distribution.

Let a shared latent channel-survival probability be drawn for an array/group:

```
P ~ Beta(alpha, beta)
X | P ~ Binomial(n, P)
E[P] = p = 0.61
ICC = rho = 1 / (alpha + beta + 1)
alpha = p * (1/rho - 1)
beta  = (1-p) * (1/rho - 1)
```

For rho=0, this reduces to the independent binomial model. For rho>0, outcomes share a random array-level survival condition and have positive pairwise correlation rho. This is an exchangeable common-cause stress model, **not a spatial layout or wafer defect map** and not a claim that actual K26/memristor devices have these correlation values.

### Minimum channel count for at least 99% probability of retaining 32 active channels

| Survival-outcome ICC rho | Minimum installed channels | Achieved P(X ≥ 32) |
|---:|---:|---:|
| 0% | 68 | 99.2878% |
| 0.1% | 68 | 99.1112% |
| 0.5% | 71 | 99.2416% |
| 1% | 73 | 99.0157% |
| 2% | 79 | 99.0401% |
| 5% | 98 | 99.0538% |
| 10% | 137 | 99.0124% |
| 20% | 289 | 99.0008% |

With 53 installed channels, the probability of retaining at least 32 falls from 59.61% at rho=0 to 55.37% at rho=20% under this model. The 53-channel figure is only an expected-capacity estimate.

### Availability target sensitivity

| ICC rho | 95% availability | 99% availability | 99.9% availability | 99.99% availability |
|---:|---:|---:|---:|---:|
| 0% | 63 | 68 | 74 | 79 |
| 0.1% | 63 | 68 | 75 | 80 |
| 0.5% | 64 | 71 | 78 | 85 |
| 1% | 66 | 73 | 83 | 92 |
| 2% | 69 | 79 | 93 | 107 |
| 5% | 79 | 98 | 128 | 165 |
| 10% | 96 | 137 | 220 | 344 |
| 20% | 146 | 289 | 755 | 1,951 |

Values are minimum installed channels under the exchangeable beta-binomial model, for 32 active channels and mean survival 0.61. The full probabilities and n−1 checks are in the JSON fixture.

### Engineering interpretation

1. A small positive common-cause correlation materially increases redundancy at high availability targets.
2. At rho=5% and 99% availability, the model requires 98 installed channels rather than 68.
3. At rho=20%, it requires 289. This is a stress test showing how sensitive overprovisioning can be—not a prediction of actual device yield.
4. Redundancy alone may become uneconomic. The architecture should compare fault screening, spatial interleaving, independent power/control domains, spare rows/columns, remapping, compensation layers and array partitioning.
5. Measure failure locations and covariance at cell, row, column, subarray, die and lot levels. Fit empirical hierarchical/spatial models; do not infer rho from a single global fail percentage.

### Reproducible calculator

`research/physical_compute/beta_binomial_redundancy.py` is a dependency-free Python research calculator. It uses a recurrence for the beta-binomial PMF and supports rho=0 as the binomial limit. It is research tooling, not production reliability software. The script was run locally against the tabulated scenarios and its outputs were checked against an independent SciPy calculation; repository CI/workflow records are still the qualification authority for the committed version.

### Model limitations

- The beta-binomial is exchangeable: every channel pair has the same correlation. It does not represent adjacent-cell clusters, row/column geometry, spatial gradients or multiple nested common-cause domains.
- A single global random survival probability per array/group is only one possible shared-cause mechanism.
- Actual channel-level dependence must be measured across multiple arrays, dies, lots, temperatures and aging conditions.
- Correlated failure may affect outputs, power, clocks and sense amplifiers in ways not represented by a channel-count model.
- The raw parameter ratio remains an abstract parameter count, not physical area, energy, yield or cost.
