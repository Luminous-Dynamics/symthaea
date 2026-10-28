# Coverage–regret–observation-budget sweep — 2026-10-28

**Status:** locally executed synthetic benchmark; 4/4 focused regression tests passed. This is not Symthaea's real SCM likelihood, an external benchmark result, or exact-head CI evidence.

## Protocol

We used 400 calibration and 120 independent test episodes from a synthetic six-regime prior meta-population over the same 16 graph/kernel hypotheses. The test priors are held fixed across settings. Each prior estimate uses nested multinomial observation prefixes of 30, 60, 120 and 240 samples, with symmetric Dirichlet(0.5) smoothing. At each budget we calibrated split-conformal radii for nominal coverage targets 80%, 90%, 95% and 99%. The feature map comprises four graph marginals, four mechanism/kernel marginals, diagonal mass and anti-diagonal mass. For every test episode and each target/budget setting, we solved the constrained robust policy LP for graph, kernel and joint terminal cross-entropy.

Metrics: empirical feature-set coverage with a Wilson 95% interval, worst-regime observed coverage, mean and p95 realized policy regret against the known-prior oracle, feature-width proxy, and maximum two-sided numerical certificate gap.

## Results

| Observations | Nominal target | Radius | Test coverage | Worst regime | Joint regret mean (bits) | Joint regret p95 (bits) | Width proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 80% | 0.16770 | 95/120 (79.2%) | 65.2% | 0.04239 | 0.09845 | 3.354 |
| 30 | 90% | 0.18630 | 108/120 (90.0%) | 78.3% | 0.05342 | 0.10686 | 3.726 |
| 30 | 95% | 0.20350 | 110/120 (91.7%) | 82.6% | 0.06010 | 0.12863 | 4.070 |
| 30 | 99% | 0.23681 | 117/120 (97.5%) | 94.4% | 0.06908 | 0.14767 | 4.736 |
| 60 | 80% | 0.11742 | 89/120 (74.2%) | 52.2% | 0.02116 | 0.06631 | 2.348 |
| 60 | 90% | 0.13452 | 99/120 (82.5%) | 65.2% | 0.02544 | 0.07449 | 2.690 |
| 60 | 95% | 0.14552 | 106/120 (88.3%) | 69.6% | 0.02928 | 0.08518 | 2.910 |
| 60 | 99% | 0.16817 | 112/120 (93.3%) | 88.9% | 0.03922 | 0.08891 | 3.363 |
| 120 | 80% | 0.08487 | 95/120 (79.2%) | 68.4% | 0.01228 | 0.04618 | 1.697 |
| 120 | 90% | 0.09828 | 111/120 (92.5%) | 83.3% | 0.01507 | 0.05345 | 1.966 |
| 120 | 95% | 0.10943 | 118/120 (98.3%) | 94.4% | 0.01856 | 0.06031 | 2.189 |
| 120 | 99% | 0.13270 | 119/120 (99.2%) | 94.4% | 0.02466 | 0.07913 | 2.654 |
| 240 | 80% | 0.05619 | 89/120 (74.2%) | 60.0% | 0.00609 | 0.02776 | 1.124 |
| 240 | 90% | 0.06628 | 105/120 (87.5%) | 80.0% | 0.00724 | 0.03259 | 1.326 |
| 240 | 95% | 0.07306 | 118/120 (98.3%) | 90.0% | 0.00817 | 0.03559 | 1.461 |
| 240 | 99% | 0.09369 | 120/120 (100.0%) | 100.0% | 0.01252 | 0.04613 | 1.874 |

Maximum two-sided certificate gap across all settings/objectives was (2.00001\times10^{-9}) bits, with an explicit (10^{-9})-bit outward margin on each side. This is a floating-point guard, not a formal interval-arithmetic proof.

## Findings

**More observations improved the estimate and decision quality.** At a 90% nominal target, radius contracted from 0.18630 at 30 observations to 0.06628 at 240 observations (about 64% smaller). Mean joint regret fell from 0.05342 to 0.00724 bits (about 86% lower).

**Higher nominal coverage increased the robustness cost at a fixed observation budget.** At 120 observations, increasing the target from 80% to 99% increased mean joint regret from 0.01228 to 0.02466 bits while expanding the radius from 0.08487 to 0.13270.

**Observed coverage is noisy and can fall below nominal.** At 60 observations / 90% target, observed coverage was 82.5%, with Wilson 95% interval 74.7%–88.3%. This single test split is not a valid reason to claim the conformal method lacks its marginal guarantee; it is a reason to increase test/calibration sample sizes, report uncertainty, and avoid interpreting nominal marginal coverage as a subgroup safety guarantee.

**Worst-regime coverage lagged aggregate coverage.** At 120 observations / 90% target, aggregate coverage was 92.5%, but the weakest regime was at 83.3% (graph-skewed). This is exactly the distinction between marginal and conditional reliability highlighted by recent work on conditional coverage diagnostics ([Braun et al., ICML 2026](https://proceedings.mlr.press/v306/braun26c.html)). No method can generically guarantee unrestricted exact conditional coverage in finite samples; the benchmark should instead declare target subgroups/shifts and measure their coverage errors.

## Industry / SOTA comparison plan

A directly relevant baseline is **Multi-Distribution Robust Conformal Prediction** (Yang and Jin, ICML 2026), which aims at finite-sample coverage across heterogeneous source distributions and studies set-size efficiency ([paper](https://proceedings.mlr.press/v306/yang26ad.html)). Add it alongside global split calibration, per-regime calibration, and max-group-radius. Also compare the group-invariant-representation approach to group-conditional coverage ([Melki et al., 2026](https://proceedings.mlr.press/v329/melki26a.html)).

For causal discovery, TCD-Arena is relevant for assumption-violation qualification: its ICLR 2026 benchmark describes 33 distinct violations and approximately 30 million causal-discovery attempts ([TCD-Arena](https://tcd-arena.github.io/)). No external method or causal benchmark was run in this sweep.

The benchmark package is deliberately a calibration/decision-quality diagnostic, not an SOTA claim. It does not show runtime superiority; the policy-game LPs remain tiny, and the experiment reuses a synthetic risk matrix.

## Reproducibility

- Local source SHA-256: `f630526789a8de71a154d1db5d7e7e1d537927f31baa6a2a43c7fe16d8dcf7a9`.
- Focused tests: 4/4 pass; SHA-256 `7aa43a89e6e111899ae38fb6f23c048732499fbadff6b22d20820fcc0626e307`.
- Exact result JSON: 1,436,336 bytes; SHA-256 `c75704734fb5d05764f4585f8ae2ebdc40ceed6fdd9092e57ca4302bb7ee6e40`.
- Deterministic gzip result: 211,083 bytes; SHA-256 `c3fbd62ca916af360b7c9d57d4c67825043ac8400f033c376f7ab68858f8e3d4`.
- Local runtime 16.07 seconds; peak resident memory approximately 132 MB; Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0.

Next gate: 1,000+ test episodes and 1,000/5,000 calibration episodes; coverage targets 80/90/95/99%; observation budgets 30/60/120/240; confidence intervals for overall and worst-regime coverage; then integration of Symthaea's exact SCM action/observation semantics and held-out graph/mechanism families.
