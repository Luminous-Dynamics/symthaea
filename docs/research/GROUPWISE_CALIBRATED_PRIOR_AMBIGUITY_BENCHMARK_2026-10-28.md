# Groupwise calibrated ambiguity sets: coverage vs regret — 2026-10-28

**Status:** synthetic regimewise calibration diagnostic; local combined suite 24/24 passed. Not Symthaea SCM integration or an external benchmark result.

## Research question

A single split-conformal radius can hit target *marginal* coverage while under-covering an important subgroup. The previous calibration run hit 106/120 (88.33%) at a 90% target, but sparse priors covered only 16/22 (72.7%). This lane tests three alternatives using a fresh seed and independent calibration/test samples:

1. **Global calibration:** one radius from all calibration examples.
2. **Groupwise oracle:** one conformal radius per synthetic regime, using the true regime label at test time. This is a diagnostic upper bound, not deployable unless the regime is observed or reliably classified.
3. **Worst-group radius:** use the maximum of all six group-calibrated radii for every test case; this requires no regime label at inference but can be conservative.

All three methods construct feature-constrained prior polytopes over the same fixed 16-hypothesis support and solve graph, kernel and joint minimax policy games. Regimes are sampled from the declared synthetic meta-population; this does not calibrate real-world uncertainty.

## Results

Configuration: 400 calibration episodes, 120 independent test episodes, 120 multinomial observations per prior estimate, nominal 90% target. Group calibration counts were 69, 71, 68, 65, 70 and 57 for diffuse, sparse, graph-skewed, kernel-skewed, diagonal-correlated and anti-correlated regimes respectively.

| Method | Global coverage | Wilson 95% interval | Mean radius | Joint regret mean (bits) | Joint regret p95 (bits) |
|---|---:|---:|---:|---:|---:|
| Global conformal | 111/120 = 92.5% | 86.36–96.00% | 0.091812 | 0.012046 | 0.045536 |
| Groupwise oracle | 110/120 = 91.67% | 85.34–95.41% | 0.093299 | 0.013840 | 0.048057 |
| Worst-group radius | 117/120 = 97.5% | 92.91–99.15% | 0.100041 | 0.014411 | 0.051054 |

Graph and kernel regret were zero to reported precision for all three methods in this small game. The worst-group method raised the mean radius by about 8.97% relative to the global method and increased mean joint regret by about 0.002365 bits, while improving empirical overall coverage from 92.5% to 97.5%.

### Coverage by regime

| Regime | Global | Groupwise oracle | Worst-group radius |
|---|---:|---:|---:|
| Diffuse | 16/17 = 94.1% | 16/17 = 94.1% | 17/17 = 100% |
| Sparse | 15/18 = 83.3% | 16/18 = 88.9% | 17/18 = 94.4% |
| Graph-skewed | 21/23 = 91.3% | 23/23 = 100% | 23/23 = 100% |
| Kernel-skewed | 15/16 = 93.8% | 14/16 = 87.5% | 15/16 = 93.8% |
| Diagonal-correlated | 19/20 = 95.0% | 16/20 = 80.0% | 19/20 = 95.0% |
| Anti-correlated | 25/26 = 96.2% | 25/26 = 96.2% | 26/26 = 100% |

The realized subgroup fractions are noisy because per-group test counts are only 16–26. Finite-sample groupwise conformal coverage is a probability guarantee under within-group exchangeability, not a promise that every small test subgroup will empirically exceed 90%. The universal max-radius method showed the strongest subgroup coverage in this particular test, but the set is more conservative and still missed 3 of 120 draws.

## Interpretation

- **Global conformal** has the best observed joint regret in this run, but does not meet 90% in the sparse subgroup (15/18 covered).
- **Groupwise oracle** uses test-time true regime identity, so its 91.67% aggregate coverage is a diagnostic only. Its low observed diagonal-regime rate (16/20) reflects test noise and does not falsify the method's marginal group guarantee.
- **Worst-group radius** avoids needing a regime label and covered 117/120, including 17/18 sparse priors. This comes at a larger uncertainty set and slightly higher decision regret.
- The practical goal is a calibrated coverage–regret–cost frontier, not maximum coverage or minimum regret in isolation.

## Reproducibility

- Local combined regression suite: **24/24 passed** (Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0).
- Local script SHA-256: `f509f056270b670d6ab0f00e03a1c7102f2c624169912079258bc344a183219c`; local Git blob `8621fd0bb4b4e6d4fe3a218f2d4c4804f94f84e0`; committed branch blob `d2f11ec5114cb07307e4c85fe1228a7749b8948c` (same byte length, not byte-identical).
- Local test SHA-256: `39bb8e67f7aff18a2b7b13e09334fb4a6a3576bad988a41fd743f80f9fae2f4c`; test is byte-identical to branch blob `2ca8d6ca2be60af6bebdf2660353fcd903bb7836`.
- Exact local result JSON: 73,433 bytes, SHA-256 `3cfb6350f12db29e6cb871409f2c3e79cfe390fce00e73ea456062de22033085` (local-only, not committed).
- Lossless gzip result: 6,100 bytes, SHA-256 `be8953084b09c72e49d1296cc141adfe15920deb5acbedea7462f201f27fb9da` (local-only, not committed).

The local executed script and test should be compared byte-for-byte with their committed branch copies before using this result as exact-head evidence. The synthetic risk model is inherited from the previous standalone pilot; the code is not yet integrated with Symthaea's actual SCM likelihood/observation semantics.

## Literature context

- [Multi-Distribution Robust Conformal Prediction (ICML 2026)](https://proceedings.mlr.press/v306/yang26ad.html) studies finite-sample coverage across multiple source distributions; relevant to the regimewise diagnostic.
- [When Robustness Meets Conservativeness: Conformalized Uncertainty Calibration for Balanced Decision Making](https://arxiv.org/abs/2510.07750) explicitly studies the finite-sample miscoverage–regret frontier.
- [Front-Loaded Robust Conformal Prediction (ICML 2026)](https://proceedings.mlr.press/v306/h-zargarbashi26a.html) explores spending more computation at calibration time to reduce test-time conservatism.
- [CausalArena](https://arxiv.org/abs/2609.11897), [ACDB](https://arxiv.org/abs/2609.31675) and [TCD-Arena](https://tcd-arena.github.io/) remain the intended external qualifications, none yet run in this lane.

These are related prior art; this experiment does not claim novelty or SOTA.

## Next gates

1. Byte-align local/committed source and rerun the full grid from exact branch bytes.
2. Increase per-regime calibration sample size and report confidence intervals, not just point coverage.
3. Compare global, groupwise, worst-group, and multi-distribution conformal aggregation over target levels 80/90/95/99%, with coverage, regret and set-size proxies.
4. Learn a regime classifier from prior-estimate features on a training split; report groupwise-oracle performance only as a ceiling and evaluate classification errors separately.
5. Integrate the actual Symthaea SCM action and likelihood semantics; then run held-out mechanism families and matched ACDB/CausalArena/TCD-Arena baselines.
