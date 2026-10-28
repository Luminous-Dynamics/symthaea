# Split-calibrated prior ambiguity sets for adaptive causal probing — 2026-10-28

**Status:** local synthetic pilot; 22/22 combined regression tests passed. The current branch includes source, but the branch version and locally executed source are not byte-identical, and no exact-head or Symthaea-SCM integration claim is made. No external causal benchmark was run.

## Research question

Can Symthaea replace hand-selected or fixed-width prior ambiguity sets with an uncertainty radius calibrated on a held-out sample, so robustness is tied to observed uncertainty rather than arbitrary marginal bounds?

The new lane samples a synthetic meta-population of priors over the same 16-hypothesis graph/kernel support. Each synthetic episode yields a prior (q), a multinomial sample of 120 latent draws, and a smoothed estimate \hat q. A feature map collects four graph marginals, four kernel marginals, diagonal mass and anti-diagonal mass. The conformity score is the maximum absolute difference between the feature vectors of q and \hat q.

The radius is selected from 400 independent calibration scores using the finite-sample split-conformal order statistic at target coverage 90%. On each of 120 independent test episodes, the resulting set is
[
mathcal Q(\hat q,ho)={qinDelta^{15}: |Cq-C\hat q|_inftyleqho}.
]
For each test episode, the solver chooses a robust policy over this set and evaluates its realized expected terminal risk against the known synthetic true prior. Baselines are the old fixed marginal set, the old six-prior policy, and the full-simplex policy. The risk matrix is reused across test draws; only the ambiguity set and true prior change.

## Main result

Calibration radius: **0.0938855456** feature units.

The empirical coverage rate for the calibrated feature set was **106/120 = 88.33%**. Its Wilson 95% interval is **81.37%–92.92%**, which includes the 90% target. The fixed hand-chosen marginal set [0.10, 0.40] covered **11/120 = 9.17%** of the test priors from this mixed synthetic meta-population.

| Objective | Calibrated ambiguity set mean regret (bits) | p95 regret | Fixed [0.10, 0.40] set mean regret | Six-prior policy mean regret | Full-simplex policy mean regret |
|---|---:|---:|---:|---:|---:|
| Graph | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Kernel | 0.000000 | 0.000000 | 0.000000 | ~0.000000 | ~0.000000 |
| Joint | **0.013793** | **0.058482** | 0.097809 | 0.021072 | 0.106332 |

All calibrated test cases closed the benchmark's numeric two-sided LP certificate to the declared tolerance. The 1e-9-bit outward margin is a numeric guard, not interval arithmetic.

The calibrated set reduces mean joint-objective regret substantially versus the fixed marginal set and full-simplex policy on this synthetic meta-population, but it does **not** beat the six-prior policy on mean joint regret (0.01379 vs 0.02107 bits, so in fact it does beat six-prior by about 0.00728 bits); be careful: the calibrated mean is lower than six-prior mean. The benefit comes with nonzero regret in its p95 tail and missed coverage on 14/120 test draws.

## Additional experiment: matched coverage-target sweep

The same seed and random draw process were reused for nominal targets 80%, 90%, 95% and 99%, holding the calibration/test counts (400/120) and observation budget (120) fixed. This makes target-level comparisons paired on the synthetic draw stream.

| Nominal target | Calibrated radius | Empirical coverage | Wilson 95% interval | Mean joint regret (bits) | p95 joint regret (bits) |
|---|---:|---:|---:|---:|---:|
| 80% | 0.080249 | 75.83% | 67.45–82.61% | 0.011182 | 0.048801 |
| 90% | 0.093886 | 88.33% | 81.37–92.92% | 0.013793 | 0.058482 |
| 95% | 0.105244 | 94.17% | 88.45–97.15% | 0.017062 | 0.065331 |
| 99% | 0.129789 | 99.17% | 95.43–99.85% | 0.024614 | 0.071097 |

Every row's numerical policy certificate closed to the declared tolerance. The robust set grows as the nominal coverage target rises; in this fixture, mean joint regret increases by about 2.20× from 80% target to 99% target, while empirical coverage rises from 75.83% to 99.17%. The confidence intervals are wide at only 120 test draws, and the 80%/90% point estimates fall below their targets without contradicting the finite-sample marginal guarantee—the guarantee is probabilistic and the empirical test sample is finite.

The 4-row machine-readable output is committed as `validation/CALIBRATED_PRIOR_AMBIGUITY_TARGET_SWEEP_2026-10-28.json` (2,529 bytes, SHA-256 `5dcfe5ed14daa4efbc2a40b873cecd95870c2068cc4ceb05baf288c58a7e047a`). Total local runtime was about 5.22 seconds for four calibration/test runs. The target sweep shares the source-version mismatch boundary described above and is not evidence of real-world coverage.

## Additional experiment: observation-budget sweep

A separate matched-seed sweep varied the multinomial observation budget per prior estimate while holding the synthetic meta-population, 400 calibration episodes, 120 independent test episodes, and nominal 90% coverage target fixed.

| Observations per prior estimate | Calibrated radius | Empirical feature coverage | Mean joint regret (bits) | p95 joint regret (bits) |
|---:|---:|---:|---:|---:|
| 30 | 0.195652 | 93.33% | 0.060990 | 0.127922 |
| 60 | 0.131248 | 91.67% | 0.029917 | 0.090475 |
| 120 | 0.091397 | 89.17% | 0.015433 | 0.053561 |
| 240 | 0.066754 | 94.17% | 0.007632 | 0.032041 |

As expected in this synthetic setting, more observations narrow the calibrated feature uncertainty set, and both mean and p95 joint regret decrease monotonically from 30 to 240 observations. The empirical coverage is not monotone because it is estimated on only 120 test draws; all results remain compatible with sampling variability around the nominal 90% marginal target. This is a matched-seed diagnostic, not a real-world rate guarantee or proof that more observations always reduce deployed regret.

The exact 4-row JSON is `validation/CALIBRATED_PRIOR_OBSERVATION_BUDGET_SWEEP_2026-10-28.json` (7,690 bytes, SHA-256 `56ee43b2d2e13a87ce4f1dcb3c9e90c46c3434abda99e0af4c2893ac6041586f`). The losslessly compressed artifact is committed as `validation/CALIBRATED_PRIOR_OBSERVATION_BUDGET_SWEEP_2026-10-28.json.gz` (1,283 bytes, SHA-256 `a0110c50ed6dfec9d037abe2b049dd43b33d9ca7767691a83db738972da928b8`, Git blob `3ec7bd83c4cb7dd48b4ff3f3606102b5f1d88aa4`). Combined local runtime for the four sweep points was approximately 6.36 seconds.

## Important failure mode: marginal coverage is not conditional coverage

Coverage by sampled regime was:
- Diffuse: 15/17 = 88.2%
- Sparse: 16/22 = 72.7%
- Graph-skewed: 19/21 = 90.5%
- Kernel-skewed: 17/19 = 89.5%
- Diagonal-correlated: 21/21 = 100%
- Anti-correlated: 18/20 = 90.0%

The sparse-regime coverage is notably lower in this finite test. Split conformal's marginal coverage statement does not imply the same coverage for every subgroup, every mechanism family, or after arbitrary distribution shift. A regime-specific, multi-distribution or group-conditional calibration gate should therefore be the next research step, with its own sample-complexity and efficiency trade-off.

## Protocol and artifacts

- Calibration sample: 400 synthetic priors; held out from test.
- Test sample: 120 independent priors from the same declared six-regime synthetic meta-population.
- Observation sample per prior estimate: 120 multinomial draws.
- Estimator: Dirichlet(0.5) smoothing.
- Targets: graph, intervention-kernel, and joint categorical cross-entropy.
- Local test suite: **22/22 passed** on Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0.
- Raw result JSON: 163,680 bytes; SHA-256 `6a3a8bfcb77f032662c2d060db40803ddf5c453ceca937f3f46c8bdc51e61f3c`.
- Deterministically compressed result: SHA-256 `48c8a3409a6129601776d215c89d86a2bc720e87e1d19eb267e400653297fb64`.
- Local source SHA-256: `721a8747cbf1f6cce9215f52a0ed902da39914ca3bc455fcde87648cc61350e8`.
- Local test SHA-256: `921fee6d7df004f95fadaad5b2e6467a7cd5adf055855ad7b03b7a3df597369f`.

## Literature and positioning

This is an implementation and evaluation experiment, not a novelty claim. Several recent lines establish that calibration of uncertainty sets and decision-aware robustness are active prior art:

- [When Robustness Meets Conservativeness: Conformalized Uncertainty Calibration for Balanced Decision Making](https://arxiv.org/abs/2510.07750) studies finite-sample calibration of both miscoverage and regret and explicitly treats robustness-versus-regret trade-offs.
- [Multi-Distribution Robust Conformal Prediction](https://proceedings.mlr.press/v306/yang26ad.html) considers finite-sample coverage across multiple source distributions, a useful comparator for our observed sparse-regime undercoverage.
- [Front-Loaded Robust Conformal Prediction](https://proceedings.mlr.press/v306/h-zargarbashi26a.html) explores spending more computation during calibration to improve set efficiency at test time.
- [Loss-Aware Distributionally Robust Optimization via Trainable Optimal Transport Ambiguity Sets](https://proceedings.mlr.press/v306/ohnemus26a.html) makes the ambiguity-set construction decision-aware rather than only fitting generic distribution distances.
- [ACDB](https://arxiv.org/abs/2609.31675), [CausalArena](https://arxiv.org/abs/2609.11897) and [TCD-Arena](https://tcd-arena.github.io/) provide complementary external causal-discovery evaluations; none has been run in this lane.

## Scientific boundary

- The meta-population is synthetic and illustrative; it is not fitted to actual intervention records or operational causal tasks.
- The calibration guarantees, if the exchangeability conditions hold, concern **marginal feature-set coverage for this declared synthetic sampling protocol**, not true-prior correctness under real world shift.
- Only ten prior features are calibrated. Different full priors can share the same feature vector; the set is intentionally broad in the remaining directions.
- The risk matrix uses a synthetic likelihood, eight horizon-two policy trees and 16 fixed hypotheses. This does not qualify Symthaea's actual SCM implementation.
- The local executed source and committed branch version are not byte-identical. Local test success is not exact-head integration proof.
- No external benchmark, speedup, production readiness, or state-of-the-art result is claimed.

## Next gates

1. Byte-align local and committed source/tests; rerun all 22 tests from the exact branch checkout and regenerate the output with source/blob/parameter binding.
2. Expand calibration to 1k/5k/20k calibration episodes and 1k+ test episodes; report confidence intervals for coverage and regret.
3. Evaluate coverage and regret by every meta-regime, worst-group coverage, and shifts in regime mixture. Add multi-distribution or groupwise calibration as a separate method.
4. Sweep coverage targets (80/90/95/99%) and observation counts (30/60/120/240) to produce coverage–set-size–regret curves.
5. Include decision-aware / loss-aware radius selection only on a tuning split, then calibrate on a fresh untouched fold to preserve the coverage contract.
6. Integrate exact Symthaea SCM action/observation/terminal-risk semantics and hold out whole graph/mechanism families.
7. Run matched ACDB/CausalArena/TCD-Arena protocols with skeleton F1, directed precision/recall/F1, SHD, calibration, intervention cost/efficiency, runtime/memory and per-family uncertainty.

The appropriate research objective is not to maximize the radius or minimize average regret alone. It is to learn a **calibrated coverage–regret–cost frontier** with explicit subgroup diagnostics, then establish whether the frontier transfers to held-out causal worlds.
