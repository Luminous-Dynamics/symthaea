# SDCB Dependence-Regime Shift Confirmatory Grid v1

**Status:** exact PR-head Rust 1.96 workflow passed on executable source `079b0d330f87e3d067ca05099d4e8e8457744a44`: formatting, 7/7 tests, and all 8 conditions × 50,000 trials (400,000 total). CSV and source/toolchain manifest were downloaded and inspected ([workflow run](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46005734569), [artifact 15200587284](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46005734569/artifacts/15200587284)). This increment uses fresh seeds and does not alter production SDCB assessment.

## Purpose

The exploratory #13986 Gaussian and #13990 standardized Student-t(5) common-scale grids each used 2,500 trials per condition. They showed strong directional effects when the within-block dependence regime changes while each one-point marginal remains unchanged. This follow-on increases precision on the most informative contrasts with an independent seed domain.

## Pre-registered confirmation set

Eight conditions, each with **50,000 Monte Carlo trials**:
- marginal family: Gaussian; standardized Student-t(5) common-scale;
- calibration rho → test rho: 0.05 → 0.95 and 0.95 → 0.05;
- block lengths: 8 and 32;
- 39 calibration blocks;
- nominal alpha: 0.10;
- independent-block sampling only.

The independent-block design deliberately isolates the distribution shift in the block-max score from dependence between consecutive calibration blocks. Every calibration and test block is independently seeded. In the Student-t(5) arm, one independent chi-square scale is shared within each block/path, preserving standardized t5 one-point marginals while introducing the declared common-scale dependence.

Total planned execution: 8 × 50,000 = **400,000 trials**. The seed domain `0x5344434350000000` is distinct from the exploratory runs. The full report records marginal family, rho pair, block length, exact seed, alarm rate, Wilson interval and empirical coverage. Main execution is compiled with `-C opt-level=2`; the manifest records the exact source commit, Rust version and compile flag.

## Exact-head confirmatory results — 27 November 2026

The exact source manifest reports commit `079b0d330f87e3d067ca05099d4e8e8457744a44`, Rust 1.96.0 and `-C opt-level=2`. The workflow passed formatting and **7/7 unit tests**; the full report contains eight rows at 50,000 trials per row = **400,000 Monte Carlo trials**. See [workflow run](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46005734569) and [CSV + manifest artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46005734569/artifacts/15200587284).

| Marginal family | Calibration rho → test rho | Block length | Alarm rate | Wilson 95% interval | Coverage |
|---|---|---:|---:|---:|---:|
| Gaussian | 0.05 → 0.95 | 8 | **3.810%** | 3.646–3.981% | 96.190% |
| Gaussian | 0.05 → 0.95 | 32 | **3.214%** | 3.063–3.372% | 96.786% |
| Gaussian | 0.95 → 0.05 | 8 | **26.872%** | 26.485–27.262% | 73.128% |
| Gaussian | 0.95 → 0.05 | 32 | **33.052%** | 32.641–33.466% | 66.948% |
| Student-t(5), common scale | 0.05 → 0.95 | 8 | **4.370%** | 4.194–4.553% | 95.630% |
| Student-t(5), common scale | 0.05 → 0.95 | 32 | **4.206%** | 4.034–4.386% | 95.794% |
| Student-t(5), common scale | 0.95 → 0.05 | 8 | **22.360%** | 21.997–22.727% | 77.640% |
| Student-t(5), common scale | 0.95 → 0.05 | 32 | **22.876%** | 22.510–23.246% | 77.124% |

All eight Wilson intervals exclude the nominal 10% alarm rate by a wide margin. This substantially narrows Monte Carlo uncertainty around the exploratory result:

- **Low-to-high dependence** (`rho=.05 → .95`) makes the test block maximum less extreme than calibration block maxima from weakly correlated blocks. The resulting alarms fall to 3.214–4.370%, i.e. the procedure becomes over-conservative for this test regime.
- **High-to-low dependence** (`rho=.95 → .05`) makes the test block maxima more extreme than calibration maxima from strongly correlated blocks. Alarms rise to 22.360–33.052%, producing substantial undercoverage.
- The effect is visible under both Gaussian and standardized t5 common-scale marginals. Its magnitude differs, but direction is consistent.
- Block length changes the magnitude, not the existence, of the mismatch. Longer blocks did not consistently improve coverage.

**Interpretation:** the result is now a high-precision counterexample for the declared synthetic generator and score: stable pointwise marginals do not protect a block-max conformal rank test from a shift in within-block dependence. The independent-block construction isolates non-identical block-score distributions from dependence between calibration blocks. This is not a statement that a specific real deployment has rho=.05 or .95, and not a general coverage theorem.

## Confirmation questions

1. Do both marginal families reproduce the low-to-high over-conservative shift?
2. Do both marginal families reproduce high-to-low undercoverage?
3. Do 50,000-trial Wilson intervals remain clearly separated from nominal alpha=10% for the selected block lengths?
4. Are Gaussian and t5 effects materially different in magnitude, despite the same qualitative direction?
5. Do the exact-source tests and fresh-seed report reproduce deterministically?

## Interpretation limits

- This is a confirmatory simulation for eight selected model/score conditions, not a general SDCB validity theorem.
- It tests two positive rho regimes only; negative autocorrelation, multivariate common causes, nonstationarity and adversarial diagnostic evasion remain open.
- It only evaluates independent-block sampling, intentionally isolating the block-score distribution shift; serial dependence across blocks is covered separately by #13959 and #13986/#13990.
- The 50,000 trials reduce Monte Carlo uncertainty for these conditions but do not establish external validity or a universal dependence envelope.
- Results must not be promoted into production acceptance thresholds without independent seed replications, held-out regimes and real-data calibration.

