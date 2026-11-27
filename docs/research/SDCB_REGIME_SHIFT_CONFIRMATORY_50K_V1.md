# SDCB Dependence-Regime Shift Confirmatory Grid v1

**Status:** standalone confirmatory executable and Rust 1.96 workflow authored. The exact-head format/test/full-run gate and report inspection are pending. This increment uses fresh seeds and raises the selected conditions to 50,000 trials each; it does not alter production SDCB assessment.

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

