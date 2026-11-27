# SDCB Gaussian and standardized Student-t(5) common-scale dependence stress v1

**Status:** standalone Rust 1.96 executable authored; exact-head formatting, unit controls, full-grid execution and artifact inspection are the qualification gate. No empirical result is claimed until the exact source revision passes and its report is inspected.

## Research question

Does a dependence-regime shift break block-conformal calibration when the one-step marginal distribution remains unchanged, including under a heavy-tailed marginal and a shared latent scale?

The preceding #13986 experiment used stationary Gaussian AR(1) paths. This follow-on adds a multivariate Student-t(5)-style scale mixture:

`X_t = S Z_t,quad S = \sqrt{3/V},quad V\sim\chi^2_5`

where `Z_t` is a unit-variance Gaussian AR(1) process with coefficient `rho`. The scale `S` is drawn once per independent path and is shared across all observations on that path. Because `Z_t\sim N(0,1)` at every time and the scale is independent of the Gaussian path, each `X_t` has a standardized Student-t(5) marginal with mean zero and variance one. Changing `rho` therefore preserves the one-step marginal while changing temporal dependence and the distribution of a block maximum.

The shared scale is a simple common-cause mechanism: all observations on one path are jointly amplified or attenuated by the same latent factor. It is not an independent scale draw for each tick.

## Experimental design

Executable: `scripts/research/sdcb_student_t5_common_scale_stress.rs`.

Both marginal families are evaluated:
- Gaussian AR(1), the baseline from #13986;
- standardized Student-t(5) common-scale AR(1), with a path-level latent chi-square scale.

The grid contains:
- calibration rho: `0.05`, `0.95`;
- test rho: `0.05`, `0.95`;
- block lengths: `1`, `8`, `32`;
- 39 calibration blocks;
- nominal alarm rate `alpha=0.10`;
- 2,500 trials per condition;
- independent-block controls and serial-path arms with embargo gaps `0` and `64`.

This creates 72 conditions × 2,500 trials = 180,000 Monte Carlo trials. Both shift directions and no-change controls are included. The independent-block arm draws an independent path-level scale for every block, including the test block; the serial arm shares one scale across its complete calibration/embargo/test path.

## Why Student-t(5) and the common scale matter

A single marginal family can hide substantial dependence differences. The t5 mixture has heavier tails than a Gaussian while retaining finite variance. Sharing the latent scale across a path introduces tail co-movement, so extreme residuals can cluster. A robust calibration protocol should not assume that a Gaussian AR(1) diagnostic transfers to heavy-tailed, common-cause observations.

The path-level scale model is a narrow, interpretable common-cause stressor. It is not a universal model of sensor faults, cross-channel dependence or arbitrary non-Gaussian time series.

## Statistic and decision rule

Every block uses the maximum absolute residual:

`S_j = max_{t in block j} |X_t|`

The conservative conformal rank p-value is:

`p = (1 + count(S_j >= S*)) / 40`

An alarm occurs at `p <= 0.10`. The CSV records the marginal family, design, calibration/test rho, block length, embargo, exact seed, trial count, alarm rate, Wilson 95% alarm interval and empirical coverage.

## Controls

- **Unit-variance control:** 50,000 independent standardized Student-t(5) samples must have sample variance within a predeclared tolerance of one.
- **Marginal-invariance control:** one-point independent-block calibration/test samples must retain a near-nominal rank alarm rate after the rho switch under Gaussian and t5 marginals.
- **Direction controls:** low-to-high and high-to-low dependence shifts must both be reported; one direction may become over-conservative while the other undercovers.
- **No-change controls:** equal calibration/test rho values distinguish regime shift from serial dependence between blocks.
- **Replay control:** identical source, seed, toolchain and parameters must produce identical report structs.

## Interpretation boundaries

- This is a deterministic synthetic simulation, not a general coverage theorem or real sensor validation.
- The Student-t(5) path-scale model has a shared latent scale; it is not equivalent to iid heavy-tailed innovations and should be named precisely.
- The independent-block arm intentionally compares independent but non-identically distributed block maxima after a rho switch. It is not a valid exchangeability construction under the shift.
- The serial arm combines within-block score-distribution shift, cross-block dependence and a calibration/test boundary transition.
- The model has a single scalar dependence parameter and a path-level common scale. It does not address multivariate channel networks or arbitrary nonlinear dependence.
- Any apparent near-nominal rate for a tested setting is not a transferable embargo or calibration policy.

## Qualification and promotion gates

1. Exact-head Rust 1.96 formatting and every unit control must pass.
2. Full 72-row CSV and source/toolchain manifest must be generated and independently inspected.
3. Both Gaussian and t5 rows must remain visible; do not pool the two marginal families.
4. Repeat confirmatory conditions with independent seeds and at least 50,000–100,000 trials per condition before promoting numerical thresholds.
5. Extend to negative rho, skewed marginals, time-varying common scales, multiple shared factors, cross-channel covariance shifts and gradual regime changes.
6. Freeze any candidate dependence envelope using development regimes and evaluate it on held-out regimes.
7. Outside-envelope or ambiguous dependence must produce `NotQualified` or `Inconclusive`; marginal goodness-of-fit alone must not authorize a positive adequacy result.

This increment is a standalone research stress benchmark stacked on #13986. It does not modify production SDCB assessment or prove that any statistical detector can recognize every form of dependence shift.
