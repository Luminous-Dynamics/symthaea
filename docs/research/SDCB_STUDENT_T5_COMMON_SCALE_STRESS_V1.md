# SDCB Gaussian and standardized Student-t(5) common-scale dependence stress v1

**Status:** exact-head Rust 1.96 workflow passed on source `394919be85c65b04510a8ca33b490116a56dcff2`: rustfmt, **7/7 unit controls**, full 72-condition × 2,500-trial grid (180,000 simulations), and report artifact inspection. See [workflow run](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46003315297) and [CSV + manifest artifact 15199048151](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46003315297/artifacts/15199048151). Results remain diagnostic, not a general coverage guarantee.

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

## Exact-head results — 27 November 2026

The exact executable source head `394919be85c65b04510a8ca33b490116a56dcff2` passed Rust 1.96 formatting and **7/7 tests**, including the 50,000-sample standardized t5 variance control, t5 one-point rank-invariance control, both regime-shift directions and deterministic replay. The full 72-condition grid completed at 2,500 trials per condition (180,000 total). The manifest binds source commit and Rust 1.96.0. See [workflow run](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46003315297) and [CSV + manifest artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46003315297/artifacts/15199048151).

Selected independent-block results for the standardized Student-t(5) common-scale marginal (nominal alarm rate = 10%):

| Calibration rho → test rho | Block length | Alarm rate | Wilson 95% interval |
|---|---:|---:|---:|
| 0.05 → 0.95 | 1 | 10.60% | 9.453–11.868% |
| 0.05 → 0.95 | 8 | **4.56%** | 3.810–5.450% |
| 0.05 → 0.95 | 32 | **3.92%** | 3.227–4.754% |
| 0.95 → 0.05 | 1 | 9.64% | 8.544–10.860% |
| 0.95 → 0.05 | 8 | **23.20%** | 21.587–24.895% |
| 0.95 → 0.05 | 32 | **22.60%** | 21.003–24.281% |

At block length 1, the independent-block rates remain close to the nominal 10% in both dependence-shift directions, consistent with the same one-point marginal distribution. At lengths 8 and 32, the block-max score distribution changes sharply: low-to-high dependence becomes over-conservative, while high-to-low dependence produces substantial undercoverage. This repeats the directional pattern observed under Gaussian marginals, although the magnitudes differ under the heavier-tailed, path-level scale mixture.

The serial-path arms retain the additional cross-block dependence failure seen in #13959 and #13986. For example, high-to-low rho at block length 1 yields 18.68% alarms with no gap and 10.00% with gap 64; at block length 32, the serial-path rates are 32.12% with no gap and 33.72% with gap 64. The shared t5 scale cancels when comparing ranks within a single path, so those serial-path rates coincide with the Gaussian path's rank outcomes for the same seeds. The independent-block arm is therefore the clearest view of how the heavy-tail marginal interacts with a within-block dependence shift.

**Interpretation:** the same calibration principle is fragile under both a Gaussian marginal and this standardized Student-t(5) common-scale marginal. The result is not an artifact of a mean/variance shift: each pointwise marginal remains fixed and unit variance. The scale-mixture model adds one shared latent factor per path, but does not model every common-cause or multivariate failure mode. These 2,500-trial results are exploratory; confirmatory replication should use more independent seeds and at least 50,000–100,000 trials per condition.

## Why Student-t(5) and the common scale matter

A single marginal family can hide substantial dependence differences. The t5 mixture has heavier tails than a Gaussian while retaining finite variance. Sharing the latent scale across a path introduces tail co-movement, so extreme residuals can cluster. A robust calibration protocol should not assume that a Gaussian AR(1) diagnostic transfers to heavy-tailed, common-cause observations.

The path-level scale model is a narrow, interpretable common-cause stressor. It is not a universal model of sensor faults, cross-channel dependence or arbitrary non-Gaussian time series. In the current implementation, the t5 scale is shared across an independent AR path; this makes blocks within a serial path share the same latent scale, while each independent-block arm gets a separately drawn scale.

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
