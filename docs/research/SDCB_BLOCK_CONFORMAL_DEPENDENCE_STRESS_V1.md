# SDCB Block-Conformal Dependence Stress Experiment v1

**Status:** deterministic AR(1) stress-test executable authored; exact-head Rust 1.96 workflow is the execution gate. No empirical coverage result is claimed until the workflow report artifact is generated and inspected.

## Question

When is a block a defensible calibration unit for dependent observations, and how much residual dependence remains between adjacent blocks?

A conformal rank guarantee requires exchangeability of the calibration and test units. Grouping serial observations into blocks does not, by itself, create exchangeability. This experiment measures the empirical alarm/coverage consequences of:
1. independent stationary AR(1) trajectories, one block per trajectory (exchangeable-block control);
2. non-overlapping blocks cut from one stationary AR(1) trajectory with no embargo;
3. the same serial path with embargo gaps between blocks.

The initial stress point uses stationary Gaussian AR(1), `rho=0.95`, block maxima of absolute residuals, 39 calibration blocks, a single test block, and nominal `alpha=0.10`. It varies block lengths 1, 8 and 32 and serial-path embargo gaps 0, 16 and 64.

## Statistic and decision rule

Each block score is the maximum absolute residual in that block:

`S_j = max_{t in block j} |X_t|`

For a test block score `S*`, the conservative finite-sample rank p-value is:

`p = (1 + count(S_j >= S*)) / (n_cal + 1)`

The diagnostic alarm occurs when `p <= 0.10`. With 39 calibration blocks the p-value resolution is 1/40 = 0.025; ties are counted conservatively. The report includes empirical alarm rate, empirical coverage (1 - alarm rate), and Wilson 95% intervals for the binomial alarm rate and its complement.

The independent-block control restarts each block from the stationary marginal distribution using a distinct deterministic seed. In the serial-path conditions, calibration and test blocks are consecutive segments of one AR(1) realization, separated by the declared embargo. Embargo reduces correlation approximately according to the AR(1) decay; it does not prove independence or exchangeability.

## Reproducibility

Executable source:

`scripts/research/sdcb_block_conformal_dependence_stress.rs`

The source uses only the Rust standard library:
- SplitMix64 for deterministic pseudo-random streams;
- Box–Muller normal innovations;
- stationary AR(1) initialization;
- explicit per-condition seeds;
- Wilson intervals for observed alarm frequency.

The workflow records the exact source commit and Rust version in `manifest.txt` and uploads:
- `coverage.csv`, one row per design/block-length/embargo setting;
- `manifest.txt`, binding source revision, toolchain, sample count and grid.

The seed used for every output row is included in the CSV. This experiment is deterministic for the same source, seed, toolchain and math runtime; cross-platform floating-point library differences may produce small differences.

## Interpretation boundaries

- An independent-block control tests the finite-sample implementation under an exchangeable-block construction; it does not establish validity for arbitrary time series.
- Serial-path conditions are diagnostic counterexamples/stress points. Their empirical rate is not a theorem and must not be generalized beyond the tested stationary AR(1) process.
- Embargo is an operational approximation. It is not an independence certificate, and `rho^gap` is not a complete dependence measure for nonlinear, nonstationary or latent-common-cause systems.
- Block maxima may be conservative and can discard useful within-block structure. Alternative block scores and vector-valued residual scores require separate, predeclared experiments.
- The standalone experiment does not yet call the production SDCB conformal assessment helper; integration must use the same frozen score definition and preserve test/calibration unit custody.

## Required follow-on grid

The first executable grid is a smoke experiment, not a final benchmark. Extend the confirmatory grid before making a coverage claim:

| Factor | Initial values to test |
|---|---|
| AR coefficient | 0, 0.5, 0.8, 0.9, 0.95, 0.99 |
| Block length | 1, 2, 4, 8, 16, 32, 64 |
| Embargo | 0, 1, 4, 16, 32, 64, dependence-adaptive |
| Score | maximum absolute residual, mean absolute residual, block norm, model-set residual |
| Drift | stationary, mean shift, variance shift, gradual drift, abrupt regime change |
| Common cause | shared latent factor, correlated channels, shared sensor/calibration failure |
| Tail | Gaussian, Student-t, skewed, rare contamination and bounded mixtures |
| Sample size | calibration blocks 19, 39, 99, 199, 499 |

Compare:
- iid pointwise split conformal (deliberately misspecified control for serial data);
- random point split (leakage-sensitive comparator);
- contiguous non-overlapping block conformal;
- embargoed block conformal;
- stationary/moving-block bootstrap intervals;
- correctly specified AR(1) predictive residual model;
- misspecified scalar model versus coupled vector/covariance model.

Report nominal-versus-empirical alarm rate, Wilson intervals, coverage conditional on regime and tail, decision regret from false alarms/missed alarms, calibration sample size in blocks and effective independent information. Use independent trajectory seeds and preregister the scoring/threshold definition. Never label a block length as universally safe.

## Promotion gates

Before using block conformal as an engineering adequacy gate:
1. Independent-block control must reproduce the expected finite-sample rank behaviour.
2. All serial-dependence and drift regimes must publish their empirical alarm rates and uncertainty intervals.
3. The chosen block/embargo protocol must be justified on held-out regimes; it cannot be chosen after seeing the test outcomes.
4. Model mismatch and common-cause tests must be included.
5. Any condition outside the validated dependence envelope must return `NotQualified` or `Inconclusive`, not a positive adequacy result.
6. The production SDCB helper must consume calibration and test units once and retain the complete cross-stage ledger, as implemented in #13946.

This experiment is a step toward falsifiable coverage evidence; it is not a conformal-validity proof, a physical calibration campaign or an external benchmark result.
