# SDCB Block-Conformal Dependence-Regime Shift Experiment v1

**Status:** exact-head Rust 1.96 workflow passed on source `969a7344bb392ab63b98d40e43b2d673bfe640ec`: rustfmt, 4/4 unit controls and the full 36-condition × 2,500-trial grid (90,000 simulations). CSV and source/toolchain manifest were generated, downloaded and inspected ([workflow run #16](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46002051862), [artifact 15197659974](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46002051862/artifacts/15197659974)). Results are diagnostic and do not establish a general coverage guarantee. This is a research increment stacked on #13959, not a change to the production SDCB assessor.

## Research question

Can calibration remain trustworthy when every one-step residual has the same marginal distribution, but the temporal dependence regime changes between calibration and test?

This is a stricter failure mode than ordinary mean/variance drift. The experiment uses a unit-variance Gaussian AR(1) process:

`X_t = \rho X_{t-1} + \sqrt{1-\rho^2}\,\epsilon_t,quad \epsilon_t\sim N(0,1)`

For every `|rho| < 1`, the stationary one-step marginal is exactly `N(0,1)`. A switch from `rho_calibration` to `rho_test` therefore preserves the individual residual marginal while changing the dependence between observations inside a block. For block-max scores, that can change the score distribution even though a univariate marginal check sees no change.

The critical distinction is:
- **marginal stability**: the distribution of each `X_t` is unchanged;
- **block-score stability**: the distribution of `max |X_t|` over a block is unchanged;
- **exchangeability**: calibration and test block scores have the joint symmetry required by a conformal rank argument.

The first does not imply the second or third.

## Experiment design

Executable:

`scripts/research/sdcb_block_conformal_regime_shift.rs`

The deterministic grid covers:
- calibration AR coefficient: `0.05`, `0.95`;
- test AR coefficient: `0.05`, `0.95`;
- block lengths: `1`, `8`, `32`;
- 39 calibration blocks;
- nominal alarm rate `alpha = 0.10`;
- 2,500 trials per grid point;
- independent-block controls plus one serial path with embargo gaps `0` and `64`;
- 36 conditions and 90,000 total Monte Carlo trials.

The two calibration/test directions are both included:
- low dependence in calibration, high dependence in test;
- high dependence in calibration, low dependence in test.

Equal-regime controls (`0.05 → 0.05` and `0.95 → 0.95`) are included as well.

### Independent-block control

Every calibration block is produced by a separately seeded stationary AR(1) stream using `rho_calibration`; the test block comes from a separate stream using `rho_test`. This isolates a change in the within-block score distribution from dependence between calibration blocks.

When the two rho values differ, the blocks are independent but not identically distributed for block lengths greater than one. This intentionally tests the exchangeability boundary.

### Serial-path arm

All calibration blocks are taken from one path using `rho_calibration`; each block is followed by the declared embargo under that calibration regime. The test block follows the embargo using `rho_test`. The normalized innovation scale preserves the `N(0,1)` one-step marginal through the regime switch.

The serial arm therefore combines:
1. dependence among calibration blocks;
2. dependence between the final calibration block and test boundary;
3. a possible change in within-block dependence and block-score distribution.

The independent arm isolates item 3.

## Exact-head results — 27 November 2026

The exact-source run passed Rust 1.96 formatting and **4/4 unit controls** and generated 36 rows × 2,500 trials = **90,000 trials**. The manifest binds source commit `969a7344bb392ab63b98d40e43b2d673bfe640ec`, Rust 1.96.0 and the full grid. See [workflow run #16](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46002051862) and [CSV + manifest artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46002051862/artifacts/15197659974).

The following table shows empirical alarm percentages. The independent-block arm isolates the within-block score-distribution change; serial-path arms also contain dependence across calibration/test blocks.

| Calibration rho → test rho | Block length | Independent blocks | Serial path, gap 0 | Serial path, gap 64 |
|---|---:|---:|---:|---:|
| 0.05 → 0.05 | 1 | 9.44% | 10.08% | 10.40% |
| 0.05 → 0.05 | 8 | 9.76% | 9.60% | 10.16% |
| 0.05 → 0.05 | 32 | 9.92% | 9.52% | 10.20% |
| 0.05 → 0.95 | 1 | 10.96% | 9.56% | 9.36% |
| 0.05 → 0.95 | 8 | **3.88%** | **3.68%** | **3.48%** |
| 0.05 → 0.95 | 32 | **3.72%** | **2.72%** | **2.92%** |
| 0.95 → 0.05 | 1 | 9.40% | **18.68%** | 10.00% |
| 0.95 → 0.05 | 8 | **27.12%** | **28.92%** | **26.56%** |
| 0.95 → 0.05 | 32 | **30.12%** | **32.12%** | **33.72%** |
| 0.95 → 0.95 | 1 | 10.16% | **14.32%** | 9.64% |
| 0.95 → 0.95 | 8 | 9.96% | 11.16% | 10.76% |
| 0.95 → 0.95 | 32 | 10.08% | 10.12% | 9.72% |

The most important finding is the **directional block-score shift despite identical pointwise marginals**:

- Low-to-high dependence, independent blocks: alarm rate falls to 3.88% for length 8 (Wilson 95%: 3.191–4.711%) and 3.72% for length 32 (3.046–4.536%). Calibration blocks with weak correlation have more extreme maxima than highly correlated test blocks, so the nominal 10% test becomes materially conservative.
- High-to-low dependence, independent blocks: alarm rate rises to 27.12% for length 8 (25.413–28.897%) and 30.12% for length 32 (28.353–31.948%). Calibration maxima from strongly correlated blocks are too small for weakly correlated test blocks, resulting in substantial undercoverage.
- At block length 1, independent arms remain near nominal in both directions (10.96% and 9.40%), because the one-point marginal is unchanged. The serial-path arm can still violate nominal through dependence between sequential scores: for the `0.95 → 0.05` case it reports 18.68% with no gap.
- Embargo reduces some serial dependence effects, but it does not repair the independent-arm change in block-score distribution. In the `0.95 → 0.05` block-length 32 condition, even gap 64 has a 33.72% alarm rate.

The equal-regime arms provide controls: independent-block rates remain near 10% for the `0.05 → 0.05` and `0.95 → 0.95` settings; the serial high-correlation, block-length-1 condition remains elevated without an embargo (14.32%), consistent with the separate dependence-stress finding in #13959.

**Interpretation:** pointwise marginal fit is insufficient to validate block-conformal calibration when the score aggregates multiple dependent observations. Dependence shifts can change the distribution of the block statistic in either direction; it is not accurate to describe all such shifts as simply “more conservative” or “less conservative.” These results are limited to the specified stationary Gaussian AR(1), block-max score, two rho values, sample count and fixed seeds. They are not proof of a universal detector or calibration guarantee.

## Scoring

Each block score is the maximum absolute residual:

`S_j = max_{t in block j} |X_t|`

The conservative rank p-value is:

`p = (1 + count(S_j >= S*)) / 40`

The alarm occurs at `p <= 0.10`. The report includes empirical alarm rate, Wilson 95% interval, coverage complement, exact seed and regime parameters.

## Expected falsification controls

1. **Length-1 marginal control:** under independent-block sampling, a one-point block has the same standard-normal score marginal for every rho. A rho switch should not systematically move its alarm rate away from nominal.
2. **Block-max regime control:** for block lengths 8 and 32, low-to-high and high-to-low rho changes can move the block-max distribution in opposite directions despite identical one-step marginals.
3. **No-change controls:** equal calibration/test rho values expose the baseline effect of serial dependence alone.
4. **Embargo control:** a gap may reduce cross-boundary dependence, but it cannot restore identical block-score distributions after a rho change.

The direction is important. If test blocks are more correlated, their block maxima tend to be less extreme than calibration maxima from weakly correlated blocks, reducing alarm frequency. If test blocks are less correlated, their maxima tend to become larger, increasing alarms. Both are forms of calibration mismatch; neither is safely summarized as "more conservative."

## Paradigm shift: qualify the dependence envelope, not only the marginal fit

The observed result supports a later production-facing contract. A calibration receipt should bind at least:
- source episode IDs and exact calibration/test unit custody;
- residual score definition and block construction;
- block length and embargo protocol;
- marginal distribution/model family;
- dependence/regime diagnostics used at calibration time;
- validated parameter/regime envelope and its evidence digest;
- calibration/test time boundary;
- drift/common-cause monitor status and diagnostic uncertainty.

A runtime result should distinguish:
- `QualifiedWithinEnvelope`: the observed regime remains inside a frozen, independently validated envelope;
- `Inconclusive`: diagnostics cannot establish whether the current regime is inside the envelope;
- `NotQualified`: evidence indicates the regime is outside the envelope.

These statuses are not a proof that all dependence has been detected. Correlation summaries alone are insufficient for nonlinear, multivariate or latent-common-cause dependence. The point is to prevent an untested regime from silently inheriting a positive adequacy result.

## Validation and promotion gates

Before using block-conformal calibration as an engineering adequacy gate:
1. Run the full grid on the exact committed source head and retain CSV + manifest.
2. The independent length-1 controls must remain within a predeclared Monte Carlo tolerance of nominal.
3. Equal-regime independent controls must behave as the reference; equal-regime serial controls must be reported separately.
4. Both directions of dependence-regime shift must be retained; do not select only the favorable direction.
5. Repeat with more seeds and at least 50,000–100,000 trials per confirmatory condition before promoting numerical limits.
6. Extend to Student-t/heavy-tailed innovations, negative autocorrelation, multivariate common factors, variance shifts and gradual/abrupt transitions.
7. Choose any accepted dependence envelope on development regimes, freeze it, then test on held-out regimes.
8. Outside-envelope or diagnostically ambiguous operation must return `NotQualified` or `Inconclusive`, not positive adequacy.

The Rust workflow runs formatting, unit controls and the full deterministic grid, and uploads a source-commit/toolchain manifest. This is a synthetic stress benchmark only. It does not call the production SDCB assessment helper, validate real sensor data, prove exchangeability or establish a universal embargo.
