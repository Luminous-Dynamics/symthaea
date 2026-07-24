# Migrating from 0.2 to 0.3

Version 0.3 is additive. Existing 0.2 call sites remain source-compatible.

## New deterministic resampling

Use an explicit seed for evidence that can be regenerated exactly:

```rust
use symthaea_statistics::try_bootstrap_mean;

let result = try_bootstrap_mean(&samples, 10_000, 0.95, 0x5eed)?;
```

`BootstrapResult` retains replicate values so callers can archive, visualize, or independently audit the empirical sampling distribution.

## Binary outcomes

Use `try_proportion_estimate` for a Wilson interval and the one-/two-sample score-test APIs for hypothesis testing. These avoid the pathological boundary behavior of naive Wald intervals.

## Categorical studies

- `try_chi_square_independence` handles rectangular count tables.
- `Contingency2x2::fisher_exact` provides directional and probability-ordering two-sided exact p-values.
- Odds ratios, corrected odds ratios, risk ratios, and risk differences are exposed on `Contingency2x2`.

## Multi-group studies

`try_one_way_anova` reports the full sums-of-squares decomposition, F statistic, direct-tail p-value, eta-squared, and omega-squared.

## Long-running and distributed simulations

- `FixedHistogram` bounds memory and merges across workers.
- `WeightedMoments` and `WeightedBivariateMoments` support reliability or importance weights.
- Configuration mismatches fail explicitly rather than silently producing invalid aggregates.

## Distribution-free and temporal diagnostics

- `try_mann_whitney_u`, `try_spearman_correlation`
- `EmpiricalDistribution`, `try_one_sample_ks`, `try_two_sample_ks`
- `try_autocorrelation_function`, `try_partial_autocorrelation_function`, `try_durbin_watson`

## Information metrics

The `information` module validates normalized probability mass and provides entropy, cross entropy, KL, Jensen-Shannon, total variation, Hellinger, and mutual information.

## New error variants

The non-exhaustive `StatisticsError` enum now distinguishes invalid iteration counts, incompatible accumulators, oversized exact computations/counts, invalid success/trial counts, invalid lags, invalid probability mass, and invalid logarithm bases. Downstream exhaustive matches were already disallowed by `#[non_exhaustive]`.
