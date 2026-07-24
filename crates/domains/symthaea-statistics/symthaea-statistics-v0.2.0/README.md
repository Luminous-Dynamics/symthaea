# symthaea-statistics

A dependency-free statistical foundation for Symthaea and the wider Luminous Dynamics workspace.

The crate is designed around four rules:

1. **Validated APIs are explicit.** `try_*` functions return `StatsResult<T>` with a structured `StatisticsError` rather than hiding unrelated failures behind `None` or `NaN`.
2. **Compatibility APIs stay lightweight.** Familiar functions such as `mean`, `linear_regression`, and `students_t_cdf` remain available as `Option`/`f64` wrappers.
3. **Tail probabilities are computed as tails.** Student-t and chi-square p-values use direct survival functions instead of subtracting a rounded CDF from one.
4. **Streaming and parallel aggregation are first-class.** `RunningMoments`, `BivariateMoments`, and `CalibrationAccumulator` can be merged without retaining raw observations.

## Included layers

- Stable descriptive and online moments
- Robust summaries: IQR, MAD, trimmed and winsorized means
- Normal, binomial, Poisson, Student-t, and chi-square densities/CDFs/tails
- One-sample, Welch, and paired t-tests with intervals and effect estimates
- Chi-square goodness-of-fit
- OLS regression with coefficient uncertainty and prediction intervals
- Bayesian probability and log-odds updating
- Binary-classification diagnostics
- Brier score, log loss, ECE/MCE, and reliability bins
- Bonferroni, Holm, and Benjamini-Hochberg corrections

## Example

```rust
use symthaea_statistics::{try_one_sample_t_test, CalibrationAccumulator};

let test = try_one_sample_t_test(&[4.0, 5.0, 6.0, 4.0, 6.0], 3.0, 0.95)?;
assert!(test.p_two_sided < 0.05);
assert!(test.confidence_interval.contains(test.estimate));

let mut calibration = CalibrationAccumulator::new(10)?;
calibration.push(0.8, true)?;
calibration.push(0.2, false)?;
let report = calibration.summary()?;
assert!(report.brier_score < 0.1);
# Ok::<(), symthaea_statistics::StatisticsError>(())
```

## Verification

`cargo test --all-targets` runs closed-form tests, committed differential reference vectors, complement/symmetry/monotonicity invariants, normalization checks, and merge-partition tests. See [NUMERICAL_SCOPE.md](NUMERICAL_SCOPE.md) for the exact claims and limitations.
