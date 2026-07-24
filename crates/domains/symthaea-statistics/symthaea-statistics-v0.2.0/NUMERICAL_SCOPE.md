# Numerical scope and guarantees

This crate uses `f64` throughout. It is intended for simulation, calibration, scientific diagnostics, and ordinary statistical inference—not arbitrary-precision proof work.

## Error and domain contracts

Validated `try_*` APIs reject empty or insufficient samples, non-finite observations, invalid probabilities, non-positive scales/degrees of freedom, negative counts, length mismatches, degenerate designs, and iterative non-convergence.

Compatibility wrappers preserve the original ergonomic surface:

- `Option<T>` wrappers return `None` on invalid or degenerate inputs.
- Scalar distribution wrappers return `NaN` on invalid parameters.
- Mathematically meaningful endpoint values remain explicit (`±∞` quantiles, zero/infinite boundary densities, and degenerate PMFs).

## Algorithms

- Mean: max-magnitude scaling with Neumaier compensated summation.
- Variance/covariance: Welford and parallel-merge recurrences.
- Quantiles: type-7 linear interpolation after total ordering of validated finite values.
- Log-gamma: Lanczos approximation for positive arguments.
- `erf`/`erfc`: direct-tail approximation with roughly `1e-7` relative tail accuracy.
- Incomplete beta: modified Lentz continued fraction with explicit iteration budget.
- Incomplete gamma: power series or modified Lentz fraction; a Temme transition approximation is used for large shape near `x ≈ a`.
- Normal quantile: Acklam rational approximation.
- Student-t quantile: bracketed monotone solve over the validated CDF.
- Regression: centered OLS sufficient statistics and classical homoskedastic standard errors.

## Known limits

- Extremely small probabilities may underflow to zero below the representable `f64` range. Log-tail APIs should be preferred where provided.
- Normal CDF accuracy is bounded by the dependency-free `erfc` approximation rather than the platform C library.
- Regression uncertainty assumes independent, homoskedastic residuals and a correctly specified linear mean model.
- Chi-square approximations are only statistically meaningful when expected-count assumptions are appropriate; the crate validates arithmetic domains, not study design.
- ECE/MCE depend on the selected bin count and are descriptive, not uniquely defined population parameters.

## Truth harness

The test suite includes committed reference vectors generated independently with SciPy/mpmath. Runtime and test execution remain dependency-free. Reference regimes include ordinary values, distribution boundaries, deep tails, and large-shape incomplete-gamma transition points.
