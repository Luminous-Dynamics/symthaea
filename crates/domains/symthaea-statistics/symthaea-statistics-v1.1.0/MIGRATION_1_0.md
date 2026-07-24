# Migration to 1.0

Version 1.0 stabilizes a conservative high-level import surface while preserving the domain modules introduced throughout the 0.x series.

## Stable import path

Applications that prefer a deliberately small compatibility anchor can use:

```rust
use symthaea_statistics::prelude::*;
```

The full root and module surfaces remain available. `API_LEVEL` is `1`, `VERSION` reflects the Cargo package version, and `NUMERICAL_CONTRACT` is `validated-v1`.

## Compatibility

No intentional 0.9 API removals are included. Existing validated `try_*` functions and lightweight compatibility wrappers remain available. The package version changes to `1.0.0`; downstream version constraints should be updated accordingly.

## New finite-sample APIs

Use `try_exact_binomial_test` when an exact one-sample Bernoulli test is required. Its two-sided value uses probability ordering, matching conventional SciPy/R exact tests. `try_clopper_pearson_interval` is conservative and frequentist; `try_jeffreys_interval` is an equal-tailed Bayesian interval.

Hypergeometric APIs model sampling without replacement. Beta-binomial APIs model predictive overdispersion induced by a beta-distributed Bernoulli probability.

## New geometry and simulation APIs

Compositional log-ratio transforms require strictly positive components. `try_closure` permits zero components but logarithmic transforms do not silently add pseudocounts.

Deming regression requires the caller to declare `variance_y / variance_x`; it does not estimate that measurement-error ratio from the same paired observations.

MCMC diagnostics require at least two equal-length chains. The reported convergence measure is rank-normalized split R-hat with a folded-tail companion. Effective sample size is a Geyer initial-positive/monotone estimate over split rank-normalized chains.

Randomized quasi-Monte Carlo estimates integrate over the unit hypercube. Reported standard errors come from independently shifted Halton replications, not from treating individual low-discrepancy points as IID observations.

## Verification

Run the ordinary Rust lane with `./scripts/verify.sh`. Independent Python references are separate and can be regenerated with `./scripts/verify_v1_0_references.sh`.
