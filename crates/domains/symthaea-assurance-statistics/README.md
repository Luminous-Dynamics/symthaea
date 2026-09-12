# symthaea-assurance-statistics

Small statistical primitives for safety/assurance evidence where **zero observed failures does not mean zero true failure probability**.

The initial scope is intentionally narrow and dependency-free:

- exact one-sided Bernoulli upper bound after zero observed events
- required independent Bernoulli exposure count for a target upper probability
- exact one-sided Poisson rate upper bound after zero observed events over known exposure time
- required Poisson exposure for a target upper event rate

## Why

If a perception system records zero false alarms over a validation campaign, the defensible statement is not:

> the false-positive probability is zero

It is closer to:

> under the stated statistical assumptions, zero events over this amount of exposure places an upper confidence bound on the true event probability/rate.

For Bernoulli exposures with zero events, the exact one-sided upper bound is:

`p_upper = 1 - alpha^(1/n)`

where `alpha = 1 - confidence` and `n` is the number of **independently justified exposure units**.

For a Poisson process with zero events over exposure `T`:

`lambda_upper = -ln(alpha) / T`

## Important assumption boundary

Raw video frames are usually correlated and should not automatically be counted as independent Bernoulli trials. The caller must justify the exposure unit (for example independent scenes, separately randomized trials, or a Poisson exposure-time model).

The crate does not certify independence, stationarity, representativeness, or deployment equivalence. Those remain part of the safety case.

## Verification

```bash
cargo test -p symthaea-assurance-statistics
```
