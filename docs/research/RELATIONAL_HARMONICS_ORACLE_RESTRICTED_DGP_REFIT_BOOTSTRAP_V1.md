# RH-006 Oracle Restricted-DGP Refit Bootstrap — benchmark v1

## Status

**Research diagnostic only. Not approved for formal inference.**

This benchmark is intentionally oracle-only: the simulation knows the true null DGP parameters. It is designed to isolate whether the proposed origin-level studentized statistic can behave sensibly when the bootstrap reference distribution preserves the full nested rolling refit process.

## What was tested

The scientific null is:

~~~text
beta_relational = 0
~~~

For each Monte Carlo dataset:

1. generate one complete synthetic feature/outcome series under the null;
2. keep the feature path fixed in the bootstrap world;
3. generate each bootstrap outcome path from the **known true null DGP**;
4. refit both RH-006 nested estimators at every rolling origin;
5. recompute training-only standardization and fixed ridge;
6. recompute all held-out losses and origin means;
7. recompute the same Bartlett-studentized T statistic;
8. estimate the one-sided tail from the bootstrap T* distribution.

This is an oracle benchmark, not the final empirical-data bootstrap.

## Geometry

~~~text
train_samples = 48
test_samples  = 16
gap_samples   = 4
origin_count  = 24
step_samples  = 16
horizon       = 1
ridge_lambda  = 1e-8
Bartlett K    = 3
alpha         = 0.05
Monte Carlo   = 40 per scenario
bootstrap     = 99 per Monte Carlo replication
~~~

The simulation uses the same pinned deterministic SplitMix64/Rademacher generation family as the block diagnostic.

## Results

| Scenario | Rejections / 40 | Empirical size | Mean T | Mean p |
|---|---:|---:|---:|---:|
| IID | 2 | 0.050 | -3.421 | 0.565 |
| AR(0.5) | 3 | 0.075 | -3.299 | 0.591 |
| AR(0.8) | 2 | 0.050 | -2.713 | 0.531 |
| AR(0.5) + heteroskedasticity | 1 | 0.025 | -2.656 | 0.425 |

The Monte Carlo sample is intentionally small. These numbers are not a formal size proof. The useful observation is structural: once the bootstrap world recreates the complete estimator/refit process, the origin statistic can have a reasonably calibrated null reference even though its finite-sample mean is strongly negative under beta_relational = 0.

That contrasts with the coefficient-null behavior of the retained-loss bootstrap, where centering the loss process at zero cannot reproduce the nested estimator effect.

## Interpretation

The result supports the following decomposition:

~~~text
problem A: choosing the right statistic
problem B: preserving nested rolling estimation in the null reference
problem C: estimating the null DGP and its dependence from real data
~~~

The oracle experiment provides evidence that problem B is tractable for this candidate statistic.

Problems A and C remain open.

In particular, this benchmark does not justify treating observed features as exogenous in real data. It also does not establish how the restricted null DGP parameters should be estimated without contaminating the scientific evaluation.

## Next empirical candidate

The next implementation should replace the oracle parameters with a frozen nuisance-estimation contract:

~~~text
observed RH-006 source
        |
        v
declared restricted null model
        |
        +--> declared nuisance-parameter estimator
        |
        +--> declared dependent residual/innovation bootstrap
        |
        v
contiguous bootstrap outcome series
        |
        v
full RH-006 rolling refits
        |
        v
origin T*
~~~

The nuisance-estimation choice must be frozen before inspecting the observed relational forecast advantage.

The first serious comparison should be between:

1. a restricted null fit using a non-evaluation calibration span, if one exists;
2. a restricted null fit using the first eligible training window only;
3. a theoretically justified full-sample nuisance fit, with its leakage/conditioning implications documented.

No option should be selected by observed forecast performance.

## Hard boundary

This benchmark does **not** modify:

~~~text
applicability = not-approved-for-execution
selection     = stop-assumption-failure
formal p      = disabled
formal CI     = disabled
~~~

Its role is to justify continued development of the refit-bootstrap direction, not to authorize its use.