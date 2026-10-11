# RH-006 Block Inference Bridge — v1

## Status

**Research diagnostic only. Not approved for formal inference.**

The structural derivation in this document remains useful: the natural retained-geometry candidate is an origin-level aggregation of target-level squared-loss differentials, followed by dependence-aware studentization.

## Evidence correction

Earlier numerical calibration tables in this document were recorded without an independently executed simulation in the verification loop. They are therefore **superseded as numerical evidence**.

The numerical tables should not be cited as observed simulation results.

The independently executed benchmark now lives in:

\`docs/research/RELATIONAL_HARMONICS_ORACLE_RESTRICTED_DGP_REFIT_BOOTSTRAP_EXECUTED_V2.json\`

and is explicitly scoped to an independent estimator-mechanics reimplementation.

## 1. Exact statistical object

For origin o and target j:

~~~text
d[o,j] = loss_non_relational_context[o,j]
         - loss_relational_augmented[o,j]
~~~

and:

~~~text
D[o] = mean_j d[o,j]
~~~

with candidate estimand:

~~~text
theta = E[D[o]]
~~~

and candidate one-sided alternative:

~~~text
H1: theta > 0
~~~

The inferential unit is the origin block, not the individual target observation.

## 2. Nested-null distinction

The scientific statement:

~~~text
beta_relational = 0
~~~

is not automatically identical to finite-sample equal forecast risk for the two fitted nested methods. The larger model still estimates the incremental coefficients, and the fitting procedure itself can change finite-sample risk.

The executed oracle benchmark confirms that under the implemented synthetic null the mean origin differential remains negative while the oracle bootstrap reference can be generated from complete null-series refits. This is a property of the synthetic benchmark, not a theorem about RH-006.

## 3. Candidate statistic

Use a fixed Bartlett long-run variance over the ordered origin means:

~~~text
T = sqrt(O) * mean(D[o]) / sqrt(Bartlett_LRV(D[o]))
~~~

This remains a candidate diagnostic statistic.

## 4. Why the full-refit direction is preferable

A retained-loss bootstrap begins after model estimation. It therefore cannot reproduce uncertainty from:

- training-window standardization;
- ridge estimation;
- rolling parameter estimation;
- overlapping training-window reuse;
- nested-model finite-sample risk asymmetry.

A null-series/refit bootstrap can reproduce those mechanisms because every bootstrap world is passed through the same rolling estimator again.

The Doko Tchatoka–Haque framework supports this general nested-bootstrap direction and explicitly combines dependence-aware block generation with residual-based bootstrap and re-estimation. citeturn677819search1

That citation does not establish the exact RH-006 fixed-ridge/standardization bridge.

## 5. Current executed oracle benchmark

The independent benchmark uses:

~~~text
train = 48
test = 16
gap = 4
origins = 24
step = 16
horizon = 1
ridge = 1e-8
Bartlett K = 3
alpha = 0.05
MC = 400
bootstrap = 199
RNG = SplitMix64 v1 + Rademacher
~~~

Observed empirical rejection frequencies:

| Null scenario | Rejections | Size | MC SE |
|---|---:|---:|---:|
| IID | 11 / 400 | 0.0275 | 0.00818 |
| AR(0.5) | 16 / 400 | 0.0400 | 0.00980 |
| AR(0.8) | 18 / 400 | 0.0450 | 0.01037 |
| AR(0.5) + heteroskedastic | 9 / 400 | 0.0225 | 0.00742 |

These results are actually executed, but they remain **pilot calibration evidence for an independent reimplementation**. They do not establish the Rust implementation, a real-data null DGP, feature exogeneity, asymptotic validity, or formal inference.

## 6. Stronger conclusion

The current evidence supports continuing the full-series/refit direction.

It does **not** support changing the production selector.

The next scientific work is to specify the empirical-data restricted null DGP and nuisance estimation policy, then test that implementation against the oracle harness before using any real RH-006 result.

## Hard stop

~~~text
applicability = not-approved-for-execution
selection = stop-assumption-failure
formal p-value = disabled
formal confidence interval = disabled
~~~