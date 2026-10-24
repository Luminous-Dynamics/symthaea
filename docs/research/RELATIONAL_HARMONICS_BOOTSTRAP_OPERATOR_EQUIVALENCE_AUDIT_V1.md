# RH-006 Bootstrap Operator-Equivalence Audit — v1

## Status

Research diagnostic only.

- applicability = not-approved-for-execution
- selection = stop-assumption-failure
- formal inference = disabled

## Audit question

The plug-in restricted-null bootstrap reuses the forecast operators `ops` across bootstrap outcome draws. The audit asks whether this is an accidental omission of estimator refitting or an exact representation of refitting under the declared fixed-feature ridge estimator.

## Result

Under the RH-006 estimator mechanics, reusing `W` is algebraically equivalent to refitting the ridge estimator for every bootstrap outcome, provided:

1. training and test feature matrices are held fixed;
2. training-only standardization is held fixed as a function of those features;
3. ridge `lambda` is fixed;
4. the fitted model is linear in the training outcome.

Let

`A = [1, standardized(X_train)]`

and

`B = [1, standardized(X_test)]`.

The ridge fit for bootstrap outcome `y*` is

`beta* = (A'A + lambda P)^(-1) A' y*`

and the forecast is

`B beta* = W y*`

where

`W = B(A'A + lambda P)^(-1)A'`.

Therefore `W` depends on features and ridge only, not on `y*`. Reusing `W` is exactly the same linear operation as refitting the estimator on each bootstrap outcome.

A deterministic numerical audit with 100 independent outcome draws produced maximum absolute fresh-fit versus reused-operator difference:

`5.55e-16`

at ordinary double precision.

## What this closes

This audit closes a possible false-positive concern:

`bootstrap operator reuse != failure to refit`

for the declared fixed-feature linear ridge estimator.

The bootstrap code can legitimately cache the outcome-linear forecast operator.

## What remains open

The equivalence is conditional on the feature path.

It does **not** establish validity when the scientific DGP requires joint resampling of:

- features and outcomes;
- feature/outcome endogeneity;
- feature-dependent nuisance evolution;
- data-dependent preprocessing not represented by the cached operator;
- adaptive projection or model selection whose state changes with bootstrap outcomes.

Consequently, the important remaining distinction is:

`conditional fixed-feature calibration`

versus

`joint feature/outcome calibration`.

The existing RH-006 applicability boundary already treats feature endogeneity as uncovered.

## Policy

The research lane should explicitly document operator caching as a mathematically exact optimization under the fixed-feature estimator.

It should not describe that caching as proof that the full scientific bootstrap is valid.

The empirical bridge still requires a DGP and resampling rule that match how features and outcomes arise in the intended scientific application.

## Nonclaims

No formal p-value.

No confidence interval.

No uniform size guarantee.

No empirical validation.

No claim that conditional fixed-feature bootstrap validity transfers to a joint feature/outcome process.
