# Symthaea Causal Estimate Evidence

This bridge defines the evidence boundary for **numerical** causal-effect estimates used by Planetary Perception.

It deliberately separates three questions:

1. **Identification** — is the causal estimand recoverable under the reviewed causal model?
2. **Estimation** — what numerical magnitude was produced from a specific dataset and estimator?
3. **Decision** — how, if at all, should humans or governed systems use that estimate?

This crate addresses only the first two. It contains no policy ranking and no execution authority.

## Zero is not a failure sentinel

The current `symthaea-causal-reasoning` estimators contain several paths where insufficient samples or degenerate variance return `0.0`. That behavior is convenient inside an exploratory numerical routine but is ambiguous at an evidence boundary.

Planetary Perception therefore does **not** directly wrap those return values yet. A numerical effect enters `CausalEffectEstimateEvidence::estimated` only with:

- an identified estimand;
- explicit estimator identity/version;
- dataset identity/digest/sample count;
- a finite effect interval and units;
- non-degenerate treatment variance;
- structured diagnostics.

Failures such as insufficient samples, positivity violations, missing variables, or unavailable diagnostics use `NumericalEstimateStatus::NotEstimable` instead.

A genuine point estimate of zero remains perfectly representable when those requirements are met.

## Confidence separation

`identification_confidence` is retained separately from the numerical effect interval. It must never be shown as though it were the uncertainty of the effect magnitude.

## Next integration

A later estimator adapter may populate this envelope only after the underlying estimator API reports structured failure/diagnostic states instead of using numeric sentinels for estimation failure.
