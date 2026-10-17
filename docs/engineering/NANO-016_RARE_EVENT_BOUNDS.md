# NANO-016 — Assumption-gated rare-event upper bound

The NANO-013 scorer estimates weighted rates from a stratified probability sample of eligible trajectory windows. Those rates and its approximate parent-cluster bootstrap are useful diagnostics, but they do not automatically produce a valid upper confidence bound on catastrophic risk.

This work adds an exact one-sided Clopper–Pearson upper bound for a narrower, explicit model: one binary outcome per independent, identically distributed unit, all units observed at the same exposure horizon, and equal-probability sampling. The exact binomial method is documented by NIST: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm.

## Run the illustrative utility

From the repository root:

~~~sh
cargo run -p symthaea-nanotech-audit --bin nano-risk-bound -- \
  0 299 0.95 parent-system-fixed-horizon true true true true true
~~~

Arguments are: observed events, number of independent units, confidence level, unit/exposure label, independence justified, common event probability justified, common exposure horizon, all outcomes observed, and equal-probability sampling. Boolean arguments must be the literal words true or false.

The flags are declarations, not evidence. The CLI cannot establish independence, homogeneity, complete reference outcomes, equal inclusion probability or external provenance. Those assumptions must be justified from the preregistered experiment and its records. If an assumption flag is false, the command emits a not_estimable report and no bound. Invalid counts fail closed.

## Method

For observed event count \(k\) out of \(n\) complete IID Bernoulli trials, the one-sided upper bound \(p_U\) solves:

~~~text
Pr[X <= k | X ~ Binomial(n, p_U)] = 1 - confidence_level
~~~

For zero observed events, the exact result simplifies to:

~~~text
p_U = 1 - (1 - confidence_level)^(1/n)
~~~

The implementation uses the stable expm1 form for the zero-event equation and bisection of the binomial CDF for positive event counts. It reports the observed rate and bound separately. Unit tests include reference values for 0/10, 0/299 and 1/100, monotonicity, all-events-observed, missing outcomes, unequal-probability sampling and clustered-unit guards.

## Why it is deliberately not wired into the window scorer

NANO-013 samples trajectory windows with recorded inclusion probabilities, and several windows may share one trajectory or parent material. Such a sample is not automatically equivalent to IID Bernoulli trials. Applying the binomial bound to window count or parent IDs without proving complete fixed-horizon outcomes and a compatible sampling design would be a false safety claim.

Therefore this utility is a separate method for an admitted, explicitly IID endpoint. It must not be run on the raw NANO-013 window sample merely because event and window counts are available. For unequal-probability or stratified samples, use a design-consistent bound and retain the sampling design; for clustered windows, inference must respect the real independent sampling unit. If no defensible method fits the realized design, the correct result is not_estimable.

## Sanity targets

Under the IID fixed-horizon assumptions and a one-sided 95% confidence level:

| Events / independent units | Approximate exact upper bound | Interpretation |
|---|---:|---|
| 0 / 10 | 25.89% | A small zero-event sample gives weak evidence about risk |
| 0 / 299 | 0.997% | Roughly 299 complete independent zero-event units are needed for a bound below 1% |
| 1 / 100 | 4.656% | One event materially increases the upper bound |

These are mathematical sanity checks for the declared binomial model, not Symthaea results, not power calculations for heterogeneous materials, and not a bound on transfer risk to unseen systems.

## Promotion rule

A bound should only be one part of the promotion decision. Freeze the event definition, exposure horizon, confidence level, independent unit, sampling design, target risk budget and worst-group rule before unblinding. A pooled bound must not compensate for a preregistered material-group failure, and zero observed failures must never be reported as zero risk. A parent with no successfully evaluated reference outcome cannot be classified as safe.
