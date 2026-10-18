# FDM spatial benchmark: synthetic harness v1

## Purpose

This offline harness tests whether spatially varying synthetic dimensional error can be estimated more informatively than by one global calibration value. It is an experimental fixture—not a physical printer model, quality certificate, machine setting recommendation, or proof that a spatial policy is better.

## Research motivation

ISO/ASTM 52902:2023 describes geometric-capability test artifacts for additive manufacturing systems and calibration. It includes different geometries and discusses measurement uncertainty, without mandating a universal measurement method or machine settings. Its scope supports predeclaring artifacts and features and reporting their positions/orientations alongside measurements. See the official standard abstract: <https://www.iso.org/standard/79683.html>.

A position-varying dynamic response has also been studied for delta-printer vibration compensation. That is a machine-specific research result, not evidence that an identical effect or compensator applies to Cartesian, CoreXY, or other systems: <https://arxiv.org/abs/2209.06791>.

## Corpus contract

- Schema: `symthaea.fdm-spatial-synthetic.v1`.
- Evidence class: `synthetic`, always.
- Discovery data: a 3 x 3 normalized grid with deterministic, seed-controlled synthetic perturbations.
- Confirmation data: off-grid interior points plus deliberate probes outside the measured discovery rectangle.
- Response law: versioned smooth quadratic surface; this is a test fixture, not a fitted or validated physical model.
- Candidate prediction receives discovery observations only. Confirmation oracle values are read only by the evaluator to independently recompute errors.
- Duplicate identities, overlapping discovery/confirmation positions, non-finite values, unsupported versions, and normalized coordinates outside [0, 1] invalidate the corpus.

## Policy arms

1. Global mean: baseline; predicts at every confirmation point.
2. Nearest neighbor: uses the nearest discovery observation only within the axis-aligned rectangle covered by discovery points; otherwise returns Unknown.
3. Inverse-distance-squared weighted interpolation: considers a fixed number of nearest observations within a declared radius and within the observed rectangle; otherwise returns Unknown.

The rectangle is a conservative empirical coverage rule, not a validated physical applicability domain. The benchmark deliberately does not extrapolate local policies beyond observed bounds.

## Metrics, kept separate

- Coverage and Unknown count.
- Mean signed prediction error (bias).
- Mean absolute error.
- Root mean squared error.
- Maximum absolute error.
- Per-point prediction/error records to expose local failures.

No policy is guaranteed to win. Error metrics must be read with coverage, and no weighted aggregate turns missing predictions into passes.

## Proposed target gates

- 100% deterministic replay for identical schema, seed, corpus, policy and code.
- 100% rejection of the frozen schema/range/identity/split negative controls.
- Confirmation positions are not reused discovery positions or IDs.
- Report all policies on identical held-out points; preserve per-point results.
- No claims about physical accuracy, calibration savings, machine-wide capability or ISO/ASTM conformity from synthetic results.

## Next falsifiable step

Add measured replay data only after source, units, machine/profile, material lot, measurement method, calibration, uncertainty, failure and missingness semantics are explicit. Compare policies under equal calibration budgets, freeze the primary endpoint before confirmation, then validate on a preregistered physical campaign with independent metrology. Simulation and synthetic response surfaces are never physical confirmation.


## Spatial interval benchmark v1 (dependent draft increment)

The companion module `fdm_spatial_uncertainty` extends the synthetic spatial harness without changing the v1 point-estimate baseline. It uses three disjoint splits:

- **Discovery (9 observations):** fit each policy.
- **Interval calibration (16 points):** compute absolute residuals using discovery-only predictions and choose the declared residual quantile.
- **Confirmation (20 points):** final held-out interval evaluation; these targets do not determine the interval width.

For nominal coverage (1-\alpha), the one-based residual order-statistic rank is `ceil((n + 1) * (1 - alpha))`. Evaluation fails closed when that rank exceeds the number of usable calibration residuals. Reports preserve interval width, empirical coverage, prediction coverage, calibration abstentions, Unknown count and OutOfDomain count separately. The interval wrapper refuses to issue an interval outside the discovery rectangle even for the global-mean point estimator.

**Statistical limitation:** the calibration and confirmation locations are a fixed, spatially structured synthetic design. Spatial dependence can violate exchangeability, so this is a residual-quantile / conformal-style benchmark—not a claim of a distribution-free coverage guarantee. Empirical synthetic coverage is not physical printer validation. A future physical path must preserve unit, instrument, calibration, machine/profile, material lot, location, orientation, repeat, uncertainty and missingness provenance, and should use preregistered independent metrology.

### New acceptance tests

- The calibration split is identity- and position-disjoint from discovery and confirmation.
- Calibration-oracle, split-overlap, non-finite and identity corruption must fail closed.
- Prediction intervals expose `Predicted`, `Unknown` and `OutOfDomain` distinctly.
- Calibration quantile rank and usable sample count are explicit; insufficient calibration rejects rather than silently shrinking the interval.
- Replay is deterministic; interval widths and empirical coverage are reported separately from prediction coverage.
- A parity test checks the duplicate predictor implementation against the v1 evaluator on in-domain confirmation positions.
