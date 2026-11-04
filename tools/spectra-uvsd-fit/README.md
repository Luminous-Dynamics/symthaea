# SPECTRA UVSD likelihood fitter

This crate fits the single-observer unequal-variance Gaussian signal-detection model to ordered type-1 response-category counts. It is deliberately separate from the SPECTRA observation adapter, shared ROC implementation, and Symthaea's HDC optimizer.

## Model

The noise evidence distribution is N(0, 1). Signal evidence is N(mu, sigma^2), with sigma strictly positive. For K ordered response categories, the K-1 criteria are strictly increasing. Each category is an interval between adjacent criteria (with infinite outer bounds).

The un-smoothed multinomial log likelihood is

    sum_j noise_j * log(Phi(c_j) - Phi(c_(j-1)))
  + sum_j signal_j * log(Phi((c_j-mu)/sigma) - Phi((c_(j-1)-mu)/sigma))

The estimator reports d_a = mu / sqrt((1 + sigma^2) / 2). It does not silently add pseudo-counts. Empty observed cells contribute zero; an impossible positive-count cell makes the objective non-finite and is not hidden.

## Inputs and analysis unit

Construct a CategoryTable with one row for every predeclared ordered category, including categories with zero observed counts. The order field is an opaque strictly increasing category rank; it is not a numeric score spacing. Preserve the same category definitions for all fits. For SPECTRA adapter output, group the stratified category counts by participant and session and fit each table separately. Do not fit one pooled curve to a heterogeneous population and then call it a single-observer estimate.

If a category is absent from an adapter's emitted sparse count vector, insert its zero-count row from the frozen observation profile before constructing the table. The estimator cannot infer an omitted category's definition.

## Numerical and optimizer design

- Uses shared symthaea-statistics log-CDF, log-survival and log-interval primitives; lower-tail probabilities are evaluated by a Mills-ratio continued fraction.
- Ordered thresholds are parameterized by a free first threshold and exponentiated positive gaps.
- The primary likelihood has no continuity correction or arbitrary pseudo-count smoothing.
- Multi-start BFGS with finite-difference gradients is available as the default fit.
- Nelder-Mead is provided as a derivative-free cross-check, not as an assumed truth oracle.
- Reports objective, gradient norm, iteration count, start count, optimizer and explicit Converged / Boundary / NotIdentifiable / NotConverged / NotEvaluable disposition.
- Boundary and non-convergence are not converted into finite-point success.
- This initial version does not report confidence intervals, fit hierarchical UVSD, or claim asymptotic coverage.

The fit crate intentionally does not depend on symthaea-core or HDC-specific optimization types. A statistical likelihood should have a transparent numerical contract independent of consciousness/HDC state or optimizer telemetry.

## Interpretation and fail-closed boundaries

- `NotIdentifiable` is currently a conservative category-support screen, not a formal proof from the Fisher information matrix or profile likelihood. A table passing the screen can still be practically weakly identified.
- The numerical parameter bounds in the optimizer are computational guardrails, not scientifically justified limits. A fit at one is returned as `Boundary`, not as an interior estimate.
- `Converged` means the selected optimizer met its numerical stopping rule. It does not certify a global optimum, model adequacy, correct Gaussian assumptions, or statistical identifiability.
- `solution_parameter_spread` is the coordinate-wise max-minus-min across converged starts in the internal coordinates (mu, log-sigma, first cut, log-gaps). It is a multistart stability diagnostic, not a confidence interval.
- `diagonal_curvature` is a coordinate-wise second derivative of the objective. It is not a full Hessian, covariance matrix, or uncertainty interval.
- The best finite objective is selected even when that run did not converge; in that case the final status is `NotConverged`. A worse converged start cannot mask a lower-objective failed start.
- The primary likelihood remains unsmoothed. A separate paper-compatibility smoothing mode is not yet implemented; do not compare exact numerical outputs until the reference preprocessing/correction convention is matched.

## Reproduction and benchmark programme

1. Known-answer parameter recovery over mu, sigma, criteria, category count, trial budget, class imbalance and sparse cells.
2. Equal-variance negative controls (sigma=1) to measure false UVSD declarations separately from parameter RMSE.
3. Unequal-variance alternatives on both sides of one, including small departures near the practical-identifiability boundary.
4. Held-out category log likelihood, threshold recovery, boundary and non-convergence rates, and optimizer disagreement.
5. Stress conditions: missing categories, lapses, response ties, speed-accuracy effects, participant heterogeneity and dependence.
6. Reproduce all 11 visual-detection data sets from the public Miyoshi, Rahnev & Lau reference implementation; report per-dataset results, not only a pooled average. Treat recognition-memory dissociations as a domain-transfer negative control.
7. Compare separately with the published RT/confidence UVSD workflow, SDT Assistant's rating-scale UVSD fits, equal-variance SDT, and Lages' hierarchical binary UVSD. These methods have different data and pooling assumptions.

Reference work:
- Miyoshi, Rahnev & Lau (2026), iScience 29(3), 114998: https://doi.org/10.1016/j.isci.2026.114998
- Their public implementation: https://github.com/kiyomiyoshi/rt_type1_roc
- Lages (2024), Psychonomic Bulletin & Review 31, 2534-2557: https://doi.org/10.3758/s13423-024-02504-5
- SDT Assistant V3.1: https://hautus.org/sdt-assistant/

## Run

    cargo test --manifest-path tools/spectra-uvsd-fit/Cargo.toml
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin recovery_sweep > uvsd-recovery.csv

The recovery runner emits one CSV row per frozen synthetic campaign across mu, sigma and independent seeds. It is a parameter-recovery smoke benchmark only; it does not calculate calibrated power, interval coverage, a false-discovery rate or empirical performance.

This code is an estimator prototype, not an empirical result. It has not been compiled or run in the current environment. Human-subject collection, physical actuation and causal/anomalous information claims are out of scope.
