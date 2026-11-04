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
- The primary likelihood has no continuity correction or arbitrary pseudo-count smoothing. An explicit `CountCorrection::MiyoshiReferencePerCategory` compatibility mode adds `1/K` to each cell in each stimulus class, matching the public R source and `rt-da` Python implementation; it is never enabled by default.
- Multi-start BFGS uses an analytic gradient in natural parameters, transformed to the ordered-cut parameterization. An independent central finite-difference test is included; finite differences remain a fallback for non-finite analytic-gradient evaluations.
- Nelder-Mead is provided as a derivative-free cross-check, not as an assumed truth oracle.
- Reports objective, gradient norm, start count and dispersion, selected optimizer, expected-Fisher eigenvalues/rank/condition, and explicit Converged / Boundary / NotIdentifiable / WeaklyIdentified / NotConverged / NotEvaluable dispositions.
- For raw-likelihood, converged, interior fits with full-rank expected information, reports local asymptotic standard errors and 95% Wald intervals for sigma and d_a. These are approximations, not profile-likelihood or bootstrap intervals; they are suppressed for boundary, non-converged, weakly identified and reference-corrected fits.
- Boundary and non-convergence are not converted into finite-point success.
- This prototype does not fit hierarchical UVSD, compute profile-likelihood intervals, or claim calibrated interval coverage.

The fit crate intentionally does not depend on symthaea-core or HDC-specific optimization types. A statistical likelihood should have a transparent numerical contract independent of consciousness/HDC state or optimizer telemetry.

## Interpretation and fail-closed boundaries

- `NotIdentifiable` is an early conservative category-support screen. `WeaklyIdentified` is assigned after fitting when the expected Fisher matrix in internal coordinates is rank-deficient under a relative `1e-10` eigenvalue threshold or has condition number above `1e10`. These are engineering diagnostics, not universal inferential cutoffs or proofs of global identifiability.
- Expected-Fisher intervals use an inverse local information matrix and delta method. They are not robust to unmodeled dependence, misspecification, boundary effects or poor finite-sample asymptotics; separate profile-likelihood/bootstrap calibration remains required.
- Numerical parameter bounds are optimizer guardrails, not scientifically justified parameter limits. A fit touching one is returned as `Boundary`.
- `Converged` means the selected optimizer met its numerical stopping rule. It does not certify a global optimum, model adequacy, the Gaussian assumption, or statistical identifiability.
- `solution_parameter_spread` is a coordinate-wise max-minus-min across converged starts in internal coordinates (mu, log-sigma, first threshold, log-gaps). It is a multistart stability diagnostic, not a confidence interval.
- `diagonal_curvature` is a coordinate-wise second derivative of the objective. It is not a full Hessian, covariance matrix, or uncertainty interval.
- Expected Fisher information is computed from the fitted Gaussian category probabilities and the observed class totals. Wald intervals are emitted only for an interior converged fit using the unsmoothed primary likelihood.
- The lowest finite objective is selected even when its run did not converge; in that case the result is `NotConverged`. A higher-objective converged start cannot mask a lower-objective failed start.
- The primary likelihood has no pseudo-counts. The separately named `MiyoshiReferencePerCategory` mode matches the external `1/K` compatibility convention; exact comparisons must still pin category order, preprocessing and correction mode.

## Reproduction and benchmark programme

1. Known-answer parameter recovery over mu, sigma, criteria, category count, trial budget, class imbalance and sparse cells.
2. Equal-variance negative controls (sigma=1) to measure false UVSD declarations separately from parameter RMSE.
3. Unequal-variance alternatives on both sides of one, including small departures near the practical-identifiability boundary.
4. Held-out category log likelihood, threshold recovery, boundary and non-convergence rates, expected-information rank/condition, Wald/profile/bootstrap interval coverage, and optimizer disagreement.
5. Stress conditions: missing categories, lapses, response ties, speed-accuracy effects, participant heterogeneity and dependence.
6. Reproduce all 11 visual-detection data sets from the public Miyoshi, Rahnev & Lau reference implementation; report per-dataset results, not only a pooled average. Treat recognition-memory dissociations as a domain-transfer negative control.
7. Compare separately with the published RT/confidence UVSD workflow, SDT Assistant's rating-scale UVSD fits, equal-variance SDT, and Lages' hierarchical binary UVSD. These methods have different data and pooling assumptions.
8. Match the independent `rt-da` Python package on its published six-cell example and its 80 subject-fit comparisons before claiming estimator parity; then reproduce all 11 Miyoshi visual-detection datasets per dataset, not only in aggregate. Also include its RT-validity checks, bootstrap intervals and group-level summaries as parity targets.

Reference work:
- Miyoshi, Rahnev & Lau (2026), iScience 29(3), 114998: https://doi.org/10.1016/j.isci.2026.114998
- Their public implementation: https://github.com/kiyomiyoshi/rt_type1_roc
- Independent Python implementation and published-example/subject-fit validation: https://github.com/trevcaru/rt-da
- Lages (2024), Psychonomic Bulletin & Review 31, 2534-2557: https://doi.org/10.3758/s13423-024-02504-5
- SDT Assistant V3.1: https://hautus.org/sdt-assistant/

### Published-example compatibility regression

The R reference function defaults to adding `1/K` to every category cell in each stimulus class (`add_constant = TRUE`); the Rust estimator makes this an explicit `CountCorrection::MiyoshiReferencePerCategory` compatibility mode, never a default. This distinction matters for sparse cells and must be pinned when comparing numerical fits. The independent `rt-da` package also documents matching the R output on 80 subject fits (with per-fit d_a differences no larger than 2.5e-3); that is a stronger benchmark than a single example.

The public R example uses `nr_s1 = [10, 7, 16, 27, 29, 10]` and `nr_s2 = [43, 21, 10, 12, 8, 3]`. Its cell order is strongest-Yes to strongest-No, whereas this crate uses ascending evidence order, so the vectors are reversed before fitting. The explicit reference correction adds `1/K` to each of the K cells in each stimulus class. The regression test targets the published `mu=1.231417`, `sigma=1.252275`, `d_a=1.086691`, and `logL=-315.8781`.

An independent Python reference replay of the same six-start design obtained `mu=1.231422`, `sigma=1.252292`, `d_a=1.086687`, and `logL=-315.878077`. This is a Python algorithm check, not execution of the Rust test.

## Class-stratified bootstrap intervals

The crate exposes `bootstrap_intervals` for a deterministic nonparametric multinomial bootstrap conditional on the observed signal and noise trial totals. It resamples category assignments independently inside each stimulus class, holds the ordered category design fixed, and refits every replicate. The default requests 2,000 replicates and requires at least 1,600 successful interior fits plus a 90% success fraction before emitting percentile 95% intervals for mu, sigma and d_a.

Only `Converged` fits enter the interval distribution. Boundary, weakly identified, non-identifiable, non-converged and non-evaluable replicates remain in the denominator and are returned separately. Insufficient success returns no intervals. The seed and count-correction mode are recorded.

The `bootstrap_coverage` binary simulates known-answer Gaussian category counts and compares the returned mu, sigma and d_a intervals with the generating values. It reports the full-fit disposition counts, interval-return rate, coverage conditional on a returned qualified interval, mean interval widths and bootstrap failure fractions. Its defaults (50 outer campaigns × 100 bootstrap replicates) are a smoke/calibration pilot only; increase outer campaigns, bootstrap replicates and trial budgets for formal Monte Carlo assessment, and use disjoint calibration/held-out seed sets.

These are conditional category-count bootstrap intervals, not participant-level cluster bootstrap intervals or a proof of calibrated 95% coverage. They do not account for sampling variability across participants, sessions or sites, and their coverage must be benchmarked against known-answer synthetic campaigns and the external reference workflow before use for scientific conclusions. A large failed-fit fraction is itself a design/identifiability warning, not a nuisance to be silently filtered out.

## Run

    cargo test --manifest-path tools/spectra-uvsd-fit/Cargo.toml
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin recovery_sweep > uvsd-recovery.csv
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin bootstrap_coverage -- --outer=50 --bootstraps=100 --trials-per-class=400 > bootstrap-coverage.csv

The recovery runner emits one CSV row per frozen synthetic campaign across mu, sigma and independent seeds, including optimizer and Fisher-information diagnostics. It is a parameter-recovery smoke benchmark only; its Wald intervals are not yet coverage-calibrated and the runner does not estimate false UVSD declaration rates, power, or empirical performance.

This code is an estimator prototype, not an empirical result. It has not been compiled or run in the current environment. Human-subject collection, physical actuation and causal/anomalous information claims are out of scope.
