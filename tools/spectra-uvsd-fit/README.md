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

## Pre-study design search

The public `expected_fisher_information_for_design` API computes local expected information for proposed category boundaries before observations exist. The `design_search` binary explores all 1,287 five-cut designs from a fixed [-3, 3] grid for six ordinal categories, scoring the worst-case expected standard error of d_a across mu={0.5, 1.0, 1.5, 2.0} and sigma={0.70, 1.0, 1.30, 1.60}, with 250 signal and 250 noise trials per design cell. It reports equal-probability and fixed-spacing baselines plus the top 20 minimax candidates.

This is a **model-based design sensitivity analysis**, not evidence that an optimized category scale improves real participant measurements. It assumes the specified Gaussian distributions and known design parameters; the next qualification step is Monte Carlo recovery under lapses, RT/confidence noise, criterion changes, class imbalance and missingness, with frozen design choices and disjoint held-out seeds. Use worst-case and average precision together; do not choose a design solely from one parameter point.

## Latent-threshold finite-sample design recovery

The `latent_design_recovery` binary compares four six-category evidence-threshold vectors: equal-probability, fixed-spacing, the current minimax candidate, and the trade-off candidate. Every design is simulated at the same signal/noise trial totals. It reports per-cell fit dispositions, parameter bias/RMSE, expected-Fisher diagnostics at the generating values, and local Wald interval availability, conditional coverage, Wilson bounds and widths.

All four design arms use common random numbers: the same underlying latent Gaussian draws for a given (mu, sigma, replicate) are categorized by each threshold vector. This permits paired comparisons and reduces avoidable Monte Carlo noise between candidate designs. The seed root and PRNG are emitted in the CSV receipt. Use a seed root reserved for evaluation; do not use the same campaigns to select the design and claim its performance.

Example:

```bash
cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin latent_design_recovery -- --outer=2000 --trials-per-class=250 --seed=0x5350454344455349 > latent-design-recovery.csv
```

The default 50 outer campaigns are a smoke pilot, not a coverage qualification. Coverage is conditional on returned local Wald intervals; availability and Wilson intervals are reported separately, along with the fit disposition denominator. The output labels every row `LATENT_CONDITIONAL_ONLY`: the candidate evidence thresholds are not controlled participant settings, and this runner does not operationalize confidence criteria or RT measurement. A future operational design must add the observer/channel mapping and burden qualification specified by #10464. These are Gaussian simulation results only, not participant data, a protocol recommendation, or evidence of superiority over external methods.

## Run

    cargo test --manifest-path tools/spectra-uvsd-fit/Cargo.toml
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin recovery_sweep > uvsd-recovery.csv
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin design_search > spectra-design.csv

The recovery runner emits one CSV row per frozen synthetic campaign across mu, sigma and independent seeds, including optimizer and Fisher-information diagnostics. It is a parameter-recovery smoke benchmark only; its Wald intervals are not yet coverage-calibrated and the runner does not estimate false UVSD declaration rates, power, or empirical performance.

This code is an estimator prototype, not an empirical result. It has not been compiled or run in the current environment. Human-subject collection, physical actuation and causal/anomalous information claims are out of scope.


## Sequential policy core (experimental)

The crate exports an experimental `sequential` module for choosing among a frozen set of **experimenter-controlled** trial actions. It intentionally has no action field for participant confidence criteria, internal evidence thresholds or latent response bias.

The selector currently provides:
- typed action-space validation and canonical action ordering;
- hard exclusion of actions without qualified action-to-observation mappings or qualified model adequacy;
- minimum representativeness and model-discrimination floors;
- forced, randomized audit-stratum coverage before ordinary exploration/exploitation;
- deterministic SplitMix64 research randomization;
- exploration plus exploitation of a caller-supplied worst-case target utility;
- marginal propensity for every eligible action, including the combined exploration and exploitation probability mass;
- replay receipts containing policy configuration, model-set version, action-set/history/assessment digests, complete candidate assessments, random draws, exclusions and selected action.

The selector is intentionally a **policy primitive**, not a Bayesian adaptive inference engine. Its caller must supply and freeze the assessments, robust utility, model-adequacy decisions, digest commitments, estimand, ambiguity set and action-to-observation calibration. The module does not compute EIG, posterior updates, model-discrepancy diagnostics, confidence/RT validity, or sequential p-values. A qualified action set and complete receipt do not establish statistical calibration.

SplitMix64-v1 is reproducible simulation randomness, not cryptographic randomness or a deployment-grade randomization service. Do not use this prototype for live human-subject adaptive presentation. See the [SPECTRA sequential-design specification PR #10491](https://github.com/Luminous-Dynamics/symthaea/pull/10491) and [tracking issue #10490](https://github.com/Luminous-Dynamics/symthaea/issues/10490).
