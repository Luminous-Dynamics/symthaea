# Validation record — v0.7.0

Date: 2026-07-21

## Source and API audit

- All 84 Rust source and integration-test files parsed successfully with the tree-sitter Rust grammar.
- The public-surface audit resolved all 418 root exports.
- The source tree contains 290 `#[test]` cases.
- `git diff --check` passed.
- Every production source module contains no `unwrap`, `expect`, `panic!`, or `unreachable!` paths before its `#[cfg(test)]` section; documentation examples are excluded from this executable-surface check.
- Seeded and grid-selected procedures are deterministic under identical inputs.

## Independent numerical references

The retained generator `validation/generate_v0_7_references.py` uses NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and statsmodels 0.14.6. These packages are validation-only dependencies.

Committed reference values cover:

- Negative-binomial NB2 log-PMF `-3.2956292587802802`, CDF `0.844479919949704`, and direct survival probability `0.15552008005029602`.
- Fixed-dispersion NB2 regression coefficients `[0.24919256239736642, 0.19198405994266482]` and deviance `6.242553710441211`.
- Standardized lasso coefficients `[2.9826740767270317, 0.0]` on the original predictor scale.
- Theil-Sen slope `2.0`, intercept `1.0`, and Kendall tau-b `0.8` for tied data.
- Exact McNemar p-value `0.021484375` and Cochran Q statistic `7.6` with p-value `0.022370771856165598`.
- Horvitz-Thompson total `24.0`, Hájek mean `4.0`, Kish effective sample size `3.0`, and stratified mean `8.0`.
- Nelson-Aalen cumulative hazard `1.75` and target cumulative incidence `0.5` in hand-auditable examples.

The Rust integration suite uses explicit tolerances selected for algorithmic and platform arithmetic rather than decimal-string identity.

## Contract and adversarial coverage

The v0.7 adversarial suite checks:

- invalid NB2 dispersion and all-zero count outcomes;
- constant predictors in standardized elastic net;
- entirely unobserved columns during imputation;
- samples larger than declared stratum populations;
- absent and invalid competing-risk causes;
- degenerate all-success or all-failure empirical beta priors; and
- invalid smoothing parameters and non-finite series.

## Toolchain limitation

The environment exposed rustup/Cargo/rustc launchers but no installed Rust toolchain, and no cached or network-resolvable toolchain was available. Consequently `cargo check`, `cargo test`, Clippy, rustfmt, rustdoc, and package construction were not executed here. `scripts/verify.sh` remains mandatory before publication under Rust 1.85+.
