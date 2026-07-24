# Validation record — v0.5.0

Date: 2026-07-21

## Source and API audit

- All Rust source and integration-test files parsed successfully with the tree-sitter Rust grammar.
- The public-surface audit resolved 312 root exports.
- The source tree contains 205 `#[test]` cases.
- `git diff --check` passed.
- New streaming charts reject invalid observations before mutating state.
- New randomized partitioning and matching procedures are deterministic under identical inputs and seeds.

## Independent numerical references

The retained generator `validation/generate_v0_5_references.py` uses NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and statsmodels 0.14.6. They are validation-only dependencies.

Committed reference values include:

- ROC AUC `0.875` and average precision `0.8333333333333333` for a tied-score binary example.
- Poisson log-linear coefficients `[0.05822004434200412, 0.2367202261396152]`, deviance `0.9831018361910427`, and null deviance `36.56779558175958`.
- HC3 OLS standard errors `[0.09551732243170924, 0.006974344526421904]`.
- PCA sample-covariance eigenvalues `[12.5, 0.0]`.
- Log-rank chi-square `7.344406814715234` and p-value `0.006727172585530289`.
- Cox coefficient `0.649776433611313` with standard error `0.7780085325594036` under Breslow ties.
- DerSimonian-Laird tau-squared `3.9900000000000007` for the committed heterogeneous-study example.
- Conditional AR(1) coefficients approximately `[1.0, 0.5]`.

The integration reference suite uses explicit tolerances reflecting algorithm and platform arithmetic rather than exact string equality.

## Contract and adversarial coverage

The v0.5 adversarial suite checks:

- disjoint and exhaustive folds;
- minority-class stratification refusal;
- non-finite predictive input rejection;
- singular Poisson and robust-OLS design refusal;
- asymmetric eigensystem refusal and zero-variance PCA refusal;
- negative survival times and all-censored Cox refusal;
- strict propensity overlap;
- deterministic caliper matching;
- impossible beta moments and invalid meta-analysis scales;
- zero-cost threshold-policy refusal; and
- transactional EWMA/CUSUM updates.

## Toolchain limitation

The environment exposed `rustup`, Cargo, rustc, and rust-analyzer shims but no installed Rust toolchain. Network access was unavailable, so Rust 1.85 could not be downloaded. Consequently `cargo check`, `cargo test`, Clippy, rustfmt, rustdoc, and packaging were not executed here. `scripts/verify.sh` remains mandatory before publication.
