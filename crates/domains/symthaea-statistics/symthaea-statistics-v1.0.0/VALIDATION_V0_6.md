# Validation record — v0.6.0

Date: 2026-07-21

## Source and API audit

- All 72 Rust source and integration-test files parsed successfully with the tree-sitter Rust grammar.
- The public-surface audit resolved 364 root exports.
- The source tree contains 256 `#[test]` cases.
- `git diff --check` passed.
- Every production source module contains no `unwrap`, `expect`, `panic!`, or `unreachable!` paths before its `#[cfg(test)]` section; documentation examples are excluded from this executable-surface check.
- Randomized assignment, randomization inference, conformal calibration, isotonic fitting, and BCa resampling are deterministic under identical inputs and seeds.

## Independent numerical references

The retained generator `validation/generate_v0_6_references.py` uses NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and statsmodels 0.14.6. These packages are validation-only dependencies.

Committed reference values cover:

- QR OLS coefficients `[1.1670454545454567, 2.0093749999999986, -0.7255681818181818]`, leverage, Cook's distance, and PRESS.
- Standardized ridge coefficients `[1.4949381326066773, 1.4336759311279288, -0.3773627080147452]`, effective degrees of freedom `2.479493704352424`, and GCV `0.5913954659681645`.
- VIF `1.171875` for both predictors and Breusch-Pagan statistic `2.520840397631898`.
- statsmodels CR1 cluster and Bartlett Newey-West covariance matrices.
- Weighted isotonic predictions `[0.0, 0.5, 0.5, 0.5, 1.0, 1.0]` for a tied-score example.
- Split-conformal radius `0.5` under the committed finite-sample order statistic.
- AIPW ATE `2.4166666666666665` with standard error `0.2006932429798716`.
- Repeated-cross-section DiD interaction `2.0` with standard error `0.2`.
- Exact randomization p-value `1/3` for all six fixed-arm-size assignments.
- Gaussian KDE density `0.1852332088252571`, CDF `0.5786286389957912`, and DKW epsilon `0.13581015157406195`.
- Ljung-Box statistic `22.95561839540373`, Jarque-Bera statistic `0.8133633473934934`, and deterministic BCa interval `[1.75, 7.962667273429325]`.

The Rust integration suite uses explicit tolerances selected for algorithmic and platform arithmetic rather than exact decimal-string identity.

## Contract and adversarial coverage

The v0.6 adversarial suite checks:

- wide and rank-deficient QR refusal;
- invalid ridge penalties and unidentified predictor standardization;
- saturated OLS influence designs;
- one-cluster and invalid-HAC-lag refusal;
- non-positive isotonic weights;
- invalid conformal scales and probabilities;
- strict AIPW overlap and both-arm requirements;
- incomplete DiD cells;
- singleton randomized blocks;
- deterministic Monte Carlo fallback when exact enumeration is capped;
- invalid or unidentified KDE bandwidths;
- DKW and Ljung-Box domain checks; and
- BCa minimum-sample and iteration contracts.

## Toolchain limitation

The environment exposed rustup/Cargo/rustc launchers but no installed Rust toolchain, and no cached or network-resolvable toolchain was available. Consequently `cargo check`, `cargo test`, Clippy, rustfmt, rustdoc, and package construction were not executed here. `scripts/verify.sh` remains mandatory before publication under Rust 1.85+.
