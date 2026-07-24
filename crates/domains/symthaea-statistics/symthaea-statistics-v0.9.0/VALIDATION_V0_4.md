# Validation record — v0.4.0

Date: 2026-07-21

## Source and API audit

- 40 Rust source/test files parsed successfully with the tree-sitter Rust grammar.
- 251 unique root exports were resolved to public functions, structs, enums, traits, or type aliases.
- No `.unwrap(`, `.expect(`, `panic!(`, or `unreachable!(` calls were found in the production portions of the eleven new v0.4 modules.
- `git diff --check` passed.
- The source tree contains 153 `#[test]` cases.

## Independent numerical comparisons

Reference calculations used SciPy 1.17.0, NumPy 2.3.5, and statsmodels 0.14.6. These packages are validation tools only; the Rust crate remains dependency-free. The generator and resulting JSON are retained under `validation/`.

### Distribution quantiles

Selected SciPy reference quantiles:

- Beta(2, 5), p=0.1: `0.09259525891312873`
- Beta(8, 2), p=0.9: `0.9392309503492315`
- Gamma(shape=0.8, rate=2), p=0.1: `0.02649091090008842`
- Gamma(shape=10, rate=0.5), p=0.99: `37.56623478662507`
- Gamma(shape=100, rate=3), p=0.999: `44.590087970459535`

The committed v0.4 integration tests require beta/gamma CDF-quantile round trips within `1e-10` for ordinary and tail cases. Quantiles use bounded monotone bisection over the crate's validated CDFs.

### Logistic regression

For the committed penalized binary dataset (`ridge=0.1`):

- IRLS reconstruction: `[0.5430698023749847, 2.123813674832848]`
- Independent BFGS penalized-likelihood optimizer: `[0.5430698009988928, 2.123813674309244]`
- Maximum absolute coefficient difference: `1.38e-9`
- IRLS iterations: 7

### Multiple and robust regression

The multiple-regression reference system recovered coefficients `[2.975, 2.01, -0.98]` to floating-point precision. A Huber regression with one gross endpoint outlier recovered `[1.0, 2.0]`; statsmodels assigned the outlier a weight below `1e-16`.

### Power and sequential evidence

- Two-sided equal-group normal-approximation design, d=0.5, alpha=0.05, target power=0.8: 63 observations per group; achieved power `0.8013023941`; 62 gives `0.7950080284`.
- Two-sided correlation design, r=0.3, alpha=0.05, target power=0.8: n=85; achieved power `0.8003462499`.
- Bernoulli SPRT p0=0.3, p1=0.7, alpha=beta=0.05: upper boundary `2.9444389792`; four consecutive successes cross the alternative boundary.
- Bounded-mean confidence sequence at t=2000, alpha=0.05, observed mean=0.75: radius `0.06962109`, interval approximately `[0.68038, 0.81962]`.

## Toolchain limitation

The artifact environment had `rustup` shims but no installed Rust toolchain, and outbound network access was unavailable. Therefore `cargo test`, `cargo clippy`, `cargo fmt`, `cargo doc`, and `cargo package` were not executed here. `scripts/verify.sh` remains the authoritative Rust-enabled verification lane and must be run before release publication.
