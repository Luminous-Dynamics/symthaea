# SALib Rust reference validation 001

This stand-alone research harness tests one carefully selected method path in
Rust `salib 0.3.0` without adding the dependency to Symthaea's production
workspace or allowing optional Arrow/Polars/surrogate features into the build.

## Why this exists

Symtropy #3784 requires global sensitivity and identifiability diagnostics.
Symthaea already contains a small Morris screening implementation, while the
external Rust `salib` 0.3.0 package advertises a broader sensitivity suite.
Before adopting it, validate estimator outputs against analytic references and
measure the dependency/feature cost.

## Reproducible run

```sh
cargo generate-lockfile --manifest-path research/salib-validation-001/Cargo.toml
cargo test --manifest-path research/salib-validation-001/Cargo.toml
cargo tree --manifest-path research/salib-validation-001/Cargo.toml --depth 2
cargo clippy --manifest-path research/salib-validation-001/Cargo.toml --all-targets -- -D warnings
```

The dependency is pinned to exactly `salib = 0.3.0`; the feature selection is
limited to `samplers` and `estimators` and omits the crate's parallel,
surrogate, Shapley, Arrow and Polars features. CI generates and uses a lockfile
within the isolated run. A production dependency decision requires committing
a reviewed lockfile or workspace lock update, not merely relying on CI's
ephemeral generated lock.

## Analytic test

The Ishigami function has three independent uniform inputs over [-π, π]:

```
f(x) = sin(x1) + 7 sin²(x2) + 0.1 x3⁴ sin(x1)
```

Its analytic first-order Sobol indices are approximately
[0.3139, 0.4424, 0.0000], and total-order indices approximately
[0.5576, 0.4424, 0.2437]. The third factor has essentially no first-order
effect but a non-zero total-order effect through interaction with x1.

The test uses a fixed 8,192-row base design, records the exact evaluation
budget N(d+2), and checks the independent analytic references with a declared
0.03 sampling tolerance. It also checks that reusing the full RNG state seed
reproduces identical index vectors.

## Claim ceiling

A pass validates only this Saltelli 2010/Ishigami path. It does not validate
all estimators in `salib`, dependent-input Shapley correctness, arbitrary
physical models, parameter identifiability, or Bayesian calibration. Those
require separate references and validation fixtures.
