<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Microdisplay yield reference

This isolated Rust crate holds analytical yield/repair reference calculations for [MICRODISP-YIELD-001 #7868](https://github.com/Luminous-Dynamics/symthaea/issues/7868). It is deliberately not a semiconductor-process model.

## Run

```sh
cargo fmt --manifest-path tools/microdisplay-yield-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/microdisplay-yield-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/microdisplay-yield-reference/Cargo.toml
```

## Reference functions

- Expected good/defect counts under a simple independent equal-probability Bernoulli model.
- Observed fraction and Wilson score interval for a declared sample denominator.
- Stage-yield composition only when the caller explicitly declares independence.
- Repair-adjusted yield with explicit repair-success and retest assumptions.
- Fraction meeting a declared quality floor, for separating yield from optical feasibility.

The unit tests include negative controls for invalid probabilities and denominators, correlated/unknown stage assumptions, impossible success counts and non-finite quality inputs. They check formulas only; they do not validate the 20-case corpus schema, establish defect independence, or imply production data.

A successful test run means the implementation matches these local analytical references. It is not manufacturing-yield evidence, physical micro-LED characterization, panel reliability qualification, or proof of any cost reduction.
