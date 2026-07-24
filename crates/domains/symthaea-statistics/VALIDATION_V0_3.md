# v0.3 validation record

Validation performed on 2026-07-21 against commit `40e3c2a` and its ancestors in the v0.3 series.

## Repository-static checks

- Parsed 27 Rust source/test files with the tree-sitter Rust grammar: no syntax errors.
- Verified all 22 `pub mod` declarations resolve to source files.
- Verified every root `pub use` name resolves to a top-level public declaration in its module.
- Counted 116 unit and integration tests.
- `git diff --check`: clean.
- `scripts/verify.sh` shell syntax: clean.
- Cargo manifest parsed successfully as TOML and reports version 0.3.0.
- Audited all new production modules for `unwrap`, `expect`, `panic!`, and `unreachable!`: none remain before their test modules.

## Independent differential reconstruction

The new formulas were independently reconstructed in Python and compared with SciPy. This does not execute the Rust implementation, but it tests the same equations, branch choices, tail parameterizations, and continuity corrections.

### F distribution

3,000 randomized cases with numerator/denominator degrees of freedom sampled log-uniformly from `0.1..10,000` and `x` from `1e-6..1e6`:

- maximum absolute CDF/SF error: `5.23e-8`
- maximum CDF+SF complement error: `5.23e-8`

The largest errors occurred in highly asymmetric, very small-tail regimes and are consistent with the crate's dependency-free incomplete-beta accuracy.

### Fisher exact test

500 random 2×2 tables with cell counts from 0 through 30:

- maximum absolute directional/two-sided p-value error: `8.20e-14`

The comparison used SciPy's fixed-margin Fisher exact implementation.

### Mann-Whitney U

500 randomized two-sample cases with ties, compared with SciPy's asymptotic, tie-corrected, continuity-corrected method:

- U statistic and p-value differences: zero at displayed precision

### One-way ANOVA

500 randomized multi-group cases:

- maximum absolute F-statistic difference: `2.07e-13`

## Toolchain limitation

A Rust toolchain was not installed in the artifact environment, and network access was unavailable for `rustup`. Therefore `cargo fmt`, `cargo check`, `cargo clippy`, `cargo test`, and `cargo doc` were not executed here. The included `scripts/verify.sh` runs all of those checks in a Rust 1.85+ environment.
