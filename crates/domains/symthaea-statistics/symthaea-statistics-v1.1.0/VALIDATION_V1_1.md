# v1.1 validation record

This record describes the checks performed on the `symthaea-statistics` v1.1
release candidate in the artifact environment.

## Source surface

- 108 public modules plus one private model-matrix module
- 591 resolved root exports
- 110 Rust source files and 21 integration-test files
- 455 `#[test]` functions present in source and integration tests
- no `unsafe` token outside the crate-level prohibition
- no production `unwrap`, `expect`, `panic!`, or `unreachable!` surface
- every declared module has a matching source file and every module file is declared
- every item in the stable v1 prelude resolves through the root export surface

## Syntax and repository checks

- all 131 Rust source and integration-test files parse without tree-sitter error nodes
- `git diff --check` passes
- the working tree is clean after the release-evidence commit
- the independent v1.1 reference generator reproduces its committed JSON byte-for-byte

## Independent numerical landmarks

The retained generator uses NumPy, SciPy, scikit-learn, and statsmodels to
record independent values for:

- multivariate energy distance and nonlinear distance correlation
- Gaussian-kernel MMD and the pooled median-distance bandwidth
- Euclidean one-factor PERMANOVA
- Kruskal-Wallis inference
- fixed and generalized-DL random-effects meta-regression
- oracle-approximating covariance shrinkage
- normal-inverse-gamma sufficient-statistic updating
- multiclass Brier score, log loss, and top-label accuracy
- Sidak, Hochberg, and Benjamini-Yekutieli multiplicity corrections

The Rust integration vectors bind those values with explicit numerical
tolerances. Seeded permutation and streaming APIs also carry deterministic and
transactional adversarial tests.

## Resource and contract hardening

Distance, kernel, and PERMANOVA matrix requests above 25,000,000 pairwise
elements fail with `ComputationTooLarge`. Invalid probability mass, singular
moderator designs, degenerate covariance, all-tied ranks, invalid bandwidths,
and non-finite streaming updates fail explicitly rather than fabricating a
result or partially mutating state.

## Unavailable checks

The environment contains Rustup launchers but no installed Rust toolchain and
cannot resolve an outbound toolchain download. Therefore Cargo compilation,
execution of the 455 Rust tests, Clippy, rustfmt, rustdoc, and `cargo package`
were not performed here. `scripts/verify.sh` remains the authoritative Rust
1.85+ verification lane, and `scripts/verify_v1_1_references.sh` is the separate
Python reference-regeneration lane.
