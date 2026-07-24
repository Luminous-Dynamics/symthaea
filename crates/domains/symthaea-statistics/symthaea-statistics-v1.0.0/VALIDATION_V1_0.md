# Validation record — v1.0.0

## Scope

v1.0 adds exact hypergeometric and beta-binomial probabilities, exact one-proportion inference, compositional geometry, Deming regression, rank-normalized MCMC diagnostics, multivariate-normal density/sampling/conditioning, randomized Halton integration, continuous EDF goodness-of-fit diagnostics, Qn robust scale, and a curated stable prelude.

## Independent references

`validation/generate_v1_0_references.py` retains independently generated values from:

- NumPy 2.3.5,
- SciPy 1.17.0,
- ArviZ 1.1.0, and
- statsmodels 0.14.6.

The snapshot covers hypergeometric and beta-binomial probabilities/moments, exact binomial tests and intervals, CLR/ALR/Aitchison geometry, Deming coefficients, rank-normalized R-hat and bulk ESS, multivariate-normal density and conditioning, Cramer-von Mises/Anderson-Darling/KS statistics, and Qn scale.

Run `./scripts/verify_v1_0_references.sh` in an environment containing those Python libraries. The generator must reproduce `validation/v1_0_reference_results.json` byte-for-byte.

## Repository checks completed in the artifact environment

- Portable root-export and complete production panic-surface audit: **555 resolved root exports**
- Tree-sitter parsing: **120 Rust source and integration-test files**
- Retained test inventory: **423 unit and integration tests**
- Public module inventory: **99 modules**
- Stable prelude import-surface check
- `git diff --check`
- Independent-reference regeneration and JSON parse
- Adversarial coverage for impossible null outcomes, zero-mass compositions, degenerate Deming fits, non-positive-definite covariance, unsupported QMC dimensions, unequal MCMC chain lengths, non-finite Qn samples, and exact support boundaries

## Rust execution limitation

The artifact environment exposes Rustup launchers but has no installed Rust toolchain and no available network path to install one. Cargo compilation, rustfmt, Clippy, rustdoc, package construction, and execution of Rust tests are therefore not claimed here.

`scripts/verify.sh` runs the complete Rust 1.85+ lane. Independent Python reference comparison remains a separate evidence lane through `scripts/verify_v1_0_references.sh`.

Exact patch reconstruction, Git-bundle verification, release tree hash, and artifact checksums are recorded after the release tag is created.
