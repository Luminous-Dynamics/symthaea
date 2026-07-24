# Validation record — v0.9.0

## Scope

v0.9 adds finite Markov chains, categorical hidden Markov inference, multinomial logistic regression, excess-zero count models, paired/repeated rank inference, uncertainty propagation, Rubin multiple-imputation pooling, Bayesian bootstrap posteriors, and univariate Gaussian mixtures.

## Independent references

`validation/generate_v0_9_references.py` retains independently generated values from:

- NumPy 2.3.5,
- SciPy 1.17.0,
- scikit-learn 1.8.0, and
- statsmodels 0.14.6.

The retained snapshot covers Markov and HMM recursions, penalized multinomial-logit coefficients and predictions, zero-inflated Poisson likelihood fitting, hurdle-rate root finding, Wilcoxon and Friedman inference, delta-method variance, Rubin pooling, and Gaussian-mixture parameters.

Run `./scripts/verify_v0_9_references.sh` in an environment containing those Python libraries. The generator must reproduce `validation/v0_9_reference_results.json` byte-for-byte.

## Repository checks completed in the artifact environment

- Portable root-export and production panic-surface audit: **509 resolved root exports**
- Tree-sitter parsing: **107 Rust source and integration-test files**
- Retained test inventory: **381 unit and integration tests**
- Source-module inventory: **91 root source modules**
- `git diff --check`
- Independent-reference regeneration and byte comparison
- Adversarial coverage for probability endpoints, ragged/asymmetric covariance, impossible hidden observations, malformed stochastic matrices, missing declared outcome classes, and degenerate mixtures

## Rust execution limitation

The artifact environment exposes Rustup launchers but has no installed Rust toolchain and no available network path to install one. Cargo compilation, rustfmt, Clippy, rustdoc, package construction, and execution of Rust tests are therefore not claimed here.

`scripts/verify.sh` runs the complete Rust 1.85+ lane. Independent Python reference comparison remains a separate evidence lane through `scripts/verify_v0_9_references.sh`.

Exact patch reconstruction, Git-bundle verification, release tree hash, and artifact checksums are recorded after the release tag is created.
