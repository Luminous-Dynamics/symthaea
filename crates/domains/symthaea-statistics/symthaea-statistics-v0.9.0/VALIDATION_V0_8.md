# Validation record — v0.8.0

## Scope

v0.8 adds multicategory probability, multiclass metrics, agreement/reliability, circular statistics, dependent-data bootstrap procedures, local-level Kalman estimation, generalized Pareto/Hill tail analysis, and Hotelling multivariate mean tests.

## Independent references

`validation/generate_v0_8_references.py` retains independently generated values from:

- NumPy 2.3.5,
- SciPy 1.17.0, and
- scikit-learn 1.8.0.

The retained snapshot covers:

- Dirichlet log density, moments, and covariance,
- multinomial and Dirichlet-multinomial count-vector mass,
- multiclass precision/recall/F1 and Cohen kappa,
- weighted ordinal kappa,
- Bland-Altman limits and Lin concordance,
- raw/standardized Cronbach alpha and ICC variants,
- circular summaries and the finite-sample Rayleigh approximation,
- every local-level Kalman update and RTS-smoothed state,
- generalized Pareto density/CDF/direct tail/quantile values,
- Hill tail-index estimation, and
- one- and two-sample Hotelling T² with SciPy F tails.

Run `./scripts/verify_v0_8_references.sh` in an environment containing the named Python libraries. The generator must reproduce `validation/v0_8_reference_results.json` byte-for-byte.

## Repository checks completed in the artifact environment

- Portable root-export and production panic-surface audit: **470 resolved root exports**
- Tree-sitter parsing: **96 Rust source and integration-test files**
- Retained test inventory: **337 unit and integration tests**
- Source-module inventory: **82 root source modules**
- `git diff --check`
- Independent-reference regeneration and byte comparison
- Adversarial tests retained for invalid simplex mass, category ranges, degenerate reliability, invalid block configuration, nonphysical state variances, finite generalized-Pareto endpoints, and singular Hotelling covariance

## Rust execution limitation

The artifact environment exposes Rustup launchers but has no installed Rust toolchain and no available network path to install one. Therefore Cargo compilation, rustfmt, Clippy, rustdoc, package construction, and execution of the Rust tests are not claimed here.

`scripts/verify.sh` runs the complete Rust 1.85+ lane:

1. portable source audit,
2. rustfmt check,
3. Cargo check for all targets,
4. Clippy with warnings denied,
5. all unit/integration tests and doctests,
6. rustdoc,
7. metadata validation, and
8. package construction.

The independent Python reference comparison remains a separate evidence lane through `scripts/verify_v0_8_references.sh`.

Exact patch reconstruction, Git-bundle verification, release tree hash, and artifact checksums are recorded after the release tag is created.
