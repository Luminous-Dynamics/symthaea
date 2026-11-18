# Exact finite benchmark oracle

The standalone Rust crate in this directory is an independent exact oracle for the first benchmark fixtures in `../fixtures/v0.1/cases.json`.

It checks the mathematical ground truth directly:
- D4 closure, identity and inverses;
- exact unordered point-set stabilizers for D4, C4 and identity-only near-symmetry cases;
- invariance of the D4 quartic and C4-only mixed quartic under the full D4 matrix set;
- orientation and input-order invariance of the point-set oracle;
- checked integer polynomial expansion and overflow failure.

These tests are not learned-model benchmark runs. They validate the fixture labels used as truth for future baseline comparisons. The exact oracle uses small integer matrices and polynomial coefficients, avoiding floating-point tolerance in these first labels.

Run:

    cargo fmt --manifest-path research/symmetry-discovery-benchmark-001/reference/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-discovery-benchmark-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-discovery-benchmark-001/reference/Cargo.toml --all-targets -- -D warnings

The benchmark remains `protocol_fixtures_not_benchmark_results` until independent baseline adapters have been run on a frozen split manifest.
