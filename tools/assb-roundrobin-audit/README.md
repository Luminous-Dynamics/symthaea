# ASSB round-robin reproducibility audit v0.1

Dependency-free Rust 1.96 read-only audit for ASSB interlaboratory round-robin evidence.

Checks laboratory/fixture/cell identity, fixture ownership and calibration references, raw-data digests, rights references, independent cell identity, exact common-protocol grouping, non-pooling of lab-native/deviation arms, and minimum replicate/laboratory counts.

A synthetic 3-lab × 3-cell fixture is explicitly reported as `KnownAnswerFixtureOnly`; it never qualifies as physical reproducibility. A physical common-protocol group with at least three labs and at least three distinct cells per lab is only a `CandidateCrossLabVarianceAnalysis`. The software does not calculate or certify statistical reproducibility.

## Commands

    cargo metadata --manifest-path tools/assb-roundrobin-audit/Cargo.toml --locked --no-deps
    cargo fmt --manifest-path tools/assb-roundrobin-audit/Cargo.toml -- --check
    cargo test --manifest-path tools/assb-roundrobin-audit/Cargo.toml --locked
    cargo clippy --manifest-path tools/assb-roundrobin-audit/Cargo.toml --all-targets --locked -- -D warnings
