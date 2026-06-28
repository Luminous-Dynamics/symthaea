# Alpha.10 Release Checklist

Before publishing results from this release, run locally:

- `cargo fmt --check`
- `cargo test --all-features`
- `cargo run --bin symthaea-quantum-comp -- gate smoke-binding`
- `cargo run --bin symthaea-quantum-comp -- matrix local-research`
- `cargo run --bin symthaea-quantum-comp -- verify-matrix`
- `cargo run --bin symthaea-quantum-comp -- beta`
- `cargo run --bin symthaea-quantum-comp -- snapshot`

Keep blocked claims attached to reports:

- no quantum consciousness claim
- no quantum advantage claim
- no physical backend execution claim without external backend metadata
- no Mycelix attestation claim from local fingerprints
