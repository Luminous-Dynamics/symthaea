# Alpha.8 Release Checklist

Before treating an alpha.8 report as publishable pilot evidence, run:

1. `cargo fmt --check`
2. `cargo test --all-features`
3. `cargo run --bin symthaea-quantum-comp -- fixtures`
4. `cargo run --bin symthaea-quantum-comp -- replay local-research`
5. `cargo run --bin symthaea-quantum-comp -- gate demo-binding`
6. `cargo run --example release_gate`
7. `cargo run --example interop_boundary`

Copy all warnings and caveats into any report.

Never rewrite local fingerprints as cryptographic receipts.
