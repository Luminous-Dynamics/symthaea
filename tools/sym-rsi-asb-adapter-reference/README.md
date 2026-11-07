# Autonomous Science Bench compatibility reference

This standalone offline Rust crate validates the checked-in, pinned AS-Bench compatibility manifest:

`docs/research/adapters/SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json`

It checks the upstream repository/commit/API/schema pin, eight Git blob identities, exact v0 campaign identities, round × batch budget arithmetic, discovery and held-out-learning gate mappings, upstream reference-result labels, and the provenance distinction between replay and digital-twin outputs.

It is deliberately **not** a campaign runner, metric evaluator, oracle, receipt/authentication service, scheduler, memory service or promotion authority. A valid manifest proves only that the local compatibility declaration matches this crate's frozen contract.

## Run

```bash
cargo fmt --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- docs/research/adapters/SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json
```

## Claim boundary

- The upstream pin is a reproducibility anchor, not a claim that it is the latest upstream revision.
- The upstream reference scores in the manifest are not Symthaea results.
- Replay results are published measurements; twin results are model-mediated and are not live laboratory results.
- The checker does not fetch upstream content or cryptographically attest the source. File identities are recorded Git blob SHA-1 identifiers from the pinned commit; they do not authenticate a run or establish truth.
- The JSON Schema is structural. This crate adds semantic checks for cross-field relationships and the two exact campaign contracts.
- No Rust compilation, formatting, Clippy, test or CI result is claimed until those commands execute on an available toolchain.
