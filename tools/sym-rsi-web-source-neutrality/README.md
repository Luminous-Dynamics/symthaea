# Web-research source-neutrality contract

This standalone Rust crate imports the repository's production
`src/web_research/types.rs` and `src/web_research/verifier.rs` by path.
It does not copy or reimplement their logic.

## Contract

- The production retrieval path uses `SourceEvidence::unclassified`, which always sets `supports = None`; this is tested directly.
- Unclassified retrieval results contribute neither support nor contradiction, regardless of query similarity or source confidence.
- A prior `WebResearchResult.confidence` is not evidence that a source supports a particular claim.
- Explicit `Some(true)` and `Some(false)` relations remain available for callers that have performed claim-specific evaluation.

The production researcher still needs a real claim/span attribution stage, source-family independence checks and independent adjudication. This harness tests only that missing attribution cannot be silently converted into support.

## Run

```sh
cargo test --locked --manifest-path tools/sym-rsi-web-source-neutrality/Cargo.toml
cargo clippy --locked --manifest-path tools/sym-rsi-web-source-neutrality/Cargo.toml --all-targets -- -D warnings
```
