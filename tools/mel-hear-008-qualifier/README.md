# MEL-HEAR-008 incidence qualifier

Standalone Rust 1.96 harness for the research-only incidence/estimand manifest.

Run from the repository root:

```bash
cargo +1.96.0 fmt --manifest-path tools/mel-hear-008-qualifier/Cargo.toml --check
cargo +1.96.0 test --locked --manifest-path tools/mel-hear-008-qualifier/Cargo.toml
cargo +1.96.0 clippy --locked --manifest-path tools/mel-hear-008-qualifier/Cargo.toml --all-targets -- -D warnings
cargo +1.96.0 run --locked --manifest-path tools/mel-hear-008-qualifier/Cargo.toml -- research/mel-hear-008/incidence-panel-v0.json
```

## v0 scope

- Requires an explicit row for every possible work × performer pair.
- Counts only rows marked observed when computing the incidence graph.
- Validates single-performer cell weights, observed outcomes, missingness reasons and declared support diagnostics.
- Recomputes observed-cell, equal-work and equal-performer means from the explicit cell weights.
- Rejects a numeric estimate for a target that includes missing cells without an authorized model.
- Rejects set-valued/ensemble performer membership. It does not expand one ensemble observation into one pseudo-independent row per member.

The parser is intentionally stricter than the draft schema for v0. Ensemble/multiple-membership rows need an explicit group-level or membership-weighted estimand before they are eligible for inference.

This tool is a deterministic contract checker. It does not choose a confidence-interval method, authenticate performer identities, validate audio or annotations, establish sampling exchangeability, or qualify real-cohort inference.
