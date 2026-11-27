# SAT memory trace semantic verifier v0.1

This standalone Rust crate checks semantic invariants over SAT memory lifecycle JSONL traces.

## Run

```sh
cargo test --manifest-path research/sat-memory-trace-verifier-v0-1/Cargo.toml
cargo run --manifest-path research/sat-memory-trace-verifier-v0-1/Cargo.toml -- docs/research/fixtures/sat-memory-lifecycle-trace-v0.1.jsonl
```

## Checks

- Trace IDs and event IDs are unique.
- Event sequence numbers are contiguous and predecessor links match.
- Invariant references resolve within their trace.
- Paired control references resolve within the corpus.
- Adaptive rounds stay within the declared budget.
- Store-only traces cannot claim agent decisions or external effects.
- Confirmed effects and vulnerable positive controls require independent sink evidence.
- T10 traces order erasure, restore, witness reconciliation and post-restore read.
- L2 counterfactual-repair traces require at least two attack records, a no-effect counterfactual, correction, invalidation, rebuild and residual-influence absence.
- Executed/reproduced traces require a run manifest and SHA-256 for every evidence artifact.

## Boundaries

This is a semantic verifier, not a complete Draft 2020-12 JSON Schema validator. It does not dereference evidence URIs, authenticate signatures, verify external witnesses, run agent trials, calculate benchmark scores or establish causal effects on its own. The counterfactual fixture is only a structural example.

The source and tests are authored. In the present authoring environment, `cargo`, `rustc` and `rustfmt` are unavailable; compilation, tests, formatting and Clippy have not been run. Treat the source as **not yet compiled** until the exact-head Rust gates pass.
