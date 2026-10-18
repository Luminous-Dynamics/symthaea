# Metal Stock-to-Part Benchmark

Standalone Rust validator for a frozen synthetic stock-to-part, secondary-feedstock and repair/remanufacture corpus.

From the repository root:

```sh
cargo fmt --manifest-path tools/metal-stock-part-bench/Cargo.toml -- --check
cargo test --manifest-path tools/metal-stock-part-bench/Cargo.toml
cargo run --quiet --manifest-path tools/metal-stock-part-bench/Cargo.toml -- \
  docs/tech/metal-stock-part-corpus-v0.1.json \
  --source-commit "$(git rev-parse HEAD)"
```

The v0.1 evaluator validates explicit material and lot identity, source/grade evidence, secondary-feedstock custody and contamination status, process history, state transitions, equipment capabilities, final physical inspection evidence, and mass balance by stream.

The 20-scenario corpus is synthetic. A passing receipt establishes the declared software contract only; it is not evidence that a real stock lot, alloy, process, machine, part, secondary stream or facility is qualified. It does not claim industrial productivity, energy savings or circularity improvements.
