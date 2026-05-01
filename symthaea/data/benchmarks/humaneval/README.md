# Rust-Adapted HumanEval Testbench

`examples/benchmark_humaneval_verified.rs` runs Symthaea's verified generation path against Rust function-synthesis tasks and reports pass@1-style results.

Run the checked-in smoke set:

```bash
cargo run --example benchmark_humaneval_verified --features code_generation -- --input data/benchmarks/humaneval/rust_smoke.jsonl --json
```

Each JSONL row is one task:

```json
{"id":"rust/add","name":"add","purpose":"Add two integers","signature":"fn add(a: i32, b: i32) -> i32","examples":[{"input":"add(2, 3)","output":"5"}]}
```

Required fields:

- `id`: stable benchmark identifier.
- `name`: Rust function name.
- `purpose`: natural language task description.
- `signature`: Rust function signature beginning with `fn`.
- `examples`: one or more input/output assertions used by the verifier.

The harness requires real execution. If the local Rust toolchain cannot link test binaries, tasks should fail with explicit compile errors rather than being counted as verified.
