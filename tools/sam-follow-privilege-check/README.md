# SAM Follow Privilege Checker v0.1

Standalone validator for the SAM Follow Privilege-Separation Benchmark report.

Run from repository root:

```bash
cargo fmt --manifest-path tools/sam-follow-privilege-check/Cargo.toml -- --check
cargo test --manifest-path tools/sam-follow-privilege-check/Cargo.toml
cargo run --manifest-path tools/sam-follow-privilege-check/Cargo.toml -- \
  docs/autonomy/SAM_FOLLOW_PRIVILEGE_BENCHMARK_SCHEMA_V0_1.json \
  docs/autonomy/fixtures/SAM_FOLLOW_PRIVILEGE_BENCHMARK_V0_1.json
cargo clippy --manifest-path tools/sam-follow-privilege-check/Cargo.toml --all-targets -- -D warnings
```

This is a nested Cargo workspace, intentionally independent of the root workspace globs.

The validator applies full JSON Schema Draft 2020-12 validation plus SAM semantic checks:
- arm and trial ID uniqueness and reference resolution;
- no simulator-oracle/V2X visibility in local-only observation profiles;
- correct source role for Follow target by arm;
- blend, supervision, evaluation-only and reference-only consistency;
- Arm C no-teacher/no-update/replay/exploration invariants;
- exact parameter digest invariance in Arm C;
- explicit recurrent-state evolution consistency;
- evaluator truth strictly after action commitment and never policy-visible;
- specification-only reports contain no trials.

A passing check establishes only structural and declared information-flow semantics for the supplied report. It does not establish perception quality, statistical validity, physical safety or deployment authority.
