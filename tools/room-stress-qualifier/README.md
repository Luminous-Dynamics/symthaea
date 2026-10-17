# Room Stress Qualifier

Independent Rust CLI for validating the synthetic room-imagination stress manifest. It checks JSON Schema Draft 2020-12 and additional semantic invariants; it is deliberately outside the main Symthaea workspace.

## Run from repository root

```sh
cargo test --manifest-path tools/room-stress-qualifier/Cargo.toml
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml -- --json
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml -- --evaluate RUN_REPORT.json
```

The default paths point to the checked-in benchmark and schema relative to the repository root. To validate alternate files, pass both paths, optionally after `--json`. Invalid input exits nonzero.

In JSON mode the tool emits a deterministic, content-addressed receipt with SHA-256 digests of the exact benchmark and schema bytes, the validator source and Cargo manifest, the validator version, the case/family/risk counts and an explicit claim ceiling. The receipt proves only that this validator accepted those manifest inputs; it is not a signed attestation and does not qualify spatial geometry or building performance.

## What this does — and does not — prove

It checks the schema itself, the manifest/schema contract, exact case ordering and IDs, non-empty required lists, allowed risk values, unknown-field rejection and duplicate IDs. Unit tests include negative controls for malformed JSON, missing/extra fields, unsupported risk values, empty required lists, sequence gaps, wrong counts, invalid schemas, deterministic receipts, incorrect observations, missing evidence references, benchmark digest mismatches, and duplicate result cases. The synthetic golden-answer fixture tests evaluator mechanics only; it is not a simulated design system.

It does **not** run geometry predicates, daylight/energy/thermal/acoustic simulation, accessibility assessments, code compliance, safety reviews, professional approval or human studies. Its success means only that the fixture manifest passes these checks. A future geometry/simulation adapter needs its own qualified contract, golden corpus, uncertainty model and independent checker.
