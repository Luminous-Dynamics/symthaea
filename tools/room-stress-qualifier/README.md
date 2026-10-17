# Room Stress Qualifier

Independent Rust CLI for validating the synthetic room-imagination stress manifest. It checks JSON Schema Draft 2020-12 and additional semantic invariants; it is deliberately outside the main Symthaea workspace.

## Run from repository root

```sh
cargo test --manifest-path tools/room-stress-qualifier/Cargo.toml
cargo run --manifest-path tools/room-stress-qualifier/Cargo.toml -- \
  docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json \
  docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.schema.json
```

With no CLI arguments, the executable reads the default benchmark and schema paths relative to the current working directory (repository root). With exactly two arguments, it reads the given benchmark and schema files. Invalid input exits nonzero.

## What this does — and does not — prove

It checks the manifest/schema contract, exact case ordering and IDs, non-empty required lists, allowed risk values, unknown-field rejection and duplicate IDs. Unit tests include negative controls for malformed JSON, missing/extra fields, unsupported risk values, empty required lists, sequence gaps, wrong counts, and an invalid schema.

It does **not** run geometry predicates, daylight/energy/thermal/acoustic simulation, accessibility assessments, code compliance, safety reviews, professional approval or human studies. Its success means only that the fixture manifest passes these checks. A future geometry/simulation adapter needs its own qualified contract, golden corpus, uncertainty model and independent checker.
