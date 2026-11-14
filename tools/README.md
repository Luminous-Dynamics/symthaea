# Tools

Debugging and inspection tools for development.

## symthaea-inspect

Interactive debugger for consciousness traces.

### Usage

```bash
# Build
cargo build -p symthaea-inspect

# Run with trace file
./target/debug/symthaea-inspect trace.json
```

### Features

- Visualize consciousness graph evolution
- Inspect Φ values over time
- Debug brain subsystem messages
- Trace HDC vector operations

## development-map-validator

Standalone read-only validator for EGDDM v0.1 map snapshots, temporal decisions, development plans, fixed-budget retrieval cases, and federation manifests v0.1.0/v0.1.1.

Run it from its own directory (it has a separate Cargo workspace and is not a Symthaea root-workspace member):

```bash
cd tools/development-map-validator
cargo run --release -- ../../docs/development-map/readme-and-workstreams-v0.1.json
cargo run --release -- ../../docs/development-map/fixtures/retrieval-exploration-v0.1.json
cargo run --release -- ../../docs/development-map/federation-v0.1.1.json
cargo test
```

The validator checks graph references, provenance/lifecycle shape, temporal invalidation consistency, decision-time evidence/authority, development-plan feasibility references, retrieval case revision/configuration isolation, and federation contract/receipt identity. It does not execute repository code, fetch source URIs, verify signatures or replace full JSON Schema validation. Fifty unit-test cases are authored but uncompiled/unexecuted; execution remains subject to #10064.

## trace-schema-v1.json

JSON schema for trace file format. Used by symthaea-inspect and other analysis tools.
