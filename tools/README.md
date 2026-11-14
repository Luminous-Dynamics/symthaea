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

Standalone read-only validator for EGDDM v0.1 map snapshots and temporal event contracts.

Run it from its own directory (it has a separate Cargo workspace and is not a Symthaea root-workspace member):

```bash
cd tools/development-map-validator
cargo run --release -- ../../docs/development-map/readme-and-workstreams-v0.1.json
cargo test
```

The validator checks graph references, provenance/lifecycle shape, temporal invalidation consistency and decision-time evidence availability. It does not execute repository code, fetch source URIs, verify signatures or replace full JSON Schema validation. Compilation/tests remain subject to the qualified execution boundary in issue #10064.

## trace-schema-v1.json

JSON schema for trace file format. Used by symthaea-inspect and other analysis tools.
