# Tools

**Execution status (2026-11-05):** The `symthaea-inspect` Cargo package and its source tree are not in the current checkout. The `cargo build -p symthaea-inspect` command below is an intended interface only, not a runnable command or evidence of an implemented tool.

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

## trace-schema-v1.json

JSON schema for trace file format. Used by symthaea-inspect and other analysis tools.
