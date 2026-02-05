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

## trace-schema-v1.json

JSON schema for trace file format. Used by symthaea-inspect and other analysis tools.
