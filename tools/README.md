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

## neurotwin-cohort-audit

Privacy-minimizing, local BIDS metadata preflight for NeuroTwin research intake.

```bash
cargo test --manifest-path tools/neurotwin-cohort-audit/Cargo.toml
cargo run --release --manifest-path tools/neurotwin-cohort-audit/Cargo.toml -- /secure/path/to/bids-dataset
```

The receipt contains aggregate subject/session/modality counts and finding codes only. It does not open scan contents or emit subject labels, filenames, paths or participant-table values. It is not a substitute for the official BIDS Validator, data-use/ethics review, or clinical qualification. See `tools/neurotwin-cohort-audit/README.md` and `docs/neuroscience/NEUROTWIN_DATASET_INTAKE_PROTOCOL_V0_1.md`.
