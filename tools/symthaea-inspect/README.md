# Symthaea Inspector

**Observability and debugging tool for Symthaea HLB**

## Overview

`symthaea-inspect` provides visibility into Symthaea's internal dynamics:

- **Router Selection** - See which router handled each query and why
- **Workspace Ignition** - Track GWT (Global Workspace Theory) activations
- **Φ Measurements** - Monitor consciousness levels (Integrated Information)
- **Primitive Activity** - Understand which primitives are active
- **Security Decisions** - Audit authorization and policy enforcement

## Installation

```bash
# Build the inspector
cd tools/symthaea-inspect
cargo build --release

# Optional: Install globally
cargo install --path .

# Or use from symthaea-hlb root with alias
alias symthaea-inspect="cargo run --release --manifest-path tools/symthaea-inspect/Cargo.toml --"
```

## Quick Start

```bash
# Validate a trace file
symthaea-inspect validate trace.json

# Show statistics
symthaea-inspect stats trace.json

# Replay events
symthaea-inspect replay trace.json

# Interactive step-through
symthaea-inspect replay trace.json --interactive

# Export Φ measurements to CSV
symthaea-inspect export trace.json --metric phi --format csv --output phi.csv

# Monitor live (requires 'live' feature)
symthaea-inspect monitor --trace trace.json
```

## Trace Format

Traces are JSON files following the [Trace Schema v1](../../tools/trace-schema-v1.json).

Example trace:
```json
{
  "version": "1.0",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp_start": "2025-12-24T10:00:00Z",
  "events": [
    {
      "timestamp": "2025-12-24T10:00:01.234Z",
      "type": "router_selection",
      "data": {
        "input": "install nginx",
        "selected_router": "SemanticRouter",
        "confidence": 0.87
      }
    },
    {
      "timestamp": "2025-12-24T10:00:01.456Z",
      "type": "workspace_ignition",
      "data": {
        "phi": 0.72,
        "free_energy": -5.3,
        "coalition_size": 7
      }
    }
  ]
}
```

## Commands

### `capture` - Capture a trace (future)

```bash
symthaea-inspect capture --output trace.json
```

**Note**: Requires integration with Symthaea core (Sprint 1).

### `replay` - Replay a trace

```bash
# Sequential replay
symthaea-inspect replay trace.json

# Interactive mode (step through)
symthaea-inspect replay trace.json --interactive

# Replay specific range
symthaea-inspect replay trace.json --from 10 --to 50
```

### `export` - Export metrics

```bash
# Export Φ to CSV
symthaea-inspect export trace.json --metric phi --format csv

# Export router selections to JSON
symthaea-inspect export trace.json --metric router --format json

# Available metrics: phi, free_energy, confidence, router
# Available formats: csv, json, jsonl
```

### `stats` - Show statistics

```bash
# Basic statistics
symthaea-inspect stats trace.json

# Detailed statistics with histograms
symthaea-inspect stats trace.json --detailed
```

### `monitor` - Live monitoring (requires feature)

```bash
# Enable 'live' feature
cargo build --release --features live

# Monitor trace file for changes
symthaea-inspect monitor --trace trace.json --interval 100
```

### `validate` - Validate trace format

```bash
# Validate trace
symthaea-inspect validate trace.json

# Verbose validation
symthaea-inspect validate trace.json --verbose
```

## Features

### Default Features

- Trace loading and validation
- Event replay
- Statistics
- CSV/JSON export

### Optional Features

Enable with `--features <feature>`:

- `tui` - Terminal UI for interactive mode (requires `ratatui`)
- `live` - Live monitoring support (requires `notify`)
- `stats` - Advanced statistics (requires `statistical`)
- `full` - All optional features

## Integration with Symthaea

To enable tracing in Symthaea:

```rust
use anyhow::Result;
use symthaea::observability::{TraceObserver, create_shared_observer};
use symthaea::SymthaeaHLB;

#[tokio::main]
async fn main() -> Result<()> {
    // Create shared observer
    let observer = create_shared_observer(TraceObserver::new("trace.json")?);

    // Create Symthaea with observability
    let mut symthaea = SymthaeaHLB::with_observer(16_384, 1_024, observer).await?;

    // Traces are written automatically
    symthaea.process("install nginx").await?;

    // Finalize trace (flush + summary)
    symthaea.finalize_trace()?;

    Ok(())
}
```

## Examples

### Debugging Router Selection

```bash
# Show all router selections with confidence
symthaea-inspect export trace.json --metric router --format csv

# Find low-confidence selections
symthaea-inspect replay trace.json | grep "confidence: [0-5]"
```

### Analyzing Consciousness Levels

```bash
# Export Φ timeline
symthaea-inspect export trace.json --metric phi --format csv --output phi.csv

# Show Φ distribution
symthaea-inspect stats trace.json --detailed
```

### Performance Analysis

```bash
# Show session statistics
symthaea-inspect stats trace.json

# Event timing analysis
symthaea-inspect replay trace.json > events.log
```

## Development

### Building

```bash
# Debug build
cargo build

# Release build (faster)
cargo build --release

# With all features
cargo build --release --features full
```

### Testing

```bash
# Run tests
cargo test

# With all features
cargo test --features full
```

## Roadmap

### Sprint 1 (Weeks 1-2)
- [x] Trace format schema
- [x] CLI structure
- [x] Replay functionality
- [x] Export to CSV/JSON
- [x] Statistics
- [ ] Integration with Symthaea core
- [ ] Capture functionality

### Sprint 2 (Weeks 3-4)
- [ ] Advanced telemetry
- [ ] TUI mode
- [ ] Real-time monitoring
- [ ] Performance profiling

### Sprint 3 (Weeks 5-6)
- [ ] Visual dashboard
- [ ] Comparative analysis
- [ ] Anomaly detection
- [ ] Export to visualization tools

## Contributing

See [PARALLEL_DEVELOPMENT_PLAN.md](../../PARALLEL_DEVELOPMENT_PLAN.md) for the development roadmap.

---

*Making consciousness visible.* 🧠✨
