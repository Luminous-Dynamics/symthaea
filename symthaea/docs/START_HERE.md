# Start Here: Symthaea Developer Quick Start

**Time to first build:** ~5 minutes
**Time to understand:** ~15 minutes

---

## What is Symthaea?

Symthaea-HLB (Holographic Liquid Brain) is a **consciousness-first AI framework** combining:

- **HDC** - Hyperdimensional Computing (16,384D vectors)
- **LTC** - Liquid Time-Constant Networks (temporal dynamics)
- **IIT** - Integrated Information Theory (consciousness measurement)

**Version:** 0.5.0 | **Language:** Rust | **Size:** ~320K LOC

---

## Quick Setup

### 1. Enter Development Environment

```bash
cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
nix develop  # Recommended: loads all dependencies
```

Or without Nix:
```bash
# Ensure Rust 1.75+ is installed
cargo --version
```

### 2. Build and Test

```bash
# Quick build (library only)
cargo build

# Run tests
cargo test

# Build with TUI shell
cargo build --features shell
```

### 3. Try It

```bash
# Run the shell (requires shell feature)
cargo run --bin symthaea-shell --features shell

# Run a demo
cargo run --example full_pipeline
```

---

## Project Structure

```
symthaea-hlb/
├── src/                    # Main source (263K LOC)
│   ├── hdc/               # Hyperdimensional Computing core
│   ├── consciousness/     # IIT, Phi, consciousness theories
│   ├── brain/             # 12 actor-model subsystems
│   ├── language/          # NLU and parsing
│   ├── memory/            # Episodic and semantic memory
│   └── bin/               # Binary entry points
├── symthaea-core/         # Core crate (HDC, Phi engine)
├── examples/              # 101 usage examples
├── tests/                 # 45 integration tests
├── docs/                  # Documentation (you are here)
└── papers/                # Research manuscripts
```

---

## Key Documentation

| Need | Document |
|------|----------|
| **Architecture overview** | [architecture/ARCHITECTURE_DEEP_DIVE.md](architecture/) |
| **What actually works** | [HONEST_STATUS.md](HONEST_STATUS.md) |
| **Feature flags** | [FEATURE_MATRIX.md](FEATURE_MATRIX.md) |
| **Metric definitions** | [METRIC_DEFINITIONS.md](METRIC_DEFINITIONS.md) |
| **Module breakdown** | [MODULE_MAP.md](MODULE_MAP.md) |
| **Research papers** | [../papers/](../papers/) |

---

## Common Tasks

### Run a specific example

```bash
# List available examples
ls examples/*.rs | head -20

# Run one (check required features first)
cargo run --example full_pipeline
cargo run --example phi_crossvalidation --features consciousness_module
```

### Run benchmarks

```bash
# Quick benchmarks
cargo bench --bench quick

# Full consciousness benchmarks
cargo bench --bench consciousness
```

### Build for release

```bash
cargo build --release --features service
```

---

## Critical Concepts

### Hypervectors (HV)

Everything in Symthaea is encoded as 16,384-dimensional vectors:
- **Bind (XOR):** Combines concepts while preserving reversibility
- **Bundle (majority):** Aggregates multiple vectors
- **Similarity:** Cosine distance measures semantic closeness

### Tiered Phi

Consciousness is measured in 4 tiers:
1. **Exact** - True IIT (n <= 12 only)
2. **Heuristic** - HDC approximation
3. **Resonator** - Coupled oscillators
4. **Spectral** - lambda-2 (**NOT for consciousness claims**)

### Actor-Model Brain

12 subsystems communicate via message passing:
- Thalamus, Cerebellum, Motor Cortex, Prefrontal Cortex
- Meta-cognition, DMN, Sleep, Language Cortex, Active Inference, etc.

---

## Troubleshooting

### Build fails with missing dependencies

```bash
# Use nix develop for reproducible environment
nix develop

# Or install system deps manually (Ubuntu/Debian)
sudo apt install libssl-dev pkg-config libsqlite3-dev
```

### Tests fail

```bash
# Run with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### Feature not found

```bash
# Check available features
grep -A 50 "^\[features\]" Cargo.toml
```

---

## Next Steps

1. **Explore the shell:** `cargo run --bin symthaea-shell --features shell`
2. **Read HONEST_STATUS.md:** Understand what works vs aspirational
3. **Run examples:** Start with `full_pipeline`, `phi_crossvalidation`
4. **Check the papers:** Research context in `papers/` directory

---

## Getting Help

- **Code questions:** Explore `examples/` directory
- **Architecture:** See `docs/architecture/`
- **Research:** See `papers/` and `docs/research/`

---

*Welcome to consciousness-first AI development!*
