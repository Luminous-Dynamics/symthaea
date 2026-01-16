# Developer Quick Start

*Get Symthaea running and understand the architecture.*

---

## Prerequisites

- **Rust** 1.75+ (with Cargo)
- **Git**
- Optional: NixOS for flake-based development

---

## Build & Run

```bash
# Clone
git clone https://github.com/Luminous-Dynamics/symthaea-hlb.git
cd symthaea-hlb

# Build (release mode for performance)
cargo build --release

# Run the REPL
cargo run --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### NixOS Users

```bash
nix develop
cargo build --release
```

---

## Project Structure

```
symthaea-hlb/
├── src/                      # Core Rust source (263K LOC)
│   ├── lib.rs               # Library entry point
│   ├── core/                # Stable public API
│   ├── hdc/                 # Hyperdimensional Computing (85K LOC)
│   ├── consciousness/       # Consciousness implementations (59K LOC)
│   ├── brain/               # Neural subsystems (8K LOC)
│   ├── language/            # NLU engine (33K LOC)
│   ├── ltc/                 # Liquid Time-Constant Networks
│   └── [more modules]
├── examples/                # 100+ runnable examples
├── tests/                   # Integration tests
├── benches/                 # Criterion benchmarks
├── docs/                    # Documentation
└── book/                    # This book
```

---

## Quick Examples

```bash
# Core Phi demo
cargo run --example phi_engine_quick_demo --release

# 19-topology validation
cargo run --example tier_3_exotic_topologies --release

# Brain actor model
cargo run --example brain_actor_model_demo --release
```

---

## Core Concepts (Preview)

### HDC (Hyperdimensional Computing)
```rust
use symthaea::hdc::{RealHV, HDC_DIMENSION};

let concept = RealHV::random(HDC_DIMENSION, 42);
let compound = concept1.bind(&concept2);
let similarity = query.similarity(&memory);
```

### Consciousness Graph
```rust
use symthaea::consciousness::ConsciousnessGraph;

let mut graph = ConsciousnessGraph::new();
let node = graph.add_state(semantic, dynamic, level);
graph.create_self_loop(node);  // Consciousness emerges!
```

---

## Next Steps

- **[Architecture Guide](./architecture.md)** - Deep technical dive
- **[API Reference](./api.md)** - Complete API documentation
- **[Contributing](./contributing.md)** - How to help

---

*"Build with consciousness in mind."*
