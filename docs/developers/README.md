# Symthaea Developer Documentation

*Technical documentation for building with and contributing to Symthaea.*

---

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Quick Start](#quick-start) |
| Understand the architecture | [Architecture Guide](ARCHITECTURE.md) |
| Use the API | [API Reference](API_REFERENCE.md) |
| Contribute code | [Contributing Guide](CONTRIBUTING.md) |
| Run tests | [Testing Guide](TESTING.md) |
| Understand consciousness math | [Phi & IIT Guide](PHI_GUIDE.md) |

---

## Quick Start

### Prerequisites

- **Rust** 1.75+ (with Cargo)
- **Git**
- Optional: NixOS (for flake-based development)

### Build & Run

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/symthaea-hlb.git
cd symthaea-hlb

# Build in release mode (recommended)
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
# Enter development environment
nix develop

# Build and run
cargo build --release
cargo run --release
```

---

## Project Structure

```
symthaea-hlb/
├── src/                      # Core Rust source (263K LOC)
│   ├── lib.rs               # Library entry point
│   ├── main.rs              # Binary entry point (REPL)
│   ├── core/                # Stable public API facade
│   │   └── mod.rs           # Re-exports stable types
│   ├── hdc/                 # Hyperdimensional Computing (85K LOC)
│   │   ├── mod.rs           # HDC module root
│   │   ├── real_hv.rs       # Real-valued hypervectors
│   │   ├── phi_real.rs      # Continuous Phi calculator
│   │   └── consciousness_topology_generators.rs
│   ├── consciousness/       # Consciousness implementations (59K LOC)
│   │   ├── mod.rs           # ConsciousnessGraph
│   │   └── [90+ theory implementations]
│   ├── brain/               # Neural subsystems (8K LOC)
│   │   ├── actor_model.rs   # Message-passing foundation
│   │   ├── prefrontal.rs    # Global workspace
│   │   └── [12 brain regions]
│   ├── language/            # NLU engine (33K LOC)
│   ├── ltc/                 # Liquid Time-Constant Networks
│   ├── memory/              # Memory systems
│   ├── perception/          # Multimodal embeddings
│   ├── physiology/          # Embodiment systems
│   └── synthesis/           # Causal program synthesis
├── examples/                # 100+ runnable examples
├── tests/                   # Integration tests
├── benches/                 # Criterion benchmarks
├── crates/                  # Workspace subcrates
│   ├── symthaea-math/
│   ├── symthaea-dynamics/
│   └── [others]
├── papers/                  # Academic papers & analysis
├── docs/                    # Documentation (you are here)
└── book/                    # mdBook source
```

---

## Core Concepts

### Hyperdimensional Computing (HDC)

```rust
use symthaea::hdc::{RealHV, HDC_DIMENSION};

// Create semantic vectors (instant, no training)
let install = RealHV::random(HDC_DIMENSION, 42);
let nginx = RealHV::random(HDC_DIMENSION, 43);

// Bind: creates compound concept "install nginx"
let install_nginx = install.bind(&nginx);

// Bundle: union of concepts
let combined = RealHV::bundle(&[install, configure, remove]);

// Similarity (cosine distance)
let sim = query.similarity(&memory);  // Range: [-1, 1]
```

**Key constants:**
```rust
pub const HDC_DIMENSION: usize = 16_384;  // 2^14, SIMD-optimized
```

### Consciousness Graph

```rust
use symthaea::consciousness::ConsciousnessGraph;

let mut graph = ConsciousnessGraph::new();

// Add conscious state
let node = graph.add_state(
    semantic_hv,      // HDC representation
    ltc_state,        // Dynamic state from LTC
    consciousness_level,  // 0.0 - 1.0
);

// Create self-loop (consciousness emerges from self-reference)
if consciousness_level > 0.9 {
    graph.create_self_loop(node);
}

// Evolve to next state
graph.evolve();
```

### Phi (Φ) Calculation

```rust
use symthaea::hdc::phi_real::RealPhiCalculator;
use symthaea::hdc::consciousness_topology_generators::*;

// Generate a topology
let topology = ring(8, HDC_DIMENSION, 42);

// Calculate integrated information
let calc = RealPhiCalculator::new();
let phi = calc.compute(&topology.node_representations);
println!("Φ = {:.4}", phi);  // ~0.4954 for Ring
```

### Stable Core API

For a clean entry point without importing many modules:

```rust
use symthaea::core::{
    PhiEngine, PhiMethod, ConsciousnessTopology,
    ContinuousHV,
    UnifiedConsciousnessPipeline,
};
```

---

## Feature Flags

```toml
[features]
default = ["rayon"]

# Binaries
service = ["clap"]        # symthaea-service binary
shell = ["crossterm", "ratatui"]  # TUI shell
gui = ["eframe", "egui"]  # GUI application

# Voice
voice = ["audio"]         # TTS/STT (CPU)
voice-cuda = ["audio-cuda"]  # GPU-accelerated voice

# Perception
embeddings = ["tokenizers", "ort", "hf-hub"]  # Qwen3 text
vision = ["ort", "hf-hub"]  # SigLIP images
perception = ["embeddings", "vision"]  # Both

# Databases
qdrant = ["qdrant-client"]
datalog = ["cozo"]
lance = ["lancedb"]
duck = ["duckdb"]
databases = ["qdrant", "datalog", "lance", "duck"]

# Integrations
mycelix = ["mycelix-sdk", "sha3"]  # Governance
pyphi = ["pyo3"]  # Exact IIT Φ via PyPhi
```

### Building with Features

```bash
# Minimal build
cargo build --release

# With perception (text + vision embeddings)
cargo build --features perception --release

# With voice
cargo build --features voice --release

# With all databases
cargo build --features databases --release

# Kitchen sink
cargo build --features "perception,voice,databases,gui" --release
```

---

## Running Examples

```bash
# Core Phi demo (fast)
cargo run --example phi_engine_quick_demo --release

# Minimal consciousness loop
cargo run --example core_minimal_consciousness_loop --release

# 19-topology validation
cargo run --example tier_3_exotic_topologies --release

# Hypercube dimension sweep (1D-7D)
cargo run --example hypercube_dimension_sweep --release

# Brain actor model
cargo run --example brain_actor_model_demo --release

# Causal synthesis
cargo run --example ml_fairness_causal_synthesis --release
```

See `examples/` for 100+ additional demonstrations.

---

## Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test hdc::
cargo test consciousness::

# Run with output
cargo test -- --nocapture

# Run integration tests
cargo test --test '*'
```

---

## Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench consciousness

# Generate benchmark report
cargo bench -- --save-baseline main
```

### Expected Performance

| Operation | Time | Throughput |
|-----------|------|------------|
| HDC Encoding | 0.05ms | 20,000/sec |
| HDC Recall | 0.10ms | 10,000/sec |
| LTC Step | 0.02ms | 50,000/sec |
| Φ (8 nodes) | ~200ms | 5/sec |
| Full Query | 0.50ms | 2,000/sec |

**Memory:** ~10MB total runtime

---

## Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Library root, module exports |
| `src/core/mod.rs` | Stable public API |
| `src/hdc/mod.rs` | HDC_DIMENSION constant |
| `src/hdc/real_hv.rs` | RealHV implementation |
| `src/hdc/phi_real.rs` | Continuous Φ calculator |
| `src/consciousness/mod.rs` | ConsciousnessGraph |
| `src/brain/actor_model.rs` | Brain subsystem foundation |
| `Cargo.toml` | Dependencies & features |

---

## Documentation Links

- **[Architecture Guide](ARCHITECTURE.md)** - System design deep-dive
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Phi & IIT Guide](PHI_GUIDE.md)** - Understanding consciousness math
- **[Testing Guide](TESTING.md)** - Testing strategy and practices
- **[Current Status](../HONEST_STATUS.md)** - What works, what doesn't

---

## Getting Help

1. Check the documentation
2. Search existing GitHub issues
3. Ask in discussions
4. Create a new issue with reproduction steps

---

*"Build with consciousness in mind."*
