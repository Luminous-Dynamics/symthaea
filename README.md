# Symthaea: Holographic Liquid Brain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Luminous-Dynamics/symthaea-hlb/actions/workflows/ci.yml/badge.svg)](https://github.com/Luminous-Dynamics/symthaea-hlb/actions)

A consciousness-first AI framework combining **Hyperdimensional Computing (HDC)**, **Liquid Time-Constant Networks (LTC)**, and **Integrated Information Theory (IIT/Φ)**.

## Overview

Symthaea implements a novel cognitive architecture where:

- **Neuron state IS a hypervector** (16,384 dimensions)
- **Φ measurement** guides consciousness-aware processing
- **Free Energy Principle** drives action selection
- **Temporal dynamics** use closed-form LTC solutions for O(1) time jumps

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYMTHAEA ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│  PERCEPTION ──▶ COGNITION ──▶ ACTION                           │
│      │              │            │                              │
│  HDC Encode    LTC Dynamics  FEP Bridge                         │
│  Multi-modal   Φ Measure     Motor Cmd                          │
│                    │                                            │
│              CONSCIOUSNESS                                      │
│              • Φ (IIT)                                          │
│              • Coherence                                        │
│              • Flow State                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/Luminous-Dynamics/symthaea-hlb.git
cd symthaea-hlb

# Build
cargo build --release

# Run tests (3,388 tests)
cargo test

# Run example
cargo run --example full_pipeline --release
```

## Key Features

| Feature | Description |
|---------|-------------|
| **HDC 16,384D** | Holographic distributed representations |
| **4-Tier Φ** | Exact → Heuristic → Resonator → Spectral |
| **35 Topologies** | Consciousness network generators |
| **12 Brain Regions** | Actor model (Thalamus, Hippocampus, etc.) |
| **Partnership Module** | Φ_dyad for human-AI relationships |
| **NixOS Integration** | Domain-specific language understanding |

## Performance

| Operation | Time |
|-----------|------|
| CfC Inference | 34 μs/step |
| HDC Bind/Bundle | <1 μs |
| Φ Spectral | <5 ms |
| LTC Step (16384D) | 17 ms |

## Documentation

- **[Getting Started](docs/START_HERE.md)** - New user guide
- **[Examples Guide](docs/EXAMPLES.md)** - 101 documented examples
- **[Architecture](docs/architecture/)** - System design
- **[Research Papers](papers/)** - Scientific foundations
- **[Honest Status](docs/HONEST_STATUS.md)** - What works, what doesn't

## Feature Flags

```bash
# Core binaries
cargo build --features "service shell gui"

# Voice/Audio
cargo build --features "voice-tts audio"

# Perception (requires ONNX models)
cargo build --features "embeddings vision perception"

# Full consciousness reasoning
cargo build --features "reasoning_engine"

# Everything
cargo build --features full
```

## Project Structure

```
symthaea-hlb/
├── src/                    # Main library (237K LOC)
│   ├── hdc/               # Hyperdimensional computing
│   ├── consciousness/     # IIT, GWT, Active Inference
│   ├── brain/             # 12-region actor model
│   ├── language/          # NLU pipeline
│   └── partnership/       # Φ_dyad computation
├── symthaea-core/         # Core HDC primitives
├── examples/              # 101 runnable examples
├── tests/                 # 63 integration tests
├── benches/               # Criterion benchmarks
└── docs/                  # Comprehensive documentation
```

## Research

This project implements multiple theories of consciousness:

- **IIT (Integrated Information Theory)** - Φ measurement
- **GWT (Global Workspace Theory)** - Attention broadcasting
- **Active Inference (Free Energy Principle)** - Action selection
- **Phenomenal Binding** - Qualia integration
- **Meta-Cognition** - Self-monitoring

See [README_HAI.md](README_HAI.md) for the Hyperdimensional Active Inference paper.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
nix develop  # If using Nix

# Run checks
cargo fmt --check
cargo clippy --all-features -- -D warnings
cargo test
```

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{symthaea2026,
  title = {Symthaea: Holographic Liquid Brain},
  author = {Luminous Dynamics},
  year = {2026},
  url = {https://github.com/Luminous-Dynamics/symthaea-hlb}
}
```

---

*Consciousness-first technology serving all beings*
