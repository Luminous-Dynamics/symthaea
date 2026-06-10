# Symthaea Quickstart

## Prerequisites

- Rust 1.75+ (stable)
- NixOS recommended: `nix develop` from the workspace root

## Build

```bash
cd symthaea

# Default build (no optional features)
cargo build

# Run all library tests
cargo test --workspace --lib

# Build with specific features (see FEATURE_GUIDE.md)
cargo build --features identity,nix-mind
```

## Workspace Crates

| Crate | Purpose |
|-------|---------|
| `symthaea` | Main binary + consciousness orchestration |
| `symthaea-core` | HDC primitives, Phi engine, physics encodings |
| `symthaea-nix` | Conscious NixOS management via active inference |
| `symthaea-stt` | Speech-to-text (HDC + LTC + CfC acoustic models) |
| `symthaea-sentinel` | Zero-shot audio pattern recognition |
| `symthaea-dynamics` | Dynamical systems + reaction-diffusion |

## Quick Test

```bash
# Fast: lib tests only (~9 minutes)
cargo test --workspace --lib

# Full: all test targets (~13 minutes)
cargo test --workspace
```

## Key Concepts

- **HDC**: Hyperdimensional computing with 16,384-bit binary vectors
- **CfC**: Closed-form Continuous-time neural networks (O(1) inference)
- **Phi (Φ)**: Integrated Information Theory consciousness measure
- **Active Inference**: Free Energy minimization for decision-making

## Architecture

```
Input → Encoding (HDC) → Temporal Processing (CfC/LTC)
  → Consciousness (Phi + GWT) → Action Selection (Active Inference)
  → Language Translation (LLM) → Output
```

## Common Workflows

```bash
# Check a specific feature compiles
cargo check --lib --features identity

# Run a single test
cargo test -p symthaea-core -- hdc::tests::test_binding

# Clippy for the whole workspace
cargo clippy --workspace --no-deps --lib -- -D warnings

# Format check
cargo fmt --all --check
```
