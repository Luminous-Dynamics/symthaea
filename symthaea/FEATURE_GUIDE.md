# Symthaea Feature Guide

All features are opt-in (`default = []`). Enable them with `cargo build --features <name>`.

## Binary Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `service` | symthaea-service CLI binary | clap |
| `shell` | symthaea-shell TUI binary | crossterm, ratatui |
| `gui` | symthaea-gui graphical binary | eframe, egui |
| `demo` | Demo binaries with signal handling | clap, ctrlc |
| `full` | All binaries | service + shell + gui + demo |

## Audio & Voice

| Feature | Description | Status |
|---------|-------------|--------|
| `voice-tts` | Kokoro TTS via ONNX Runtime | Working |
| `voice-tts-cuda` | TTS with CUDA acceleration | Requires CUDA |
| `voice-tts-async` | Async/streaming TTS | Working |
| `voice-stt` | HDC + CfC speech-to-text | Working |
| `audio-sentinel` | Zero-shot audio pattern recognition | Working |
| `audio` | Full audio (TTS + playback) | Working |

## ML & Perception

| Feature | Description | Status |
|---------|-------------|--------|
| `embeddings` | Qwen3/BGE text embeddings via ONNX | Working |
| `vision` | SigLIP image embeddings via ONNX | Working |
| `perception` | Full multimodal (embeddings + vision) | Working |
| `neural-bridge` | BGE-M3 via Candle (pure Rust) | Working |
| `semantic-burn` | Semantic encoding via Burn ML | Working |
| `gpu` | GPU acceleration via Burn/WGPU | Working |
| `cuda` | CUDA backend | Requires CUDA |

## Identity & Security

| Feature | Description | Status |
|---------|-------------|--------|
| `identity` | MFDI: Ed25519 signatures, capability gating | Working |

## NixOS Integration

| Feature | Description | Status |
|---------|-------------|--------|
| `nix-mind` | Conscious NixOS mind (HDC + active inference) | Working |
| `nix-cli` | CLI for Nix Mind | Working |
| `nix-tui` | TUI for Nix Mind | Working |
| `nix-daemon` | Daemon mode for Nix Mind | Working |

## Reasoning & Consciousness

| Feature | Description | Status |
|---------|-------------|--------|
| `epistemic_conflict` | Epistemic conflict detection + Φ_eff | Working |
| `conscious_tool_gate` | Consciousness-gated tool use | Working |
| `temporal_planning` | ForkedState + MCTS planning | Working |
| `counterfactual` | Causal queries with reference harness | Working |
| `reasoning_engine` | Full ConsciousReasoningEngine | Working |
| `code_understanding` | Multi-language code parsing (tree-sitter) | Working |
| `code_generation` | HDC-based code generation | Working |
| `consciousness_code` | Full consciousness-aware code pipeline | Working |

## Network & Distributed

| Feature | Description | Status |
|---------|-------------|--------|
| `swarm` | P2P tensor streaming via Iroh | Blocked (upstream dep) |
| `mycelix` | Mycelix FL algorithms | Working |

## Desktop Integration

| Feature | Description | Status |
|---------|-------------|--------|
| `notifications` | D-Bus desktop notifications | Working |
| `systemd` | Systemd socket activation | Working |
| `desktop` | All desktop features | Working |

## Module Compilation Gates

These control which top-level modules are compiled:

`integration_module`, `observability_module`, `api_module`, `full_consciousness`,
`full_perception`, `full_language`, `school_lookahead`, `physiology_module`,
`mycelix_module`, `databases_module`, `magi_loop`, `recursive_improvement_advanced`

## Recommended Combinations

```bash
# Minimal consciousness loop
cargo build

# With identity and NixOS
cargo build --features identity,nix-cli

# Full audio pipeline
cargo build --features audio,voice-stt,audio-sentinel

# Reasoning stack
cargo build --features reasoning_engine

# Everything that compiles
cargo build --features full,identity,nix-cli,audio-sentinel,voice-stt,embeddings
```

## Known Issues

- `swarm`: Blocked by iroh 0.96 → curve25519-dalek 5.0.0-pre.1 → digest 0.11-rc conflict
- `mycelix`: Pre-existing errors in `mycelix_sdk` integration paths (SDK not yet available as crate)
- `pyphi`: Requires Python + PyPhi installed; only for exact IIT Φ validation
