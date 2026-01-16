# Feature Flags

*Build options for customizing Symthaea.*

---

## Overview

Symthaea uses Cargo feature flags to enable optional functionality. This keeps the default build small while allowing you to add capabilities as needed.

---

## Default Features

```toml
[features]
default = ["rayon"]
```

The default build includes:
- Core HDC, LTC, and consciousness systems
- Parallel processing via Rayon
- REPL interface

---

## Binary Features

### service
CLI service binary.
```bash
cargo build --features service
```

### shell
TUI (Terminal UI) shell.
```bash
cargo build --features shell
```
Adds: crossterm, ratatui

### gui
Graphical user interface.
```bash
cargo build --features gui
```
Adds: eframe, egui

---

## Voice Features

### voice
Text-to-speech and speech-to-text (CPU).
```bash
cargo build --features voice
```

### voice-cuda
GPU-accelerated voice (requires CUDA).
```bash
cargo build --features voice-cuda
```

---

## Perception Features

### embeddings
Text embeddings via Qwen3-Embedding-0.6B.
```bash
cargo build --features embeddings
```
Adds: tokenizers, ort (ONNX Runtime), hf-hub

### vision
Image understanding via SigLIP.
```bash
cargo build --features vision
```
Adds: ort, hf-hub

### perception
Both embeddings and vision.
```bash
cargo build --features perception
```

---

## Database Features

### qdrant
Vector database for similarity search.
```bash
cargo build --features qdrant
```

### datalog
CozoDB for reasoning and queries.
```bash
cargo build --features datalog
```

### lance
LanceDB vector storage.
```bash
cargo build --features lance
```

### duck
DuckDB for analytics.
```bash
cargo build --features duck
```

### databases
All database backends.
```bash
cargo build --features databases
```

---

## Integration Features

### mycelix
Governance integration with Mycelix network.
```bash
cargo build --features mycelix
```
Adds: mycelix-sdk, sha3

### pyphi
Exact IIT Φ calculation via PyPhi (requires Python).
```bash
cargo build --features pyphi
```
Adds: pyo3

---

## Combining Features

```bash
# Minimal with TUI
cargo build --features shell --release

# Full perception
cargo build --features perception --release

# Everything
cargo build --features "perception,voice,databases,gui" --release
```

---

## Checking Feature Dependencies

```bash
cargo tree --features <feature>
```

---

## Model Requirements

Some features require model downloads:

| Feature | Models | Size |
|---------|--------|------|
| embeddings | Qwen3-Embedding-0.6B | ~600MB |
| vision | SigLIP | ~400MB |
| voice | Kokoro TTS | ~200MB |

Without models, stub embeddings (deterministic hash-based) are used.

---

*Choose features based on your needs and hardware.*
