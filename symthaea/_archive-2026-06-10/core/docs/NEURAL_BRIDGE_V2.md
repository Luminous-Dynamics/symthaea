# Neural Bridge v2: BGE-M3 Semantic Encoding

Neural Bridge v2 provides high-quality semantic encoding for Symthaea's consciousness pipeline using BGE-M3, a state-of-the-art multilingual embedding model.

## Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Text      │     │   BGE-M3     │     │   Linear     │     │    HDC       │
│  "Justice"   │────▶│  Encoder     │────▶│   Probe      │────▶│   Vector     │
│              │     │  [1024-dim]  │     │  [16384-dim] │     │  PackedBip.  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Key Features

- **Pure Rust**: No Python/Ollama dependency, self-contained binary
- **High Quality**: BGE-M3 provides semantic understanding vs hash-based encoding
- **Multilingual**: 100+ languages supported out of the box
- **Fast**: ~380ms CPU, <2µs cached, ~60-100ms GPU (estimated)
- **Cached**: Two-tier caching with 269,000x speedup on repeated queries

## Quick Start

### Build with Neural Bridge

```bash
# CPU-only build
cargo build --release --features neural-bridge

# CUDA build (requires CUDA toolkit)
CUDA_COMPUTE_CAP=75 cargo build --release --features neural-bridge-cuda
```

### Usage

```rust
use symthaea::perception::NeuralBridgeV2;

// Load model (downloads BGE-M3 on first run, ~2.2GB)
let mut bridge = NeuralBridgeV2::load_default()?;

// Encode text to HDC vector
let hdc = bridge.encode_to_hdc("What is consciousness?")?;

// Compare semantic similarity
let justice = bridge.encode_to_hdc("justice")?;
let fairness = bridge.encode_to_hdc("fairness")?;
let similarity = justice.xor_similarity(&fairness);
println!("Similarity: {:.3}", similarity); // ~0.68

// Check cache statistics
let stats = bridge.stats();
println!("Cache hit rate: {:.1}%", stats.cache_hit_rate() * 100.0);
```

### With Symthaea

```rust
use symthaea::Symthaea;

// Neural Bridge auto-initializes if feature is enabled
let mut symthaea = Symthaea::new(512, 64).await?;

// Check if neural bridge is active
if symthaea.has_neural_bridge() {
    println!("Using BGE-M3 semantic encoding");
}

// Process normally - neural bridge used automatically
let response = symthaea.process("What is justice?").await?;
```

## Performance

### Latency Comparison

| Mode | Encoding Time | Notes |
|------|---------------|-------|
| Hash-based (fallback) | <1ms | Low quality, no semantics |
| Neural Bridge (cached) | **<2µs** | 269,000x speedup |
| Neural Bridge (CPU) | ~380ms | Release build |
| Neural Bridge (CUDA) | ~60-100ms | Estimated, requires GPU |
| Ollama/Python (old) | 5-18s | Replaced by v2 |

### Cache Architecture

Two-tier caching minimizes redundant computation:

1. **HDC Cache**: Stores final `PackedBipolar` vectors
   - Hit: Returns in <2µs
   - Saves full pipeline (~380ms)

2. **Embedding Cache**: Stores intermediate 1024-dim embeddings
   - Hit: Only runs probe (~30ms)
   - Useful when same text encoded with different probes

### Memory Usage

| Component | Size |
|-----------|------|
| BGE-M3 model | ~2.2GB (downloaded once) |
| Probe weights | ~67MB |
| HDC cache (1000 entries) | ~32MB |
| Embedding cache (1000 entries) | ~4MB |

## Configuration

### Builder Pattern

```rust
use symthaea::perception::NeuralBridgeV2Builder;

let bridge = NeuralBridgeV2Builder::new()
    .model_id("BAAI/bge-m3")           // HuggingFace model
    .probe_path("custom/probe.npy")     // Custom probe weights
    .track_epistemic(true)              // Enable uncertainty tracking
    .enable_cache(true)                 // Enable caching
    .max_cache_size(2000)               // Cache capacity
    .build()?;
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CUDA_COMPUTE_CAP` | GPU compute capability | Auto-detect |
| `HF_HOME` | HuggingFace cache dir | `~/.cache/huggingface` |

## Epistemic Metadata

Neural Bridge v2 can track uncertainty in encodings:

```rust
let esv = bridge.encode_epistemic("Maybe consciousness emerges from complexity")?;

println!("Confidence: {:.3}", esv.confidence);
println!("Stability: {:?}", esv.stability);

for source in &esv.uncertainty_sources {
    println!("Uncertainty: {:?}", source);
}
```

Uncertainty sources include:
- **ProjectionMargin**: How close values are to decision boundary
- **EmbeddingNorm**: Deviation from expected L2 norm (potential OOD)

## Troubleshooting

### Model Download Fails

```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/hub/models--BAAI--bge-m3/

# Manual download
huggingface-cli download BAAI/bge-m3
```

### CUDA Errors

```
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
```

This indicates driver/library version mismatch. Solutions:
1. Reboot to sync kernel module with user-space libraries
2. Set `CUDA_COMPUTE_CAP` manually: `CUDA_COMPUTE_CAP=75`
3. Fall back to CPU (automatic if CUDA fails)

### Probe Weights Not Found

Ensure probe weights exist at `models/neural_bridge/probe_weights_bge_m3.npy`.

Generate with:
```bash
python scripts/train_probe.py --encoder bge-m3 --output models/neural_bridge/
```

## Architecture Details

### BGE-M3 Model

- **Architecture**: XLM-RoBERTa (560M parameters)
- **Embedding Dimension**: 1024
- **Languages**: 100+ (zero-shot multilingual)
- **Format**: SafeTensors (2.2GB)

### Linear Probe

- **Input**: 1024-dim BGE-M3 embedding
- **Output**: 16,384-dim continuous values
- **Binarization**: Threshold at 0 → PackedBipolar
- **Training**: Contrastive learning on concept pairs

### HDC Integration

The 16,384-dim PackedBipolar vectors integrate with Symthaea's:
- `ContinuousMind.perceive_text()` for perception
- XOR-based similarity for memory retrieval
- Consciousness pipeline for reasoning

## Files

```
symthaea-hlb/
├── src/perception/
│   ├── bge_m3.rs           # BGE-M3 encoder (pure Rust/Candle)
│   ├── neural_bridge.rs    # Linear probe projection
│   └── neural_bridge_v2.rs # Unified pipeline + caching
├── models/neural_bridge/
│   └── probe_weights_bge_m3.npy  # Trained probe (67MB)
└── tests/
    └── neural_bridge_integration.rs  # E2E tests
```

## Benchmarks

Run benchmarks:
```bash
cargo bench --features neural-bridge -- neural_bridge
```

Key metrics tracked:
- `neural_bridge_encode_uncached`: Raw encoding latency
- `neural_bridge_encode_cached`: Cache hit latency
- `neural_bridge_probe_only`: Probe projection time

---

*Neural Bridge v2 enables Symthaea to truly understand language semantics, not just process text.*
