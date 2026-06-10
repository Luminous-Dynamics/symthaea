# Model Inference Integration

## Overview

Symthaea supports neural embedding models for semantic understanding:

| Model | Type | Dimension | Purpose |
|-------|------|-----------|---------|
| **Qwen3-Embedding-0.6B** | Text | 1024D | Semantic grounding |
| **SigLIP-SO400M** | Vision | 768D | Visual understanding |
| **BGE-Base-EN** | Text | 768D | Lightweight alternative |

## Architecture

```
Input (Text/Image)
        │
        ▼
┌─────────────────┐
│  Model Hub      │ ← Auto-download from HuggingFace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ONNX Runtime   │ ← SigLIP/Qwen3 inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JL Projection  │ ← Dense → Sparse HDC (16,384D)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HDC Space      │ ← Holographic consciousness
└─────────────────┘
```

## Text Embedding (Qwen3)

### Configuration

```rust
use symthaea::embeddings::{Qwen3Embedder, Qwen3Config};

let config = Qwen3Config {
    model_path: Some("models/qwen3-embedding".into()),
    max_length: 8192,
    ..Default::default()
};

let embedder = Qwen3Embedder::new(config)?;
```

### Usage

```rust
let result = embedder.embed("Hello, consciousness!")?;
println!("Embedding dim: {}", result.embedding.len()); // 1024
println!("Truncated: {}", result.truncated);
println!("Time: {}ms", result.time_ms);
```

### Model Download

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B-ONNX
```

## Vision Embedding (SigLIP)

### Configuration

```rust
use symthaea::perception::{SemanticVision, SigLipModel};

let mut vision = SemanticVision::new(1000); // 1000 cache entries
vision.initialize()?;
```

### Usage

```rust
let image = image::open("input.jpg")?;
let embedding = vision.embed_image(&image)?;
println!("Embedding dim: {}", embedding.vector.len()); // 768
```

### Model Download

```bash
optimum-cli export onnx \
  --model google/siglip-so400m-patch14-384 \
  models/siglip-so400m/
```

## Feature Flags

Enable model inference via Cargo features:

```toml
[dependencies]
symthaea = { features = ["embeddings", "vision"] }
```

| Feature | Description |
|---------|-------------|
| `embeddings` | Qwen3/BGE text embeddings |
| `vision` | SigLIP image embeddings |
| `perception` | Both embeddings + vision |

## HDC Projection

Neural embeddings are projected to HDC space using Johnson-Lindenstrauss:

```rust
use symthaea::perception::MultiModalIntegrator;

let integrator = MultiModalIntegrator::new();

// Text → HDC (1024D → 16,384D)
let text_hdc = integrator.project_text_embedding(&embedding)?;

// Image → HDC (768D → 16,384D)  
let image_hdc = integrator.project_image_embedding(&img_embedding)?;
```

## Performance

| Operation | Latency (GPU) | Latency (CPU) |
|-----------|---------------|---------------|
| Qwen3 embed (512 tokens) | ~15ms | ~80ms |
| SigLIP embed (384x384) | ~20ms | ~150ms |
| JL projection | ~1ms | ~5ms |

## Caching

Embedding cache for repeated inputs:

```rust
let vision = SemanticVision::new(1000);

// First call: computes embedding
let emb1 = vision.embed_image(&image)?;

// Second call: returns cached
let emb2 = vision.embed_image(&image)?;

// Check cache stats
let stats = vision.cache_stats();
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
```

## Integration with Consciousness

Embeddings flow into the consciousness pipeline:

```rust
use symthaea::perception::PerceptionBridge;

let mut bridge = PerceptionBridge::default();
bridge.initialize()?;

// Text → Attention Bid
let bid = bridge.process_text("I feel happy")?;
println!("Salience: {}", bid.salience);
println!("Urgency: {}", bid.urgency);

// Image → Attention Bid
let img_bid = bridge.process_image(&image)?;
```

## Troubleshooting

### ONNX Runtime Errors

```bash
# Check ONNX version
pip show onnxruntime

# For GPU support
pip install onnxruntime-gpu
```

### Model Loading Failures

```bash
# Verify model files
ls models/qwen3-embedding/
# Should contain: model.onnx, tokenizer.json, config.json
```

### Out of Memory

Reduce batch size or use smaller model:

```rust
let config = Qwen3Config {
    max_batch_size: 1,  // Process one at a time
    ..Default::default()
};
```
