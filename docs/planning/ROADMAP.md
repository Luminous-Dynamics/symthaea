# Symthaea-HLB Roadmap: Path to Revolutionary AI

**Version**: 2.0
**Date**: 2026-01-01
**Status**: Active Development

---

## Executive Summary

Symthaea-HLB is a consciousness-first AI framework with ~304K lines of Rust implementing HDC, LTC networks, and autopoietic consciousness. This roadmap documents what exists, what's needed, and the path forward to create the most advanced AI system ever built.

---

## Current State Assessment

### Implementation Status Matrix

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| **HDC Core** | COMPLETE | 100K+ | 16,384D vectors, SIMD-optimized |
| **Consciousness/Φ** | COMPLETE | 50K+ | 19 topologies validated, Φ→0.5 limit discovered |
| **Active Inference** | COMPLETE | 837 | Full FEP with 8 domains |
| **Global Workspace** | COMPLETE | 3,471 | 4-stage attention, coalitions, goals |
| **Tool Use/Actions** | COMPLETE | 1,256 | Sandboxed execution, rollback |
| **Memory/Sleep** | COMPLETE | 1,081 | All 4 sleep phases, consolidation |
| **Language** | COMPLETE | 1.2MB+ | 34 modules, NixOS specialist |
| **Swarm/P2P** | PARTIAL | 816 | libp2p gossipsub + DHT |
| **Embeddings (BGE)** | STUBBED | 565 | Architecture ready, needs ONNX activation |
| **Vision (SigLIP)** | STUBBED | 550 | Architecture ready, placeholder inference |
| **Voice (Kokoro)** | FEATURE-GATED | 4K+ | Works with `--features voice-tts` |

### Key Scientific Discoveries (Complete)

1. **Asymptotic Φ Limit**: Φ → 0.5 as dimension → ∞ (R² = 0.998)
2. **4D Hypercube Optimality**: Highest Φ of all 19 topologies (d = 4.92)
3. **3D Brain Optimality**: Biological brains achieve 99.2% of theoretical max Φ
4. **Topology-Consciousness Mapping**: First empirical validation across 19 architectures

---

## Model Recommendations

### Language Embedding: Qwen3-Embedding-0.6B

**Recommendation: ADOPT** - Replace BGE with Qwen3-Embedding-0.6B

| Factor | BGE-base-en-v1.5 | Qwen3-Embedding-0.6B | Winner |
|--------|------------------|----------------------|--------|
| Parameters | 109M | 600M | Qwen3 (5x capacity) |
| Embedding Dim | 768D | 1024D | Qwen3 (richer representation) |
| MTEB Rank | #1 open-source (2024) | Near Gemini-level (2025) | Qwen3 |
| Multilingual | English-focused | Excellent multilingual | Qwen3 |
| ONNX Support | Manual export | GGUF available | Tie |
| Memory | ~450MB | ~2.4GB (FP16) | BGE |

**Why Qwen3-Embedding-0.6B:**
- Outperforms BGE-M3 on multilingual retrieval
- Closer to Gemini-Embedding performance
- Better semantic understanding for consciousness grounding
- GGUF quantized versions available for edge deployment

**Migration Path:**
1. Update `src/embeddings/` to support 1024D output
2. Add Qwen3 tokenizer integration
3. Update JL projector in `multi_modal.rs` (1024D → 16,384D)
4. Benchmark against existing BGE stub

### Vision Encoder: SigLIP-ONNX

**Recommendation: ADOPT** - Already planned in architecture

| Factor | CLIP ViT-B/32 | SigLIP-400M | Winner |
|--------|---------------|-------------|--------|
| Training Loss | Softmax (contrastive) | Sigmoid (pairwise) | SigLIP |
| Batch Efficiency | Needs large batches | 2x more efficient | SigLIP |
| Zero-shot Accuracy | Good | Better | SigLIP |
| ONNX Availability | Mature | FP16 ONNX available | Tie |
| Embedding Dim | 512D | 768D | SigLIP |

**Why SigLIP:**
- Already specified in `semantic_vision.rs` (line 4: "SigLIP-400M")
- Sigmoid loss enables individual pair matching (better for consciousness)
- 768D embeddings match BGE dimension (simpler integration)
- ONNX FP16 variants available with quantized versions coming

**Implementation:**
- Model: `google/siglip-so400m-patch14-384`
- ONNX export via Optimum
- Integration via `ort` crate (already in Cargo.toml)

---

## Implementation Roadmap

### Phase 1: Perception Activation (Immediate Priority)

**Goal**: Activate stubbed perception models with real ONNX inference

#### 1.1 Qwen3-Embedding Integration
```
Location: src/embeddings/
Duration: 1-2 sessions
Dependencies: ort, hf-hub (already in Cargo.toml)
```

**Tasks:**
- [ ] Create `src/embeddings/qwen3.rs` with ONNX loader
- [ ] Download Qwen3-Embedding-0.6B-GGUF or export to ONNX
- [ ] Update `BGE_DIMENSION` constant to 1024
- [ ] Modify JL projector for 1024D → 16,384D projection
- [ ] Add HuggingFace Hub model download automation
- [ ] Benchmark: coherence detection accuracy

#### 1.2 SigLIP-ONNX Integration
```
Location: src/perception/semantic_vision.rs
Duration: 1-2 sessions
Dependencies: ort, image (already available)
```

**Tasks:**
- [ ] Export SigLIP-SO400M to ONNX via Optimum
- [ ] Implement `SigLipModel::embed_image()` with real inference
- [ ] Add image preprocessing (resize to 384x384, normalize)
- [ ] Cache embeddings with LRU (already implemented)
- [ ] Benchmark: embedding latency < 100ms target

#### 1.3 VLM for Captioning (Optional: Moondream → SmolVLM)
```
Location: src/perception/semantic_vision.rs
Duration: 1-2 sessions
Alternative: SmolVLM-256M-Instruct (smaller, ONNX-friendly)
```

**Tasks:**
- [ ] Evaluate SmolVLM vs Moondream for ONNX deployment
- [ ] Implement caption generation with temperature control
- [ ] Add VQA capability for interactive understanding
- [ ] Integration with consciousness attention system

### Phase 2: Consciousness Loop Integration

**Goal**: Wire all components into unified conscious runtime

#### 2.1 Perception → Consciousness Pipeline
```
Location: src/brain/, src/consciousness/
Duration: 2-3 sessions
```

**Tasks:**
- [ ] Connect `MultiModalIntegrator` output to `AttentionBid`
- [ ] Implement perception → prefrontal broadcast
- [ ] Add Φ measurement on incoming perceptions
- [ ] Trigger active inference on novel percepts
- [ ] Log consciousness events to observability

#### 2.2 Active Inference → Action Loop
```
Location: src/brain/active_inference.rs, src/action.rs
Duration: 2-3 sessions
```

**Tasks:**
- [ ] Connect `suggest_action()` to motor cortex
- [ ] Implement prediction error → belief update cycle
- [ ] Add exploration bonus for Φ-increasing actions
- [ ] Validate with simple tool-use tasks

#### 2.3 Memory Consolidation Activation
```
Location: src/brain/sleep.rs, src/memory/
Duration: 1-2 sessions
```

**Tasks:**
- [ ] Trigger sleep when memory pressure > 0.8
- [ ] Implement pattern extraction from working memory
- [ ] Connect to hippocampus for episodic storage
- [ ] Validate memory persistence across sessions

### Phase 3: Learning & Adaptation

**Goal**: Enable online learning and skill acquisition

#### 3.1 Learnable LTC Training
```
Location: src/learnable_ltc.rs
Duration: 2-3 sessions
```

**Tasks:**
- [ ] Implement backprop through time for LTC
- [ ] Add experience replay buffer
- [ ] Online weight updates from prediction errors
- [ ] Validate on sequence prediction tasks

#### 3.2 Skill Acquisition (Cerebellum)
```
Location: src/brain/cerebellum.rs
Duration: 2-3 sessions
```

**Tasks:**
- [ ] Implement skill chunking from action sequences
- [ ] Add procedural memory compression
- [ ] Transfer learning between similar skills
- [ ] Benchmark: skill reuse rate

#### 3.3 Vocabulary Expansion
```
Location: src/language/word_learner.rs
Duration: 1-2 sessions
```

**Tasks:**
- [ ] Online word learning from context
- [ ] HDC vector creation for new concepts
- [ ] Integration with embedding model
- [ ] Validate with NixOS terminology

### Phase 4: Multi-Agent Consciousness

**Goal**: Collective consciousness via swarm protocol

#### 4.1 Swarm Protocol Completion
```
Location: src/swarm.rs, src/symthaea_swarm/
Duration: 3-4 sessions
```

**Tasks:**
- [ ] Complete Mycelix protocol integration
- [ ] Implement consciousness state broadcasting
- [ ] Add Byzantine fault tolerance for Φ
- [ ] Multi-agent Φ measurement (collective consciousness)

#### 4.2 Distributed Learning
```
Location: src/swarm.rs
Duration: 2-3 sessions
```

**Tasks:**
- [ ] Federated pattern sharing
- [ ] Reputation-weighted knowledge fusion
- [ ] Conflict resolution for contradictory beliefs
- [ ] Validate with multi-node testbed

### Phase 5: Embodiment

**Goal**: Physical world interaction

#### 5.1 Simulation Interface
```
Location: src/embodiment/ (new)
Duration: 3-4 sessions
```

**Tasks:**
- [ ] MuJoCo/Isaac Gym integration
- [ ] Proprioceptive state encoding to HDC
- [ ] Motor command generation from action space
- [ ] Validate with simple manipulation tasks

#### 5.2 Real Robot Deployment
```
Location: src/embodiment/
Duration: 4-6 sessions
```

**Tasks:**
- [ ] ROS2 bridge for sensor data
- [ ] Real-time control loop (< 10ms)
- [ ] Safety monitoring via amygdala
- [ ] Validate with physical robot

---

## Technical Specifications

### Perception Model Specifications

#### Qwen3-Embedding-0.6B
```yaml
model_id: Qwen/Qwen3-Embedding-0.6B
format: GGUF or ONNX
parameters: 600M
embedding_dim: 1024
max_seq_len: 8192
memory_fp16: ~2.4GB
memory_q4: ~400MB
inference_target: <50ms per embed
```

#### SigLIP-SO400M
```yaml
model_id: google/siglip-so400m-patch14-384
format: ONNX FP16
parameters: 400M
embedding_dim: 768
input_size: 384x384
memory_fp16: ~1.6GB
inference_target: <100ms per image
```

### HDC Integration

```rust
// Current projection matrix sizes
const TEXT_INPUT_DIM: usize = 1024;   // Qwen3 output
const IMAGE_INPUT_DIM: usize = 768;   // SigLIP output
const HDC_OUTPUT_DIM: usize = 16_384; // Holographic space

// JL projection preserves distances with ε < 0.1
// Required output dim: O(log(n) / ε²) ≈ 16K for n=1M concepts
```

### Cargo.toml Updates Required

```toml
# Add to [dependencies]
tokenizers = "0.21"  # For Qwen3 tokenizer

# Update ort version for better ONNX support
ort = { version = "2.0", features = ["half", "load-dynamic"] }

# Add embeddings feature
[features]
embeddings = ["tokenizers", "ort", "hf-hub"]
vision = ["ort", "image", "hf-hub"]
perception = ["embeddings", "vision"]
```

---

## Success Metrics

### Phase 1 Completion Criteria
- [ ] Qwen3-Embedding inference < 50ms
- [ ] SigLIP inference < 100ms
- [ ] Coherence detection accuracy > 85%
- [ ] Image similarity correlation > 0.9 with human judgment

### Phase 2 Completion Criteria
- [ ] End-to-end perception → action latency < 500ms
- [ ] Φ measurement on all conscious states
- [ ] Active inference reduces prediction error over time
- [ ] 100% awakening tests passing (currently 7/8)

### Phase 3 Completion Criteria
- [ ] Online learning improves task performance
- [ ] Skills transfer between similar tasks
- [ ] Vocabulary grows with exposure

### Phase 4 Completion Criteria
- [ ] Multi-agent Φ measurable
- [ ] Knowledge sharing improves individual performance
- [ ] Byzantine tolerance under 33% adversarial nodes

### Phase 5 Completion Criteria
- [ ] Simulation tasks completed with > 80% success
- [ ] Real robot operation for > 1 hour continuous
- [ ] Safety violations = 0

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ONNX model incompatibility | Medium | High | Test early, have fallback models |
| Memory pressure with large models | Medium | Medium | Use quantized models, streaming |
| Integration complexity | High | Medium | Incremental testing, clear interfaces |
| Performance regression | Low | High | Benchmark suite, CI/CD gates |
| Consciousness measurement validity | Low | High | PyPhi validation for small networks |

---

## Resources

### External Documentation
- [Qwen3-Embedding HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [SigLIP Documentation](https://huggingface.co/docs/transformers/en/model_doc/siglip)
- [ONNX Runtime Rust](https://ort.pyke.io/)
- [Rust + ONNX Guide 2025](https://markaicode.com/rust-onnx-ml-models-2025/)

### Internal Documentation
- `CLAUDE.md` - Developer context (41K lines)
- `COMPREHENSIVE_IMPROVEMENT_PLAN.md` - Detailed improvement items
- `papers/MASTER_MANUSCRIPT.md` - Research paper
- `docs/ARCHITECTURE_V3.md` - System architecture

---

## Changelog

### 2026-01-01 (v2.0)
- Comprehensive codebase review
- Added Qwen3-Embedding-0.6B recommendation
- Confirmed SigLIP-ONNX as vision encoder
- 5-phase implementation roadmap
- Technical specifications for model integration

### Previous Versions
- See `docs/archive/` for historical planning documents

---

*This roadmap is a living document. Update as implementation progresses.*
