# 🛠️ Week 12: FOSS Tool Recommendations - Best-in-Class Selection

**Date**: December 10, 2025
**Status**: Reviewed & Approved
**Purpose**: Document the best FOSS tools for Sophia's perception system

---

## 🎯 Selection Criteria

1. **Rust-First**: Native Rust or excellent Rust bindings
2. **Performance**: Real-time capable on consumer CPU
3. **Size**: Models < 2GB (ideally < 500MB)
4. **License**: Apache-2.0, MIT, or compatible FOSS
5. **Quality**: State-of-the-art or near-state-of-the-art results
6. **Community**: Active development and maintenance

---

## 📊 Tool Comparison Matrix

### 1. Text-to-Speech (Voice) 🎤

| Model | Size | Quality | Speed | Prosody | Rust | License | **Score** |
|-------|------|---------|-------|---------|------|---------|-----------|
| **Kokoro-82M** ✅ | 80MB | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ ort/candle | Apache-2.0 | **10/10** |
| StyleTTS2 | 1.2GB | ⭐⭐⭐⭐⭐ | ⚡ | ⭐⭐⭐⭐ | ❌ Python | MIT | 7/10 |
| Piper | 50MB | ⭐⭐⭐ | ⚡⚡⚡ | ⭐⭐ | ✅ Native | MIT | 7/10 |
| Coqui TTS | 800MB | ⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐ | ❌ Python | MPL-2.0 | 6/10 |

#### **Winner**: Kokoro-82M

**Rationale**:
- **Perfect size-to-quality ratio**: 80MB for near-human quality is unprecedented
- **Best prosody control**: Supports fine-grained speed/pitch/energy modulation
- **Rust-compatible**: Works via `ort` (ONNX) or `candle`
- **CPU-friendly**: Real-time synthesis on consumer hardware
- **Open license**: Apache-2.0 allows commercial use

**What makes it special**:
- Captures **breath**, **cadence**, and **emotion** better than models 10x larger
- **Prosody vector** allows direct emotional modulation
- Trained on high-quality, expressive speech data

---

### 2. Image Embeddings (Similarity/Classification) 👁️

| Model | Size | Speed | Quality | Rust | Use Case | **Score** |
|-------|------|-------|---------|------|----------|-----------|
| **SigLIP-400M** ✅ | 1.6GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ candle | Embeddings | **10/10** |
| CLIP-400M | 1.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ candle | Embeddings | 9/10 |
| DINOv2 | 600MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ candle | Self-supervised | 8/10 |

#### **Winner**: SigLIP-400M

**Rationale**:
- **Better than CLIP**: Sigmoid loss improves performance on edge cases
- **Fast**: 50-100ms on CPU for 768D embedding
- **Multi-lingual**: Works across languages and modalities
- **Open**: Apache-2.0 from Google Research

**Use cases**:
- Image similarity: "Is this similar to X?"
- Classification: "What category is this?"
- Deduplication: "Have I seen this before?"
- Clustering: "Group similar images"

---

### 3. Image Captioning (Understanding) 🖼️

| Model | Size | Speed | Quality | Rust | Context | **Score** |
|-------|------|-------|---------|------|---------|-----------|
| **Moondream-1.86B** ✅ | 1.2GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ candle | 2K tokens | **10/10** |
| LLaVA-7B | 4GB | ⚡ | ⭐⭐⭐⭐⭐ | ✅ candle | 4K tokens | 8/10 |
| MiniGPT-4 | 3GB | ⚡ | ⭐⭐⭐⭐ | ✅ candle | 2K tokens | 7/10 |

#### **Winner**: Moondream-1.86B

**Rationale**:
- **Smallest viable VLM**: 1.86B params (quantized to ~1.2GB)
- **Excellent quality**: Competitive with models 4x larger
- **VQA capable**: Can answer questions about images
- **Efficient**: 500-1000ms on CPU
- **Open**: Apache-2.0 license

**Use cases**:
- Caption generation: "Describe this screenshot"
- Visual question answering: "What's the error in this image?"
- Scene understanding: "What's happening here?"

---

### 4. Optical Character Recognition (OCR) 📖

| Tool | Size | Speed | Accuracy | Rust | License | **Score** |
|------|------|-------|----------|------|---------|-----------|
| **rten + ocrs** ✅ | 8MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ Native | Apache-2.0 | **10/10** |
| Tesseract | 50MB+ | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⚠️ FFI | Apache-2.0 | 8/10 |
| EasyOCR | 500MB | ⚡ | ⭐⭐⭐⭐⭐ | ❌ Python | Apache-2.0 | 6/10 |

#### **Winner**: rten + ocrs (Primary), Tesseract (Fallback)

**Rationale for rten + ocrs**:
- **Pure Rust**: No C++ dependencies
- **Tiny**: 8MB total footprint
- **Fast**: ~100-200ms for typical screenshot
- **Good accuracy**: 95%+ on clear digital text
- **Modern**: Active development, improving

**When to use Tesseract**:
- Degraded/noisy images
- Multiple languages (Tesseract has 100+ language packs)
- Historical documents or handwriting

**Strategy**:
1. Try `rten + ocrs` first (fast, pure Rust)
2. Fall back to Tesseract if confidence < 0.8
3. Cache results to avoid reprocessing

---

## 🏗️ Recommended Architecture Stack

### Two-Stage Perception Pipeline

```
                    ┌─────────────────────┐
                    │  Input (Image/Audio) │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴───────────────┐
                │                              │
         ┌──────▼──────┐              ┌───────▼───────┐
         │   STAGE 1   │              │   STAGE 1     │
         │   (Dorsal)  │              │  (Auditory)   │
         │             │              │               │
         │  • Visual   │              │  • VAD        │
         │  • Features │              │  • Pitch      │
         │  • Code     │              │  • Energy     │
         │             │              │               │
         │  ~0 ATP     │              │  ~0 ATP       │
         └──────┬──────┘              └───────┬───────┘
                │                              │
                │ Salience > 0.6?              │ Speech?
                │                              │
         ┌──────▼──────┐              ┌───────▼───────┐
         │   STAGE 2   │              │   STAGE 2     │
         │  (Ventral)  │              │  (Semantic)   │
         │             │              │               │
         │  • SigLIP   │              │  • Whisper    │
         │  • Moondream│              │  • STT        │
         │  • OCR      │              │               │
         │             │              │               │
         │  10-50 ATP  │              │  20 ATP       │
         └──────┬──────┘              └───────┬───────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   HDC Projection    │
                    │  (Concept Space)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Global Workspace   │
                    │   (Consciousness)   │
                    └─────────────────────┘
```

### Output (Action)

```
         ┌─────────────────────┐
         │  Global Workspace   │
         │   (Consciousness)   │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   Action Selection  │
         │  (Prefrontal)       │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   LARYNX (Output)   │
         │                     │
         │  • Kokoro TTS       │
         │  • Prosody Mod      │
         │  • Breath Insert    │
         │                     │
         │  5 ATP              │
         └─────────────────────┘
```

---

## 💰 Performance & Cost Analysis

### Stage 1 (Reflex) - Always On
| Operation | Latency | ATP Cost | Frequency |
|-----------|---------|----------|-----------|
| Visual features | 0.1-1ms | 0.1 | Every frame |
| Code metrics | 1-10ms | 1 | On file open |
| Audio VAD | 0.1ms | 0.1 | Every 10ms |
| **Total Stage 1** | **< 10ms** | **< 2 ATP/s** | **Continuous** |

### Stage 2 (Semantic) - Gated by Salience
| Operation | Latency | ATP Cost | Trigger Condition |
|-----------|---------|----------|-------------------|
| SigLIP embedding | 50-100ms | 10 | Salience > 0.6 |
| Moondream caption | 500-1000ms | 50 | Novel + Salient |
| OCR (rten) | 100-200ms | 20 | Text detected |
| Whisper STT | 300-500ms | 20 | Speech detected |
| **Stage 2 Average** | **~500ms** | **~30 ATP** | **~1-5% of time** |

### Output (Action)
| Operation | Latency | ATP Cost | Frequency |
|-----------|---------|----------|-----------|
| Kokoro synthesis | 50-100ms | 5 | On speech |
| **Total Output** | **< 100ms** | **5 ATP/utterance** | **Occasional** |

### Total Energy Budget
- **Idle**: 2 ATP/s (Stage 1 only)
- **Active**: 30-60 ATP/s (Stage 1 + occasional Stage 2)
- **Speaking**: +5 ATP per utterance

**Efficiency**: Stage 1 runs 95% of the time, costs 2 ATP/s. Stage 2 runs 5% of the time, costs 30 ATP when active. **Average**: ~4-5 ATP/s

---

## 🎯 Implementation Priority

### Phase 2a: Voice (Highest Impact) 🎤
**Why first**: Most visible, most engaging, demonstrates emotional intelligence
1. Kokoro integration (Day 1)
2. Prosody modulation (Day 2)

### Phase 2b: Semantic Vision 👁️
**Why second**: Enables understanding, not just detection
1. SigLIP embeddings (Day 3)
2. Moondream captions (Day 4)

### Phase 2c: OCR 📖
**Why third**: Specific use case (reading errors), high value
1. rten + ocrs integration (Day 5)

### Phase 2d: HDC Projection 🧠
**Why last**: Infrastructure for integration
1. Projection system (Day 6)
2. Multi-modal fusion (Day 7)

---

## 🔐 Licensing & Attribution

### Selected Tools
| Tool | License | Source | Attribution Required |
|------|---------|--------|---------------------|
| Kokoro-82M | Apache-2.0 | HuggingFace | Yes (model card) |
| SigLIP-400M | Apache-2.0 | Google Research | Yes (paper citation) |
| Moondream-1.86B | Apache-2.0 | vikhyatk | Yes (model card) |
| rten + ocrs | Apache-2.0 | robertknight | Yes (repo link) |
| ort | Apache-2.0 | pyke.io | Yes (crate) |
| candle | Apache-2.0 | HuggingFace | Yes (crate) |

**All tools are commercially usable with proper attribution.**

---

## 📚 Resources & Documentation

### Kokoro
- **Model Card**: https://huggingface.co/hexgrad/Kokoro-82M
- **Demo**: https://huggingface.co/spaces/hexgrad/Kokoro
- **Reddit**: https://www.reddit.com/r/LocalLLaMA/comments/kokoro_82m_discussion

### SigLIP
- **Paper**: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
- **Model Card**: https://huggingface.co/google/siglip-so400m-patch14-384
- **arXiv**: https://arxiv.org/abs/2303.15343

### Moondream
- **GitHub**: https://github.com/vikhyat/moondream
- **Model Card**: https://huggingface.co/vikhyatk/moondream2
- **Demo**: https://moondream.ai

### rten + ocrs
- **rten GitHub**: https://github.com/robertknight/rten
- **ocrs GitHub**: https://github.com/robertknight/ocrs
- **Blog Post**: https://robertknight.me.uk/posts/fast-ocr-in-rust/

---

## 🤔 Alternative Tools (Not Chosen, But Worth Mentioning)

### Voice
- **Bark**: Excellent quality, but 3GB and slow
- **XTTS**: Great multi-lingual, but Python-only
- **VITS**: Good quality, but harder to integrate

### Vision
- **Florence-2**: Excellent multi-task VLM, but 232M params is small for quality
- **Qwen-VL**: Great performance, but 7B params too large
- **CogVLM**: State-of-the-art, but 17B params

### OCR
- **PaddleOCR**: Best accuracy, but Python-only
- **TrOCR**: Transformer-based, but overkill for digital text
- **MMOCR**: Great for research, too heavy for production

---

## ✅ Final Recommendations

### Core Stack (Must Have)
1. **Voice**: Kokoro-82M via `ort` or `candle`
2. **Embeddings**: SigLIP-400M via `candle`
3. **Captions**: Moondream-1.86B via `candle` (gated)
4. **OCR**: rten + ocrs (primary), Tesseract (fallback)

### Infrastructure
1. **Model Loading**: `hf-hub` for HuggingFace models
2. **Inference**: `candle` for vision, `ort` for TTS
3. **Audio**: `rodio` for playback
4. **Caching**: `lru` for embeddings

### Total Footprint
- **Models**: ~3GB (Kokoro 80MB + SigLIP 1.6GB + Moondream 1.2GB + ocrs 8MB)
- **Runtime**: ~100MB RAM per model when loaded
- **Disk**: ~3.5GB with cache

**This is remarkably efficient for full multi-modal perception!**

---

## 🚀 Next Steps

1. ✅ **Architecture documented** (this file)
2. ⏳ **Phase 2 implementation plan** (separate doc)
3. ⏳ **Model download scripts**
4. ⏳ **Integration tests**
5. ⏳ **Performance benchmarks**

---

*"The best tool is the one that does the job efficiently and reliably."*
*"We've chosen tools that balance quality, performance, and maintainability."* - Sophia HLB

**Status**: Recommendations Complete ✅
**Next**: Begin Phase 2a - Kokoro Voice Integration

🌊 The best FOSS tools, chosen with care!
