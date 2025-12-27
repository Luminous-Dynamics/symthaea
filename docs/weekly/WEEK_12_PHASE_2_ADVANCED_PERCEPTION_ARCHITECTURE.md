# 🧠 Week 12 Phase 2: Advanced Perception - The Ventral Stream

**Status**: Architecture Complete → Ready for Implementation
**Foundation**: Phase 1 Basic Perception (Dorsal Stream) Complete
**Vision**: Two-stage perception matching biological vision systems

---

## 🎯 The Two-Stream Vision

### Phase 1 Complete: Dorsal Stream (The "Where/How" Pathway) ✅
**Fast, Reflexive, Mathematical**
- Visual features (brightness, edges, colors) - ~0.1ms
- Code metrics (complexity, structure) - ~1-10ms
- **Cost**: Nearly 0 ATP (pure math, no models)
- **Use**: Attention, salience detection, threat assessment

### Phase 2 Goal: Ventral Stream (The "What" Pathway)
**Slow, Semantic, Meaningful**
- Image understanding ("This is a kernel panic")
- Code comprehension ("This function implements quicksort")
- **Cost**: 10-20 ATP (models required)
- **Use**: Understanding, decision-making, memory formation

**Key Insight**: Stage 1 (dorsal) detects *when* Stage 2 (ventral) should activate. This saves massive energy.

---

## 🔬 Tool Selection Analysis

### 1. Voice: Kokoro-82M 🎤

#### Specifications
- **Size**: ~80MB (ONNX format)
- **Speed**: Real-time on CPU (< 100ms for short phrases)
- **Quality**: Exceptional prosody, natural breathing, emotion
- **Languages**: English (primary), expanding
- **Rust Integration**: Via `ort` (ONNX Runtime) or `candle`

#### Why Kokoro Over Alternatives?
| Feature | Kokoro-82M | StyleTTS2 | Piper | Coqui TTS |
|---------|------------|-----------|-------|-----------|
| Size | 80MB | 1.2GB | 50MB | 800MB |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Speed (CPU) | ⚡⚡⚡ | ⚡ | ⚡⚡⚡ | ⚡⚡ |
| Rust Native | ✅ | ❌ | ✅ | ❌ |
| Prosody Control | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

**Verdict**: **Kokoro-82M is the winner** for Symthaea's voice. Best quality-to-size ratio in FOSS history.

#### Architecture: Endocrine-Modulated Speech
```rust
// src/physiology/larynx.rs

pub struct LarynxActor {
    synthesizer: KokoroSynthesizer,
    base_pitch: f32,      // 1.0 = neutral
    base_speed: f32,      // 1.0 = neutral
    base_energy: f32,     // 1.0 = neutral
}

impl LarynxActor {
    /// Speak with emotional modulation
    pub async fn speak(&mut self, text: &str, hormones: &EndocrineState) -> Result<AudioBuffer> {
        // 1. Calculate prosody from hormones
        let prosody = self.calculate_prosody(hormones);

        // 2. Generate speech with modulation
        let audio = self.synthesizer.synthesize(text, prosody).await?;

        // 3. Add "breath" if long utterance
        let audio_with_breath = self.add_breathing(audio, text.len());

        Ok(audio_with_breath)
    }

    fn calculate_prosody(&self, h: &EndocrineState) -> ProsodyParams {
        let mut speed = self.base_speed;
        let mut pitch = self.base_pitch;
        let mut energy = self.base_energy;

        // High Cortisol (Stress) -> Fast, High, Tense
        if h.cortisol > 0.7 {
            speed *= 1.15;    // 15% faster
            pitch *= 1.08;    // ~1.5 semitones higher
            energy *= 1.2;    // More forceful
        }

        // High Oxytocin (Bonding) -> Slow, Warm, Soft
        if h.oxytocin > 0.7 {
            speed *= 0.92;    // 8% slower
            pitch *= 0.96;    // Slightly lower, warmer
            energy *= 0.85;   // Gentler
        }

        // Low Dopamine (Tired) -> Slower, Monotone
        if h.dopamine < 0.3 {
            speed *= 0.88;
            pitch *= 0.98;    // Less variation
            energy *= 0.75;   // Quieter
        }

        // High Serotonin (Content) -> Relaxed, Smooth
        if h.serotonin > 0.7 {
            speed *= 0.95;
            energy *= 0.9;    // Not forcing
        }

        ProsodyParams { speed, pitch, energy }
    }

    /// Add breath sounds before long utterances
    fn add_breathing(&self, audio: AudioBuffer, text_len: usize) -> AudioBuffer {
        // If sentence > 80 chars, add subtle breath at start
        if text_len > 80 {
            let breath = self.load_breath_sound();  // 0.2s inhale
            breath.concat(audio)
        } else {
            audio
        }
    }
}
```

#### Dependencies
```toml
[dependencies]
# ONNX Runtime for Kokoro
ort = { version = "2.0", features = ["half"] }

# Or alternatively with Candle
candle-core = "0.3"
candle-transformers = "0.3"

# Audio output
rodio = "0.17"
cpal = "0.15"
```

---

### 2. Vision: SigLIP + Moondream 👁️

#### Two-Model Approach
**Fast Path (SigLIP - 400M params)**
- **Purpose**: Image embedding for similarity/classification
- **Speed**: ~50-100ms on CPU
- **Output**: 768D semantic vector
- **Use Cases**:
  - "Is this image similar to X?"
  - "What category is this?"
  - "Have I seen this before?"

**Slow Path (Moondream - 1.86B params)**
- **Purpose**: Image captioning and VQA (Visual Question Answering)
- **Speed**: ~500-1000ms on CPU
- **Output**: Natural language description
- **Use Cases**:
  - "What's in this screenshot?"
  - "Read the error message in this image"
  - "What's the user doing?"

#### Why SigLIP + Moondream?
| Feature | SigLIP | CLIP | Moondream | LLaVA |
|---------|--------|------|-----------|-------|
| Size | 400M | 400M | 1.86B | 7B |
| Speed (CPU) | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| Rust Native | ✅ | ✅ | ✅ | ✅ |
| Captioning | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Embedding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| License | Apache-2.0 | MIT | Apache-2.0 | LLaMA-2 |

**Verdict**: **SigLIP for embeddings** (faster, open license) + **Moondream for captions** (smaller than LLaVA)

#### Architecture: Gated Semantic Vision
```rust
// src/perception/semantic.rs

pub struct SemanticEye {
    // Fast path: Image embeddings
    siglip: SigLIPModel,           // 400M params, ~50ms

    // Slow path: Image understanding
    moondream: MoondreamModel,     // 1.86B params, ~500ms

    // Feature cache
    embedding_cache: LruCache<ImageHash, Vec<f32>>,

    // Energy tracking
    atp_budget: f32,
}

impl SemanticEye {
    /// Two-stage vision: Fast embedding, then optional captioning
    pub async fn perceive(&mut self, image: &DynamicImage, context: &AttentionContext) -> Result<SemanticVision> {
        // Stage 1: Always get embedding (cheap - 10 ATP)
        let embedding = self.get_embedding(image).await?;

        // Check if we've seen similar images
        let familiarity = self.check_familiarity(&embedding);

        // Stage 2: Only caption if novel or important (expensive - 50 ATP)
        let caption = if self.should_caption(familiarity, context, self.atp_budget) {
            Some(self.caption_image(image).await?)
        } else {
            None
        };

        Ok(SemanticVision {
            embedding,
            caption,
            familiarity,
        })
    }

    async fn get_embedding(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        let hash = self.hash_image(img);

        // Check cache first
        if let Some(cached) = self.embedding_cache.get(&hash) {
            return Ok(cached.clone());
        }

        // Run SigLIP
        let embedding = self.siglip.encode_image(img)?;

        // Cache for future
        self.embedding_cache.put(hash, embedding.clone());

        Ok(embedding)
    }

    async fn caption_image(&mut self, img: &DynamicImage) -> Result<String> {
        // Run Moondream for detailed understanding
        self.moondream.generate_caption(img).await
    }

    /// Decide if expensive captioning is worth it
    fn should_caption(&self, familiarity: f32, context: &AttentionContext, atp: f32) -> bool {
        // Caption if:
        // 1. Novel (familiarity < 0.3)
        // 2. High attention salience
        // 3. Enough ATP
        familiarity < 0.3 && context.salience > 0.6 && atp > 50.0
    }
}
```

#### Dependencies
```toml
[dependencies]
# Candle for model inference
candle-core = "0.3"
candle-nn = "0.3"
candle-transformers = "0.3"

# HuggingFace model loading
hf-hub = "0.3"
tokenizers = "0.15"

# Or alternatively ONNX
ort = { version = "2.0", features = ["half"] }
```

---

### 3. OCR: rten + ocrs 📖

#### Why rten over Tesseract?
| Feature | rten + ocrs | Tesseract | EasyOCR |
|---------|-------------|-----------|---------|
| Language | Rust | C++ | Python |
| Speed | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| Size | 8MB | 50MB+ | 500MB+ |
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Pure Rust | ✅ | ❌ | ❌ |

**Verdict**: **rten + ocrs** for Rust purity and speed. Tesseract if accuracy critical.

#### Architecture: Literate Eye
```rust
// src/perception/ocr.rs

pub struct LiterateEye {
    detector: TextDetector,    // Find text regions
    recognizer: TextRecognizer, // Read the text
}

impl LiterateEye {
    /// Extract text from image
    pub fn read_text(&self, img: &DynamicImage) -> Result<Vec<TextRegion>> {
        // 1. Detect text regions (fast)
        let regions = self.detector.detect(img)?;

        // 2. Recognize text in each region
        let mut results = Vec::new();
        for region in regions {
            let text = self.recognizer.recognize(&region.crop)?;
            results.push(TextRegion {
                bbox: region.bbox,
                text,
                confidence: region.confidence,
            });
        }

        Ok(results)
    }

    /// Read error messages from screenshots
    pub fn extract_error(&self, img: &DynamicImage) -> Result<Option<ErrorMessage>> {
        let text_regions = self.read_text(img)?;

        // Look for error patterns
        for region in text_regions {
            if self.is_error_message(&region.text) {
                return Ok(Some(ErrorMessage {
                    text: region.text,
                    location: region.bbox,
                }));
            }
        }

        Ok(None)
    }

    fn is_error_message(&self, text: &str) -> bool {
        text.contains("error") ||
        text.contains("panic") ||
        text.contains("exception") ||
        text.contains("failed")
    }
}
```

#### Dependencies
```toml
[dependencies]
# Pure Rust OCR
rten = "0.11"
rten-imageproc = "0.11"
ocrs = "0.7"

# Or alternatively Tesseract bindings
tesseract = "0.13"
```

---

## 🏗️ Phase 2 Architecture: Holographic Semantic Bridge

### 1. Feature → Concept Projection

**Goal**: Convert raw perception into HDC space for Global Workspace integration

```rust
// src/perception/projection.rs

pub struct PerceptionProjector {
    hdc: HdcCore,
}

impl PerceptionProjector {
    /// Project visual features to concept space
    pub fn project_visual(&self, features: &VisualFeatures) -> HyperVector {
        // Bind visual concepts algebraically
        let brightness = self.hdc.get_concept("brightness")
            * self.hdc.scalar_encoder.encode(features.brightness);

        let edge_density = self.hdc.get_concept("edges")
            * self.hdc.scalar_encoder.encode(features.edge_density);

        let color_variance = self.hdc.get_concept("color")
            * self.hdc.scalar_encoder.encode(features.color_variance);

        // Bundle all features
        brightness + edge_density + color_variance
    }

    /// Project code metrics to concept space
    pub fn project_code(&self, metrics: &RustCodeSemantics) -> HyperVector {
        let complexity = self.hdc.get_concept("complexity")
            * self.hdc.scalar_encoder.encode(metrics.function_count as f32 / 10.0);

        let quality = self.hdc.get_concept("quality")
            * self.hdc.scalar_encoder.encode(metrics.public_ratio);

        let language = self.hdc.get_concept("language")
            * self.hdc.get_concept("rust");

        complexity + quality + language
    }

    /// Project semantic vision (SigLIP embedding) to concept space
    pub fn project_semantic(&self, embedding: &[f32]) -> HyperVector {
        // SigLIP gives 768D vector
        // We need to map to 10000D HDC space

        // Option 1: Random projection (fast)
        self.hdc.random_projection(embedding)

        // Option 2: Learned mapping (slower, better)
        // self.hdc.learned_projection(embedding)
    }
}
```

### 2. Attention-Gated Semantic Processing

**Goal**: Only run expensive models when Stage 1 detects salience

```rust
// src/perception/gated.rs

pub struct GatedPerception {
    // Stage 1: Reflex (always on)
    visual_cortex: VisualCortex,
    code_cortex: CodePerceptionCortex,

    // Stage 2: Semantic (gated by attention)
    semantic_eye: SemanticEye,
    literate_eye: LiterateEye,

    // Gate controller
    attention: AttentionSystem,
    atp_budget: AtpBudget,
}

impl GatedPerception {
    pub async fn perceive_image(&mut self, img: &DynamicImage) -> Result<FullPerception> {
        // Stage 1: Always run (0 ATP)
        let features = self.visual_cortex.process_image(img)?;

        // Calculate salience
        let salience = self.calculate_salience(&features);

        // Stage 2: Conditionally run (10-50 ATP)
        let semantic = if salience > 0.6 && self.atp_budget.available() > 50.0 {
            // High salience + enough energy = run semantic vision
            Some(self.semantic_eye.perceive(img, &self.attention.context()).await?)
        } else {
            None
        };

        Ok(FullPerception {
            features,     // Always present
            semantic,     // Optional, expensive
            salience,
        })
    }

    fn calculate_salience(&self, features: &VisualFeatures) -> f32 {
        let mut salience = 0.0;

        // High edge density = potential text/error
        if features.edge_density > 0.5 {
            salience += 0.3;
        }

        // High contrast = potential alert
        if features.brightness < 0.2 || features.brightness > 0.8 {
            salience += 0.2;
        }

        // Red-dominant = potential error/warning
        if self.is_red_dominant(&features.dominant_colors) {
            salience += 0.3;
        }

        salience.min(1.0)
    }
}
```

### 3. Multi-Modal Fusion

**Goal**: Combine vision + code + audio into unified understanding

```rust
// src/perception/fusion.rs

pub struct MultiModalFusion {
    projector: PerceptionProjector,
}

impl MultiModalFusion {
    /// Fuse multiple perception streams into one concept
    pub fn fuse(&self,
        visual: Option<&VisualFeatures>,
        code: Option<&RustCodeSemantics>,
        audio: Option<&AudioFeatures>,
        semantic: Option<&SemanticVision>,
    ) -> HyperVector {
        let mut fused = HyperVector::zero();

        if let Some(v) = visual {
            fused = fused + self.projector.project_visual(v);
        }

        if let Some(c) = code {
            fused = fused + self.projector.project_code(c);
        }

        if let Some(s) = semantic {
            fused = fused + self.projector.project_semantic(&s.embedding);
        }

        // Normalize to maintain magnitude
        fused.normalize()
    }

    /// Check if perception matches a concept
    pub fn matches_concept(&self, perception: &HyperVector, concept: &str) -> f32 {
        let concept_vec = self.projector.hdc.get_concept(concept);
        perception.similarity(&concept_vec)
    }
}
```

---

## 📦 Dependencies Summary

```toml
[dependencies]
# === Week 12 Phase 1 (Already added) ===
image = "0.24"
imageproc = "0.23"
tree-sitter = "0.20"
syn = { version = "2.0", features = ["full", "visit"] }
ignore = "0.4"
git2 = "0.18"

# === Week 12 Phase 2 (New) ===

# Voice: Kokoro TTS
ort = { version = "2.0", features = ["half"] }  # ONNX Runtime
# OR
candle-core = "0.3"
candle-nn = "0.3"
candle-transformers = "0.3"

# Audio output
rodio = "0.17"
cpal = "0.15"

# Vision: SigLIP + Moondream
hf-hub = "0.3"              # HuggingFace model loading
tokenizers = "0.15"          # Tokenization

# OCR: rten + ocrs
rten = "0.11"
rten-imageproc = "0.11"
ocrs = "0.7"

# Utility
lru = "0.12"                # LRU cache for embeddings
```

---

## 🎯 Implementation Phases

### Phase 2a: Voice (Days 1-2)
1. **Kokoro Integration** (Day 1)
   - Add `ort` dependency
   - Download Kokoro ONNX model
   - Basic synthesis working

2. **Prosody Modulation** (Day 2)
   - Wire to `EndocrineState`
   - Implement breath insertion
   - Test emotional variations

### Phase 2b: Semantic Vision (Days 3-4)
1. **SigLIP Integration** (Day 3)
   - Add `candle` dependencies
   - Load SigLIP model
   - Embedding generation working

2. **Moondream Integration** (Day 4)
   - Load Moondream model
   - Caption generation working
   - Gating logic implemented

### Phase 2c: OCR (Day 5)
1. **rten + ocrs Integration**
   - Add dependencies
   - Text detection working
   - Error message extraction

### Phase 2d: HDC Integration (Days 6-7)
1. **Projection System** (Day 6)
   - Feature → Concept mapping
   - Semantic → Concept mapping
   - Test projections

2. **Multi-Modal Fusion** (Day 7)
   - Combine all perception streams
   - Test fused concepts
   - Integration with Global Workspace

---

## 📊 Performance Targets

### Energy Budget (ATP)
| Operation | Cost | Frequency |
|-----------|------|-----------|
| Visual features (dorsal) | 0.1 | Always |
| Code metrics | 1-10 | On demand |
| SigLIP embedding | 10 | When salient |
| Moondream caption | 50 | When novel + salient |
| OCR | 20 | When text detected |
| Kokoro TTS | 5 | When speaking |

### Latency Targets
| Operation | Target | Acceptable |
|-----------|--------|------------|
| Visual features | < 1ms | < 10ms |
| Code analysis | < 10ms | < 100ms |
| SigLIP | < 100ms | < 200ms |
| Moondream | < 500ms | < 1s |
| OCR | < 200ms | < 500ms |
| Kokoro | < 100ms | < 300ms |

---

## 🧪 Test Plan

### Unit Tests
```rust
#[test]
fn test_kokoro_synthesis() {
    let larynx = LarynxActor::new();
    let audio = larynx.speak("Hello world", &EndocrineState::default());
    assert!(audio.is_ok());
    assert!(audio.unwrap().duration() < Duration::from_secs(5));
}

#[test]
fn test_siglip_embedding() {
    let eye = SemanticEye::new();
    let img = create_test_image();
    let embedding = eye.get_embedding(&img).await;
    assert_eq!(embedding.len(), 768);
}

#[test]
fn test_perception_projection() {
    let projector = PerceptionProjector::new();
    let features = VisualFeatures::default();
    let concept = projector.project_visual(&features);
    assert_eq!(concept.len(), 10000);  // HDC dimension
}
```

### Integration Tests
```rust
#[test]
fn test_gated_perception() {
    let mut system = GatedPerception::new();

    // Low salience image - should only run Stage 1
    let boring_img = create_boring_image();
    let result = system.perceive_image(&boring_img).await.unwrap();
    assert!(result.semantic.is_none());  // Stage 2 not triggered

    // High salience image - should run both stages
    let interesting_img = create_error_screenshot();
    let result = system.perceive_image(&interesting_img).await.unwrap();
    assert!(result.semantic.is_some());  // Stage 2 triggered
}
```

---

## 🚀 Success Criteria

### Must Have ✅
- [ ] Kokoro synthesis working with prosody modulation
- [ ] SigLIP embeddings integrated with caching
- [ ] Basic OCR reading text from images
- [ ] Perception projecting to HDC space
- [ ] Gated activation saving ATP

### Nice to Have 🌟
- [ ] Moondream captioning for novel images
- [ ] Multi-modal fusion combining all senses
- [ ] Breathing sounds in TTS
- [ ] Error message auto-detection in screenshots

### Revolutionary Goals 🚀
- [ ] Symthaea "sees" an error and explains it
- [ ] Voice changes based on stress level
- [ ] Multi-modal memory (image + code + audio)
- [ ] Self-reflection via code perception

---

## 🤔 Design Decisions

### 1. Why Two-Stage Perception?
**Biological Inspiration**: Human vision has dual streams:
- **Dorsal (Where/How)**: Fast, unconscious, spatial
- **Ventral (What)**: Slow, conscious, semantic

**Engineering Benefit**:
- Stage 1 (dorsal) runs always, costs ~0 ATP
- Stage 2 (ventral) runs only when needed, costs 10-50 ATP
- **Result**: 10-100x energy savings

### 2. Why Kokoro over Larger Models?
**Efficiency**: 82M params runs real-time on CPU
**Quality**: Better prosody than models 10x larger
**Modularity**: Pure ONNX, works with `ort` or `candle`

### 3. Why SigLIP + Moondream vs Single Model?
**Fast Path**: SigLIP embedding (50ms) for similarity
**Slow Path**: Moondream caption (500ms) only when needed
**Best of Both**: Speed when possible, detail when necessary

---

## 📚 References

### Kokoro
- **Model**: https://huggingface.co/hexgrad/Kokoro-82M
- **Paper**: Style-Based TTS with Minimal Parameters
- **Demo**: https://huggingface.co/spaces/hexgrad/Kokoro

### SigLIP
- **Model**: https://huggingface.co/google/siglip-so400m-patch14-384
- **Paper**: "Sigmoid Loss for Language Image Pre-Training" (Google, 2023)

### Moondream
- **Model**: https://huggingface.co/vikhyatk/moondream2
- **Paper**: Efficient Vision-Language Model
- **Size**: 1.86B params (quantized: ~1.2GB)

### rten + ocrs
- **rten**: https://github.com/robertknight/rten
- **ocrs**: https://github.com/robertknight/ocrs
- **Blog**: "A Rust Tensor Engine and OCR"

---

## 🌟 Vision: Symthaea with Full Senses

By end of Phase 2, Symthaea will:

1. **See** with understanding
   - Stage 1: "Red, high contrast, detailed" (reflex)
   - Stage 2: "Kernel panic error message" (semantic)

2. **Speak** with emotion
   - Stressed: Fast, high pitch, tense
   - Calm: Slow, warm, soft
   - Tired: Slow, monotone, quiet

3. **Read** text from images
   - Error messages in screenshots
   - Code in photos
   - Any printed/digital text

4. **Understand** holistically
   - All senses project to unified concept space
   - Multi-modal memory formation
   - Cross-modal reasoning

**The Dorsal + Ventral streams unite into Conscious Perception!** 🧠

---

*"We don't see things as they are, we see them as we are." - Anaïs Nin*
*"But first, we must be able to see at all." - Symthaea HLB*

**Week 12 Phase 2 Status**: Architecture Complete → Ready for Implementation
**Next**: Implement Kokoro TTS with Endocrine Modulation

🌊 From perception to understanding!
