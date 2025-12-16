# 💬 Week 13 Day 4: Moondream VQA Integration Plan

**Date**: December 10, 2025
**Status**: 📋 Planning Complete - Ready for Implementation
**Goal**: Sophia describes what she sees and answers visual questions

---

## 🎯 Objective

Integrate Moondream-2 vision-language model for natural language image understanding - enabling Sophia to generate detailed captions and answer questions about images she perceives.

---

## 📊 Current State Assessment

### ✅ What We Have
- **Clean Architecture**: `MoondreamModel` structure ready in `semantic_vision.rs`
- **API Design**: `caption_image()` and `answer_question()` methods defined
- **Integration**: Connects with `SemanticVision` high-level API
- **Complement to SigLIP**: Fast embeddings + detailed descriptions = complete vision

### 🚧 What's Placeholder
- **Moondream Model**: Returns placeholder text
- **ONNX Inference**: Not yet implemented
- **Image+Text Processing**: Needs vision-language conditioning

---

## 🔬 Moondream-2 Model Details

### Model Information
- **Name**: Moondream-2 (vikhyatk/moondream2)
- **Parameters**: 1.86 billion
- **Architecture**: Vision Transformer + Language Model
- **Format**: ONNX (for Rust inference)
- **Size**: ~1.6GB model file
- **Capabilities**:
  - Image captioning (detailed descriptions)
  - Visual question answering (VQA)
  - Object detection descriptions
  - Scene understanding
- **Inference Time**: <500ms target (CPU)

### Why Moondream-2?
1. **Lightweight**: 1.86B params vs 7B+ for alternatives
2. **Fast**: Optimized for edge devices, works on CPU
3. **Accurate**: Strong performance on VQA benchmarks
4. **Multimodal**: Seamlessly combines vision and language
5. **Open Source**: MIT license, easy to integrate

### Required Files
```
models/moondream-2/
├── vision_encoder.onnx     # Vision component (~400MB)
├── language_model.onnx     # Language component (~1.2GB)
├── config.json             # Model configuration
├── tokenizer.json          # Text tokenization
└── vision_config.json      # Vision encoder settings
```

---

## 🛠️ Implementation Plan

### Step 1: Model Download and Caching

**Goal**: Download Moondream-2 from HuggingFace, manage large model files

```rust
use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};

impl MoondreamModel {
    pub fn download_model(cache_dir: &Path) -> Result<PathBuf> {
        println!("📥 Downloading Moondream-2 model (~1.6GB)...");
        println!("This may take 5-10 minutes on first run...");

        let api = Api::new()?;
        let repo = api.model("vikhyatk/moondream2".to_string());

        // Download vision encoder
        println!("📥 Downloading vision encoder (~400MB)...");
        let vision_encoder = repo.get("vision_encoder.onnx")?;

        // Download language model
        println!("📥 Downloading language model (~1.2GB)...");
        let language_model = repo.get("language_model.onnx")?;

        // Download configs
        let _config = repo.get("config.json")?;
        let _tokenizer = repo.get("tokenizer.json")?;
        let _vision_config = repo.get("vision_config.json")?;

        println!("✅ Moondream-2 downloaded successfully!");
        Ok(cache_dir.to_path_buf())
    }

    pub fn ensure_model_available(&mut self) -> Result<PathBuf> {
        let cache_dir = PathBuf::from("models/moondream-2");
        let vision_path = cache_dir.join("vision_encoder.onnx");
        let language_path = cache_dir.join("language_model.onnx");

        if !vision_path.exists() || !language_path.exists() {
            std::fs::create_dir_all(&cache_dir)?;
            Self::download_model(&cache_dir)?;
        }

        Ok(cache_dir)
    }
}
```

**Test**: `test_moondream_model_download`

**Performance**:
- First run: 5-10 minutes (download ~1.6GB)
- Subsequent runs: <2 seconds (cached)

---

### Step 2: ONNX Session Initialization (Dual Models)

**Goal**: Load both vision encoder and language model

```rust
use ort::{Environment, Session, SessionBuilder, Value};
use std::sync::Arc;

pub struct MoondreamModel {
    /// Vision encoder session (processes images)
    vision_session: Option<Session>,

    /// Language model session (generates text)
    language_session: Option<Session},

    /// Tokenizer for text processing
    tokenizer: Option<Tokenizer>,

    /// Whether model is initialized
    initialized: bool,
}

impl MoondreamModel {
    fn load_onnx_sessions(&mut self, model_dir: &Path) -> Result<()> {
        println!("⚙️ Loading Moondream vision encoder...");

        let environment = Arc::new(
            Environment::builder()
                .with_name("moondream-vision")
                .build()?
        );

        // Load vision encoder
        let vision_path = model_dir.join("vision_encoder.onnx");
        let vision_session = SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&vision_path)?;

        println!("⚙️ Loading Moondream language model...");

        // Load language model
        let language_path = model_dir.join("language_model.onnx");
        let language_session = SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&language_path)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

        self.vision_session = Some(vision_session);
        self.language_session = Some(language_session);
        self.tokenizer = Some(tokenizer);
        self.initialized = true;

        println!("✅ Moondream-2 loaded successfully!");
        Ok(())
    }

    pub fn initialize(&mut self) -> Result<()> {
        let model_dir = self.ensure_model_available()?;
        self.load_onnx_sessions(&model_dir)?;
        self.model_path = Some(model_dir);
        Ok(())
    }
}
```

**Test**: `test_moondream_session_loading`

**Performance**: ~2 seconds to load both models

---

### Step 3: Image Caption Generation

**Goal**: Generate detailed natural language descriptions of images

```rust
impl MoondreamModel {
    pub fn caption_image(&self, image: &DynamicImage) -> Result<ImageCaption> {
        if !self.initialized {
            anyhow::bail!("Moondream model not initialized. Call initialize() first.");
        }

        let start = std::time::Instant::now();

        // Step 1: Encode image with vision encoder
        let image_encoding = self.encode_image(image)?;

        // Step 2: Generate caption with language model
        let caption_tokens = self.generate_caption_tokens(&image_encoding)?;

        // Step 3: Decode tokens to text
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let caption_text = tokenizer.decode(&caption_tokens, true)?;

        // Step 4: Extract confidence (from logits if available)
        let confidence = 0.85; // TODO: Calculate from actual logits

        let elapsed = start.elapsed();
        println!("💬 Caption generated in {:?}", elapsed);

        Ok(ImageCaption {
            text: caption_text,
            confidence,
            timestamp: std::time::Instant::now(),
        })
    }

    fn encode_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let vision_session = self.vision_session.as_ref().unwrap();

        // Preprocess image (similar to SigLIP but may have different size)
        let preprocessed = self.preprocess_image_for_vision(image)?;

        // Run vision encoder
        let input_tensor = Value::from_array(
            vision_session.allocator(),
            &preprocessed
        )?;

        let outputs = vision_session.run(vec![input_tensor])?;
        let encoding_tensor = &outputs[0];
        let encoding: Vec<f32> = encoding_tensor
            .try_extract()?
            .view()
            .to_slice()?
            .to_vec();

        Ok(encoding)
    }

    fn generate_caption_tokens(&self, image_encoding: &[f32]) -> Result<Vec<i64>> {
        let language_session = self.language_session.as_ref().unwrap();

        // Prepare prompt: <image> Describe this image.
        let prompt_tokens = vec![1, 2, 3]; // TODO: Actual prompt encoding

        // Autoregressive generation
        let mut generated_tokens = Vec::new();
        let max_length = 100; // Max caption length

        for _ in 0..max_length {
            // Prepare inputs: image encoding + current tokens
            let inputs = self.prepare_generation_inputs(
                image_encoding,
                &prompt_tokens,
                &generated_tokens
            )?;

            // Run language model forward pass
            let outputs = language_session.run(inputs)?;

            // Get next token logits
            let logits_tensor = &outputs[0];
            let logits: &[f32] = logits_tensor.try_extract()?.view().to_slice()?;

            // Sample next token (greedy or sampling)
            let next_token = self.sample_next_token(logits)?;

            // Check for end token
            if next_token == 2 { // EOS token
                break;
            }

            generated_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }

    fn sample_next_token(&self, logits: &[f32]) -> Result<i64> {
        // Greedy sampling: Pick highest probability token
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(max_idx as i64)
    }
}
```

**Test**: `test_moondream_caption_generation`

**Performance**: <500ms per caption

---

### Step 4: Visual Question Answering (VQA)

**Goal**: Answer specific questions about image content

```rust
impl MoondreamModel {
    pub fn answer_question(&self, image: &DynamicImage, question: &str) -> Result<VqaResponse> {
        if !self.initialized {
            anyhow::bail!("Moondream model not initialized. Call initialize() first.");
        }

        let start = std::time::Instant::now();

        // Step 1: Encode image
        let image_encoding = self.encode_image(image)?;

        // Step 2: Tokenize question
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let question_tokens = tokenizer.encode(question, false)?
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();

        // Step 3: Generate answer with question conditioning
        let answer_tokens = self.generate_answer_tokens(
            &image_encoding,
            &question_tokens
        )?;

        // Step 4: Decode answer
        let answer_text = tokenizer.decode(&answer_tokens, true)?;

        // Step 5: Calculate confidence
        let confidence = 0.80; // TODO: From actual logits

        let elapsed = start.elapsed();
        println!("❓ Question answered in {:?}", elapsed);

        Ok(VqaResponse {
            answer: answer_text,
            confidence,
            question: question.to_string(),
        })
    }

    fn generate_answer_tokens(
        &self,
        image_encoding: &[f32],
        question_tokens: &[i64]
    ) -> Result<Vec<i64>> {
        let language_session = self.language_session.as_ref().unwrap();

        // Similar to caption generation but with question prefix
        let mut generated_tokens = Vec::new();
        let max_length = 50; // Max answer length

        for _ in 0..max_length {
            let inputs = self.prepare_vqa_inputs(
                image_encoding,
                question_tokens,
                &generated_tokens
            )?;

            let outputs = language_session.run(inputs)?;
            let logits_tensor = &outputs[0];
            let logits: &[f32] = logits_tensor.try_extract()?.view().to_slice()?;

            let next_token = self.sample_next_token(logits)?;

            if next_token == 2 { // EOS
                break;
            }

            generated_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }
}
```

**Test**: `test_moondream_vqa`

**Performance**: <500ms per question

---

### Step 5: Integration with SemanticVision

**Goal**: Expose Moondream through high-level API

```rust
impl SemanticVision {
    /// Generate detailed caption for an image
    pub fn caption_image(&self, image: &DynamicImage) -> Result<ImageCaption> {
        self.moondream.caption_image(image)
    }

    /// Answer a question about an image
    pub fn answer_question(&self, image: &DynamicImage, question: &str) -> Result<VqaResponse> {
        self.moondream.answer_question(image, question)
    }

    /// Get both embedding and caption (full understanding)
    pub fn understand_image(&mut self, image: &DynamicImage) -> Result<(ImageEmbedding, ImageCaption)> {
        let embedding = self.embed_image(image)?;
        let caption = self.caption_image(image)?;
        Ok((embedding, caption))
    }

    /// Multi-modal search: Find images matching text description
    pub fn search_by_description(
        &mut self,
        description: &str,
        candidates: &[&DynamicImage],
        top_k: usize
    ) -> Result<Vec<(usize, f32, String)>> {
        // For each candidate, generate caption and compare to description
        let mut matches: Vec<(usize, f32, String)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, img)| {
                let caption = self.caption_image(img)
                    .unwrap_or_else(|_| ImageCaption {
                        text: "".to_string(),
                        confidence: 0.0,
                        timestamp: std::time::Instant::now(),
                    });

                let similarity = self.text_similarity(&description, &caption.text);
                (idx, similarity, caption.text)
            })
            .collect();

        // Sort by similarity
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(matches.into_iter().take(top_k).collect())
    }

    fn text_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple word overlap similarity (can be improved with embeddings)
        let words1: std::collections::HashSet<_> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}
```

**Test**: `test_semantic_vision_full_understanding`

---

## 🧪 Testing Strategy

### New Tests to Add

```rust
#[test]
fn test_moondream_caption_quality() {
    // Test caption generation on known images
    let mut model = MoondreamModel::new();
    model.initialize().unwrap();

    let red_image = create_test_image(100, 100, [255, 0, 0]);
    let caption = model.caption_image(&red_image).unwrap();

    // Caption should mention color
    assert!(caption.text.to_lowercase().contains("red") ||
            caption.text.to_lowercase().contains("color"),
            "Caption should describe image color");

    assert!(caption.confidence > 0.5, "Caption should have reasonable confidence");
}

#[test]
fn test_moondream_vqa_accuracy() {
    // Test VQA on simple questions
    let mut model = MoondreamModel::new();
    model.initialize().unwrap();

    let red_image = create_test_image(100, 100, [255, 0, 0]);

    let response = model.answer_question(&red_image, "What color is this?").unwrap();

    // Answer should mention red
    assert!(response.answer.to_lowercase().contains("red"),
            "Answer should identify red color");
}

#[test]
fn test_caption_generation_speed() {
    // Verify captioning meets performance target
    let mut model = MoondreamModel::new();
    model.initialize().unwrap();

    let image = create_test_image(100, 100, [128, 128, 128]);

    let start = std::time::Instant::now();
    let _caption = model.caption_image(&image).unwrap();
    let duration = start.elapsed();

    assert!(duration.as_millis() < 1000,
        "Captioning should complete in <1000ms, took {:?}", duration);
}

#[test]
fn test_vqa_response_speed() {
    // Verify VQA meets performance target
    let mut model = MoondreamModel::new();
    model.initialize().unwrap();

    let image = create_test_image(100, 100, [255, 0, 0]);

    let start = std::time::Instant::now();
    let _response = model.answer_question(&image, "What is this?").unwrap();
    let duration = start.elapsed();

    assert!(duration.as_millis() < 1000,
        "VQA should complete in <1000ms, took {:?}", duration);
}

#[test]
fn test_moondream_and_siglip_integration() {
    // Test full semantic vision pipeline
    let mut vision = SemanticVision::new(100);
    vision.initialize().unwrap();

    let image = create_test_image(100, 100, [255, 0, 0]);

    // Get both embedding and caption
    let (embedding, caption) = vision.understand_image(&image).unwrap();

    // Both should be meaningful
    assert!(embedding.vector.iter().any(|&x| x != 0.0),
        "Embedding should not be zero vector");

    assert!(!caption.text.is_empty(),
        "Caption should not be empty");
}
```

### Test Coverage Goals
- Existing tests keep passing ✅
- Add 5+ new tests for Moondream functionality
- Verify caption quality (mentions key features)
- Validate VQA correctness (answers match questions)
- Performance benchmarks (<500ms target)

---

## 📊 Success Criteria

### Functional Requirements ✅
- [ ] Model downloads automatically on first use
- [ ] Vision encoder and language model load successfully
- [ ] Captions describe image content accurately
- [ ] VQA answers are relevant to questions
- [ ] Integration with SigLIP works seamlessly
- [ ] All existing tests still pass

### Performance Requirements 🚀
- [ ] Model download: <10 minutes (1.6GB, first run only)
- [ ] Model loading: <2 seconds
- [ ] Caption generation: <500ms per image
- [ ] VQA inference: <500ms per question
- [ ] Full understanding (embed + caption): <600ms

### Quality Requirements 🎯
- [ ] Captions are coherent and grammatical
- [ ] Captions mention key visual elements
- [ ] VQA answers are relevant to questions
- [ ] Confidence scores are reasonable (0.5-1.0)
- [ ] No hallucinations (making up non-existent details)

---

## 🚧 Implementation Phases

### Phase 4a: Model Download & Loading (1-2 hours)
- Implement dual-model download (vision + language)
- Load both ONNX sessions
- Initialize tokenizer
- **Deliverable**: Models load successfully

### Phase 4b: Caption Generation (2-3 hours)
- Implement image encoding
- Add autoregressive text generation
- Connect vision and language models
- **Deliverable**: Generates captions

### Phase 4c: Visual Question Answering (2 hours)
- Implement question conditioning
- Generate answers with image context
- Handle various question types
- **Deliverable**: Answers questions

### Phase 4d: Integration & Testing (1-2 hours)
- Connect to SemanticVision API
- Add comprehensive tests
- Benchmark performance
- **Deliverable**: Full system working

**Total Estimated Time**: 6-9 hours

---

## 🔄 Fallback Strategy

Current placeholder implementation remains working:
- Returns placeholder text for captions
- Returns placeholder answers for VQA
- All structure tests pass

Migration path:
```rust
pub fn caption_image(&self, image: &DynamicImage) -> Result<ImageCaption> {
    if self.initialized && self.vision_session.is_some() {
        // Real Moondream inference
        self.generate_real_caption(image)
    } else {
        // Fallback to placeholder
        Ok(ImageCaption {
            text: format!("An image ({} x {})",
                image.width(), image.height()),
            confidence: 0.0,
            timestamp: Instant::now(),
        })
    }
}
```

---

## 📚 Dependencies

### Already Available ✅
- `image = "0.25"` - Image manipulation
- `ort = "1.16"` - ONNX Runtime
- `hf-hub = "0.3"` - HuggingFace integration
- `anyhow` - Error handling

### Needs Adding 🆕
- `tokenizers = "0.15"` - Text tokenization (needed for both Moondream and Kokoro)

```toml
[dependencies]
tokenizers = "0.15"
```

---

## 🎯 Revolutionary Impact

### What This Enables

1. **Natural Language Understanding**
   - Sophia can describe what she sees in words
   - Bridges vision and language modalities

2. **Interactive Visual Dialogue**
   - Ask questions about images
   - Get specific information from visual scenes

3. **Semantic Search by Description**
   - "Find images with mountains and sunset"
   - Natural language image retrieval

4. **Multi-Modal Reasoning**
   - Combine visual perception with linguistic knowledge
   - Answer complex questions requiring both

5. **Accessibility Features**
   - Describe images for visually impaired users
   - Automatic alt-text generation

### Synergy with SigLIP

| Capability | SigLIP | Moondream | Together |
|------------|--------|-----------|----------|
| Speed | <100ms | <500ms | <600ms total |
| Semantic Search | ✅ Fast | ❌ Slow | ✅ Best of both |
| Natural Language | ❌ No | ✅ Yes | ✅ Complete |
| Visual Details | ❌ Limited | ✅ Rich | ✅ Detailed |
| Memory Efficient | ✅ 768D | ❌ Heavy | ✅ Cached |

**Strategy**: Use SigLIP for fast semantic search, Moondream for detailed understanding!

---

## 🌟 User Experience Examples

### Scenario 1: Image Description
```
User: "What's in this photo?"
Sophia: [Generates embedding with SigLIP (~100ms)]
Sophia: [Generates caption with Moondream (~500ms)]
Result: "A sunset over mountains with orange and pink clouds in the sky"
```

### Scenario 2: Visual Q&A
```
User: "How many people are in this image?"
Sophia: [Encodes image with Moondream vision encoder]
Sophia: [Generates answer with language model]
Result: "There are three people visible in the image"
```

### Scenario 3: Semantic Search
```
User: "Find photos with dogs playing"
Sophia: [Captions all images with Moondream]
Sophia: [Compares captions to query]
Result: Returns images whose captions mention "dog" and "play"
```

---

## 🚀 Next Steps After Moondream Integration

### Week 13 Day 5: OCR Integration
- Text extraction from images
- Document understanding
- Complements visual understanding

### Week 14+: Advanced Multi-Modal Features
- Visual reasoning chains
- Image-to-code generation
- Multi-image understanding
- Temporal visual understanding (video)

---

## 📝 Notes & Decisions

### Decision 1: Moondream-2 Over LLaVA
- **Why**: Smaller (1.86B vs 7B), faster, good enough quality
- **Trade-off**: Slightly less detailed than larger models
- **Benefit**: Runs on CPU, faster inference

### Decision 2: Separate Vision & Language Models
- **Why**: More flexible, can optimize each separately
- **How**: Vision encoder extracts features, language model generates text
- **Benefit**: Can reuse vision encoding for multiple questions

### Decision 3: Autoregressive Generation
- **Why**: Standard for language models, flexible
- **Trade-off**: Slower than single-shot prediction
- **Benefit**: Can generate variable-length, high-quality text

### Decision 4: Greedy Sampling Default
- **Why**: Deterministic, reproducible, fast
- **Alternative**: Can add temperature sampling later
- **Trade-off**: Less creative than sampling
- **Benefit**: Consistent results, easier to test

### Decision 5: Caption + VQA Over Single Model
- **Why**: More versatile, covers both use cases
- **Implementation**: Shared vision encoder, different prompting
- **Benefit**: One model serves multiple purposes

---

**Status**: 📋 Plan complete, ready for implementation

**Current State**: Architecture working with placeholders, ready for real model

**Priority**: Implement alongside SigLIP for complete vision system

🌊 Sophia will speak about what she sees! 💬👁️
