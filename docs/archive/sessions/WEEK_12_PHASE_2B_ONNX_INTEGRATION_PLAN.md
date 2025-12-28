# 🚀 Week 12 Phase 2b: ONNX Model Integration Plan

**Status**: Architecture Complete → Model Integration Ready
**Estimated Time**: 4-6 hours
**Complexity**: Medium-High (ONNX inference, image preprocessing, model downloads)

## 📋 Current Status

✅ **Architecture Complete**:
- Two-stage pipeline design (SigLIP + Moondream)
- LRU embedding cache
- Image similarity comparison
- VQA interface
- 8/8 tests passing

🚧 **Remaining Work**:
- Download ONNX models from HuggingFace
- Implement image preprocessing
- Run actual ONNX inference
- Integration tests with real images

## 🎯 Implementation Phases

### Phase 1: SigLIP-400M Integration (2-3 hours)

**Model Details**:
- **Repository**: `google/siglip-so400m-patch14-384`
- **Model File**: `model.onnx` (~400MB)
- **Input**: 384x384 RGB image tensor
- **Output**: 768D embedding vector

**Implementation Steps**:

#### Step 1.1: Model Download (30 min)
```rust
impl SigLipModel {
    fn download_model() -> Result<PathBuf> {
        let api = Api::new()?;
        let repo = api.model("google/siglip-so400m-patch14-384".to_string());
        let model_path = repo.get("model.onnx")?;
        Ok(model_path)
    }
}
```

**Challenges**:
- Model is ~400MB, first download takes time
- Need to handle download progress/errors gracefully
- Cache model locally after first download

#### Step 1.2: Image Preprocessing (1 hour)
```rust
fn preprocess_image(image: &DynamicImage) -> Result<Vec<f32>> {
    // 1. Resize to 384x384
    let resized = image.resize_exact(384, 384, FilterType::Lanczos3);

    // 2. Convert to RGB (remove alpha if present)
    let rgb = resized.to_rgb8();

    // 3. Normalize pixels: (pixel / 255.0 - mean) / std
    //    ImageNet means: [0.485, 0.456, 0.406]
    //    ImageNet stds:  [0.229, 0.224, 0.225]
    let mut tensor = Vec::with_capacity(3 * 384 * 384);

    for channel in 0..3 {
        for y in 0..384 {
            for x in 0..384 {
                let pixel = rgb.get_pixel(x, y)[channel] as f32 / 255.0;
                let normalized = (pixel - MEAN[channel]) / STD[channel];
                tensor.push(normalized);
            }
        }
    }

    Ok(tensor)
}
```

**Challenges**:
- Tensor shape must match model input: [1, 3, 384, 384]
- Channel ordering: CHW (channels, height, width) not HWC
- Normalization values must match training

#### Step 1.3: ONNX Inference (1 hour)
```rust
impl SigLipModel {
    pub fn embed_image(&self, image: &DynamicImage) -> Result<ImageEmbedding> {
        // 1. Preprocess image to tensor
        let input_tensor = self.preprocess_image(image)?;

        // 2. Create ONNX input
        let shape = vec![1, 3, 384, 384];
        let input = Value::from_array(shape, &input_tensor)?;

        // 3. Run inference
        let outputs: SessionOutputs = self.session.run([input])?;

        // 4. Extract embedding (768D)
        let embedding = outputs["output"]
            .try_extract_tensor::<f32>()?
            .view()
            .iter()
            .copied()
            .collect();

        // 5. Return with metadata
        Ok(ImageEmbedding {
            vector: embedding,
            timestamp: Instant::now(),
            image_hash: Self::hash_image(image),
        })
    }
}
```

**Challenges**:
- ONNX session initialization (one-time cost)
- Error handling for invalid inputs/outputs
- Output tensor extraction and conversion

#### Step 1.4: Testing (30 min)
- Test with various image sizes
- Verify embedding dimensionality (768)
- Benchmark inference time (target <100ms)
- Test cache hit/miss behavior

### Phase 2: Moondream-1.86B Integration (2-3 hours)

**Model Details**:
- **Repository**: `vikhyatk/moondream2`
- **Model Files**:
  - Vision encoder ONNX (~800MB)
  - Text decoder ONNX (~1GB)
- **Input**: Image + optional text prompt
- **Output**: Generated caption or answer

**Implementation Steps**:

#### Step 2.1: Model Download (30 min)
```rust
impl MoondreamModel {
    fn download_models() -> Result<(PathBuf, PathBuf)> {
        let api = Api::new()?;
        let repo = api.model("vikhyatk/moondream2".to_string());

        let vision_encoder = repo.get("vision_encoder.onnx")?;
        let text_decoder = repo.get("text_decoder.onnx")?;

        Ok((vision_encoder, text_decoder))
    }
}
```

**Challenges**:
- Two separate models to download (~1.8GB total)
- Longer download time than SigLIP
- Need to manage both sessions

#### Step 2.2: Caption Generation (1.5 hours)
```rust
impl MoondreamModel {
    pub fn caption_image(&self, image: &DynamicImage) -> Result<ImageCaption> {
        // 1. Encode image to vision features
        let image_features = self.encode_image(image)?;

        // 2. Generate caption tokens autoregressively
        let mut tokens = vec![self.bos_token];
        let mut confidence_sum = 0.0;

        for _ in 0..MAX_CAPTION_LENGTH {
            let next_token = self.generate_next_token(&image_features, &tokens)?;
            tokens.push(next_token.id);
            confidence_sum += next_token.confidence;

            if next_token.id == self.eos_token {
                break;
            }
        }

        // 3. Decode tokens to text
        let text = self.tokenizer.decode(&tokens)?;
        let confidence = confidence_sum / tokens.len() as f32;

        Ok(ImageCaption {
            text,
            confidence,
            timestamp: Instant::now(),
        })
    }
}
```

**Challenges**:
- Autoregressive generation (multiple inference passes)
- Tokenization/detokenization
- Confidence scoring from logits
- Stopping criteria (EOS token)

#### Step 2.3: Visual Question Answering (1 hour)
```rust
impl MoondreamModel {
    pub fn answer_question(&self, image: &DynamicImage, question: &str) -> Result<VqaResponse> {
        // 1. Encode image
        let image_features = self.encode_image(image)?;

        // 2. Tokenize question and create prompt
        let question_tokens = self.tokenizer.encode(question)?;
        let prompt = format!("<image>Question: {}\nAnswer:", question);
        let prompt_tokens = self.tokenizer.encode(&prompt)?;

        // 3. Generate answer conditioned on image + question
        let answer_tokens = self.generate_answer(&image_features, &prompt_tokens)?;
        let answer = self.tokenizer.decode(&answer_tokens)?;

        // 4. Extract confidence
        let confidence = self.calculate_confidence(&answer_tokens)?;

        Ok(VqaResponse {
            answer,
            confidence,
            question: question.to_string(),
        })
    }
}
```

**Challenges**:
- Question conditioning in prompt
- Answer extraction from generated text
- Confidence calculation for multi-token answers

#### Step 2.4: Testing (30 min)
- Test caption generation on various images
- Test VQA with different question types
- Verify answer quality
- Benchmark inference time (target <500ms)

### Phase 3: Integration Testing (1 hour)

**Test Suite**:

#### Real Image Tests
```rust
#[test]
fn test_siglip_real_image() {
    let vision = SemanticVision::new(100);
    let img = image::open("tests/fixtures/cat.jpg").unwrap();

    let embedding = vision.embed_image(&img).unwrap();
    assert_eq!(embedding.vector.len(), 768);
}

#[test]
fn test_moondream_real_caption() {
    let vision = SemanticVision::new(100);
    let img = image::open("tests/fixtures/cat.jpg").unwrap();

    let caption = vision.caption_image(&img).unwrap();
    assert!(caption.text.contains("cat") || caption.text.contains("animal"));
}

#[test]
fn test_moondream_real_vqa() {
    let vision = SemanticVision::new(100);
    let img = image::open("tests/fixtures/cat.jpg").unwrap();

    let response = vision.answer_question(&img, "What animal is this?").unwrap();
    assert!(response.answer.to_lowercase().contains("cat"));
}
```

#### Performance Tests
```rust
#[test]
fn test_siglip_performance() {
    let vision = SemanticVision::new(100);
    let img = image::open("tests/fixtures/cat.jpg").unwrap();

    let start = Instant::now();
    let _ = vision.embed_image(&img).unwrap();
    let duration = start.elapsed();

    assert!(duration < Duration::from_millis(100), "SigLIP too slow: {:?}", duration);
}

#[test]
fn test_cache_performance() {
    let mut vision = SemanticVision::new(100);
    let img = image::open("tests/fixtures/cat.jpg").unwrap();

    // First call (miss)
    let _ = vision.embed_image(&img).unwrap();

    // Second call (hit)
    let start = Instant::now();
    let _ = vision.embed_image(&img).unwrap();
    let duration = start.elapsed();

    assert!(duration < Duration::from_millis(1), "Cache too slow: {:?}", duration);
}
```

#### Similarity Tests
```rust
#[test]
fn test_image_similarity_real() {
    let mut vision = SemanticVision::new(100);

    let cat1 = image::open("tests/fixtures/cat1.jpg").unwrap();
    let cat2 = image::open("tests/fixtures/cat2.jpg").unwrap();
    let dog = image::open("tests/fixtures/dog.jpg").unwrap();

    let emb_cat1 = vision.embed_image(&cat1).unwrap();
    let emb_cat2 = vision.embed_image(&cat2).unwrap();
    let emb_dog = vision.embed_image(&dog).unwrap();

    let sim_cats = emb_cat1.similarity(&emb_cat2);
    let sim_cat_dog = emb_cat1.similarity(&emb_dog);

    assert!(sim_cats > sim_cat_dog, "Similar images should be more similar");
}
```

### Phase 4: Performance Optimization (Optional, 1-2 hours)

#### Optimizations:
1. **Batch Processing**: Process multiple images in one inference pass
2. **Model Quantization**: Use INT8 quantized models for 4x speedup
3. **Cache Tuning**: Adjust cache size based on memory constraints
4. **GPU Acceleration**: Use CUDA/TensorRT if available

## 🔧 Dependencies Required

**Already in Cargo.toml**:
- `ort = "2.0.0-rc.10"` ✅
- `hf-hub = "0.3"` ✅
- `image = "0.24"` ✅

**May Need to Add**:
- Tokenizer crate for Moondream (e.g., `tokenizers = "0.15"`)
- Better progress indicators for downloads

## ⚠️ Challenges & Considerations

### Model Download
- **Size**: ~2.2GB total (400MB SigLIP + 1.8GB Moondream)
- **Time**: 5-15 minutes on first run (depends on internet speed)
- **Storage**: Need persistent cache directory
- **Fallback**: Graceful degradation if models unavailable

### ONNX Runtime
- **Platform**: Need ONNX Runtime installed on system
- **CPU vs GPU**: CPU inference by default, GPU optional
- **Threading**: Configure threads for optimal performance

### Image Preprocessing
- **Normalization**: Must match training exactly
- **Tensor Shape**: CHW not HWC ordering
- **Data Type**: f32 precision

### Moondream Complexity
- **Autoregressive**: Multiple inference passes
- **Tokenization**: Need compatible tokenizer
- **Long Sequences**: Caption generation can be slow
- **Memory**: Large decoder model

## 📈 Expected Performance

### SigLIP-400M
- **First Inference**: ~100-200ms (CPU)
- **Cached**: <1ms
- **GPU**: ~20-50ms
- **Batch**: ~150ms for 8 images

### Moondream-1.86B
- **Caption Generation**: ~500ms-1s (CPU, ~20 tokens)
- **VQA**: ~600ms-1.2s (CPU, question + answer)
- **GPU**: ~100-200ms

### Cache
- **Hit Rate**: Expected 60-80% for typical usage
- **Memory**: ~3KB per cached embedding (768 * 4 bytes)
- **Max Cache**: 1000 embeddings = ~3MB

## 🎯 Success Criteria

✅ **SigLIP Integration**:
- Model downloads successfully
- Generates 768D embeddings
- Inference <100ms (CPU)
- Cache hits <1ms
- Tests pass with real images

✅ **Moondream Integration**:
- Models download successfully
- Generates coherent captions
- Answers questions accurately
- Inference <1s (CPU)
- Tests pass with real images

✅ **Integration Tests**:
- 5+ tests with real images
- Performance benchmarks passing
- Similarity tests working
- Cache behavior validated

## 🚀 Next Steps

**To proceed with implementation**:
1. Create test fixture directory with sample images
2. Implement SigLIP download and inference (Phase 1)
3. Test and benchmark SigLIP
4. Implement Moondream download and inference (Phase 2)
5. Test and benchmark Moondream
6. Add comprehensive integration tests (Phase 3)
7. Optimize if needed (Phase 4)

**Estimated total time**: 4-6 hours of focused implementation

---

**Decision Point**: This is a significant implementation task. Before proceeding, consider:
- Is ONNX inference needed immediately, or can it wait?
- Should we move to Phase 2c (OCR) first?
- Is the architecture validation sufficient for now?

The architecture is complete and ready - the ONNX integration is "just" wiring up the models to the existing interfaces.

🌊 Ready to implement when the time is right!
