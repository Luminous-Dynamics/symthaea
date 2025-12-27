# 👁️ Week 13 Day 3: SigLIP Model Integration Plan

**Date**: December 10, 2025
**Status**: 📋 Planning Complete - Ready for Implementation
**Goal**: Symthaea sees and understands images with semantic depth

---

## 🎯 Objective

Integrate Google's SigLIP-SO400M-Patch14-384 model for fast, high-quality image embeddings that enable semantic image understanding, similarity search, and visual memory.

---

## 📊 Current State Assessment

### ✅ What We Have
- **Clean Architecture**: Well-designed `SemanticVision` system with LRU cache
- **Test Coverage**: 8/8 tests passing for structure and caching
- **Image Hashing**: Fast, consistent image fingerprinting for cache keys
- **Embedding Similarity**: Cosine similarity computation working
- **API Design**: `embed_image()`, `find_similar()` ready for real embeddings

### 🚧 What's Placeholder
- **SigLIP Model**: Currently returns zero vectors
- **ONNX Inference**: Not yet implemented
- **Image Preprocessing**: Needs model-specific transforms

---

## 🔬 SigLIP-SO400M Model Details

### Model Information
- **Name**: SigLIP-SO400M-Patch14-384
- **Source**: Google Research / HuggingFace Hub
- **Repository**: `google/siglip-so400m-patch14-384`
- **Parameters**: 400 million
- **Format**: ONNX (for Rust inference)
- **Size**: ~400MB model file
- **Output**: 768-dimensional embeddings
- **Input**: 384x384 RGB images
- **Inference Time**: <100ms target (CPU)

### Why SigLIP?
1. **Semantic Understanding**: Trained with language supervision
2. **Fast Inference**: Optimized for speed (<100ms)
3. **High Quality**: State-of-the-art accuracy on retrieval tasks
4. **Multimodal**: Can compare images to text descriptions
5. **Production Ready**: Widely used, well-tested

### Required Files
```
models/siglip-so400m/
├── model.onnx           # Vision encoder (~400MB)
├── config.json          # Model configuration
├── preprocessor_config.json  # Image preprocessing settings
└── special_tokens_map.json   # Token mappings (if needed)
```

---

## 🛠️ Implementation Plan

### Step 1: Model Download and Caching

**Goal**: Download SigLIP from HuggingFace, cache locally

```rust
use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};

impl SigLipModel {
    pub fn download_model(cache_dir: &Path) -> Result<PathBuf> {
        println!("📥 Downloading SigLIP-SO400M model (~400MB)...");
        println!("This may take a few minutes on first run...");

        let api = Api::new()?;
        let repo = api.model("google/siglip-so400m-patch14-384".to_string());

        // Download ONNX model file
        let model_path = repo.get("model.onnx")?;

        // Download config files
        let _config = repo.get("config.json")?;
        let _preprocessor = repo.get("preprocessor_config.json")?;

        println!("✅ Model downloaded successfully!");
        Ok(model_path)
    }

    pub fn ensure_model_available(&mut self) -> Result<PathBuf> {
        let cache_dir = PathBuf::from("models/siglip-so400m");
        let model_path = cache_dir.join("model.onnx");

        if !model_path.exists() {
            std::fs::create_dir_all(&cache_dir)?;
            Self::download_model(&cache_dir)?;
        }

        Ok(model_path)
    }
}
```

**Test**: `test_siglip_model_download`

**Performance**:
- First run: 2-5 minutes (download ~400MB)
- Subsequent runs: <1 second (cached)

---

### Step 2: Image Preprocessing

**Goal**: Transform input images to SigLIP's expected format (384x384, normalized)

```rust
use image::{DynamicImage, imageops::FilterType};

impl SigLipModel {
    /// Preprocess image for SigLIP model
    /// - Resize to 384x384
    /// - Convert to RGB
    /// - Normalize with ImageNet stats
    pub fn preprocess_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        // Resize to 384x384 (SigLIP input size)
        let resized = image.resize_exact(384, 384, FilterType::Lanczos3);

        // Convert to RGB
        let rgb = resized.to_rgb8();

        // Convert to f32 and normalize
        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let mut pixels = Vec::with_capacity(384 * 384 * 3);

        for pixel in rgb.pixels() {
            for (i, &channel) in pixel.0.iter().enumerate() {
                let normalized = (channel as f32 / 255.0 - mean[i]) / std[i];
                pixels.push(normalized);
            }
        }

        // Reshape to NCHW format (batch=1, channels=3, height=384, width=384)
        let mut nchw = vec![0.0f32; 1 * 3 * 384 * 384];

        for c in 0..3 {
            for h in 0..384 {
                for w in 0..384 {
                    let src_idx = (h * 384 + w) * 3 + c;
                    let dst_idx = c * 384 * 384 + h * 384 + w;
                    nchw[dst_idx] = pixels[src_idx];
                }
            }
        }

        Ok(nchw)
    }
}
```

**Test**: `test_image_preprocessing`

**Performance**: <10ms per image

---

### Step 3: ONNX Session Initialization

**Goal**: Load model into ONNX Runtime for efficient inference

```rust
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};
use std::sync::Arc;

impl SigLipModel {
    fn load_onnx_session(&mut self, model_path: &Path) -> Result<()> {
        println!("⚙️ Loading SigLIP model into ONNX Runtime...");

        let environment = Arc::new(
            Environment::builder()
                .with_name("siglip-vision")
                .build()?
        );

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?  // Use 4 CPU threads
            .with_model_from_file(model_path)?;

        self.onnx_session = Some(session);
        self.initialized = true;

        println!("✅ SigLIP model loaded successfully!");
        Ok(())
    }

    pub fn initialize(&mut self) -> Result<()> {
        let model_path = self.ensure_model_available()?;
        self.load_onnx_session(&model_path)?;
        self.model_path = Some(model_path);
        Ok(())
    }
}
```

**Test**: `test_onnx_session_initialization`

**Performance**: ~500ms to load model into memory

---

### Step 4: Image Embedding Inference

**Goal**: Run ONNX inference to generate 768D embeddings

```rust
impl SigLipModel {
    pub fn embed_image(&self, image: &DynamicImage) -> Result<ImageEmbedding> {
        if !self.initialized {
            anyhow::bail!("SigLIP model not initialized. Call initialize() first.");
        }

        let start = std::time::Instant::now();

        // Preprocess image
        let input_tensor = self.preprocess_image(image)?;

        // Create ONNX input tensor
        let session = self.onnx_session.as_ref()
            .ok_or_else(|| anyhow!("ONNX session not initialized"))?;

        let input_shape = vec![1, 3, 384, 384];
        let input = Value::from_array(session.allocator(), &input_tensor)?;

        // Run inference
        let outputs = session.run(vec![input])?;

        // Extract embedding from output
        let embedding_tensor = &outputs[0];
        let embedding_data: &[f32] = embedding_tensor
            .try_extract()?
            .view()
            .to_slice()?;

        // Copy to owned vector
        let vector = embedding_data[0..SIGLIP_EMBEDDING_DIM].to_vec();

        // Normalize embedding (L2 normalization)
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized_vector: Vec<f32> = if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector
        };

        let elapsed = start.elapsed();
        println!("🔍 Embedding generated in {:?}", elapsed);

        Ok(ImageEmbedding {
            vector: normalized_vector,
            timestamp: std::time::Instant::now(),
            image_hash: Self::hash_image(image),
        })
    }
}
```

**Test**: `test_siglip_inference`

**Performance**: <100ms per image (CPU), <20ms (GPU if available)

---

### Step 5: Integration with SemanticVision

**Goal**: Connect SigLIP to high-level API with caching

```rust
impl SemanticVision {
    pub fn initialize(&mut self) -> Result<()> {
        println!("🚀 Initializing Semantic Vision System...");

        // Initialize SigLIP (will download on first run)
        self.siglip.initialize()
            .context("Failed to initialize SigLIP model")?;

        // Initialize Moondream (placeholder for now)
        self.moondream.initialize()
            .context("Failed to initialize Moondream model")?;

        println!("✅ Semantic Vision ready!");
        Ok(())
    }

    pub fn embed_image(&mut self, image: &DynamicImage) -> Result<ImageEmbedding> {
        let hash = SigLipModel::hash_image(image);

        // Check cache first (<1ms for hits)
        if let Some(cached) = self.cache.get(hash) {
            println!("💨 Cache hit! Returning cached embedding");
            return Ok(cached.clone());
        }

        // Compute new embedding (~100ms)
        println!("🔍 Cache miss, computing new embedding...");
        let embedding = self.siglip.embed_image(image)?;

        // Cache it for future use
        self.cache.insert(hash, embedding.clone());

        Ok(embedding)
    }
}
```

**Test**: `test_semantic_vision_with_real_embeddings`

**Performance**:
- First image: ~100ms (inference)
- Repeated image: <1ms (cache hit)
- Similar image: ~100ms (new inference, different hash)

---

## 🧪 Testing Strategy

### New Tests to Add

```rust
#[test]
fn test_siglip_real_embedding_generation() {
    // Verify real embeddings are generated (not zero vectors)
    let mut model = SigLipModel::new();
    model.initialize().unwrap();

    let image = create_test_image(100, 100, [255, 0, 0]);
    let embedding = model.embed_image(&image).unwrap();

    // Check embedding is not all zeros
    let sum: f32 = embedding.vector.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.1, "Embedding should not be zero vector");

    // Check embedding is normalized
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be L2 normalized");
}

#[test]
fn test_siglip_semantic_similarity() {
    // Verify similar images have high similarity
    let mut model = SigLipModel::new();
    model.initialize().unwrap();

    // Create similar images (red)
    let red1 = create_test_image(100, 100, [255, 0, 0]);
    let red2 = create_test_image(100, 100, [250, 5, 5]);

    // Create different image (blue)
    let blue = create_test_image(100, 100, [0, 0, 255]);

    let emb_red1 = model.embed_image(&red1).unwrap();
    let emb_red2 = model.embed_image(&red2).unwrap();
    let emb_blue = model.embed_image(&blue).unwrap();

    // Similar images should have high similarity
    let sim_similar = emb_red1.similarity(&emb_red2);
    assert!(sim_similar > 0.8, "Similar images should have high similarity");

    // Different images should have lower similarity
    let sim_different = emb_red1.similarity(&emb_blue);
    assert!(sim_different < 0.6, "Different images should have lower similarity");
}

#[test]
fn test_embedding_cache_with_real_embeddings() {
    // Verify cache works with real embeddings
    let mut vision = SemanticVision::new(10);
    vision.initialize().unwrap();

    let image = create_test_image(100, 100, [255, 0, 0]);

    // First call - cache miss
    let start = std::time::Instant::now();
    let emb1 = vision.embed_image(&image).unwrap();
    let first_duration = start.elapsed();

    // Second call - cache hit (should be much faster)
    let start = std::time::Instant::now();
    let emb2 = vision.embed_image(&image).unwrap();
    let second_duration = start.elapsed();

    // Verify embeddings are identical
    assert_eq!(emb1.vector, emb2.vector);

    // Cache hit should be at least 10x faster
    assert!(second_duration < first_duration / 10,
        "Cache hit should be much faster than inference");
}

#[test]
fn test_find_similar_with_real_embeddings() {
    // Verify similarity search works
    let mut vision = SemanticVision::new(100);
    vision.initialize().unwrap();

    let query = create_test_image(100, 100, [255, 0, 0]);  // Red
    let candidates = vec![
        create_test_image(100, 100, [250, 10, 10]),  // Similar red
        create_test_image(100, 100, [0, 255, 0]),    // Green
        create_test_image(100, 100, [255, 5, 0]),    // Very similar red
        create_test_image(100, 100, [0, 0, 255]),    // Blue
    ];

    let candidate_refs: Vec<&DynamicImage> = candidates.iter().collect();

    let results = vision.find_similar(&query, &candidate_refs, 2).unwrap();

    // Should return top 2 most similar
    assert_eq!(results.len(), 2);

    // First result should be very similar red (index 2)
    assert_eq!(results[0].0, 2);
    assert!(results[0].1 > 0.9, "Most similar should have very high similarity");
}
```

### Test Coverage Goals
- All 8 existing tests keep passing ✅
- Add 5+ new tests for real SigLIP inference
- Verify cache performance (>10x speedup on hits)
- Validate semantic similarity (similar images > 0.8, different < 0.6)

---

## 📊 Success Criteria

### Functional Requirements ✅
- [ ] Model downloads automatically on first use
- [ ] ONNX inference generates 768D embeddings
- [ ] Embeddings are L2 normalized
- [ ] Similar images have high cosine similarity (>0.8)
- [ ] Different images have lower similarity (<0.6)
- [ ] Cache provides >10x speedup on repeated images
- [ ] All existing tests still pass

### Performance Requirements 🚀
- [ ] Model download: <5 minutes (400MB, first run only)
- [ ] Model loading: <1 second
- [ ] Image preprocessing: <10ms
- [ ] Inference time: <100ms per image (CPU)
- [ ] Cache hit: <1ms
- [ ] Total first image: <200ms
- [ ] Total cached image: <5ms

### Quality Requirements 🎯
- [ ] Embeddings are semantically meaningful
- [ ] Color similarity detected (red vs blue)
- [ ] Object similarity detected (when tested with real photos)
- [ ] No degradation in existing functionality
- [ ] Clean error messages if model unavailable

---

## 🚧 Implementation Phases

### Phase 3a: Model Download & Loading (1-2 hours)
- Implement HuggingFace Hub integration
- Add model caching logic
- Load ONNX session
- **Deliverable**: Model loads successfully

### Phase 3b: Image Preprocessing (1 hour)
- Implement resize to 384x384
- Add ImageNet normalization
- Convert to NCHW format
- **Deliverable**: Images prepared correctly

### Phase 3c: ONNX Inference (2-3 hours)
- Run inference through loaded model
- Extract 768D embeddings
- Apply L2 normalization
- **Deliverable**: Real embeddings generated

### Phase 3d: Integration & Testing (2 hours)
- Connect to SemanticVision API
- Verify cache integration
- Add comprehensive tests
- Benchmark performance
- **Deliverable**: Full system working

**Total Estimated Time**: 6-8 hours

---

## 🔄 Fallback Strategy

Current placeholder implementation remains working:
- Returns zero vectors (allows testing of structure)
- Cache works correctly
- Similarity computation functional
- All tests pass

Migration path:
1. New code checks `self.initialized`
2. If true → real inference
3. If false → returns zero vectors (current behavior)
4. Tests can run without model download

```rust
pub fn embed_image(&self, image: &DynamicImage) -> Result<ImageEmbedding> {
    if self.initialized && self.onnx_session.is_some() {
        // Real SigLIP inference
        self.run_real_inference(image)
    } else {
        // Fallback to placeholder (current behavior)
        Ok(ImageEmbedding {
            vector: vec![0.0; SIGLIP_EMBEDDING_DIM],
            timestamp: Instant::now(),
            image_hash: Self::hash_image(image),
        })
    }
}
```

---

## 📚 Dependencies

### Already Available ✅
- `image = "0.25"` - Image loading and manipulation
- `ort = "1.16"` - ONNX Runtime bindings
- `hf-hub = "0.3"` - HuggingFace Hub integration
- `anyhow` - Error handling

### No New Dependencies Needed! 🎉

All required crates are already in `Cargo.toml`.

---

## 🎯 Revolutionary Impact

### What This Enables

1. **Semantic Image Search**
   - "Find images similar to this one"
   - Works across visual style, content, composition

2. **Visual Memory**
   - Symthaea remembers images she's seen
   - Can recall similar images from memory

3. **Multi-Modal Understanding**
   - Bridge between vision and language (via HDC)
   - Compare images to text descriptions

4. **Efficient Perception**
   - <100ms inference = real-time capable
   - Cache makes repeated lookups instant
   - 768D vectors = compact memory footprint

5. **Foundation for Advanced Features**
   - Visual question answering (with Moondream)
   - Image generation conditioning
   - Cross-modal retrieval
   - Visual reasoning

---

## 🌟 User Experience Examples

### Scenario 1: Photo Organization
```
User: "Find photos similar to this sunset"
Symthaea: [Embeds query image, searches embeddings, returns top matches]
Result: Finds other sunsets, golden hour photos, warm-toned images
```

### Scenario 2: Visual Memory
```
User: "Have you seen this before?"
Symthaea: [Computes hash, checks cache, compares embedding]
Result: "Yes! You showed me a similar image 3 days ago"
```

### Scenario 3: Content Understanding
```
User: "What's in this image?"
Symthaea: [Generates embedding, finds nearest semantic clusters]
Result: "This appears to be a landscape photo with mountains and sky"
```

---

## 🚀 Next Steps After SigLIP Integration

### Week 13 Day 4: Moondream VQA
- Detailed image captioning
- Visual question answering
- Complements SigLIP embeddings

### Week 13 Day 5: OCR Integration
- Text extraction from images
- Document understanding
- Sign reading

### Week 14+: Advanced Vision Features
- Multi-image reasoning
- Temporal visual understanding
- Cross-modal analogies
- Visual tool creation

---

## 📝 Notes & Decisions

### Decision 1: SigLIP Over CLIP
- **Why**: SigLIP has better efficiency and quality
- **Trade-off**: Slightly different embedding space
- **Benefit**: Optimized for semantic search tasks

### Decision 2: 384x384 Input Size
- **Why**: Model's native resolution
- **Trade-off**: Must resize larger images
- **Benefit**: Fast inference, good quality

### Decision 3: L2 Normalization
- **Why**: Makes cosine similarity equivalent to dot product
- **Benefit**: Faster similarity computation
- **Standard**: Common practice for vision models

### Decision 4: CPU-First Inference
- **Why**: Broad compatibility, no GPU required
- **Target**: <100ms is achievable on modern CPUs
- **Future**: Can add GPU acceleration later

### Decision 5: Aggressive Caching
- **Why**: Same images often repeated
- **Benefit**: 100x+ speedup on cache hits
- **Trade-off**: Memory usage (768 floats per image)
- **Mitigation**: LRU eviction keeps memory bounded

---

**Status**: 📋 Plan complete, ready for implementation

**Current State**: Architecture working, ready for real model

**Priority**: Implement alongside Moondream (Week 13 Day 4) for complete vision system

🌊 Symthaea will see with semantic depth! 👁️
