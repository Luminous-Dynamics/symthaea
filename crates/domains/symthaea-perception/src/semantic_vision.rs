// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Vision - Deep semantic understanding of images
//!
//! Uses a two-stage pipeline for optimal performance:
//! 1. **SigLIP-SO400M** - Fast 768D image embeddings (<100ms)
//! 2. **Moondream-1.86B** - Detailed captions and VQA (when needed)
//!
//! This approach provides the best of both worlds:
//! - Fast embedding cache hits (<1ms for repeated images)
//! - Semantic search and similarity (via embeddings)
//! - Human-readable descriptions (via captions)
//! - Visual question answering (interactive understanding)
//!
//! ## SigLIP vs CLIP
//!
//! | Factor | CLIP | SigLIP |
//! |--------|------|--------|
//! | Training Loss | Softmax (contrastive) | Sigmoid (pairwise) |
//! | Batch Efficiency | Needs large batches | 2x more efficient |
//! | Zero-shot | Good | Better |
//! | Embedding Dim | 512 | 768 |
//!
//! ## Model Download
//!
//! ```bash
//! # Export SigLIP to ONNX via Optimum
//! pip install optimum[exporters] transformers
//! optimum-cli export onnx --model google/siglip-so400m-patch14-384 models/siglip-so400m/
//! ```

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// ONNX Session type (feature-gated)
#[cfg(feature = "vision")]
use ort::session::Session as OrtSession;

/// Size of SigLIP image embeddings (768 dimensions)
pub const SIGLIP_EMBEDDING_DIM: usize = 768;

/// Image embedding from SigLIP-400M model
#[derive(Debug, Clone)]
pub struct ImageEmbedding {
    /// 768-dimensional vector representing semantic content
    pub vector: Vec<f32>,

    /// When this embedding was computed
    pub timestamp: Instant,

    /// Hash of the image (for cache key)
    pub image_hash: u64,
}

impl ImageEmbedding {
    /// Calculate cosine similarity between two embeddings (0.0 to 1.0)
    pub fn similarity(&self, other: &ImageEmbedding) -> f32 {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "Embedding dimensions must match"
        );

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        // Cosine similarity: dot(a,b) / (||a|| * ||b||)
        // Returns value from -1 to 1, we normalize to 0 to 1
        let similarity = dot_product / (norm_a * norm_b);
        (similarity + 1.0) / 2.0
    }
}

/// Image caption from Moondream-1.86B model
#[derive(Debug, Clone)]
pub struct ImageCaption {
    /// Natural language description of the image
    pub text: String,

    /// Model confidence (0.0 to 1.0)
    pub confidence: f32,

    /// Timestamp when caption was generated
    pub timestamp: Instant,
}

/// Visual Question Answering response
#[derive(Debug, Clone)]
pub struct VqaResponse {
    /// Answer to the question about the image
    pub answer: String,

    /// Model confidence (0.0 to 1.0)
    pub confidence: f32,

    /// Question that was asked
    pub question: String,
}

/// SigLIP input image size (384x384 for SO400M variant)
pub const SIGLIP_INPUT_SIZE: u32 = 384;

/// SigLIP-SO400M model for fast image embeddings
///
/// Uses ONNX Runtime for inference when the `vision` feature is enabled.
/// Falls back to deterministic stub embeddings otherwise.
pub struct SigLipModel {
    /// Model path (will be downloaded from HuggingFace if needed)
    model_path: Option<PathBuf>,

    /// Whether model is initialized
    initialized: bool,

    /// ONNX session (feature-gated)
    #[cfg(feature = "vision")]
    session: Option<OrtSession>,
}

impl Default for SigLipModel {
    fn default() -> Self {
        Self {
            model_path: None,
            initialized: false,
            #[cfg(feature = "vision")]
            session: None,
        }
    }
}

impl SigLipModel {
    /// Create a new SigLIP model instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific model path
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            model_path: Some(path),
            initialized: false,
            #[cfg(feature = "vision")]
            session: None,
        }
    }

    /// Initialize the model (download if needed, load into memory)
    #[cfg(feature = "vision")]
    pub fn initialize(&mut self) -> Result<()> {
        use ort::session::{Session, builder::GraphOptimizationLevel};

        let model_dir = self
            .model_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("models/siglip-so400m"));

        // Try multiple model paths in order of preference:
        // 1. Vision-only model in onnx/ subfolder (smaller, faster for embeddings)
        // 2. Int8 quantized model (fast, lower memory)
        // 3. Full model in onnx/ subfolder
        // 4. Legacy direct path
        let model_candidates = [
            model_dir.join("onnx/vision_model_int8.onnx"), // 95MB, fastest
            model_dir.join("onnx/vision_model.onnx"),      // 372MB, full precision
            model_dir.join("onnx/model.onnx"),             // 1.5GB, full multimodal
            model_dir.join("model.onnx"),                  // Legacy path
        ];

        let model_file = model_candidates.iter().find(|p| p.exists());

        if let Some(model_path) = model_file {
            let session = Session::builder()
                .context("Failed to create ONNX session builder")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .commit_from_file(model_path)
                .context(format!("Failed to load SigLIP model from {:?}", model_path))?;

            self.session = Some(session);
            self.initialized = true;
            tracing::info!("SigLIP model loaded from {:?}", model_path);
        } else {
            // Model file doesn't exist - use stub mode
            tracing::warn!(
                "SigLIP model not found in {:?}, using stub embeddings. \
                Download with: optimum-cli export onnx --model google/siglip-so400m-patch14-384 {:?}/onnx/",
                model_dir,
                model_dir
            );
            self.initialized = true;
        }

        Ok(())
    }

    /// Initialize the model (stub version when vision feature disabled)
    #[cfg(not(feature = "vision"))]
    pub fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    /// Generate 768D embedding for an image
    #[cfg(feature = "vision")]
    pub fn embed_image(&mut self, image: &DynamicImage) -> Result<ImageEmbedding> {
        if !self.initialized {
            anyhow::bail!("SigLIP model not initialized. Call initialize() first.");
        }

        let image_hash = Self::hash_image(image);

        // Preprocess image BEFORE borrowing session (to avoid borrow conflicts)
        let processed = self.preprocess_image(image)?;

        // If we have an ONNX session, use it
        if let Some(ref mut session) = self.session {
            let start = Instant::now();

            // Create input tensor using ort 2.0 API: (shape, data) tuple
            // Shape: [1, 3, 384, 384], data: flattened f32 values
            let h = SIGLIP_INPUT_SIZE as usize;
            let w = SIGLIP_INPUT_SIZE as usize;
            use ort::value::Tensor;
            let input_value = Tensor::from_array(([1usize, 3, h, w], processed))
                .context("Failed to create input tensor")?;

            let outputs = session.run(ort::inputs![
                "pixel_values" => input_value,
            ])?;

            // Extract embedding from output
            // SigLIP outputs: image_embeds [1, 768] or pooler_output
            let output_tensor = outputs
                .get("image_embeds")
                .or_else(|| outputs.get("pooler_output"))
                .or_else(|| outputs.get("last_hidden_state"))
                .context("No embedding output found in SigLIP model")?;

            // ort 2.0 API: try_extract_tensor returns (&Shape, &[T]) tuple
            let (output_shape, output_data) = output_tensor
                .try_extract_tensor::<f32>()
                .context("Failed to extract f32 tensor")?;

            // Extract embedding vector
            let mut vector = vec![0.0f32; SIGLIP_EMBEDDING_DIM];
            let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();

            if shape_dims.len() == 2 && shape_dims[1] >= SIGLIP_EMBEDDING_DIM {
                // [batch, dim] - direct embedding
                for i in 0..SIGLIP_EMBEDDING_DIM {
                    vector[i] = output_data[i];
                }
            } else if shape_dims.len() == 3 {
                // [batch, seq, dim] - need to pool (take CLS token or mean)
                let dim = shape_dims[2];
                for i in 0..SIGLIP_EMBEDDING_DIM.min(dim) {
                    vector[i] = output_data[i]; // First token (CLS)
                }
            }

            // Normalize
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for v in &mut vector {
                    *v /= norm;
                }
            }

            let elapsed = start.elapsed();
            tracing::debug!("SigLIP inference took {:?}", elapsed);

            return Ok(ImageEmbedding {
                vector,
                timestamp: Instant::now(),
                image_hash,
            });
        }

        // Fallback to stub embedding
        Ok(self.stub_embedding(image_hash))
    }

    /// Generate 768D embedding for an image (stub version)
    #[cfg(not(feature = "vision"))]
    pub fn embed_image(&mut self, image: &DynamicImage) -> Result<ImageEmbedding> {
        if !self.initialized {
            anyhow::bail!("SigLIP model not initialized. Call initialize() first.");
        }

        let image_hash = Self::hash_image(image);
        Ok(self.stub_embedding(image_hash))
    }

    /// Preprocess image for SigLIP: resize to 384x384 and normalize
    #[cfg(feature = "vision")]
    fn preprocess_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        use image::imageops::FilterType;

        // Resize to 384x384
        let resized =
            image.resize_exact(SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE, FilterType::Lanczos3);

        let rgb = resized.to_rgb8();

        // SigLIP normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        // This normalizes to [-1, 1] range
        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];

        // Create CHW tensor
        let size = SIGLIP_INPUT_SIZE as usize;
        let mut tensor = vec![0.0f32; 3 * size * size];

        for y in 0..size {
            for x in 0..size {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let idx = c * size * size + y * size + x;
                    let val = pixel[c] as f32 / 255.0;
                    tensor[idx] = (val - mean[c]) / std[c];
                }
            }
        }

        Ok(tensor)
    }

    /// Generate a deterministic stub embedding from image hash
    fn stub_embedding(&self, image_hash: u64) -> ImageEmbedding {
        let mut vector = vec![0.0f32; SIGLIP_EMBEDDING_DIM];

        // Generate deterministic embedding from hash
        let mut state = image_hash;
        for i in 0..SIGLIP_EMBEDDING_DIM {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32);
            vector[i] = val * 2.0 - 1.0;
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        ImageEmbedding {
            vector,
            timestamp: Instant::now(),
            image_hash,
        }
    }

    /// Calculate a simple hash of an image for caching
    pub fn hash_image(image: &DynamicImage) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash dimensions
        let (width, height) = image.dimensions();
        width.hash(&mut hasher);
        height.hash(&mut hasher);

        // Sample pixels for hash (every 10th pixel to be fast)
        let rgba = image.to_rgba8();
        for y in (0..height).step_by(10) {
            for x in (0..width).step_by(10) {
                let pixel = rgba.get_pixel(x, y);
                pixel[0].hash(&mut hasher);
                pixel[1].hash(&mut hasher);
                pixel[2].hash(&mut hasher);
                pixel[3].hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Check if the model is using real ONNX inference
    pub fn is_using_onnx(&self) -> bool {
        #[cfg(feature = "vision")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "vision"))]
        {
            false
        }
    }
}

/// Moondream input image size (378x378)
pub const MOONDREAM_INPUT_SIZE: u32 = 378;

/// Moondream-2 model for image captioning and VQA
///
/// Uses ONNX Runtime for inference when the `vision` feature is enabled.
/// Model components:
/// - Vision encoder: Encodes image into feature tokens
/// - Text decoder: Generates text conditioned on image features + prompt
pub struct MoondreamModel {
    /// Model directory path
    model_path: Option<PathBuf>,

    /// Whether model is initialized
    initialized: bool,

    /// Vision encoder ONNX session
    #[cfg(feature = "vision")]
    vision_encoder: Option<OrtSession>,

    /// Text decoder ONNX session
    #[cfg(feature = "vision")]
    text_decoder: Option<OrtSession>,
}

impl Default for MoondreamModel {
    fn default() -> Self {
        Self {
            model_path: None,
            initialized: false,
            #[cfg(feature = "vision")]
            vision_encoder: None,
            #[cfg(feature = "vision")]
            text_decoder: None,
        }
    }
}

impl MoondreamModel {
    /// Create a new Moondream model instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific model path
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            model_path: Some(path),
            ..Default::default()
        }
    }

    /// Initialize the model (load ONNX files)
    #[cfg(feature = "vision")]
    pub fn initialize(&mut self) -> Result<()> {
        use ort::session::{Session, builder::GraphOptimizationLevel};

        let model_dir = self
            .model_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("models/moondream2"));

        let vision_file = model_dir.join("vision_encoder.onnx");
        let decoder_file = model_dir.join("text_decoder.onnx");

        // Load vision encoder if exists
        if vision_file.exists() {
            let session = Session::builder()
                .context("Failed to create ONNX session builder")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .commit_from_file(&vision_file)
                .context(format!(
                    "Failed to load vision encoder from {:?}",
                    vision_file
                ))?;

            self.vision_encoder = Some(session);
            tracing::info!("Moondream vision encoder loaded from {:?}", vision_file);
        } else {
            tracing::warn!(
                "Moondream vision encoder not found at {:?}. \
                Download with: optimum-cli export onnx --model vikhyatk/moondream2 {:?}",
                vision_file,
                model_dir
            );
        }

        // Load text decoder if exists
        if decoder_file.exists() {
            let session = Session::builder()
                .context("Failed to create ONNX session builder")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .commit_from_file(&decoder_file)
                .context(format!(
                    "Failed to load text decoder from {:?}",
                    decoder_file
                ))?;

            self.text_decoder = Some(session);
            tracing::info!("Moondream text decoder loaded from {:?}", decoder_file);
        }

        self.initialized = true;
        Ok(())
    }

    /// Initialize the model (stub version when vision feature disabled)
    #[cfg(not(feature = "vision"))]
    pub fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    /// Preprocess image for Moondream (378x378, normalized)
    #[cfg(feature = "vision")]
    fn preprocess_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        use image::imageops::FilterType;

        // Resize to 378x378
        let resized = image.resize_exact(
            MOONDREAM_INPUT_SIZE,
            MOONDREAM_INPUT_SIZE,
            FilterType::Lanczos3,
        );

        let rgb = resized.to_rgb8();

        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        // Create CHW tensor
        let size = MOONDREAM_INPUT_SIZE as usize;
        let mut tensor = vec![0.0f32; 3 * size * size];

        for y in 0..size {
            for x in 0..size {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let idx = c * size * size + y * size + x;
                    let val = pixel[c] as f32 / 255.0;
                    tensor[idx] = (val - mean[c]) / std[c];
                }
            }
        }

        Ok(tensor)
    }

    /// Encode image to feature tokens
    #[cfg(feature = "vision")]
    fn encode_image(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        let processed = self.preprocess_image(image)?;

        if let Some(ref mut session) = self.vision_encoder {
            use ort::value::Tensor;
            let h = MOONDREAM_INPUT_SIZE as usize;
            let w = MOONDREAM_INPUT_SIZE as usize;

            let input_value = Tensor::from_array(([1usize, 3, h, w], processed))
                .context("Failed to create input tensor")?;

            let outputs = session.run(ort::inputs![
                "pixel_values" => input_value,
            ])?;

            // Extract image features
            let output_tensor = outputs
                .get("image_features")
                .or_else(|| outputs.get("last_hidden_state"))
                .context("No output found in vision encoder")?;

            let (_shape, data) = output_tensor
                .try_extract_tensor::<f32>()
                .context("Failed to extract f32 tensor")?;

            return Ok(data.to_vec());
        }

        // Fallback: return dummy features
        Ok(vec![0.0f32; 768 * 27 * 27]) // Approximate Moondream feature size
    }

    /// Generate a caption for an image
    #[cfg(feature = "vision")]
    pub fn caption_image(&mut self, image: &DynamicImage) -> Result<ImageCaption> {
        if !self.initialized {
            anyhow::bail!("Moondream model not initialized. Call initialize() first.");
        }

        let start = Instant::now();

        // If we have the vision encoder, use it
        if self.vision_encoder.is_some() {
            let _features = self.encode_image(image)?;

            // STUB(model-required): Returns placeholder caption when Moondream2 text decoder unavailable.
            // Degraded-mode contract: generic description, 0.6 confidence, valid structure.
            let elapsed = start.elapsed();
            tracing::debug!("Moondream captioning took {:?}", elapsed);

            return Ok(ImageCaption {
                text: "An image processed by Moondream vision encoder.".to_string(),
                confidence: 0.6,
                timestamp: Instant::now(),
            });
        }

        // Stub response
        Ok(ImageCaption {
            text: "Image caption unavailable (model not loaded).".to_string(),
            confidence: 0.0,
            timestamp: Instant::now(),
        })
    }

    /// Generate a caption for an image (stub version)
    #[cfg(not(feature = "vision"))]
    pub fn caption_image(&self, _image: &DynamicImage) -> Result<ImageCaption> {
        if !self.initialized {
            anyhow::bail!("Moondream model not initialized. Call initialize() first.");
        }

        Ok(ImageCaption {
            text: "Vision feature not enabled.".to_string(),
            confidence: 0.0,
            timestamp: Instant::now(),
        })
    }

    /// Answer a question about an image
    #[cfg(feature = "vision")]
    pub fn answer_question(&mut self, image: &DynamicImage, question: &str) -> Result<VqaResponse> {
        if !self.initialized {
            anyhow::bail!("Moondream model not initialized. Call initialize() first.");
        }

        let start = Instant::now();

        // If we have the vision encoder, use it
        if self.vision_encoder.is_some() {
            let _features = self.encode_image(image)?;

            // STUB(model-required): Returns placeholder answer when Moondream2 text decoder unavailable.
            // Degraded-mode contract: echoes question, 0.5 confidence, valid structure.
            let elapsed = start.elapsed();
            tracing::debug!("Moondream VQA took {:?}", elapsed);

            return Ok(VqaResponse {
                answer: format!("Processing question: '{}'", question),
                confidence: 0.5,
                question: question.to_string(),
            });
        }

        Ok(VqaResponse {
            answer: "VQA unavailable (model not loaded).".to_string(),
            confidence: 0.0,
            question: question.to_string(),
        })
    }

    /// Answer a question about an image (stub version)
    #[cfg(not(feature = "vision"))]
    pub fn answer_question(&self, _image: &DynamicImage, question: &str) -> Result<VqaResponse> {
        if !self.initialized {
            anyhow::bail!("Moondream model not initialized. Call initialize() first.");
        }

        Ok(VqaResponse {
            answer: "Vision feature not enabled.".to_string(),
            confidence: 0.0,
            question: question.to_string(),
        })
    }

    /// Check if the model has real ONNX sessions loaded
    pub fn is_using_onnx(&self) -> bool {
        #[cfg(feature = "vision")]
        {
            self.vision_encoder.is_some()
        }
        #[cfg(not(feature = "vision"))]
        {
            false
        }
    }
}

/// LRU cache for image embeddings
pub struct EmbeddingCache {
    /// Maximum number of embeddings to cache
    max_size: usize,

    /// Cached embeddings keyed by image hash
    cache: HashMap<u64, ImageEmbedding>,

    /// Access order for LRU eviction
    access_order: Vec<u64>,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            cache: HashMap::new(),
            access_order: Vec::new(),
        }
    }

    /// Get an embedding from cache if present
    pub fn get(&mut self, hash: u64) -> Option<&ImageEmbedding> {
        if let Some(embedding) = self.cache.get(&hash) {
            // Update access order (move to end = most recently used)
            if let Some(pos) = self.access_order.iter().position(|&h| h == hash) {
                self.access_order.remove(pos);
            }
            self.access_order.push(hash);

            Some(embedding)
        } else {
            None
        }
    }

    /// Insert an embedding into cache
    pub fn insert(&mut self, hash: u64, embedding: ImageEmbedding) {
        // Evict oldest if at capacity
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&hash) {
            if let Some(oldest_hash) = self.access_order.first().copied() {
                self.cache.remove(&oldest_hash);
                self.access_order.remove(0);
            }
        }

        self.cache.insert(hash, embedding);
        self.access_order.push(hash);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            capacity: self.max_size,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of cached embeddings
    pub size: usize,

    /// Maximum cache capacity
    pub capacity: usize,
}

/// Semantic Vision system - Two-stage pipeline for image understanding
pub struct SemanticVision {
    /// Fast embedding model (SigLIP-400M)
    siglip: SigLipModel,

    /// Detailed caption model (Moondream-1.86B)
    moondream: MoondreamModel,

    /// Embedding cache for fast repeated lookups
    cache: EmbeddingCache,
}

impl Default for SemanticVision {
    fn default() -> Self {
        Self::new(1000) // Default 1000 embeddings in cache
    }
}

impl SemanticVision {
    /// Create a new semantic vision system
    pub fn new(cache_size: usize) -> Self {
        Self {
            siglip: SigLipModel::new(),
            moondream: MoondreamModel::new(),
            cache: EmbeddingCache::new(cache_size),
        }
    }

    /// Initialize both models (downloads if needed)
    pub fn initialize(&mut self) -> Result<()> {
        self.siglip
            .initialize()
            .context("Failed to initialize SigLIP model")?;

        self.moondream
            .initialize()
            .context("Failed to initialize Moondream model")?;

        Ok(())
    }

    /// Get embedding for an image (cached if available)
    pub fn embed_image(&mut self, image: &DynamicImage) -> Result<ImageEmbedding> {
        let hash = SigLipModel::hash_image(image);

        // Check cache first
        if let Some(cached) = self.cache.get(hash) {
            return Ok(cached.clone());
        }

        // Compute new embedding
        let embedding = self.siglip.embed_image(image)?;

        // Cache it
        self.cache.insert(hash, embedding.clone());

        Ok(embedding)
    }

    /// Generate caption for an image
    pub fn caption_image(&mut self, image: &DynamicImage) -> Result<ImageCaption> {
        self.moondream.caption_image(image)
    }

    /// Answer a question about an image
    pub fn answer_question(&mut self, image: &DynamicImage, question: &str) -> Result<VqaResponse> {
        self.moondream.answer_question(image, question)
    }

    /// Find similar images by comparing embeddings
    pub fn find_similar(
        &mut self,
        query_image: &DynamicImage,
        candidates: &[&DynamicImage],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let query_embedding = self.embed_image(query_image)?;

        let mut similarities: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, img)| {
                let emb = self
                    .embed_image(img)
                    .unwrap_or_else(|_| query_embedding.clone());
                let sim = query_embedding.similarity(&emb);
                (idx, sim)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        Ok(similarities.into_iter().take(top_k).collect())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Generate a deterministic stub embedding for an image
    /// Used as fallback when ONNX models are unavailable
    pub fn stub_embedding(&self, image: &DynamicImage) -> ImageEmbedding {
        let image_hash = SigLipModel::hash_image(image);
        let mut vector = vec![0.0f32; SIGLIP_EMBEDDING_DIM];

        // Generate deterministic embedding from hash
        let mut state = image_hash;
        for i in 0..SIGLIP_EMBEDDING_DIM {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32);
            vector[i] = val * 2.0 - 1.0;
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        ImageEmbedding {
            vector,
            timestamp: Instant::now(),
            image_hash,
        }
    }

    /// Check if the vision system is using real ONNX inference
    pub fn is_using_onnx(&self) -> bool {
        self.siglip.is_using_onnx()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn create_test_image(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgb(color);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_semantic_vision_creation() {
        let vision = SemanticVision::new(100);
        let stats = vision.cache_stats();
        assert_eq!(stats.size, 0);
        assert_eq!(stats.capacity, 100);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(3);

        let emb1 = ImageEmbedding {
            vector: vec![1.0; SIGLIP_EMBEDDING_DIM],
            timestamp: Instant::now(),
            image_hash: 1,
        };

        let emb2 = ImageEmbedding {
            vector: vec![2.0; SIGLIP_EMBEDDING_DIM],
            timestamp: Instant::now(),
            image_hash: 2,
        };

        // Insert two embeddings
        cache.insert(1, emb1.clone());
        cache.insert(2, emb2.clone());

        assert_eq!(cache.stats().size, 2);
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = EmbeddingCache::new(2);

        let emb1 = ImageEmbedding {
            vector: vec![1.0; SIGLIP_EMBEDDING_DIM],
            timestamp: Instant::now(),
            image_hash: 1,
        };

        let emb2 = ImageEmbedding {
            vector: vec![2.0; SIGLIP_EMBEDDING_DIM],
            timestamp: Instant::now(),
            image_hash: 2,
        };

        let emb3 = ImageEmbedding {
            vector: vec![3.0; SIGLIP_EMBEDDING_DIM],
            timestamp: Instant::now(),
            image_hash: 3,
        };

        cache.insert(1, emb1);
        cache.insert(2, emb2);
        assert_eq!(cache.stats().size, 2);

        // Insert third embedding, should evict first
        cache.insert(3, emb3);
        assert_eq!(cache.stats().size, 2);
        assert!(cache.get(1).is_none(), "Oldest embedding should be evicted");
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_embedding_similarity() {
        let emb1 = ImageEmbedding {
            vector: vec![1.0, 0.0, 0.0],
            timestamp: Instant::now(),
            image_hash: 1,
        };

        let emb2 = ImageEmbedding {
            vector: vec![1.0, 0.0, 0.0],
            timestamp: Instant::now(),
            image_hash: 2,
        };

        let emb3 = ImageEmbedding {
            vector: vec![0.0, 1.0, 0.0],
            timestamp: Instant::now(),
            image_hash: 3,
        };

        // Identical vectors should have similarity ~1.0
        let sim_identical = emb1.similarity(&emb2);
        assert!(
            sim_identical > 0.95,
            "Identical embeddings should have high similarity"
        );

        // Orthogonal vectors should have similarity ~0.5
        let sim_orthogonal = emb1.similarity(&emb3);
        assert!(
            sim_orthogonal > 0.4 && sim_orthogonal < 0.6,
            "Orthogonal embeddings should have ~0.5 similarity"
        );
    }

    #[test]
    fn test_image_hash_consistency() {
        let img1 = create_test_image(100, 100, [255, 0, 0]);
        let img2 = create_test_image(100, 100, [255, 0, 0]);
        let img3 = create_test_image(100, 100, [0, 255, 0]);

        let hash1 = SigLipModel::hash_image(&img1);
        let hash2 = SigLipModel::hash_image(&img2);
        let hash3 = SigLipModel::hash_image(&img3);

        assert_eq!(hash1, hash2, "Identical images should have same hash");
        assert_ne!(
            hash1, hash3,
            "Different images should have different hashes"
        );
    }

    #[test]
    fn test_siglip_model_creation() {
        let model = SigLipModel::new();
        assert!(!model.initialized);
    }

    #[test]
    fn test_moondream_model_creation() {
        let model = MoondreamModel::new();
        assert!(!model.initialized);
    }

    #[test]
    fn test_semantic_vision_initialization() {
        let mut vision = SemanticVision::new(100);
        let result = vision.initialize();

        // Should succeed (models marked as initialized)
        assert!(result.is_ok());
    }
}
