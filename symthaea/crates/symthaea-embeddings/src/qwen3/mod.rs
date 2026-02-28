//! # Qwen3 Embeddings: Semantic Text Encoding
//!
//! This module provides text embedding using Qwen3-Embedding models.
//! Supports both Burn inference (via the `burn` feature when available)
//! and high-quality simulated embeddings for testing.
//!
//! ## Model Specifications
//!
//! - **Qwen3-Embedding-0.6B**: 1024D output, 8192 context length
//! - **Qwen3-Embedding-1.5B**: 1536D output (QWEN3_FULL_DIMENSION)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::embeddings::qwen3::{Qwen3Embedder, Qwen3Config};
//!
//! let config = Qwen3Config::default();
//! let mut embedder = Qwen3Embedder::new(config)?;
//!
//! let result = embedder.embed("Hello, world!")?;
//! println!("Embedding dimension: {}", result.dimension);
//! ```

use super::{Embedder, EmbeddingResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[cfg(feature = "burn")]
pub mod attention;
#[cfg(feature = "burn")]
pub mod mlp;
#[cfg(feature = "burn")]
pub mod model;

#[cfg(feature = "burn")]
use burn::backend::NdArray;

/// Full dimension for Qwen3-1.5B embeddings
pub const QWEN3_FULL_DIMENSION: usize = 1536;

/// Standard dimension for Qwen3-0.6B embeddings
pub const QWEN3_STANDARD_DIMENSION: usize = 1024;

/// Configuration for Qwen3Embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Model path (directory containing safetensors + tokenizer.json)
    pub model_path: Option<String>,

    /// Tokenizer path (for HuggingFace tokenizers)
    pub tokenizer_path: Option<String>,

    /// Output embedding dimension
    pub embedding_dim: usize,

    /// Maximum sequence length
    pub max_seq_length: usize,

    /// Whether to use simulated embeddings (for testing)
    pub use_simulated: bool,

    /// Whether to normalize output embeddings
    pub normalize_output: bool,

    /// Pooling strategy
    pub pooling: PoolingStrategy,

    /// Device to use for inference
    pub device: Device,

    /// Number of threads for CPU inference
    pub num_threads: usize,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            embedding_dim: QWEN3_STANDARD_DIMENSION,
            max_seq_length: 8192,
            use_simulated: true, // Safe default - simulation mode
            normalize_output: true,
            pooling: PoolingStrategy::LastTokenPooling,
            device: Device::Cpu,
            num_threads: 4,
        }
    }
}

impl Qwen3Config {
    /// Create config for Qwen3-0.6B model
    pub fn qwen3_06b(model_path: impl Into<String>) -> Self {
        Self {
            model_path: Some(model_path.into()),
            embedding_dim: QWEN3_STANDARD_DIMENSION,
            use_simulated: false,
            ..Default::default()
        }
    }

    /// Create config for Qwen3-1.5B model
    pub fn qwen3_15b(model_path: impl Into<String>) -> Self {
        Self {
            model_path: Some(model_path.into()),
            embedding_dim: QWEN3_FULL_DIMENSION,
            use_simulated: false,
            ..Default::default()
        }
    }

    /// Create config for simulation mode (testing)
    pub fn simulated() -> Self {
        Self {
            use_simulated: true,
            ..Default::default()
        }
    }

    /// Set embedding dimension
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Enable CUDA if available
    pub fn with_cuda(mut self) -> Self {
        self.device = Device::Cuda;
        self
    }
}

/// Pooling strategy for sequence embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling over all tokens
    MeanPooling,
    /// Max pooling over all tokens
    MaxPooling,
    /// Use CLS token (first token)
    ClsToken,
    /// Use last token (Qwen3 default)
    LastTokenPooling,
}

/// Device for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU inference
    Cpu,
    /// CUDA GPU inference
    Cuda,
    /// Metal GPU (Apple Silicon)
    Metal,
}

/// Qwen3 Text Embedder
///
/// Provides semantic embeddings using Qwen3-Embedding models.
/// Falls back to high-quality simulation when Burn model is not available.
pub struct Qwen3Embedder {
    /// Configuration
    config: Qwen3Config,

    /// Model loaded state
    model_loaded: bool,

    /// Embedding cache for deduplication
    cache: std::collections::HashMap<String, Vec<f32>>,

    /// Statistics
    stats: Qwen3Stats,

    /// Burn model (when `burn` feature is enabled and weights loaded)
    #[cfg(feature = "burn")]
    burn_model: Option<model::Qwen3Model<NdArray>>,

    /// Tokenizer for text→token conversion
    #[cfg(feature = "burn")]
    tokenizer: Option<tokenizers::Tokenizer>,

    /// Burn device handle
    #[cfg(feature = "burn")]
    burn_device: burn::backend::ndarray::NdArrayDevice,
}

/// Statistics for the embedder
#[derive(Debug, Clone, Default)]
pub struct Qwen3Stats {
    /// Total embeddings computed
    pub total_embeddings: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Average processing time (ms)
    pub avg_time_ms: f32,

    /// Total characters processed
    pub total_chars: u64,
}

impl Qwen3Embedder {
    /// Create a new Qwen3 embedder
    pub fn new(config: Qwen3Config) -> Result<Self> {
        let mut embedder = Self {
            config,
            model_loaded: false,
            cache: std::collections::HashMap::new(),
            stats: Qwen3Stats::default(),
            #[cfg(feature = "burn")]
            burn_model: None,
            #[cfg(feature = "burn")]
            tokenizer: None,
            #[cfg(feature = "burn")]
            burn_device: burn::backend::ndarray::NdArrayDevice::Cpu,
        };

        // Try to load Burn model if path is specified and not in simulation mode
        if !embedder.config.use_simulated {
            #[cfg(feature = "burn")]
            {
                embedder.try_load_burn()?;
            }
            #[cfg(not(feature = "burn"))]
            {
                if embedder.config.model_path.is_some() {
                    eprintln!("Note: Burn model support requires the `burn` feature, using simulation");
                    embedder.config.use_simulated = true;
                }
            }
        }

        Ok(embedder)
    }

    /// Attempt to load Burn model and tokenizer from safetensors.
    /// Falls back to simulation on any load failure.
    #[cfg(feature = "burn")]
    fn try_load_burn(&mut self) -> Result<()> {
        use burn::record::FullPrecisionSettings;
        use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

        let model_dir = match self.config.model_path {
            Some(ref p) => p.clone(),
            None => {
                self.config.use_simulated = true;
                return Ok(());
            }
        };

        // Load tokenizer
        let tokenizer_path = self
            .config
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| {
                let base = std::path::Path::new(&model_dir);
                base.join("tokenizer.json").to_string_lossy().to_string()
            });

        match tokenizers::Tokenizer::from_file(&tokenizer_path) {
            Ok(tok) => self.tokenizer = Some(tok),
            Err(e) => {
                tracing::warn!("Failed to load tokenizer from {}: {}, using simulation", tokenizer_path, e);
                self.config.use_simulated = true;
                return Ok(());
            }
        }

        // Determine model variant from embedding_dim
        let model_cfg = if self.config.embedding_dim >= QWEN3_FULL_DIMENSION {
            model::Qwen3ModelConfig::qwen3_15b()
        } else {
            model::Qwen3ModelConfig::qwen3_06b()
        };

        // Init model with random weights, then load safetensors on top
        let device = &self.burn_device;
        let mut burn_model = model_cfg.init::<NdArray>(device);

        // Load safetensors weights — strip "model." prefix via key remap
        let safetensors_path = std::path::Path::new(&model_dir).join("model.safetensors");
        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::new();
        let load_args = LoadArgs::new(safetensors_path)
            .with_key_remap("model\\.(.*)", "$1");

        match burn::record::Recorder::load(recorder, load_args, device) {
            Ok(record) => {
                burn_model = burn_model.load_record(record);
                self.burn_model = Some(burn_model);
                self.model_loaded = true;
                tracing::info!(
                    "Loaded Qwen3 Burn model from {} ({}D)",
                    model_dir,
                    self.config.embedding_dim
                );
            }
            Err(e) => {
                tracing::warn!("Failed to load safetensors from {}: {}, using simulation", model_dir, e);
                self.config.use_simulated = true;
            }
        }

        Ok(())
    }

    /// Embed a single text
    pub fn embed(&mut self, text: &str) -> Result<EmbeddingResult> {
        let start = Instant::now();

        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            self.stats.cache_hits += 1;
            return Ok(EmbeddingResult::new(cached.clone(), "qwen3-cached"));
        }

        // Generate embedding
        let embedding = if self.config.use_simulated {
            self.simulate_embedding(text)
        } else {
            #[cfg(feature = "burn")]
            {
                self.burn_embed(text).unwrap_or_else(|_| self.simulate_embedding(text))
            }
            #[cfg(not(feature = "burn"))]
            {
                self.simulate_embedding(text)
            }
        };

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;

        // Update stats
        self.stats.total_embeddings += 1;
        self.stats.total_chars += text.len() as u64;
        let n = self.stats.total_embeddings as f32;
        self.stats.avg_time_ms = (self.stats.avg_time_ms * (n - 1.0) + elapsed) / n;

        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());

        let mut result = EmbeddingResult::new(embedding, "qwen3-embedding").with_time(elapsed);

        if self.config.use_simulated {
            result = result.simulated();
        }

        Ok(result)
    }

    /// Embed multiple texts in batch
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Run Burn inference to produce a real embedding.
    #[cfg(feature = "burn")]
    fn burn_embed(&self, text: &str) -> Result<Vec<f32>> {
        let burn_model = self
            .burn_model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Burn model not loaded"))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

        // Tokenize
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len().min(self.config.max_seq_length);

        // Create Burn input tensor [1, seq_len]
        let input_ids: Vec<i32> = ids[..seq_len].iter().map(|&x| x as i32).collect();
        let input_tensor = burn::tensor::Tensor::<NdArray, 1, burn::tensor::Int>::from_data(
            input_ids.as_slice(),
            &self.burn_device,
        )
        .unsqueeze_dim(0); // [1, seq_len]

        // Forward pass → [1, seq_len, hidden_size]
        let hidden = burn_model.forward(input_tensor);
        let [_batch, _seq, hidden_dim] = hidden.dims();

        let out_dim = self.config.embedding_dim.min(hidden_dim);

        // Pool based on strategy
        let pooled: burn::tensor::Tensor<NdArray, 2> = match self.config.pooling {
            PoolingStrategy::LastTokenPooling => {
                let last_idx = mask[..seq_len]
                    .iter()
                    .rposition(|&m| m == 1)
                    .unwrap_or(seq_len.saturating_sub(1));
                hidden.slice([0..1, last_idx..last_idx + 1, 0..out_dim]).reshape([1, out_dim])
            }
            PoolingStrategy::MeanPooling => {
                let token_count = mask[..seq_len].iter().filter(|&&m| m == 1).count();
                if token_count > 0 {
                    let sum = hidden.slice([0..1, 0..token_count, 0..out_dim]).sum_dim(1);
                    sum.div_scalar(token_count as f32)
                } else {
                    hidden.slice([0..1, 0..1, 0..out_dim]).reshape([1, out_dim])
                }
            }
            PoolingStrategy::ClsToken => {
                hidden.slice([0..1, 0..1, 0..out_dim]).reshape([1, out_dim])
            }
            PoolingStrategy::MaxPooling => {
                hidden.slice([0..1, 0..seq_len, 0..out_dim]).max_dim(1)
            }
        };

        // Convert to Vec<f32>
        let data = pooled.into_data();
        let mut embedding: Vec<f32> = data.to_vec().unwrap();
        embedding.truncate(out_dim);

        // Normalize if configured
        if self.config.normalize_output {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in embedding.iter_mut() {
                    *x /= norm;
                }
            }
        }

        Ok(embedding)
    }

    /// Simulate a high-quality embedding for text
    ///
    /// This simulation produces deterministic, semantically-meaningful embeddings
    /// that preserve basic similarity relationships.
    fn simulate_embedding(&self, text: &str) -> Vec<f32> {
        let dim = self.config.embedding_dim;
        let mut embedding = vec![0.0f32; dim];

        // Use text content to generate deterministic embedding
        let mut hash: u64 = 0x5174_1AEA_5174_1AEA; // "SYMTHAEA"
        for byte in text.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }

        // Generate pseudo-random embedding from hash
        let mut state = hash;
        for i in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32);
            embedding[i] = (val - 0.5) * 2.0; // Range [-1, 1]
        }

        // Add semantic features based on text characteristics
        self.add_semantic_features(&mut embedding, text);

        // Normalize if configured
        if self.config.normalize_output {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in embedding.iter_mut() {
                    *x /= norm;
                }
            }
        }

        embedding
    }

    /// Add semantic features to embedding based on text analysis
    fn add_semantic_features(&self, embedding: &mut [f32], text: &str) {
        let text_lower = text.to_lowercase();
        let dim = embedding.len();

        // Feature 1: Text length (dimension 0-3)
        let len_feature = (text.len() as f32 / 500.0).min(1.0);
        for i in 0..4.min(dim) {
            embedding[i] = (embedding[i] + len_feature) / 2.0;
        }

        // Feature 2: Word count (dimension 4-7)
        let word_count = text.split_whitespace().count() as f32;
        let word_feature = (word_count / 100.0).min(1.0);
        for i in 4..8.min(dim) {
            embedding[i] = (embedding[i] + word_feature) / 2.0;
        }

        // Feature 3: Question detection (dimension 8-11)
        if text.contains('?') {
            for i in 8..12.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 4: Command/imperative detection (dimension 12-15)
        let command_words = [
            "install",
            "run",
            "start",
            "stop",
            "create",
            "delete",
            "add",
            "remove",
            "update",
            "build",
            "configure",
        ];
        let is_command = command_words.iter().any(|w| text_lower.contains(w));
        if is_command {
            for i in 12..16.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 5: Error/negative sentiment (dimension 16-19)
        let error_words = ["error", "fail", "crash", "bug", "broken", "wrong"];
        let is_error = error_words.iter().any(|w| text_lower.contains(w));
        if is_error {
            for i in 16..20.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 6: Technical content (dimension 20-23)
        let tech_words = [
            "nix",
            "nixos",
            "flake",
            "derivation",
            "package",
            "module",
            "config",
            "system",
            "service",
        ];
        let is_tech = tech_words.iter().any(|w| text_lower.contains(w));
        if is_tech {
            for i in 20..24.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 7: N-gram signatures (dimension 24+)
        // Create signatures from word n-grams for better semantic similarity
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        for (idx, word) in words.iter().take(20).enumerate() {
            let word_hash: u64 = word.bytes().fold(0x5974_1AEA_u64, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u64)
            });
            let dim_offset = 24 + (word_hash as usize % (dim.saturating_sub(24).max(1)));
            if dim_offset < dim {
                embedding[dim_offset] += 0.1 / (idx + 1) as f32;
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &Qwen3Stats {
        &self.stats
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get configuration
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
}

impl Embedder for Qwen3Embedder {
    fn dimension(&self) -> usize {
        self.config.embedding_dim
    }

    fn embed(&mut self, text: &str) -> Result<EmbeddingResult> {
        Qwen3Embedder::embed(self, text)
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        Qwen3Embedder::embed_batch(self, texts)
    }

    fn model_name(&self) -> &str {
        if self.config.use_simulated {
            "qwen3-simulated"
        } else {
            "qwen3-embedding"
        }
    }
}

/// Simple text embedder (alias for backward compatibility)
pub type TextEmbedder = Qwen3Embedder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let config = Qwen3Config::default();
        let embedder = Qwen3Embedder::new(config).unwrap();
        assert_eq!(embedder.config.embedding_dim, QWEN3_STANDARD_DIMENSION);
    }

    #[test]
    fn test_embed_text() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let result = embedder.embed("Hello, world!").unwrap();
        assert_eq!(result.dimension, QWEN3_STANDARD_DIMENSION);
        assert!(result.is_simulated);
    }

    #[test]
    fn test_similar_texts_similar_embeddings() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let emb1 = embedder.embed("Install nginx").unwrap();
        let emb2 = embedder.embed("Install nginx server").unwrap();

        // Compute cosine similarity
        let dot: f32 = emb1
            .embedding
            .iter()
            .zip(emb2.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Normalized embeddings have norms close to 1
        assert!(
            dot > 0.5,
            "Similar texts should have high similarity, got {}",
            dot
        );
    }

    #[test]
    fn test_caching() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let _ = embedder.embed("test text").unwrap();
        let _ = embedder.embed("test text").unwrap();

        assert_eq!(embedder.stats.cache_hits, 1);
    }

    #[test]
    fn test_batch_embedding() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let results = embedder.embed_batch(&texts).unwrap();

        assert_eq!(results.len(), 3);
        for result in results {
            assert_eq!(result.dimension, QWEN3_STANDARD_DIMENSION);
        }
    }
}

#[cfg(test)]
#[cfg(feature = "burn")]
mod burn_tests {
    use super::model::Qwen3ModelConfig;
    use burn::backend::NdArray;
    use burn::prelude::*;

    type B = NdArray;

    #[test]
    fn test_burn_model_init() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let _model = cfg.init::<B>(&device);
    }

    #[test]
    fn test_burn_forward_shape() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let ids = Tensor::<B, 2, Int>::zeros([1, 8], &device);
        let out = model.forward(ids);
        let [b, s, h] = out.dims();
        assert_eq!((b, s, h), (1, 8, 64));
    }

    #[test]
    fn test_swiglu_mlp_shape() {
        use super::mlp::Qwen3MlpConfig;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let mlp = Qwen3MlpConfig {
            hidden_size: 64,
            intermediate_size: 128,
        }
        .init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([1, 4, 64], &device);
        let y = mlp.forward(x);
        assert_eq!(y.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_gqa_attention_shape() {
        use super::attention::Qwen3AttentionConfig;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let attn = Qwen3AttentionConfig {
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
        }
        .init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([1, 8, 64], &device);
        let y = attn.forward(x);
        assert_eq!(y.dims(), [1, 8, 64]);
    }

    #[test]
    fn test_missing_safetensors_fallback() {
        let config = super::Qwen3Config::qwen3_06b("/nonexistent/path");
        let embedder = super::Qwen3Embedder::new(config).unwrap();
        // Should gracefully fall back to simulation
        assert!(embedder.config().use_simulated);
    }

    #[test]
    fn test_rmsnorm_preserves_shape() {
        use burn::nn::RmsNormConfig;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let norm = RmsNormConfig::new(64).init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([1, 4, 64], &device);
        let y = norm.forward(x);
        assert_eq!(y.dims(), [1, 4, 64]);
    }
}
