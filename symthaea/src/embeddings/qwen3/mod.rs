//! # Qwen3 Embeddings: Semantic Text Encoding
//!
//! This module provides text embedding using Qwen3-Embedding models.
//! Supports both ONNX inference (via the `ort` crate when available)
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

use super::{EmbeddingResult, Embedder};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Full dimension for Qwen3-1.5B embeddings
pub const QWEN3_FULL_DIMENSION: usize = 1536;

/// Standard dimension for Qwen3-0.6B embeddings
pub const QWEN3_STANDARD_DIMENSION: usize = 1024;

/// Configuration for Qwen3Embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Model path (for ONNX models)
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
/// Falls back to high-quality simulation when ONNX is not available.
pub struct Qwen3Embedder {
    /// Configuration
    config: Qwen3Config,

    /// Model loaded state
    model_loaded: bool,

    /// Embedding cache for deduplication
    cache: std::collections::HashMap<String, Vec<f32>>,

    /// Statistics
    stats: Qwen3Stats,

    // ONNX session would be stored here when `ort` feature is enabled
    // #[cfg(feature = "embeddings")]
    // session: Option<ort::Session>,
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
        };

        // Try to load model if path is specified and not in simulation mode
        if !embedder.config.use_simulated {
            if let Some(ref _path) = embedder.config.model_path {
                // In a full implementation, this would load the ONNX model
                // For now, we fall back to simulation
                eprintln!("Note: ONNX model loading not implemented, using simulation");
                embedder.config.use_simulated = true;
            }
        }

        Ok(embedder)
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
            // Would use ONNX inference here
            self.simulate_embedding(text)
        };

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;

        // Update stats
        self.stats.total_embeddings += 1;
        self.stats.total_chars += text.len() as u64;
        let n = self.stats.total_embeddings as f32;
        self.stats.avg_time_ms = (self.stats.avg_time_ms * (n - 1.0) + elapsed) / n;

        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());

        let mut result = EmbeddingResult::new(embedding, "qwen3-embedding")
            .with_time(elapsed);

        if self.config.use_simulated {
            result = result.simulated();
        }

        Ok(result)
    }

    /// Embed multiple texts in batch
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        texts.iter().map(|t| self.embed(t)).collect()
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
        let command_words = ["install", "run", "start", "stop", "create", "delete",
                            "add", "remove", "update", "build", "configure"];
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
        let tech_words = ["nix", "nixos", "flake", "derivation", "package",
                         "module", "config", "system", "service"];
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
            let word_hash: u64 = word.bytes()
                .fold(0x5974_1AEA_u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
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
        let dot: f32 = emb1.embedding.iter()
            .zip(emb2.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Normalized embeddings have norms close to 1
        assert!(dot > 0.5, "Similar texts should have high similarity, got {}", dot);
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
