//! # BGE Embedding Model Integration
//!
//! Integrates BAAI/bge-base-en-v1.5 for semantic embeddings.
//!
//! ## Model Details
//!
//! - **Name**: bge-base-en-v1.5 (BAAI)
//! - **Parameters**: 109M (edge-friendly)
//! - **Embedding Dimension**: 768
//! - **Max Sequence Length**: 512 tokens
//! - **Training**: Contrastive learning (superior similarity scores)
//! - **MTEB Rank**: #1 among open-source embeddings (2025)
//!
//! ## Usage
//!
//! ```rust,ignore
//! // With embeddings feature enabled:
//! let embedder = BGEEmbedder::load("models/bge-base-en-v1.5")?;
//!
//! // Get dense embedding
//! let embedding = embedder.embed("Install nginx on NixOS")?;
//!
//! // Check coherence between input and output
//! let coherence = embedder.coherence(
//!     "Install nginx",
//!     "Added nginx to environment.systemPackages"
//! )?;
//!
//! // Detect potential hallucination
//! if coherence < 0.7 {
//!     warn!("Low coherence detected, possible hallucination");
//! }
//! ```
//!
//! ## Model Download
//!
//! ```bash
//! # Download and convert model to ONNX
//! pip install optimum[exporters] transformers
//! optimum-cli export onnx --model BAAI/bge-base-en-v1.5 models/bge-base-en-v1.5/
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// BGE embedding dimension
pub const BGE_DIMENSION: usize = 768;

/// Maximum sequence length for BGE
pub const BGE_MAX_SEQ_LEN: usize = 512;

/// Configuration for BGE embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BGEConfig {
    /// Path to model directory (contains model.onnx and tokenizer.json)
    pub model_path: String,

    /// Whether to normalize embeddings (recommended for similarity)
    pub normalize: bool,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Mean pooling (recommended for BGE)
    pub mean_pooling: bool,
}

impl Default for BGEConfig {
    fn default() -> Self {
        Self {
            model_path: "models/bge-base-en-v1.5".to_string(),
            normalize: true,
            max_seq_len: BGE_MAX_SEQ_LEN,
            mean_pooling: true,
        }
    }
}

/// Statistics for the embedder
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EmbedderStats {
    /// Total embeddings computed
    pub total_embeddings: u64,

    /// Total coherence checks
    pub total_coherence_checks: u64,

    /// Average embedding time (ms)
    pub avg_embedding_time_ms: f64,

    /// Cache hits (if caching enabled)
    pub cache_hits: u64,
}

/// Embedding result with metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Dense embedding vector (768D)
    pub embedding: Vec<f32>,

    /// Original text (truncated if needed)
    pub text: String,

    /// Whether text was truncated
    pub truncated: bool,

    /// Embedding time in milliseconds
    pub time_ms: f64,
}

// ============================================================================
// FEATURE-GATED IMPLEMENTATION: tract-onnx
// ============================================================================

#[cfg(feature = "embeddings")]
mod tract_impl {
    use super::*;
    use std::sync::Arc;
    use tract_onnx::prelude::*;
    use tokenizers::Tokenizer;

    /// BGE Embedding Model using tract-onnx
    pub struct BGEEmbedder {
        /// Configuration
        config: BGEConfig,

        /// ONNX model
        model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,

        /// Tokenizer
        tokenizer: Tokenizer,

        /// Statistics
        stats: std::sync::Mutex<EmbedderStats>,
    }

    impl BGEEmbedder {
        /// Create a new BGE embedder
        pub fn new(config: BGEConfig) -> Result<Self> {
            Self::load(&config.model_path)
        }

        /// Load model from path
        pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
            let model_dir = model_path.as_ref();

            // Load ONNX model
            let model_file = model_dir.join("model.onnx");
            let model = tract_onnx::onnx()
                .model_for_path(&model_file)
                .context(format!("Failed to load ONNX model from {:?}", model_file))?
                .into_optimized()
                .context("Failed to optimize model")?
                .into_runnable()
                .context("Failed to create runnable model")?;

            // Load tokenizer
            let tokenizer_file = model_dir.join("tokenizer.json");
            let tokenizer = Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            let config = BGEConfig {
                model_path: model_dir.to_string_lossy().to_string(),
                ..Default::default()
            };

            Ok(Self {
                config,
                model,
                tokenizer,
                stats: std::sync::Mutex::new(EmbedderStats::default()),
            })
        }

        /// Embed text to dense vector
        pub fn embed(&self, text: &str) -> Result<EmbeddingResult> {
            let start = std::time::Instant::now();

            // Tokenize
            let encoding = self.tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let truncated = encoding.get_ids().len() > self.config.max_seq_len;

            // Prepare input tensors
            let input_ids: Vec<i64> = encoding.get_ids()
                .iter()
                .take(self.config.max_seq_len)
                .map(|&id| id as i64)
                .collect();

            let attention_mask: Vec<i64> = encoding.get_attention_mask()
                .iter()
                .take(self.config.max_seq_len)
                .map(|&m| m as i64)
                .collect();

            let seq_len = input_ids.len();

            // Create tensors
            let input_ids_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
                (1, seq_len),
                input_ids,
            )?.into();

            let attention_mask_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
                (1, seq_len),
                attention_mask.clone(),
            )?.into();

            let token_type_ids: Vec<i64> = vec![0i64; seq_len];
            let token_type_ids_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
                (1, seq_len),
                token_type_ids,
            )?.into();

            // Run inference
            let outputs = self.model.run(tvec![
                input_ids_tensor.into(),
                attention_mask_tensor.into(),
                token_type_ids_tensor.into(),
            ])?;

            // Extract embeddings (last_hidden_state: [1, seq_len, 768])
            let output = outputs[0]
                .to_array_view::<f32>()?;

            // Mean pooling over sequence length (considering attention mask)
            let mut embedding = vec![0.0f32; BGE_DIMENSION];
            let mut total_weight = 0.0f32;

            for seq_idx in 0..seq_len {
                let weight = attention_mask[seq_idx] as f32;
                total_weight += weight;
                for dim_idx in 0..BGE_DIMENSION {
                    embedding[dim_idx] += output[[0, seq_idx, dim_idx]] * weight;
                }
            }

            if total_weight > 0.0 {
                for val in &mut embedding {
                    *val /= total_weight;
                }
            }

            // Normalize if configured
            if self.config.normalize {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    for val in &mut embedding {
                        *val /= norm;
                    }
                }
            }

            let time_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Update stats
            {
                let mut stats = self.stats.lock().unwrap();
                stats.total_embeddings += 1;
                stats.avg_embedding_time_ms =
                    (stats.avg_embedding_time_ms * (stats.total_embeddings - 1) as f64 + time_ms)
                    / stats.total_embeddings as f64;
            }

            Ok(EmbeddingResult {
                embedding,
                text: text.to_string(),
                truncated,
                time_ms,
            })
        }

        /// Batch embed multiple texts
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
            texts.iter().map(|t| self.embed(t)).collect()
        }

        /// Compute coherence between two texts
        pub fn coherence(&self, text_a: &str, text_b: &str) -> Result<f32> {
            let emb_a = self.embed(text_a)?;
            let emb_b = self.embed(text_b)?;

            {
                let mut stats = self.stats.lock().unwrap();
                stats.total_coherence_checks += 1;
            }

            Ok(cosine_similarity(&emb_a.embedding, &emb_b.embedding))
        }

        /// Check if output is coherent with input (hallucination detection)
        pub fn verify_coherence(&self, input: &str, output: &str, threshold: f32) -> Result<bool> {
            let coherence = self.coherence(input, output)?;
            Ok(coherence >= threshold)
        }

        /// Semantic similarity between two texts (normalized to [0, 1])
        pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
            let coh = self.coherence(text_a, text_b)?;
            Ok((coh + 1.0) / 2.0)
        }

        /// Find most similar text from candidates
        pub fn find_most_similar(&self, query: &str, candidates: &[&str]) -> Result<(usize, f32)> {
            let query_emb = self.embed(query)?;

            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;

            for (idx, candidate) in candidates.iter().enumerate() {
                let cand_emb = self.embed(candidate)?;
                let sim = cosine_similarity(&query_emb.embedding, &cand_emb.embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = idx;
                }
            }

            Ok((best_idx, best_sim))
        }

        /// Get embedder statistics
        pub fn stats(&self) -> EmbedderStats {
            self.stats.lock().unwrap().clone()
        }

        /// Get configuration
        pub fn config(&self) -> &BGEConfig {
            &self.config
        }
    }
}

// ============================================================================
// STUB IMPLEMENTATION (no embeddings feature)
// ============================================================================

#[cfg(not(feature = "embeddings"))]
mod stub_impl {
    use super::*;

    /// BGE Embedding Model (stub - requires embeddings feature)
    pub struct BGEEmbedder {
        config: BGEConfig,
        stats: EmbedderStats,
    }

    impl BGEEmbedder {
        /// Create a new BGE embedder (stub)
        pub fn new(config: BGEConfig) -> Result<Self> {
            Ok(Self {
                config,
                stats: EmbedderStats::default(),
            })
        }

        /// Load model from path (stub)
        pub fn load(_model_path: impl AsRef<Path>) -> Result<Self> {
            Self::new(BGEConfig::default())
        }

        /// Embed text to dense vector (stub - deterministic hash)
        pub fn embed(&self, text: &str) -> Result<EmbeddingResult> {
            let embedding = self.stub_embed(text);
            Ok(EmbeddingResult {
                embedding,
                text: text.to_string(),
                truncated: text.len() > self.config.max_seq_len * 4,
                time_ms: 0.1,
            })
        }

        /// Batch embed multiple texts
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
            texts.iter().map(|t| self.embed(t)).collect()
        }

        /// Compute coherence between two texts
        pub fn coherence(&self, text_a: &str, text_b: &str) -> Result<f32> {
            let emb_a = self.embed(text_a)?;
            let emb_b = self.embed(text_b)?;
            Ok(cosine_similarity(&emb_a.embedding, &emb_b.embedding))
        }

        /// Check if output is coherent with input
        pub fn verify_coherence(&self, input: &str, output: &str, threshold: f32) -> Result<bool> {
            let coherence = self.coherence(input, output)?;
            Ok(coherence >= threshold)
        }

        /// Semantic similarity between two texts
        pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
            let coh = self.coherence(text_a, text_b)?;
            Ok((coh + 1.0) / 2.0)
        }

        /// Find most similar text from candidates
        pub fn find_most_similar(&self, query: &str, candidates: &[&str]) -> Result<(usize, f32)> {
            let query_emb = self.embed(query)?;

            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;

            for (idx, candidate) in candidates.iter().enumerate() {
                let cand_emb = self.embed(candidate)?;
                let sim = cosine_similarity(&query_emb.embedding, &cand_emb.embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = idx;
                }
            }

            Ok((best_idx, best_sim))
        }

        /// Get embedder statistics
        pub fn stats(&self) -> &EmbedderStats {
            &self.stats
        }

        /// Get configuration
        pub fn config(&self) -> &BGEConfig {
            &self.config
        }

        /// Stub embedding using deterministic hash
        fn stub_embed(&self, text: &str) -> Vec<f32> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut embedding = vec![0.0f32; BGE_DIMENSION];

            // Generate deterministic pseudo-random embedding from text hash
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let seed = hasher.finish();

            // Use seed to generate embedding values
            let mut state = seed;
            for i in 0..BGE_DIMENSION {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f32) / (u32::MAX as f32);
                embedding[i] = val * 2.0 - 1.0;
            }

            // Normalize
            if self.config.normalize {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    for x in &mut embedding {
                        *x /= norm;
                    }
                }
            }

            embedding
        }
    }
}

// Re-export the appropriate implementation
#[cfg(feature = "embeddings")]
pub use tract_impl::BGEEmbedder;

#[cfg(not(feature = "embeddings"))]
pub use stub_impl::BGEEmbedder;

// ============================================================================
// SHARED UTILITIES
// ============================================================================

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-6 || norm_b < 1e-6 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute L2 (Euclidean) distance between two vectors
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_embed() {
        let embedder = BGEEmbedder::new(BGEConfig::default()).unwrap();

        let emb = embedder.embed("hello world").unwrap();
        assert_eq!(emb.embedding.len(), BGE_DIMENSION);

        // Check normalization
        let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deterministic() {
        let embedder = BGEEmbedder::new(BGEConfig::default()).unwrap();

        let emb1 = embedder.embed("test").unwrap();
        let emb2 = embedder.embed("test").unwrap();

        // Same input should give same output
        assert_eq!(emb1.embedding, emb2.embedding);
    }

    #[test]
    fn test_coherence() {
        let embedder = BGEEmbedder::new(BGEConfig::default()).unwrap();

        // Same text should have coherence ~1.0
        let coh = embedder.coherence("hello", "hello").unwrap();
        assert!((coh - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.01);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.01);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_find_most_similar() {
        let embedder = BGEEmbedder::new(BGEConfig::default()).unwrap();

        let candidates = ["apple", "banana", "cat", "dog"];
        let (idx, sim) = embedder.find_most_similar("fruit", &candidates).unwrap();

        assert!(idx < candidates.len());
        assert!(sim >= -1.0 && sim <= 1.0);
    }
}
