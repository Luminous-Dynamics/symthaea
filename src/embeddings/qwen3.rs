//! # Qwen3-Embedding-0.6B Integration
//!
//! Integrates Qwen/Qwen3-Embedding-0.6B for semantic embeddings.
//!
//! ## Model Details
//!
//! - **Name**: Qwen3-Embedding-0.6B (Alibaba/Qwen)
//! - **Parameters**: 600M (6x larger than BGE-base)
//! - **Embedding Dimension**: 1024
//! - **Max Sequence Length**: 8192 tokens
//! - **Training**: Contrastive + instruction-tuned
//! - **MTEB Rank**: Near Gemini-level (2025)
//!
//! ## Why Qwen3 over BGE?
//!
//! | Factor | BGE-base-en-v1.5 | Qwen3-Embedding-0.6B |
//! |--------|------------------|----------------------|
//! | Parameters | 109M | 600M |
//! | Embedding Dim | 768 | 1024 |
//! | Max Seq Len | 512 | 8192 |
//! | Multilingual | English-focused | Excellent |
//! | MTEB 2025 | #1 open-source | Near Gemini |
//!
//! ## Usage
//!
//! ```rust,ignore
//! // With embeddings feature enabled:
//! let embedder = Qwen3Embedder::load("models/qwen3-embedding-0.6b")?;
//!
//! // Get dense embedding (1024D)
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
//! # Option 1: Download GGUF (recommended for edge)
//! huggingface-cli download Qwen/Qwen3-Embedding-0.6B-GGUF
//!
//! # Option 2: Export to ONNX via Optimum
//! pip install optimum[exporters] transformers
//! optimum-cli export onnx --model Qwen/Qwen3-Embedding-0.6B models/qwen3-embedding-0.6b/
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Qwen3 embedding dimension (1024D)
pub const QWEN3_DIMENSION: usize = 1024;

/// Maximum sequence length for Qwen3 (8192 tokens)
pub const QWEN3_MAX_SEQ_LEN: usize = 8192;

/// Configuration for Qwen3 embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Path to model directory (contains model.onnx and tokenizer.json)
    pub model_path: String,

    /// Whether to normalize embeddings (recommended for similarity)
    pub normalize: bool,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Mean pooling (recommended for Qwen3)
    pub mean_pooling: bool,

    /// Instruction prefix for retrieval tasks
    pub instruction_prefix: Option<String>,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            model_path: "models/qwen3-embedding-0.6b".to_string(),
            normalize: true,
            max_seq_len: QWEN3_MAX_SEQ_LEN,
            mean_pooling: true,
            instruction_prefix: None,
        }
    }
}

/// Statistics for the embedder
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Qwen3Stats {
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
pub struct Qwen3EmbeddingResult {
    /// Dense embedding vector (1024D)
    pub embedding: Vec<f32>,

    /// Original text (truncated if needed)
    pub text: String,

    /// Whether text was truncated
    pub truncated: bool,

    /// Embedding time in milliseconds
    pub time_ms: f64,
}

// ============================================================================
// FEATURE-GATED IMPLEMENTATION: ort (ONNX Runtime)
// ============================================================================

#[cfg(feature = "embeddings")]
mod ort_impl {
    use super::*;
    use std::sync::Mutex;
    use ort::session::{Session, SessionOutputs, builder::GraphOptimizationLevel};
    use tokenizers::Tokenizer;

    /// Qwen3 Embedding Model using ONNX Runtime (with stub fallback)
    pub struct Qwen3Embedder {
        /// Configuration
        config: Qwen3Config,

        /// ONNX session (None if using stub mode)
        session: Option<Session>,

        /// Tokenizer (None if using stub mode)
        tokenizer: Option<Tokenizer>,

        /// Whether using stub mode (no model available)
        stub_mode: bool,

        /// Statistics
        stats: Mutex<Qwen3Stats>,
    }

    impl Qwen3Embedder {
        /// Create a new Qwen3 embedder (falls back to stub if no model)
        pub fn new(config: Qwen3Config) -> Result<Self> {
            Self::load(&config.model_path)
        }

        /// Load model from path (falls back to stub if model not found)
        pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
            let model_dir = model_path.as_ref();
            let model_file = model_dir.join("model.onnx");
            let tokenizer_file = model_dir.join("tokenizer.json");

            // Check if model exists - if not, use stub mode
            if !model_file.exists() {
                tracing::warn!(
                    "ONNX model not found at {:?}, using stub embeddings",
                    model_file
                );
                return Ok(Self {
                    config: Qwen3Config {
                        model_path: model_dir.to_string_lossy().to_string(),
                        ..Default::default()
                    },
                    session: None,
                    tokenizer: None,
                    stub_mode: true,
                    stats: Mutex::new(Qwen3Stats::default()),
                });
            }

            // Load ONNX model
            let session = Session::builder()
                .context("Failed to create ONNX session builder")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .commit_from_file(&model_file)
                .context(format!("Failed to load ONNX model from {:?}", model_file))?;

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            let config = Qwen3Config {
                model_path: model_dir.to_string_lossy().to_string(),
                ..Default::default()
            };

            Ok(Self {
                config,
                session: Some(session),
                tokenizer: Some(tokenizer),
                stub_mode: false,
                stats: Mutex::new(Qwen3Stats::default()),
            })
        }

        /// Check if using stub mode (no real model)
        pub fn is_stub(&self) -> bool {
            self.stub_mode
        }

        /// Embed text to dense vector (1024D)
        pub fn embed(&mut self, text: &str) -> Result<Qwen3EmbeddingResult> {
            let start = std::time::Instant::now();

            // If in stub mode, use deterministic hash-based embedding
            if self.stub_mode {
                return self.stub_embed(text);
            }

            // Get references to session and tokenizer (unwrap safe due to stub_mode check)
            let session = self.session.as_mut().expect("session should exist when not in stub mode");
            let tokenizer = self.tokenizer.as_ref().expect("tokenizer should exist when not in stub mode");

            // Apply instruction prefix if configured
            let input_text = if let Some(ref prefix) = self.config.instruction_prefix {
                format!("{}{}", prefix, text)
            } else {
                text.to_string()
            };

            // Tokenize
            let encoding = tokenizer
                .encode(&*input_text, true)
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

            // Create input tensors using ort 2.0 API: (shape, data) tuple
            use ort::value::Value;
            let input_ids_value = Value::from_array(([1usize, seq_len], input_ids))
                .context("Failed to create input_ids tensor")?;
            let attention_mask_value = Value::from_array(([1usize, seq_len], attention_mask.clone()))
                .context("Failed to create attention_mask tensor")?;

            let outputs: SessionOutputs = session.run(ort::inputs![
                "input_ids" => input_ids_value,
                "attention_mask" => attention_mask_value,
            ])?;

            // Extract embeddings (last_hidden_state: [1, seq_len, 1024])
            let output_tensor = outputs.get("last_hidden_state")
                .or_else(|| outputs.get("sentence_embedding"))
                .context("No embedding output found")?;

            // ort 2.0 API: try_extract_tensor returns (&Shape, &[T]) tuple
            let (output_shape, output_data) = output_tensor.try_extract_tensor::<f32>()
                .context("Failed to extract f32 tensor")?;

            // Mean pooling over sequence length (considering attention mask)
            let mut embedding = vec![0.0f32; QWEN3_DIMENSION];
            let mut total_weight = 0.0f32;

            // Handle different output shapes
            let shape_dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();
            if shape_dims.len() == 3 {
                // [batch, seq_len, dim] - need pooling
                let dim = shape_dims[2];
                for seq_idx in 0..seq_len.min(shape_dims[1]) {
                    let weight = attention_mask[seq_idx] as f32;
                    total_weight += weight;
                    for dim_idx in 0..QWEN3_DIMENSION.min(dim) {
                        // Flatten 3D index: batch * (seq * dim) + seq * dim + d
                        let idx = seq_idx * dim + dim_idx;
                        embedding[dim_idx] += output_data[idx] * weight;
                    }
                }
            } else if shape_dims.len() == 2 {
                // [batch, dim] - already pooled
                total_weight = 1.0;
                let dim = shape_dims[1];
                for dim_idx in 0..QWEN3_DIMENSION.min(dim) {
                    embedding[dim_idx] = output_data[dim_idx];
                }
            }

            if total_weight > 0.0 && shape_dims.len() == 3 {
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

            Ok(Qwen3EmbeddingResult {
                embedding,
                text: text.to_string(),
                truncated,
                time_ms,
            })
        }

        /// Batch embed multiple texts
        pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Qwen3EmbeddingResult>> {
            let mut results = Vec::with_capacity(texts.len());
            for t in texts {
                results.push(self.embed(t)?);
            }
            Ok(results)
        }

        /// Compute coherence between two texts
        pub fn coherence(&mut self, text_a: &str, text_b: &str) -> Result<f32> {
            let emb_a = self.embed(text_a)?;
            let emb_b = self.embed(text_b)?;

            {
                let mut stats = self.stats.lock().unwrap();
                stats.total_coherence_checks += 1;
            }

            Ok(cosine_similarity(&emb_a.embedding, &emb_b.embedding))
        }

        /// Check if output is coherent with input (hallucination detection)
        pub fn verify_coherence(&mut self, input: &str, output: &str, threshold: f32) -> Result<bool> {
            let coherence = self.coherence(input, output)?;
            Ok(coherence >= threshold)
        }

        /// Semantic similarity between two texts (normalized to [0, 1])
        pub fn similarity(&mut self, text_a: &str, text_b: &str) -> Result<f32> {
            let coh = self.coherence(text_a, text_b)?;
            Ok((coh + 1.0) / 2.0)
        }

        /// Find most similar text from candidates
        pub fn find_most_similar(&mut self, query: &str, candidates: &[&str]) -> Result<(usize, f32)> {
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
        pub fn stats(&self) -> Qwen3Stats {
            self.stats.lock().unwrap().clone()
        }

        /// Get configuration
        pub fn config(&self) -> &Qwen3Config {
            &self.config
        }

        /// Get embedding dimension
        pub fn dimension(&self) -> usize {
            QWEN3_DIMENSION
        }

        /// Stub embedding using deterministic hash (when no model available)
        fn stub_embed(&self, text: &str) -> Result<Qwen3EmbeddingResult> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut embedding = vec![0.0f32; QWEN3_DIMENSION];

            // Generate deterministic pseudo-random embedding from text hash
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let seed = hasher.finish();

            // Use seed to generate embedding values
            let mut state = seed;
            for i in 0..QWEN3_DIMENSION {
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

            // Update stats
            {
                let mut stats = self.stats.lock().unwrap();
                stats.total_embeddings += 1;
            }

            Ok(Qwen3EmbeddingResult {
                embedding,
                text: text.to_string(),
                truncated: text.len() > self.config.max_seq_len * 4,
                time_ms: 0.1,
            })
        }
    }
}

// ============================================================================
// STUB IMPLEMENTATION (no embeddings feature)
// ============================================================================

#[cfg(not(feature = "embeddings"))]
mod stub_impl {
    use super::*;

    /// Qwen3 Embedding Model (stub - requires embeddings feature)
    pub struct Qwen3Embedder {
        config: Qwen3Config,
        stats: Qwen3Stats,
    }

    impl Qwen3Embedder {
        /// Create a new Qwen3 embedder (stub)
        pub fn new(config: Qwen3Config) -> Result<Self> {
            Ok(Self {
                config,
                stats: Qwen3Stats::default(),
            })
        }

        /// Load model from path (stub)
        pub fn load(_model_path: impl AsRef<Path>) -> Result<Self> {
            Self::new(Qwen3Config::default())
        }

        /// Embed text to dense vector (stub - deterministic hash)
        pub fn embed(&self, text: &str) -> Result<Qwen3EmbeddingResult> {
            self.embed_full(text)
        }

        /// Full embed with complete result struct (for trait disambiguation)
        pub fn embed_full(&self, text: &str) -> Result<Qwen3EmbeddingResult> {
            let embedding = self.stub_embed(text);
            Ok(Qwen3EmbeddingResult {
                embedding,
                text: text.to_string(),
                truncated: text.len() > self.config.max_seq_len * 4,
                time_ms: 0.1,
            })
        }

        /// Batch embed multiple texts
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Qwen3EmbeddingResult>> {
            texts.iter().map(|t| self.embed_full(t)).collect()
        }

        /// Compute coherence between two texts
        pub fn coherence(&self, text_a: &str, text_b: &str) -> Result<f32> {
            let emb_a = self.embed_full(text_a)?;
            let emb_b = self.embed_full(text_b)?;
            Ok(cosine_similarity(&emb_a.embedding, &emb_b.embedding))
        }

        /// Check if output is coherent with input
        pub fn verify_coherence(&self, input: &str, output: &str, threshold: f32) -> Result<bool> {
            let coherence = self.coherence(input, output)?;
            Ok(coherence >= threshold)
        }

        /// Semantic similarity between two texts (inherent method)
        pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
            self.compute_similarity(text_a, text_b)
        }

        /// Compute similarity (for trait disambiguation)
        pub fn compute_similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
            let coh = self.coherence(text_a, text_b)?;
            Ok((coh + 1.0) / 2.0)
        }

        /// Find most similar text from candidates
        pub fn find_most_similar(&mut self, query: &str, candidates: &[&str]) -> Result<(usize, f32)> {
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
        pub fn stats(&self) -> &Qwen3Stats {
            &self.stats
        }

        /// Get configuration
        pub fn config(&self) -> &Qwen3Config {
            &self.config
        }

        /// Get embedding dimension
        pub fn dimension(&self) -> usize {
            QWEN3_DIMENSION
        }

        /// Stub embedding using deterministic hash
        fn stub_embed(&self, text: &str) -> Vec<f32> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut embedding = vec![0.0f32; QWEN3_DIMENSION];

            // Generate deterministic pseudo-random embedding from text hash
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let seed = hasher.finish();

            // Use seed to generate embedding values
            let mut state = seed;
            for i in 0..QWEN3_DIMENSION {
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
pub use ort_impl::Qwen3Embedder;

#[cfg(not(feature = "embeddings"))]
pub use stub_impl::Qwen3Embedder;

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

/// Embedding trait for unified interface across models
pub trait TextEmbedder {
    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Embed text to dense vector
    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>>;

    /// Compute similarity between two texts
    fn compute_similarity(&mut self, text_a: &str, text_b: &str) -> Result<f32>;
}

impl TextEmbedder for Qwen3Embedder {
    fn dimension(&self) -> usize {
        QWEN3_DIMENSION
    }

    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(text)?.embedding)
    }

    fn compute_similarity(&mut self, text_a: &str, text_b: &str) -> Result<f32> {
        self.similarity(text_a, text_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_embed() {
        let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

        let emb = embedder.embed("hello world").unwrap();
        assert_eq!(emb.embedding.len(), QWEN3_DIMENSION);

        // Check normalization
        let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deterministic() {
        let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

        let emb1 = embedder.embed("test").unwrap();
        let emb2 = embedder.embed("test").unwrap();

        // Same input should give same output
        assert_eq!(emb1.embedding, emb2.embedding);
    }

    #[test]
    fn test_coherence() {
        let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

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
        let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

        let candidates = ["apple", "banana", "cat", "dog"];
        let (idx, sim) = embedder.find_most_similar("fruit", &candidates).unwrap();

        assert!(idx < candidates.len());
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_dimension() {
        let embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();
        assert_eq!(embedder.dimension(), QWEN3_DIMENSION);
        assert_eq!(QWEN3_DIMENSION, 1024);
    }
}
