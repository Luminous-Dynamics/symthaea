//! ONNX Runtime backend for embedding inference.
//!
//! Provides an alternative to the Burn backend using the ONNX Runtime (via `ort`).
//! Follows the same `Embedder` trait interface. Requires an exported `.onnx` model
//! file and `tokenizer.json`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea_embeddings::qwen3::onnx::{OnnxEmbedder, OnnxEmbedderConfig};
//!
//! let config = OnnxEmbedderConfig {
//!     model_path: "/path/to/model.onnx".into(),
//!     tokenizer_path: "/path/to/tokenizer.json".into(),
//!     ..Default::default()
//! };
//! let mut embedder = OnnxEmbedder::new(config)?;
//! let result = embedder.embed("Hello world")?;
//! ```

use crate::{Embedder, EmbeddingResult};
use anyhow::Result;

/// Configuration for the ONNX embedder.
#[derive(Debug, Clone)]
pub struct OnnxEmbedderConfig {
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Path to the HuggingFace tokenizer.json.
    pub tokenizer_path: String,
    /// Output embedding dimension (1024 for Qwen3-0.6B).
    pub embedding_dim: usize,
    /// Maximum sequence length (8192 for Qwen3).
    pub max_seq_length: usize,
    /// Number of intra-op threads.
    pub num_threads: usize,
}

impl Default for OnnxEmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            embedding_dim: 1024,
            max_seq_length: 8192,
            num_threads: 4,
        }
    }
}

/// ONNX Runtime-based embedder.
pub struct OnnxEmbedder {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    config: OnnxEmbedderConfig,
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder from config.
    pub fn new(config: OnnxEmbedderConfig) -> Result<Self> {
        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(&config.model_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
            config,
        })
    }

    /// Run inference on a single text and return the embedding vector.
    pub fn run_inference(&self, text: &str) -> Result<Vec<f32>> {
        use ndarray::Array2;

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len().min(self.config.max_seq_length);

        // Build ndarray inputs [1, seq_len]
        let input_ids = Array2::from_shape_fn((1, seq_len), |(_, j)| ids[j] as i64);
        let attention_mask = Array2::from_shape_fn((1, seq_len), |(_, j)| mask[j] as i64);

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ]?)?;

        // Extract last_hidden_state [1, seq_len, hidden_dim]
        let hidden_state = outputs[0]
            .try_extract_tensor::<f32>()?;
        let hidden_view = hidden_state.view();
        let shape = hidden_view.shape();
        let hidden_dim = shape[2];
        let out_dim = self.config.embedding_dim.min(hidden_dim);

        // Last-token pooling: find last real token
        let last_idx = mask[..seq_len]
            .iter()
            .rposition(|&m| m == 1)
            .unwrap_or(seq_len.saturating_sub(1));

        let mut embedding: Vec<f32> = (0..out_dim)
            .map(|d| hidden_view[[0, last_idx, d]])
            .collect();

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    /// Get config.
    pub fn config(&self) -> &OnnxEmbedderConfig {
        &self.config
    }
}

impl Embedder for OnnxEmbedder {
    fn dimension(&self) -> usize {
        self.config.embedding_dim
    }

    fn embed(&mut self, text: &str) -> Result<EmbeddingResult> {
        let start = std::time::Instant::now();
        let embedding = self.run_inference(text)?;
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        Ok(EmbeddingResult::new(embedding, "qwen3-onnx").with_time(elapsed))
    }

    fn model_name(&self) -> &str {
        "qwen3-onnx"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_config_creation() {
        let config = OnnxEmbedderConfig {
            model_path: "/tmp/model.onnx".into(),
            tokenizer_path: "/tmp/tokenizer.json".into(),
            embedding_dim: 1024,
            max_seq_length: 8192,
            num_threads: 4,
        };
        assert_eq!(config.embedding_dim, 1024);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxEmbedderConfig::default();
        assert_eq!(config.embedding_dim, 1024);
        assert_eq!(config.max_seq_length, 8192);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    #[ignore] // Requires ONNX model file
    fn test_onnx_session_creation() {
        let model_path =
            std::env::var("ONNX_MODEL_PATH").unwrap_or_else(|_| "/tmp/qwen3.onnx".into());
        let tokenizer_path =
            std::env::var("ONNX_TOKENIZER_PATH").unwrap_or_else(|_| "/tmp/tokenizer.json".into());

        let config = OnnxEmbedderConfig {
            model_path,
            tokenizer_path,
            ..Default::default()
        };
        let embedder = OnnxEmbedder::new(config);
        assert!(embedder.is_ok(), "Should create ONNX session");
    }
}
