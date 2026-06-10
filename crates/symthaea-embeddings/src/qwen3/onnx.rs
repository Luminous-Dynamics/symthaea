// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

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
    /// Pooling strategy for extracting a single embedding from the hidden state sequence.
    pub pooling: super::PoolingStrategy,
}

impl Default for OnnxEmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            embedding_dim: 1024,
            max_seq_length: 8192,
            num_threads: 4,
            pooling: super::PoolingStrategy::LastTokenPooling,
        }
    }
}

/// ONNX Runtime-based embedder.
pub struct OnnxEmbedder {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    config: OnnxEmbedderConfig,
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder from config.
    pub fn new(config: OnnxEmbedderConfig) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
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
    pub fn run_inference(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len().min(self.config.max_seq_length);

        // Build inputs as (shape, Vec<T>) — avoids ndarray version mismatch with ort
        let input_ids_data: Vec<i64> = ids[..seq_len].iter().map(|&x| x as i64).collect();
        let attention_mask_data: Vec<i64> = mask[..seq_len].iter().map(|&x| x as i64).collect();

        let shape = vec![1i64, seq_len as i64];
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids_data))?;
        let attention_mask_tensor = Tensor::from_array((shape, attention_mask_data))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;

        // Extract last_hidden_state [1, seq_len, hidden_dim]
        let (hidden_shape, hidden_data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden_dim = hidden_shape[2] as usize;
        let out_dim = self.config.embedding_dim.min(hidden_dim);

        let mask_slice: Vec<u32> = mask[..seq_len].to_vec();
        let mut embedding = Self::pool_sequence(
            &hidden_data,
            hidden_dim,
            &mask_slice,
            seq_len,
            out_dim,
            self.config.pooling,
        );

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    /// Run batch inference on multiple texts and return embedding vectors.
    ///
    /// Tokenizes all texts, pads to max sequence length with attention_mask=0,
    /// runs a single `session.run()`, then extracts per-sequence embeddings
    /// using the configured pooling strategy.
    pub fn run_inference_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(*t, true)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_seq = self.config.max_seq_length;
        let batch_size = encodings.len();

        // Determine padded sequence length
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(max_seq))
            .max()
            .unwrap_or(0);

        // Build flattened [batch_size * max_len] tensors with padding
        let mut flat_ids = Vec::with_capacity(batch_size * max_len);
        let mut flat_mask = Vec::with_capacity(batch_size * max_len);
        let mut seq_lengths = Vec::with_capacity(batch_size);
        let mut masks_per_seq: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let seq_len = ids.len().min(max_seq);
            seq_lengths.push(seq_len);

            // Real tokens
            flat_ids.extend(ids[..seq_len].iter().map(|&x| x as i64));
            flat_mask.extend(mask[..seq_len].iter().map(|&x| x as i64));
            masks_per_seq.push(mask[..seq_len].to_vec());

            // Padding
            for _ in seq_len..max_len {
                flat_ids.push(0i64);
                flat_mask.push(0i64);
            }
            // Pad mask for per-seq storage
            let last = masks_per_seq.last_mut().unwrap();
            last.resize(max_len, 0);
        }

        let shape = vec![batch_size as i64, max_len as i64];
        let input_ids_tensor = Tensor::from_array((shape.clone(), flat_ids))?;
        let attention_mask_tensor = Tensor::from_array((shape, flat_mask))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;

        // Extract [batch_size, max_len, hidden_dim]
        let (hidden_shape, hidden_data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden_dim = hidden_shape[2] as usize;
        let out_dim = self.config.embedding_dim.min(hidden_dim);

        // Pool each sequence individually
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let offset = i * max_len * hidden_dim;
            let seq_hidden = &hidden_data[offset..offset + max_len * hidden_dim];

            let mut embedding = Self::pool_sequence(
                seq_hidden,
                hidden_dim,
                &masks_per_seq[i],
                seq_lengths[i],
                out_dim,
                self.config.pooling,
            );

            // L2 normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in embedding.iter_mut() {
                    *x /= norm;
                }
            }
            results.push(embedding);
        }

        Ok(results)
    }

    /// Extract a single embedding vector from a sequence hidden state using the given pooling strategy.
    ///
    /// `hidden_data`: row-major `[seq_len, hidden_dim]` slice
    /// `mask`: attention mask for this sequence (1 = real, 0 = padding)
    /// `seq_len`: number of tokens in this sequence (before padding)
    /// `out_dim`: output embedding dimension (may be < hidden_dim for Matryoshka)
    fn pool_sequence(
        hidden_data: &[f32],
        hidden_dim: usize,
        mask: &[u32],
        seq_len: usize,
        out_dim: usize,
        pooling: super::PoolingStrategy,
    ) -> Vec<f32> {
        match pooling {
            super::PoolingStrategy::LastTokenPooling => {
                let last_idx = mask[..seq_len]
                    .iter()
                    .rposition(|&m| m == 1)
                    .unwrap_or(seq_len.saturating_sub(1));
                let offset = last_idx * hidden_dim;
                hidden_data[offset..offset + out_dim].to_vec()
            }
            super::PoolingStrategy::ClsToken => {
                // First token (position 0)
                hidden_data[..out_dim].to_vec()
            }
            super::PoolingStrategy::MeanPooling => {
                let token_count = mask[..seq_len].iter().filter(|&&m| m == 1).count();
                if token_count == 0 {
                    return hidden_data[..out_dim].to_vec();
                }
                let mut mean = vec![0.0f32; out_dim];
                for t in 0..seq_len {
                    if mask[t] == 1 {
                        let offset = t * hidden_dim;
                        for d in 0..out_dim {
                            mean[d] += hidden_data[offset + d];
                        }
                    }
                }
                for d in 0..out_dim {
                    mean[d] /= token_count as f32;
                }
                mean
            }
            super::PoolingStrategy::MaxPooling => {
                let mut maxes = vec![f32::NEG_INFINITY; out_dim];
                for t in 0..seq_len {
                    if mask[t] == 1 {
                        let offset = t * hidden_dim;
                        for d in 0..out_dim {
                            if hidden_data[offset + d] > maxes[d] {
                                maxes[d] = hidden_data[offset + d];
                            }
                        }
                    }
                }
                // If no tokens had mask=1, fall back to zeros
                for v in &mut maxes {
                    if v.is_infinite() {
                        *v = 0.0;
                    }
                }
                maxes
            }
        }
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

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        let start = std::time::Instant::now();
        let embeddings = self.run_inference_batch(texts)?;
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        let per_item = if texts.is_empty() {
            0.0
        } else {
            elapsed / texts.len() as f32
        };
        Ok(embeddings
            .into_iter()
            .map(|emb| EmbeddingResult::new(emb, "qwen3-onnx").with_time(per_item))
            .collect())
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
            pooling: super::super::PoolingStrategy::LastTokenPooling,
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
        assert_eq!(
            config.pooling,
            super::super::PoolingStrategy::LastTokenPooling
        );
    }

    #[test]
    fn test_onnx_pooling_config() {
        use super::super::PoolingStrategy;

        let strategies = [
            PoolingStrategy::LastTokenPooling,
            PoolingStrategy::ClsToken,
            PoolingStrategy::MeanPooling,
            PoolingStrategy::MaxPooling,
        ];
        for strategy in strategies {
            let config = OnnxEmbedderConfig {
                pooling: strategy,
                ..Default::default()
            };
            assert_eq!(config.pooling, strategy);
        }
    }

    #[test]
    fn test_onnx_default_is_last_token() {
        let config = OnnxEmbedderConfig::default();
        assert_eq!(
            config.pooling,
            super::super::PoolingStrategy::LastTokenPooling
        );
    }

    #[test]
    fn test_onnx_batch_config_with_matryoshka() {
        let config = OnnxEmbedderConfig {
            embedding_dim: 256,
            pooling: super::super::PoolingStrategy::MeanPooling,
            ..Default::default()
        };
        assert_eq!(config.embedding_dim, 256);
        assert_eq!(config.pooling, super::super::PoolingStrategy::MeanPooling);
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

    #[test]
    #[ignore] // Requires ONNX model file
    fn test_onnx_batch_matches_sequential() {
        let model_path =
            std::env::var("ONNX_MODEL_PATH").unwrap_or_else(|_| "/tmp/qwen3.onnx".into());
        let tokenizer_path =
            std::env::var("ONNX_TOKENIZER_PATH").unwrap_or_else(|_| "/tmp/tokenizer.json".into());

        let config = OnnxEmbedderConfig {
            model_path,
            tokenizer_path,
            ..Default::default()
        };
        let mut embedder = OnnxEmbedder::new(config).unwrap();

        let texts = &["Hello world", "The quick brown fox", "Quantum computing"];

        // Sequential
        let sequential: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| embedder.run_inference(t).unwrap())
            .collect();

        // Batch
        let batch = embedder.run_inference_batch(texts).unwrap();
        assert_eq!(batch.len(), sequential.len());

        for (i, (s, b)) in sequential.iter().zip(batch.iter()).enumerate() {
            let dot: f32 = s.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot > 0.999,
                "Batch vs sequential mismatch at {i}: cosine={dot:.6}"
            );
        }
    }
}
