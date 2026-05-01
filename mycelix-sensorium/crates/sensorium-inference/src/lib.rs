// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local NLP inference for Mycelix Sensorium.
//!
//! Replaces Xenova Transformers (onnxruntime-web) with Candle (pure Rust WASM).
//! Provides sentence embeddings for thought similarity in the LUCID domain.
//!
//! # Architecture
//!
//! The inference pipeline runs in the main WASM thread (no Web Worker yet).
//! For production use, this should be moved to a Web Worker to avoid
//! blocking the UI during model loading and inference.
//!
//! # Model
//!
//! Uses a lightweight sentence embedding model (MiniLM-L6-v2 or similar)
//! loaded from a URL. Model weights are cached in IndexedDB after first load.
//!
//! # Usage
//!
//! ```rust,ignore
//! use sensorium_inference::SentenceEncoder;
//!
//! let encoder = SentenceEncoder::load("https://...model.safetensors").await?;
//! let embedding = encoder.encode("Consciousness is substrate-independent")?;
//! let similarity = cosine_similarity(&emb_a, &emb_b);
//! ```

pub mod model;

#[cfg(feature = "candle")]
use candle_core::Device;
use serde::{Deserialize, Serialize};

/// Cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Pairwise similarity matrix for a set of texts.
pub fn similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in i..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

/// Find the k most similar embeddings to a query.
pub fn top_k_similar(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(query, emb)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Placeholder for the full sentence encoder.
///
/// The actual model loading requires:
/// 1. Fetching safetensors weights via `web_sys::Request`
/// 2. Loading tokenizer config
/// 3. Building the model graph with candle-nn
///
/// This is architecture-ready but model loading is deferred to a future session
/// when we can test against specific model files.
#[derive(Debug)]
pub struct SentenceEncoder {
    embedding_dim: usize,
}

impl SentenceEncoder {
    /// Create a CPU-based encoder.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
        }
    }

    /// Encode text into a fixed-dimension embedding.
    ///
    /// Currently returns a deterministic hash-based embedding for testing.
    /// Will be replaced with actual model inference when model loading is wired.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        // Deterministic pseudo-embedding from text hash (for testing only)
        // Replace with actual candle model forward pass
        let mut embedding = vec![0.0_f32; self.embedding_dim];
        let bytes = text.as_bytes();
        for (i, chunk) in embedding.iter_mut().enumerate() {
            let mut hash: u32 = 5381;
            for &b in bytes {
                hash = hash.wrapping_mul(33).wrapping_add(b as u32).wrapping_add(i as u32);
            }
            *chunk = ((hash as f32 / u32::MAX as f32) * 2.0) - 1.0;
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        embedding
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_encoder_deterministic() {
        let enc = SentenceEncoder::new(32);
        let a = enc.encode("hello world");
        let b = enc.encode("hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_encoder_normalized() {
        let enc = SentenceEncoder::new(64);
        let emb = enc.encode("test sentence");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k() {
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![1.0, 0.0, 0.0];
        let top = top_k_similar(&query, &corpus, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // most similar
    }

    #[test]
    fn test_similarity_matrix() {
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let mat = similarity_matrix(&embs);
        assert!((mat[0][0] - 1.0).abs() < 1e-6);
        assert!(mat[0][1].abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_exceeds_corpus() {
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let query = vec![1.0, 0.0];
        let top = top_k_similar(&query, &corpus, 10);
        assert_eq!(top.len(), 2); // k > corpus, returns all
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let embs = vec![vec![1.0, 0.5], vec![0.3, 0.9], vec![0.7, 0.2]];
        let mat = similarity_matrix(&embs);
        for i in 0..3 {
            for j in 0..3 {
                assert!((mat[i][j] - mat[j][i]).abs() < 1e-6, "not symmetric at [{i}][{j}]");
            }
        }
    }

    #[test]
    fn test_encoder_different_texts_differ() {
        let enc = SentenceEncoder::new(32);
        let a = enc.encode("consciousness");
        let b = enc.encode("banana");
        assert_ne!(a, b);
    }
}
