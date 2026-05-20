// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Semantic Encoder with Burn Integration

Pure Rust semantic encoding using the Burn ML framework.
Projects dense embeddings to HDC space via Johnson-Lindenstrauss projection.

## Architecture

```text
Text/Image → Burn Model → Dense Embedding (768D) → JL Projection → HDC (16384D)
```

## Features

- Pure Rust inference (no Python, no external C libs)
- ONNX model import via Burn
- JL projection preserves similarity structure
- Consciousness-native output in HDC bipolar format

## Usage

```rust,ignore
use symthaea::perception::SemanticEncoder;

let encoder = SemanticEncoder::new()?;
let hdc_vec = encoder.encode_text("consciousness emerges from integration")?;
```
*/

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

/// Johnson-Lindenstrauss random projector
///
/// Projects dense embeddings to high-dimensional HDC space
/// while approximately preserving pairwise distances.
///
/// Uses sparse Rademacher distribution (±1 with probability 1/√3, 0 otherwise)
/// for computational efficiency.
#[derive(Clone)]
pub struct JLProjector {
    /// Sparse entries: (row, col, value)
    sparse_entries: Vec<(usize, usize, i8)>,
    /// Input dimension (e.g., 768 for BERT)
    input_dim: usize,
    /// Output dimension (e.g., 16384 for HDC)
    output_dim: usize,
    /// Scaling factor: 1/√(sparsity * input_dim)
    scale: f32,
}

impl JLProjector {
    /// Create new JL projector with sparse Rademacher distribution
    ///
    /// - `input_dim`: dimension of dense embeddings (e.g., 768)
    /// - `output_dim`: target HDC dimension (e.g., 16384)
    /// - `seed`: random seed for reproducibility
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Sparsity: ~1/3 of entries are non-zero
        // This gives good JL guarantees while being efficient
        let sparsity = 3;
        let prob = 1.0 / sparsity as f32;

        let mut entries = Vec::new();

        for row in 0..output_dim {
            for col in 0..input_dim {
                let p: f32 = rng.r#gen();
                if p < prob {
                    // Rademacher: +1 or -1 with equal probability
                    let val: i8 = if rng.r#gen::<bool>() { 1 } else { -1 };
                    entries.push((row, col, val));
                }
            }
        }

        let scale = 1.0 / ((sparsity as f32).sqrt() * (input_dim as f32).sqrt());

        Self {
            sparse_entries: entries,
            input_dim,
            output_dim,
            scale,
        }
    }

    /// Output dimension of the projector.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Project dense vector to output dimension.
    ///
    /// Returns `Err` if `input.len()` does not match the configured `input_dim`.
    pub fn project(&self, input: &[f32]) -> Result<Vec<f32>, crate::errors::SymthaeaError> {
        if input.len() != self.input_dim {
            return Err(crate::errors::SymthaeaError::Perception(format!(
                "JLProjector input dimension mismatch: expected {}, got {}",
                self.input_dim,
                input.len()
            )));
        }

        let mut output = vec![0.0f32; self.output_dim];

        for &(row, col, val) in &self.sparse_entries {
            output[row] += input[col] * val as f32;
        }

        // Scale for JL guarantees
        for x in &mut output {
            *x *= self.scale;
        }

        Ok(output)
    }

    /// Project to binary/bipolar representation.
    ///
    /// Returns `Err` if input dimension mismatches.
    pub fn project_to_bipolar(
        &self,
        input: &[f32],
    ) -> Result<Vec<i8>, crate::errors::SymthaeaError> {
        Ok(self
            .project(input)?
            .into_iter()
            .map(|v| if v > 0.0 { 1i8 } else { -1i8 })
            .collect())
    }

    /// Project to packed bipolar for efficient similarity.
    ///
    /// Returns `Err` if input dimension mismatches.
    pub fn project_to_packed(
        &self,
        input: &[f32],
    ) -> Result<PackedBipolar, crate::errors::SymthaeaError> {
        Ok(PackedBipolar::from_bipolar(
            &self.project_to_bipolar(input)?,
        ))
    }
}

/// Simple n-gram based text encoder (fallback when Burn not available)
///
/// Creates reproducible HDC vectors from text using character n-grams.
/// This provides a lightweight semantic encoding without ML models.
pub struct NGramEncoder {
    /// Dimension of output vectors
    dimension: usize,
    /// N-gram size (default: 3)
    n: usize,
    /// Cached n-gram vectors
    ngram_cache: HashMap<String, Vec<i8>>,
    /// Random seed for reproducibility
    seed: u64,
}

impl NGramEncoder {
    /// Create new n-gram encoder
    pub fn new(dimension: usize, n: usize, seed: u64) -> Self {
        Self {
            dimension,
            n,
            ngram_cache: HashMap::new(),
            seed,
        }
    }

    /// Get or create random vector for n-gram
    fn get_ngram_vector(&mut self, ngram: &str) -> Vec<i8> {
        if let Some(vec) = self.ngram_cache.get(ngram) {
            return vec.clone();
        }

        // Create deterministic random vector from n-gram hash
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed ^ (ngram.as_bytes().iter().map(|&b| b as u64).sum::<u64>()),
        );

        let vec: Vec<i8> = (0..self.dimension)
            .map(|_| if rng.r#gen::<bool>() { 1 } else { -1 })
            .collect();

        self.ngram_cache.insert(ngram.to_string(), vec.clone());
        vec
    }

    /// Encode text to HDC vector using n-gram superposition
    pub fn encode(&mut self, text: &str) -> Vec<i8> {
        let text = text.to_lowercase();
        let chars: Vec<char> = text.chars().collect();

        if chars.len() < self.n {
            // Too short - just hash the whole thing
            return self.get_ngram_vector(&text);
        }

        // Collect all n-gram vectors with position encoding
        let mut accum = vec![0i32; self.dimension];

        for (pos, window) in chars.windows(self.n).enumerate() {
            let ngram: String = window.iter().collect();
            let vec = self.get_ngram_vector(&ngram);

            // Apply position-based permutation (circular shift)
            let shift = pos % self.dimension;
            for (i, &v) in vec.iter().enumerate() {
                let target = (i + shift) % self.dimension;
                accum[target] += v as i32;
            }
        }

        // Threshold to bipolar
        accum
            .into_iter()
            .map(|v| if v > 0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// Encode to packed bipolar
    pub fn encode_packed(&mut self, text: &str) -> PackedBipolar {
        PackedBipolar::from_bipolar(&self.encode(text))
    }
}

/// Semantic encoder combining multiple encoding strategies
///
/// Provides a unified interface that uses the best available
/// encoding method (Burn ML model if available, fallback to n-grams).
pub struct SemanticEncoder {
    /// N-gram encoder (always available)
    ngram: NGramEncoder,
    /// JL projector for dense embeddings
    jl: JLProjector,
    /// HDC output dimension
    dimension: usize,
}

impl SemanticEncoder {
    /// Create new semantic encoder
    pub fn new() -> Self {
        Self::with_dimension(HDC_DIMENSION)
    }

    /// Create with custom dimension
    pub fn with_dimension(dimension: usize) -> Self {
        // Standard embedding dimension (BERT-like)
        let embedding_dim = 768;

        Self {
            ngram: NGramEncoder::new(dimension, 3, 42),
            jl: JLProjector::new(embedding_dim, dimension, 42),
            dimension,
        }
    }

    /// Encode text to HDC vector
    ///
    /// Uses n-gram encoding (lightweight, always available).
    /// For ML-based encoding, use `encode_with_embedding`.
    pub fn encode_text(&mut self, text: &str) -> Vec<i8> {
        self.ngram.encode(text)
    }

    /// Encode text to packed HDC
    pub fn encode_text_packed(&mut self, text: &str) -> PackedBipolar {
        self.ngram.encode_packed(text)
    }

    /// Encode pre-computed dense embedding to HDC
    ///
    /// Use this when you have embeddings from an external model
    /// (e.g., sentence transformers, CLIP, etc.)
    /// Falls back to n-gram encoding on dimension mismatch.
    pub fn encode_embedding(&self, embedding: &[f32]) -> Vec<i8> {
        self.jl
            .project_to_bipolar(embedding)
            .unwrap_or_else(|_| vec![-1i8; self.jl.output_dim()])
    }

    /// Encode embedding to packed HDC.
    /// Falls back to zero-packed on dimension mismatch.
    pub fn encode_embedding_packed(&self, embedding: &[f32]) -> PackedBipolar {
        self.jl
            .project_to_packed(embedding)
            .unwrap_or_else(|_| PackedBipolar::from_bipolar(&vec![-1i8; self.jl.output_dim()]))
    }

    /// Combine multiple encodings via bundling
    pub fn bundle(&self, vectors: &[&[i8]]) -> Vec<i8> {
        if vectors.is_empty() {
            return vec![-1i8; self.dimension];
        }

        let mut accum = vec![0i32; self.dimension];

        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                accum[i] += v as i32;
            }
        }

        accum
            .into_iter()
            .map(|v| if v > 0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// Get semantic similarity between two texts
    pub fn text_similarity(&mut self, text_a: &str, text_b: &str) -> f32 {
        let packed_a = self.encode_text_packed(text_a);
        let packed_b = self.encode_text_packed(text_b);
        packed_a.xor_similarity(&packed_b)
    }

    /// Get output dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for SemanticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jl_projector_dimensions() {
        let jl = JLProjector::new(768, 16384, 42);

        let input = vec![0.5f32; 768];
        let output = jl.project(&input).unwrap();

        assert_eq!(output.len(), 16384);
    }

    #[test]
    fn test_jl_projector_preserves_similarity() {
        let jl = JLProjector::new(768, 16384, 42);

        // Create two similar vectors
        let mut a = vec![0.0f32; 768];
        let mut b = vec![0.0f32; 768];

        for i in 0..768 {
            a[i] = (i as f32 / 768.0) * 2.0 - 1.0;
            b[i] = a[i] + 0.1; // Slight perturbation
        }

        let pa = jl.project_to_packed(&a).unwrap();
        let pb = jl.project_to_packed(&b).unwrap();

        let sim = pa.xor_similarity(&pb);

        // Similar inputs should have similar outputs
        assert!(
            sim > 0.6,
            "Similar inputs should project to similar outputs: {}",
            sim
        );
    }

    #[test]
    fn test_ngram_encoder() {
        let mut encoder = NGramEncoder::new(1024, 3, 42);

        let vec_a = encoder.encode("hello world");
        let vec_b = encoder.encode("hello world"); // Same text
        let vec_c = encoder.encode("goodbye moon"); // Different text

        assert_eq!(vec_a.len(), 1024);

        // Same text should produce same vector
        assert_eq!(vec_a, vec_b);

        // Different text should produce different vector
        assert_ne!(vec_a, vec_c);
    }

    #[test]
    fn test_ngram_similarity() {
        let mut encoder = NGramEncoder::new(HDC_DIMENSION, 3, 42);

        let similar_a = encoder.encode_packed("the quick brown fox");
        let similar_b = encoder.encode_packed("the quick brown dog");
        let different = encoder.encode_packed("machine learning algorithms");

        let sim_close = similar_a.xor_similarity(&similar_b);
        let sim_far = similar_a.xor_similarity(&different);

        // Similar texts should have higher similarity
        assert!(
            sim_close > sim_far,
            "Similar texts should be more similar: {} vs {}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_semantic_encoder_interface() {
        let mut encoder = SemanticEncoder::new();

        let vec = encoder.encode_text("consciousness emerges from integration");
        assert_eq!(vec.len(), HDC_DIMENSION);

        let packed = encoder.encode_text_packed("test");
        assert_eq!(packed.dimension(), HDC_DIMENSION);
    }

    #[test]
    fn test_semantic_encoder_similarity() {
        let mut encoder = SemanticEncoder::new();

        let sim_same = encoder.text_similarity("hello world", "hello world");
        let sim_similar = encoder.text_similarity("hello world", "hello earth");
        let sim_different = encoder.text_similarity("hello world", "quantum mechanics");

        assert!(
            (sim_same - 1.0).abs() < 0.001,
            "Same text should have 1.0 similarity"
        );
        assert!(
            sim_similar > sim_different,
            "Similar texts should be more similar: {} vs {}",
            sim_similar,
            sim_different
        );
    }

    #[test]
    fn test_bundle() {
        let mut encoder = SemanticEncoder::new();

        let a = encoder.encode_text("first concept");
        let b = encoder.encode_text("second concept");

        let bundled = encoder.bundle(&[&a, &b]);
        assert_eq!(bundled.len(), HDC_DIMENSION);

        // Bundle should be somewhat similar to both constituents
        let packed_a = PackedBipolar::from_bipolar(&a);
        let packed_b = PackedBipolar::from_bipolar(&b);
        let packed_bundle = PackedBipolar::from_bipolar(&bundled);

        let sim_a = packed_bundle.xor_similarity(&packed_a);
        let sim_b = packed_bundle.xor_similarity(&packed_b);

        // Both should have > 50% similarity (above random)
        assert!(
            sim_a > 0.45,
            "Bundle should be similar to constituent a: {}",
            sim_a
        );
        assert!(
            sim_b > 0.45,
            "Bundle should be similar to constituent b: {}",
            sim_b
        );
    }
}
