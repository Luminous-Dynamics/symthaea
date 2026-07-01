// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hub integration tests — validate full download → safetensors → Burn inference
//! with real Qwen3-Embedding-0.6B weights.
//!
//! All tests are `#[ignore]` — they require network access and ~1.2 GB download.
//!
//! Run with:
//! ```sh
//! cargo test -p symthaea-embeddings --features burn-hub -- --ignored
//! ```

#[cfg(feature = "burn-hub")]
mod hub {
    use symthaea_embeddings::qwen3::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-10)
    }

    #[test]
    #[ignore]
    fn test_hub_download_and_embed() {
        let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
        let mut embedder = Qwen3Embedder::new(config).unwrap();
        assert!(
            embedder.is_model_loaded(),
            "Model should be loaded from hub (not simulation)"
        );

        let result = embedder.embed("Hello world").unwrap();
        assert_eq!(result.dimension, QWEN3_STANDARD_DIMENSION);
        assert!(!result.is_simulated, "Should not be simulated");

        // Non-zero L2 norm
        let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm > 0.1,
            "Embedding should have non-trivial norm, got {norm}"
        );

        // Non-trivial variance (not all same value)
        let mean: f32 = result.embedding.iter().sum::<f32>() / result.dimension as f32;
        let variance: f32 = result
            .embedding
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / result.dimension as f32;
        assert!(
            variance > 1e-6,
            "Embedding should have non-trivial variance, got {variance}"
        );
    }

    #[test]
    #[ignore]
    fn test_hub_similarity_ordering() {
        let config = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
        let mut embedder = Qwen3Embedder::new(config).unwrap();
        assert!(embedder.is_model_loaded());

        let cat = embedder.embed("A cat sits on the mat").unwrap();
        let kitten = embedder.embed("A kitten rests on a rug").unwrap();
        let finance = embedder.embed("Quarterly earnings report Q3 2025").unwrap();

        let sim_related = cosine(&cat.embedding, &kitten.embedding);
        let sim_unrelated = cosine(&cat.embedding, &finance.embedding);

        assert!(
            sim_related > sim_unrelated,
            "cat-kitten ({sim_related:.4}) should be more similar than cat-finance ({sim_unrelated:.4})"
        );
        assert!(
            sim_unrelated > -1.0, // Just ensure it's a valid cosine value
            "Unrelated similarity should be a valid cosine value"
        );
    }

    #[test]
    #[ignore]
    fn test_hub_matryoshka_consistency() {
        let config_full = Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B");
        let mut embedder_full = Qwen3Embedder::new(config_full).unwrap();
        assert!(embedder_full.is_model_loaded());

        let config_half =
            Qwen3Config::from_hub("Qwen/Qwen3-Embedding-0.6B").with_matryoshka_dim(512);
        let mut embedder_half = Qwen3Embedder::new(config_half).unwrap();
        assert!(embedder_half.is_model_loaded());

        let text = "Matryoshka representation learning consistency test";
        let full = embedder_full.embed(text).unwrap();
        let half = embedder_half.embed(text).unwrap();

        assert_eq!(full.dimension, 1024);
        assert_eq!(half.dimension, 512);

        // First 512 dims of the 1024D embedding, re-normalized, should match the 512D embedding
        let truncated: Vec<f32> = full.embedding[..512].to_vec();
        let trunc_norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        let truncated_normalized: Vec<f32> = truncated.iter().map(|x| x / trunc_norm).collect();

        let sim = cosine(&truncated_normalized, &half.embedding);
        assert!(
            sim > 0.99,
            "Truncated+renormalized 1024D should match 512D MRL embedding: cosine={sim:.6}"
        );
    }
}
