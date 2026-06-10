// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for Qwen3 embeddings with real model weights.
//!
//! All tests are `#[ignore]` — they require downloaded model weights.
//!
//! Run with:
//! ```sh
//! # With local weights:
//! QWEN3_MODEL_PATH=/path/to/model cargo test -p symthaea-embeddings --features burn -- --ignored
//!
//! # With HF Hub auto-download (requires burn-hub feature + network):
//! cargo test -p symthaea-embeddings --features burn-hub -- --ignored
//! ```

#[cfg(feature = "burn")]
mod integration {
    use symthaea_embeddings::qwen3::*;

    fn model_path() -> String {
        std::env::var("QWEN3_MODEL_PATH")
            .unwrap_or_else(|_| "Qwen/Qwen3-Embedding-0.6B".to_string())
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 1e-8 && nb > 1e-8 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    #[test]
    #[ignore]
    fn test_real_model_semantic_similarity() {
        let config = Qwen3Config::qwen3_06b(model_path());
        let mut embedder = Qwen3Embedder::new(config).unwrap();
        assert!(
            embedder.is_model_loaded(),
            "Model should be loaded (not simulation)"
        );

        let dog = embedder.embed("A dog plays in the park").unwrap();
        let puppy = embedder.embed("A puppy runs in the garden").unwrap();
        let stock = embedder.embed("Stock market analysis Q3 2025").unwrap();

        let sim_related = cosine(&dog.embedding, &puppy.embedding);
        let sim_unrelated = cosine(&dog.embedding, &stock.embedding);
        assert!(
            sim_related > sim_unrelated,
            "Related texts should be more similar: related={sim_related:.4} unrelated={sim_unrelated:.4}"
        );
        assert!(
            sim_related > 0.5,
            "Related texts should have >0.5 similarity: {sim_related:.4}"
        );
    }

    #[test]
    #[ignore]
    fn test_real_model_batch_inference() {
        let config = Qwen3Config::qwen3_06b(model_path());
        let mut embedder = Qwen3Embedder::new(config).unwrap();
        assert!(embedder.is_model_loaded());

        let texts = vec![
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
        ];
        let results = embedder.embed_batch(&texts).unwrap();

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                result.dimension, QWEN3_STANDARD_DIMENSION,
                "Text {i} has wrong dimension"
            );
            assert!(!result.is_simulated, "Text {i} should not be simulated");
        }

        // Verify batch results match sequential results
        let mut embedder2 = Qwen3Embedder::new(Qwen3Config::qwen3_06b(model_path())).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let single = embedder2.embed(text).unwrap();
            let sim = cosine(&results[i].embedding, &single.embedding);
            assert!(
                sim > 0.99,
                "Batch[{i}] vs single should be near-identical: sim={sim:.4}"
            );
        }
    }

    #[test]
    #[ignore]
    fn test_real_model_normalization() {
        let config = Qwen3Config::qwen3_06b(model_path());
        let mut embedder = Qwen3Embedder::new(config).unwrap();
        assert!(embedder.is_model_loaded());

        let result = embedder.embed("Test normalization").unwrap();
        let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Normalized embedding should have unit norm, got {norm:.4}"
        );
    }

    #[test]
    #[ignore]
    fn test_model_cache_reuse() {
        let path = model_path();
        let config1 = Qwen3Config::qwen3_06b(&path);
        let mut e1 = Qwen3Embedder::new(config1).unwrap();
        assert!(e1.is_model_loaded());

        // Second embedder should reuse cached model
        let config2 = Qwen3Config::qwen3_06b(&path);
        let mut e2 = Qwen3Embedder::new(config2).unwrap();
        assert!(e2.is_model_loaded());

        // Both should produce identical embeddings
        let r1 = e1.embed("cache test").unwrap();
        let r2 = e2.embed("cache test").unwrap();
        let sim = cosine(&r1.embedding, &r2.embedding);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Cached models should produce identical output: sim={sim:.6}"
        );
    }

    #[test]
    #[ignore]
    fn test_pooling_strategies() {
        let path = model_path();
        let text = "This is a test sentence for pooling strategies";

        let pooling_configs = [
            PoolingStrategy::LastTokenPooling,
            PoolingStrategy::MeanPooling,
            PoolingStrategy::ClsToken,
            PoolingStrategy::MaxPooling,
        ];

        let mut embeddings = Vec::new();
        for &pooling in &pooling_configs {
            let config = Qwen3Config {
                model_path: Some(path.clone()),
                pooling,
                use_simulated: false,
                ..Default::default()
            };
            let mut embedder = Qwen3Embedder::new(config).unwrap();
            assert!(embedder.is_model_loaded());
            let result = embedder.embed(text).unwrap();
            assert_eq!(result.dimension, QWEN3_STANDARD_DIMENSION);
            embeddings.push(result.embedding);
        }

        // Different pooling strategies should produce different (but valid) embeddings
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                let sim = cosine(&embeddings[i], &embeddings[j]);
                assert!(
                    sim < 1.0 - 1e-6,
                    "Pooling {:?} and {:?} should differ: sim={sim:.4}",
                    pooling_configs[i],
                    pooling_configs[j]
                );
            }
        }
    }
}
