//! Integration tests for the perception pipeline
//!
//! Tests the complete perception → consciousness pipeline:
//! 1. Qwen3/SigLIP embeddings
//! 2. JL projection to HDC space
//! 3. Attention bid creation
//! 4. Consciousness integration
#![cfg(feature = "embeddings")]

use symthaea::embeddings::{Qwen3Embedder, Qwen3Config, QWEN3_DIMENSION};
use symthaea::perception::{
    SemanticVision, MultiModalIntegrator, PerceptionBridge, BridgeConfig,
    ModelHub, ModelSpec, SIGLIP_DIM, QWEN3_DIM,
};
use image::{DynamicImage, Rgb, RgbImage};

/// Create a test image for vision tests
fn create_test_image(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for pixel in img.pixels_mut() {
        *pixel = Rgb(color);
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
fn test_qwen3_embedder_stub() {
    // Test stub embedder (no ONNX model required)
    let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

    let result = embedder.embed("Hello, this is a test of the Qwen3 embedder.").unwrap();

    // Check embedding dimension
    assert_eq!(result.embedding.len(), QWEN3_DIMENSION);
    assert_eq!(result.embedding.len(), 1024);

    // Check normalization
    let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
}

#[test]
fn test_qwen3_deterministic() {
    let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

    let text = "Deterministic test input";
    let emb1 = embedder.embed(text).unwrap();
    let emb2 = embedder.embed(text).unwrap();

    // Same input should produce same output
    assert_eq!(emb1.embedding, emb2.embedding);
}

#[test]
fn test_qwen3_similarity() {
    let mut embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();

    // Self-similarity should be 1.0
    let sim = embedder.similarity("test", "test").unwrap();
    assert!((sim - 1.0).abs() < 0.01);

    // Different texts should have lower similarity
    let sim_diff = embedder.similarity("cat", "quantum physics").unwrap();
    assert!(sim_diff < 1.0);
}

#[test]
fn test_siglip_model_stub() {
    let mut model = symthaea::perception::SigLipModel::new();
    model.initialize().unwrap();

    let image = create_test_image(100, 100, [128, 128, 128]);
    let embedding = model.embed_image(&image).unwrap();

    // Check embedding dimension
    assert_eq!(embedding.vector.len(), SIGLIP_DIM);
    assert_eq!(embedding.vector.len(), 768);

    // Check normalization
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
}

#[test]
fn test_siglip_deterministic() {
    let mut model = symthaea::perception::SigLipModel::new();
    model.initialize().unwrap();

    let image = create_test_image(50, 50, [255, 0, 0]);
    let emb1 = model.embed_image(&image).unwrap();
    let emb2 = model.embed_image(&image).unwrap();

    // Same image should produce same embedding
    assert_eq!(emb1.vector, emb2.vector);
    assert_eq!(emb1.image_hash, emb2.image_hash);
}

#[test]
fn test_siglip_different_images() {
    let mut model = symthaea::perception::SigLipModel::new();
    model.initialize().unwrap();

    let red_image = create_test_image(50, 50, [255, 0, 0]);
    let green_image = create_test_image(50, 50, [0, 255, 0]);

    let emb_red = model.embed_image(&red_image).unwrap();
    let emb_green = model.embed_image(&green_image).unwrap();

    // Different images should have different hashes
    assert_ne!(emb_red.image_hash, emb_green.image_hash);

    // Different images should have different embeddings
    assert_ne!(emb_red.vector, emb_green.vector);
}

#[test]
fn test_multi_modal_integrator() {
    let integrator = MultiModalIntegrator::new();

    // Test text embedding projection (1024D → 16,384D)
    let text_embedding = vec![0.5f32; QWEN3_DIM];
    let hdc = integrator.project_text_embedding(&text_embedding).unwrap();
    assert_eq!(hdc.dim(), 16384);

    // Test image embedding projection (768D → 16,384D)
    let image_embedding = symthaea::perception::ImageEmbedding {
        vector: vec![0.5f32; SIGLIP_DIM],
        timestamp: std::time::Instant::now(),
        image_hash: 12345,
    };
    let hdc_image = integrator.project_image_embedding(&image_embedding).unwrap();
    assert_eq!(hdc_image.dim(), 16384);
}

#[test]
fn test_jl_projection_preserves_structure() {
    let integrator = MultiModalIntegrator::new();

    // Similar embeddings should have similar HDC projections
    let emb1 = vec![0.5f32; QWEN3_DIM];
    let mut emb2 = vec![0.5f32; QWEN3_DIM];
    emb2[0] = 0.51; // Slightly different

    let hdc1 = integrator.project_text_embedding(&emb1).unwrap();
    let hdc2 = integrator.project_text_embedding(&emb2).unwrap();

    // Count matching bits
    let matching: usize = hdc1.bits.iter()
        .zip(hdc2.bits.iter())
        .filter(|(a, b)| a == b)
        .count();

    // Similar embeddings should have high bit overlap
    let overlap_ratio = matching as f32 / hdc1.dim() as f32;
    assert!(overlap_ratio > 0.9, "Similar embeddings should have >90% bit overlap");
}

#[test]
fn test_perception_bridge_creation() {
    let bridge = PerceptionBridge::new(BridgeConfig::default());
    assert_eq!(bridge.stats().perceptions_processed, 0);
    assert_eq!(bridge.stats().bids_created, 0);
}

#[test]
fn test_perception_bridge_text_processing() {
    let mut bridge = PerceptionBridge::new(BridgeConfig {
        include_hdc: true,
        ..Default::default()
    });

    let bid = bridge.process_text("Test input for consciousness").unwrap();

    assert_eq!(bid.source, "TextPerception");
    assert!(bid.content.contains("Test"));
    assert!(bid.tags.contains(&"perception".to_string()));
    assert!(bid.tags.contains(&"text".to_string()));
    // HDC semantic is now Vec<i8>
    assert!(bid.hdc_semantic.is_some());
}

#[test]
fn test_perception_bridge_image_processing() {
    let mut bridge = PerceptionBridge::new(BridgeConfig {
        include_hdc: true,
        ..Default::default()
    });

    let image = create_test_image(100, 100, [64, 128, 192]);
    let bid = bridge.process_image(&image).unwrap();

    assert_eq!(bid.source, "VisionPerception");
    assert!(bid.content.contains("100x100"));
    assert!(bid.tags.contains(&"perception".to_string()));
    assert!(bid.tags.contains(&"vision".to_string()));
    assert!(bid.hdc_semantic.is_some());
}

#[test]
fn test_model_hub_spec() {
    assert_eq!(ModelSpec::Qwen3Embedding.embedding_dim(), 1024);
    assert_eq!(ModelSpec::SigLIP.embedding_dim(), 768);
    assert_eq!(ModelSpec::BGE.embedding_dim(), 768);

    assert_eq!(ModelSpec::Qwen3Embedding.local_name(), "qwen3-embedding-0.6b");
    assert_eq!(ModelSpec::SigLIP.local_name(), "siglip-so400m");
}

#[test]
fn test_embedding_cache() {
    let mut cache = symthaea::perception::EmbeddingCache::new(3);

    let emb1 = symthaea::perception::ImageEmbedding {
        vector: vec![1.0; 768],
        timestamp: std::time::Instant::now(),
        image_hash: 1,
    };

    let emb2 = symthaea::perception::ImageEmbedding {
        vector: vec![2.0; 768],
        timestamp: std::time::Instant::now(),
        image_hash: 2,
    };

    cache.insert(1, emb1.clone());
    cache.insert(2, emb2.clone());

    assert_eq!(cache.stats().size, 2);
    assert!(cache.get(1).is_some());
    assert!(cache.get(2).is_some());
    assert!(cache.get(999).is_none());
}

#[test]
fn test_semantic_vision_caching() {
    let mut vision = SemanticVision::new(100);
    vision.initialize().unwrap();

    let image = create_test_image(64, 64, [100, 150, 200]);

    // First embed
    let emb1 = vision.embed_image(&image).unwrap();

    // Second embed should hit cache
    let emb2 = vision.embed_image(&image).unwrap();

    assert_eq!(emb1.vector, emb2.vector);
    assert_eq!(emb1.image_hash, emb2.image_hash);

    // Cache should have 1 entry
    assert_eq!(vision.cache_stats().size, 1);
}

#[test]
fn test_end_to_end_perception_pipeline() {
    // Complete pipeline test:
    // 1. Create embeddings
    // 2. Project to HDC
    // 3. Create attention bid
    // 4. Verify HDC semantic is present

    let mut text_embedder = Qwen3Embedder::new(Qwen3Config::default()).unwrap();
    let integrator = MultiModalIntegrator::new();
    let mut bridge = PerceptionBridge::new(BridgeConfig::default());

    // Text path
    let text = "This is a test of the complete perception pipeline";
    let text_emb = text_embedder.embed(text).unwrap();
    assert_eq!(text_emb.embedding.len(), 1024);

    let text_hdc = integrator.project_text_embedding(&text_emb.embedding).unwrap();
    assert_eq!(text_hdc.dim(), 16384);

    let text_bid = bridge.process_text(text).unwrap();
    assert!(text_bid.hdc_semantic.is_some());

    // Vision path
    let image = create_test_image(128, 128, [50, 100, 150]);
    let image_bid = bridge.process_image(&image).unwrap();
    assert!(image_bid.hdc_semantic.is_some());

    // Stats should reflect processing
    assert_eq!(bridge.stats().perceptions_processed, 2);
}

#[test]
fn test_hdc_dimension_consistency() {
    // Verify all HDC outputs have consistent dimension
    use symthaea::hdc::HDC_DIMENSION;

    let integrator = MultiModalIntegrator::new();

    // Text embedding projection
    let text_emb = vec![0.5f32; QWEN3_DIM];
    let text_hdc = integrator.project_text_embedding(&text_emb).unwrap();
    assert_eq!(text_hdc.dim(), HDC_DIMENSION);

    // Image embedding projection
    let image_emb = symthaea::perception::ImageEmbedding {
        vector: vec![0.5f32; SIGLIP_DIM],
        timestamp: std::time::Instant::now(),
        image_hash: 42,
    };
    let image_hdc = integrator.project_image_embedding(&image_emb).unwrap();
    assert_eq!(image_hdc.dim(), HDC_DIMENSION);
}
