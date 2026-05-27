#![cfg(feature = "perception")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perception Module Integration Tests
//!
//! Tests for the perception interface components that work without
//! requiring actual ONNX models or GPU hardware.

use symthaea::perception::{
    JLProjector, ModalityType, ModalityWeights, ModelHub, ModelSpec, NGramEncoder, OcrMethod,
    QWEN3_DIM, SIGLIP_DIM, SIGLIP_EMBEDDING_DIM,
};

// ============================================================================
// JOHNSON-LINDENSTRAUSS PROJECTOR TESTS
// ============================================================================

#[test]
fn test_jl_projector_creation() {
    // Test projecting from SigLIP dimension to HDC
    let projector = JLProjector::new(SIGLIP_DIM, 16384, 42);

    // Should create without panicking
    let input = vec![0.5f32; SIGLIP_DIM];
    let output = projector.project(&input);
    assert_eq!(output.len(), 16384);
}

#[test]
fn test_jl_projector_projects() {
    let projector = JLProjector::new(768, 16384, 123);
    let input = vec![0.5f32; 768];

    let output = projector.project(&input);
    assert_eq!(output.len(), 16384);

    // Output should have varying values (not all same)
    let first = output[0];
    let has_variation = output.iter().any(|&v| (v - first).abs() > 0.0001);
    assert!(has_variation, "JL projection should produce varying output");
}

#[test]
fn test_jl_projector_deterministic() {
    let projector1 = JLProjector::new(768, 1024, 42);
    let projector2 = JLProjector::new(768, 1024, 42);
    let input = vec![1.0f32; 768];

    let output1 = projector1.project(&input);
    let output2 = projector2.project(&input);

    assert_eq!(output1, output2, "Same seed should produce same projection");
}

#[test]
fn test_jl_projector_different_seeds() {
    let projector1 = JLProjector::new(768, 1024, 42);
    let projector2 = JLProjector::new(768, 1024, 999);
    let input = vec![1.0f32; 768];

    let output1 = projector1.project(&input);
    let output2 = projector2.project(&input);

    // Should produce different outputs with different seeds
    assert_ne!(
        output1, output2,
        "Different seeds should produce different projections"
    );
}

#[test]
fn test_jl_projector_to_bipolar() {
    let projector = JLProjector::new(768, 1024, 42);
    let input = vec![0.5f32; 768];

    let bipolar = projector.project_to_bipolar(&input);
    assert_eq!(bipolar.len(), 1024);

    // All values should be +1 or -1
    for &v in &bipolar {
        assert!(v == 1 || v == -1, "Bipolar should only have +1 or -1");
    }
}

// ============================================================================
// N-GRAM ENCODER TESTS
// ============================================================================

#[test]
fn test_ngram_encoder_basic() {
    // NGramEncoder::new(dimension, n, seed)
    let mut encoder = NGramEncoder::new(1024, 3, 42); // trigrams, 1024D output

    let encoded = encoder.encode("hello world");
    assert!(!encoded.is_empty(), "Should produce non-empty encoding");
    assert_eq!(encoded.len(), 1024, "Should output correct dimension");
}

#[test]
fn test_ngram_encoder_consistency() {
    let mut encoder = NGramEncoder::new(1024, 3, 42);

    let encoded1 = encoder.encode("test");
    let encoded2 = encoder.encode("test");

    assert_eq!(encoded1, encoded2, "Same input should produce same output");
}

#[test]
fn test_ngram_encoder_different_inputs() {
    let mut encoder = NGramEncoder::new(1024, 3, 42);

    let encoded1 = encoder.encode("hello");
    let encoded2 = encoder.encode("world");

    assert_ne!(
        encoded1, encoded2,
        "Different inputs should produce different outputs"
    );
}

#[test]
fn test_ngram_encoder_different_n() {
    let mut encoder2 = NGramEncoder::new(1024, 2, 42); // bigrams
    let mut encoder3 = NGramEncoder::new(1024, 3, 42); // trigrams

    let text = "consciousness";
    let encoded2 = encoder2.encode(text);
    let encoded3 = encoder3.encode(text);

    // Different n should produce different encodings
    assert_ne!(encoded2, encoded3, "Different n-gram size should differ");
}

#[test]
fn test_ngram_encoder_deterministic() {
    // Same seed should produce same encodings
    let mut encoder1 = NGramEncoder::new(512, 3, 123);
    let mut encoder2 = NGramEncoder::new(512, 3, 123);

    let encoded1 = encoder1.encode("test phrase");
    let encoded2 = encoder2.encode("test phrase");

    assert_eq!(encoded1, encoded2, "Same seed should produce same encoding");
}

// ============================================================================
// MULTI-MODAL TYPES TESTS
// ============================================================================

#[test]
fn test_modality_weights_default() {
    let weights = ModalityWeights::default();

    // Should have default weights (vision, voice, code, ocr)
    assert!(weights.vision >= 0.0);
    assert!(weights.voice >= 0.0);
    assert!(weights.code >= 0.0);
    assert!(weights.ocr >= 0.0);
}

#[test]
fn test_modality_type_variants() {
    let types = [
        ModalityType::Vision,
        ModalityType::Voice,
        ModalityType::Code,
        ModalityType::Ocr,
    ];

    // Each should be distinct
    for (i, t1) in types.iter().enumerate() {
        for (j, t2) in types.iter().enumerate() {
            if i != j {
                assert_ne!(t1, t2);
            }
        }
    }
}

#[test]
fn test_modality_type_debug() {
    let vision = ModalityType::Vision;
    let debug = format!("{:?}", vision);
    assert!(debug.contains("Vision"));
}

// ============================================================================
// OCR METHOD TESTS
// ============================================================================

#[test]
fn test_ocr_method_variants() {
    let methods = [OcrMethod::RustOcr, OcrMethod::Tesseract, OcrMethod::None];

    assert_eq!(methods.len(), 3);
}

#[test]
fn test_ocr_method_debug() {
    let method = OcrMethod::RustOcr;
    let debug = format!("{:?}", method);
    assert!(debug.contains("RustOcr"));
}

// ============================================================================
// MODEL HUB TESTS
// ============================================================================

#[test]
fn test_model_spec_variants() {
    let specs = [
        ModelSpec::Qwen3Embedding,
        ModelSpec::SigLIP,
        ModelSpec::Moondream,
    ];

    // Each should have distinct debug output
    for spec in &specs {
        let debug = format!("{:?}", spec);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_model_hub_creation() {
    let hub = ModelHub::new().expect("Should create ModelHub");

    // Should have models directory
    assert!(!hub.models_dir().as_os_str().is_empty());
}

#[test]
fn test_model_hub_model_path() {
    let hub = ModelHub::new().expect("Should create ModelHub");

    // Each model spec should produce a valid path
    let siglip_path = hub.model_path(ModelSpec::SigLIP);
    assert!(!siglip_path.as_os_str().is_empty());

    let qwen_path = hub.model_path(ModelSpec::Qwen3Embedding);
    assert!(!qwen_path.as_os_str().is_empty());
}

// ============================================================================
// DIMENSION CONSTANTS TESTS
// ============================================================================

#[test]
fn test_dimension_constants() {
    // Verify dimension constants are sensible
    assert_eq!(SIGLIP_DIM, 768, "SigLIP should be 768D");
    assert!(QWEN3_DIM > 0, "Qwen3 dimension should be positive");
    assert_eq!(SIGLIP_EMBEDDING_DIM, 768, "SigLIP embedding should be 768D");
}

// ============================================================================
// JL PROJECTION PROPERTY TESTS
// ============================================================================

#[test]
fn test_jl_similarity_preservation() {
    // JL projection should approximately preserve similarity
    let projector = JLProjector::new(768, 4096, 42);

    // Create two similar inputs
    let input1 = vec![0.5f32; 768];
    let mut input2 = vec![0.5f32; 768];
    // Small perturbation
    input2[0] = 0.6;
    input2[1] = 0.4;

    // And one very different input
    let input3: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0) * 2.0 - 1.0).collect();

    let proj1 = projector.project(&input1);
    let proj2 = projector.project(&input2);
    let proj3 = projector.project(&input3);

    // Compute cosine similarities
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    let sim_12 = cosine_sim(&proj1, &proj2);
    let sim_13 = cosine_sim(&proj1, &proj3);

    // Similar inputs should have higher similarity than different inputs
    // (JL approximately preserves distances)
    assert!(
        sim_12 > sim_13,
        "Similar inputs should have higher similarity: {} vs {}",
        sim_12,
        sim_13
    );
}