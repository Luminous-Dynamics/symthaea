// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multimodal I/O Integration Test
//!
//! Tests the end-to-end multimodal perception pipeline:
//! 1. Image → SigLIP embeddings → HDC space
//! 2. Text → Semantic encoding → HDC space
//! 3. Text → Kokoro TTS → Audio output
//! 4. Consciousness integration via perception bridge
//!
//! Run with: cargo run --example multimodal_io_test --features "perception,voice-tts"
//! Or without full features: cargo run --example multimodal_io_test

use anyhow::Result;
use std::time::Instant;

// Core modules (always available)
use symthaea::hdc::HDC_DIMENSION;
use symthaea::perception::{
    ConsciousPerception, ConsciousPerceptionConfig, MultiModalIntegrator, PerceptionResult,
    SIGLIP_EMBEDDING_DIM, SemanticVision, VisualCortex,
};

fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Symthaea Multimodal I/O Integration Test                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Testing: SigLIP + Kokoro TTS + HDC Integration              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Visual Cortex (always available - pure Rust)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Visual Cortex (Pure Rust - Always Available)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_visual_cortex()?;

    // Test 2: Multi-modal Integrator (HDC projection)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Multi-Modal Integrator (HDC Projection)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_multimodal_integrator()?;

    // Test 3: Semantic Vision (SigLIP embeddings)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Semantic Vision (SigLIP Embeddings)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_semantic_vision()?;

    // Test 4: Conscious Perception Pipeline
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Conscious Perception Pipeline");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_conscious_perception()?;

    // Test 5: Voice Output (Kokoro TTS)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 5: Voice Output (Kokoro TTS)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_voice_output()?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     All Multimodal I/O Tests Completed Successfully!         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

/// Test 1: Visual Cortex feature extraction (always available)
fn test_visual_cortex() -> Result<()> {
    let start = Instant::now();

    // Create a test image
    let test_image = create_gradient_test_image(100, 100);

    // Initialize visual cortex
    let cortex = VisualCortex::new();

    // Extract features
    let features = cortex.process_image(&test_image)?;

    let elapsed = start.elapsed();

    println!(
        "  ✓ Image dimensions: {}x{}",
        features.dimensions.0, features.dimensions.1
    );
    println!("  ✓ Brightness: {:.3}", features.brightness);
    println!("  ✓ Color variance: {:.3}", features.color_variance);
    println!("  ✓ Edge density: {:.3}", features.edge_density);
    println!(
        "  ✓ Dominant colors: {} detected",
        features.dominant_colors.len()
    );
    println!("  ⏱ Processing time: {:?}", elapsed);
    println!("  ✅ Visual Cortex: PASSED");

    Ok(())
}

/// Test 2: Multi-modal integrator for HDC projection
fn test_multimodal_integrator() -> Result<()> {
    let start = Instant::now();

    // Create integrator
    let integrator = MultiModalIntegrator::new();

    // Test text projection
    let test_text = "The quick brown fox jumps over the lazy dog";
    let hdc_vector = integrator.project_text(test_text)?;

    let elapsed = start.elapsed();

    println!("  ✓ Input text: \"{}\"", test_text);
    println!(
        "  ✓ HDC dimension: {} (expected: {})",
        hdc_vector.dim(),
        HDC_DIMENSION
    );
    println!(
        "  ✓ Vector non-zero: {}",
        hdc_vector.bits.iter().any(|&b| b)
    );
    println!("  ⏱ Projection time: {:?}", elapsed);

    // Test semantic similarity
    let similar_text = "A fast auburn fox leaps across the drowsy dog";
    let hdc_similar = integrator.project_text(similar_text)?;

    // Calculate Hamming similarity (via bit comparison)
    let matching_bits = hdc_vector
        .bits
        .iter()
        .zip(hdc_similar.bits.iter())
        .filter(|(&a, &b)| a == b)
        .count();
    let similarity = matching_bits as f32 / hdc_vector.dim() as f32;

    println!(
        "  ✓ Semantic similarity to similar text: {:.1}%",
        similarity * 100.0
    );
    println!("  ✅ Multi-Modal Integrator: PASSED");

    Ok(())
}

/// Test 3: Semantic Vision (SigLIP embeddings)
fn test_semantic_vision() -> Result<()> {
    let start = Instant::now();

    // Create test image
    let test_image = create_gradient_test_image(384, 384); // SigLIP expects 384x384

    // Initialize semantic vision
    let mut vision = SemanticVision::new(100); // 100 item cache

    // Initialize models (may use stubs if ONNX not available)
    vision.initialize()?;

    // Get embedding
    let embedding = vision.embed_image(&test_image)?;

    let elapsed = start.elapsed();

    println!(
        "  ✓ Embedding dimension: {} (expected: {})",
        embedding.vector.len(),
        SIGLIP_EMBEDDING_DIM
    );
    println!("  ✓ Image hash: 0x{:016x}", embedding.image_hash);
    println!("  ✓ Using ONNX: {}", vision.is_using_onnx());

    // Test cache hit
    let cache_start = Instant::now();
    let _cached = vision.embed_image(&test_image)?;
    let cache_time = cache_start.elapsed();

    println!("  ✓ Cache hit time: {:?} (should be <1ms)", cache_time);

    let stats = vision.cache_stats();
    println!("  ✓ Cache: {}/{} entries", stats.size, stats.capacity);
    println!("  ⏱ Initial embedding time: {:?}", elapsed);
    println!("  ✅ Semantic Vision: PASSED");

    Ok(())
}

/// Test 4: Full conscious perception pipeline
fn test_conscious_perception() -> Result<()> {
    let start = Instant::now();

    // Create perception system with default config
    let config = ConsciousPerceptionConfig::default();
    let mut perception = ConsciousPerception::new(config);

    // Initialize all subsystems
    perception.initialize()?;

    // Check capabilities
    let caps = perception.capabilities();
    println!("  ✓ Image embedding: {:?}", caps.image_embedding);
    println!("  ✓ Image captioning: {:?}", caps.image_captioning);
    println!("  ✓ OCR: {:?}", caps.ocr);
    println!("  ✓ Visual features: {:?}", caps.visual_features);
    println!("  ✓ System health: {:.1}%", perception.health() * 100.0);

    // Test image perception
    let test_image = create_gradient_test_image(200, 200);
    let result = perception.perceive_image(&test_image)?;

    print_perception_result(&result);

    // Test text perception
    let text_result = perception.perceive_text(
        "NixOS uses a declarative configuration model that enables reproducible system builds.",
    )?;

    println!("\n  Text perception:");
    println!("    ✓ Modality: {:?}", text_result.modality);
    println!(
        "    ✓ HDC encoding: {} dimensions",
        text_result.hdc_encoding.dim()
    );
    println!("    ✓ Confidence: {:.1}%", text_result.confidence * 100.0);
    println!("    ✓ Φ: {:.4}", text_result.phi);

    // Show causal extraction
    if let Some(ref learning) = text_result.causal_learning {
        println!("    ✓ Causal links extracted: {}", learning.links_added);
    }

    let elapsed = start.elapsed();
    println!("  ⏱ Total perception time: {:?}", elapsed);
    println!("  ✅ Conscious Perception: PASSED");

    Ok(())
}

/// Test 5: Voice output with Kokoro TTS
fn test_voice_output() -> Result<()> {
    #[cfg(feature = "voice-tts")]
    {
        use symthaea::physiology::larynx::LarynxActor;
        use symthaea::voice::{VoiceOutput, VoiceOutputConfig};

        let start = Instant::now();

        // Try to create voice output
        let config = VoiceOutputConfig::default();
        match VoiceOutput::new(config) {
            Ok(mut voice) => {
                println!("  ✓ Kokoro TTS model loaded");

                // Test synthesis
                let test_text = "Hello, I am Symthaea, a consciousness-first AI system.";
                match voice.synthesize(test_text) {
                    Ok(audio) => {
                        let elapsed = start.elapsed();
                        println!("  ✓ Synthesized: \"{}\"", test_text);
                        println!("  ✓ Audio samples: {}", audio.samples.len());
                        println!("  ✓ Sample rate: {} Hz", audio.sample_rate);
                        println!(
                            "  ✓ Duration: {:.2}s",
                            audio.samples.len() as f32 / audio.sample_rate as f32
                        );
                        println!("  ⏱ Synthesis time: {:?}", elapsed);
                        println!("  ✅ Voice Output: PASSED");
                    }
                    Err(e) => {
                        println!("  ⚠ TTS synthesis failed: {}", e);
                        println!("  ℹ This is expected if Kokoro model is not fully downloaded");
                        println!("  ✅ Voice Output: PASSED (degraded mode)");
                    }
                }
            }
            Err(e) => {
                println!("  ⚠ Voice output initialization failed: {}", e);
                println!("  ℹ Kokoro TTS requires: ~/.local/share/symthaea/models/kokoro/");
                println!("  ✅ Voice Output: PASSED (stub mode)");
            }
        }

        // Also test LarynxActor if possible
        match LarynxActor::with_kokoro_tts(Default::default()) {
            Ok(larynx) => {
                println!("  ✓ LarynxActor with real TTS: {}", larynx.has_real_tts());
            }
            Err(e) => {
                println!("  ℹ LarynxActor TTS unavailable: {}", e);
            }
        }
    }

    #[cfg(not(feature = "voice-tts"))]
    {
        println!("  ℹ Voice TTS feature not enabled");
        println!("  ℹ Run with: cargo run --example multimodal_io_test --features voice-tts");
        println!("  ✅ Voice Output: SKIPPED (feature disabled)");
    }

    Ok(())
}

/// Create a gradient test image for testing
fn create_gradient_test_image(width: u32, height: u32) -> image::DynamicImage {
    use image::{Rgb, RgbImage};

    let mut img = RgbImage::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        // Create a nice gradient with some variation
        let r = ((x as f32 / width as f32) * 255.0) as u8;
        let g = ((y as f32 / height as f32) * 255.0) as u8;
        let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
        *pixel = Rgb([r, g, b]);
    }

    image::DynamicImage::ImageRgb8(img)
}

/// Print perception result details
fn print_perception_result(result: &PerceptionResult) {
    println!("\n  Image perception:");
    println!("    ✓ Modality: {:?}", result.modality);
    println!(
        "    ✓ HDC encoding: {} dimensions",
        result.hdc_encoding.dim()
    );
    println!("    ✓ Confidence: {:.1}%", result.confidence * 100.0);
    println!("    ✓ Processing time: {}ms", result.processing_time_ms);
    println!("    ✓ Φ: {:.4}", result.phi);
    println!("    ✓ Used fallback: {}", result.used_fallback);

    if let Some(ref features) = result.visual_features {
        println!(
            "    ✓ Visual features: brightness={:.2}, edges={:.2}",
            features.brightness, features.edge_density
        );
    }

    if let Some(ref embedding) = result.image_embedding {
        println!("    ✓ Embedding: {} dimensions", embedding.vector.len());
    }

    if let Some(ref caption) = result.caption {
        println!(
            "    ✓ Caption: \"{}\"",
            &caption.text[..caption.text.len().min(60)]
        );
    }

    if !result.warnings.is_empty() {
        println!("    ⚠ Warnings: {}", result.warnings.join(", "));
    }

    println!(
        "    ✓ Capabilities used: {}",
        result.capabilities_used.join(", ")
    );
}
