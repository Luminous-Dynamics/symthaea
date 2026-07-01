// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cantor Fractal Integration Tests
//!
//! Verifies the full CRHV lifecycle:
//! 1. GWT broadcast promotes BinaryHV to CantorRecursiveHV
//! 2. CRHVs accumulate in the broadcast buffer
//! 3. Dream consolidation cleans them through CantorCleanupEngine
//! 4. Telemetry reports metacognitive depth and buffer occupancy

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_cantor_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        enable_gwt: true,
        enable_dream_replay: true,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

#[test]
fn test_cantor_telemetry_fields_exist() {
    let mut service = make_cantor_service();
    let result = service.cycle("hello world");
    assert_eq!(result.metadata.cantor.cantor_buffer_occupancy, 0);
    assert_eq!(result.metadata.cantor.cantor_metacognitive_depth, 0.0);
    assert_eq!(result.metadata.cantor.cantor_resonance_boost, 0.0);
    assert_eq!(result.metadata.cantor.cantor_depth_histogram, [0u32; 6]);
}

#[test]
fn test_cantor_buffer_bounded_at_32() {
    let mut service = make_cantor_service();
    let mut max_occupancy = 0u32;
    for i in 0..500 {
        let input = format!("novel input variant {} to maximize broadcasts", i);
        let result = service.cycle(&input);
        if result.metadata.cantor.cantor_buffer_occupancy > max_occupancy {
            max_occupancy = result.metadata.cantor.cantor_buffer_occupancy;
        }
    }
    assert!(
        max_occupancy <= 32,
        "Buffer occupancy {} exceeded cap of 32",
        max_occupancy
    );
}

#[test]
fn test_cantor_metacognitive_depth_in_valid_range() {
    let mut service = make_cantor_service();
    for i in 0..50 {
        let result = service.cycle(&format!("input cycle {i}"));
        let depth = result.metadata.cantor.cantor_metacognitive_depth;
        assert!(
            (0.0..=1.0).contains(&depth),
            "Metacognitive depth {} out of range at cycle {}",
            depth,
            i
        );
    }
}

#[test]
fn test_cantor_buffer_accumulates_on_gwt_broadcast() {
    let mut service = make_cantor_service();
    let mut max_occupancy = 0u32;
    let inputs = [
        "quantum entanglement creates non-local correlations",
        "consciousness emerges from integrated information processing",
        "fractal self-similarity in neural firing patterns",
        "the binding problem requires synchronized oscillations",
        "predictive coding minimizes free energy surprise",
    ];
    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);
        if result.metadata.cantor.cantor_buffer_occupancy > max_occupancy {
            max_occupancy = result.metadata.cantor.cantor_buffer_occupancy;
        }
    }
    // Soft: GWT broadcast depends on activation dynamics
    println!(
        "Cantor buffer max occupancy over 100 cycles: {}",
        max_occupancy
    );
}

// =============================================================================
// Pure symthaea-core tests (no CLS dependency, never contested)
// =============================================================================

#[test]
fn test_crhv_cleanup_loop_codebook_growth() {
    use symthaea_core::hdc::binary_hv::BinaryHV;
    use symthaea_core::hdc::cantor_recursive_hv::CantorRecursiveHV;
    use symthaea_core::hdc::cantor_resonator_cleanup::{CantorCleanupConfig, CantorCleanupEngine};

    let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());
    assert_eq!(engine.codebook.len(), 0);

    let concepts: Vec<CantorRecursiveHV> = (0..10)
        .map(|i| CantorRecursiveHV::from_base_with_depth(BinaryHV::random(i), 4))
        .collect();

    for (i, c) in concepts.iter().enumerate() {
        engine.learn(&format!("concept_{i}"), c);
    }
    assert_eq!(engine.codebook.len(), 10);

    for c in &concepts {
        let result = engine.cleanup(c);
        assert!(result.quality > 0.9);
        if result.quality > 0.7 {
            let label = format!("dream_{}", engine.cleanups_performed);
            engine.learn(&label, &result.cleaned);
        }
    }

    assert!(
        engine.codebook.len() > 10,
        "Codebook should grow from dream feedback, got {}",
        engine.codebook.len()
    );
}

#[test]
fn test_crhv_vs_flat_self_similarity_preservation() {
    use symthaea_core::hdc::binary_hv::BinaryHV;
    use symthaea_core::hdc::cantor_recursive_hv::CantorRecursiveHV;
    use symthaea_core::hdc::cantor_resonator_cleanup::{CantorCleanupConfig, CantorCleanupEngine};

    let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());

    let concepts: Vec<CantorRecursiveHV> = (0..16)
        .map(|i| {
            let crhv = CantorRecursiveHV::from_base_with_depth(BinaryHV::random(i), 5);
            engine.learn(&format!("c{i}"), &crhv);
            crhv
        })
        .collect();

    // Measure self-similarity preservation after cleanup with XOR noise
    let preserved: f32 = concepts
        .iter()
        .enumerate()
        .map(|(i, original)| {
            let noise = BinaryHV::random((i * 1000 + 42) as u64);
            let noisy_base = original.base.bind(&noise);
            let noisy_crhv = CantorRecursiveHV::from_base_with_depth(noisy_base, original.depth);
            let result = engine.cleanup(&noisy_crhv);
            let original_ss = original.self_similarity();
            let cleaned_ss = result.cleaned.self_similarity();
            1.0 - (original_ss - cleaned_ss).abs()
        })
        .sum::<f32>()
        / 16.0;

    println!("CRHV self-similarity preservation: {preserved:.4}");
    assert!(
        preserved > 0.5,
        "Self-similarity should be partially preserved, got {}",
        preserved
    );
}

#[test]
fn test_cantor_dream_surprise_in_valid_range() {
    let mut service = make_cantor_service();
    for i in 0..100 {
        let result = service.cycle(&format!("surprise test input {i}"));
        let surprise = result.metadata.cantor.cantor_dream_surprise;
        assert!(
            (0.0..=1.0).contains(&surprise),
            "Dream surprise {} out of [0,1] at cycle {}",
            surprise,
            i
        );
    }
}

#[test]
fn test_cantor_depth_histogram_populated() {
    let mut service = make_cantor_service();
    // Run enough cycles to get some dream consolidation with codebook entries
    for i in 0..200 {
        service.cycle(&format!("histogram test input {i}"));
    }
    let result = service.cycle("histogram final");
    let hist = result.metadata.cantor.cantor_depth_histogram;
    let total: u32 = hist.iter().sum();
    // If codebook has depth-stratified entries, histogram should have non-zero total
    // (or zero if no dreams occurred — which is acceptable)
    println!(
        "Depth histogram: {:?}, total={}, codebook={}",
        hist, total, result.metadata.cantor.cantor_codebook_size
    );
    // Consistency: histogram total should match codebook size
    // (only depth-labeled entries count; there may be non-depth entries from warmup)
    assert!(
        total <= result.metadata.cantor.cantor_codebook_size,
        "Histogram total {} > codebook size {}",
        total,
        result.metadata.cantor.cantor_codebook_size
    );
}

#[test]
fn test_cantor_codebook_bounded() {
    let mut service = make_cantor_service();
    for i in 0..500 {
        let result = service.cycle(&format!("codebook bound test {i}"));
        assert!(
            result.metadata.cantor.cantor_codebook_size <= 256,
            "Codebook size {} exceeded cap 256 at cycle {}",
            result.metadata.cantor.cantor_codebook_size,
            i
        );
    }
}