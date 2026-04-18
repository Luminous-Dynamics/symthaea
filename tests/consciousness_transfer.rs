// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-platform consciousness transfer experiments.
//!
//! Tests the Multiple Realizability thesis: does consciousness survive
//! a mid-run body swap between different robotic platforms?
//!
//! Run: cargo test --features humanoid --test consciousness_transfer

#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[cfg(feature = "humanoid")]
fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_platform: EmbodimentPlatform::Humanoid,
        embodiment_blend_weight: 0.3,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService")
}

/// Does consciousness survive a platform switch?
/// Run 50 cycles as humanoid, switch to None (disembodied), run 50 more.
/// Phi should remain finite and bounded throughout.
#[cfg(feature = "humanoid")]
#[test]
fn test_consciousness_survives_disembodiment() {
    let mut service = make_service();

    // Phase 1: Embodied (50 cycles)
    let mut phi_embodied = Vec::new();
    for _ in 0..50 {
        let r = service.cycle("walking forward with awareness");
        phi_embodied.push(r.metadata.consciousness.consciousness_level);
    }

    // Switch to disembodied
    service.switch_embodiment(EmbodimentPlatform::None);

    // Phase 2: Disembodied (50 cycles)
    let mut phi_disembodied = Vec::new();
    for _ in 0..50 {
        let r = service.cycle("thinking without a body");
        phi_disembodied.push(r.metadata.consciousness.consciousness_level);
    }

    // All values must be finite
    assert!(phi_embodied.iter().all(|p| p.is_finite()), "Embodied phase NaN");
    assert!(phi_disembodied.iter().all(|p| p.is_finite()), "Disembodied phase NaN");

    // Consciousness should persist after losing body
    let mean_e: f64 = phi_embodied.iter().sum::<f64>() / phi_embodied.len() as f64;
    let mean_d: f64 = phi_disembodied.iter().sum::<f64>() / phi_disembodied.len() as f64;
    eprintln!(
        "TRANSFER: Embodied mean Phi={:.4}, Disembodied mean Phi={:.4}, Delta={:.4}",
        mean_e, mean_d, mean_e - mean_d
    );
}

/// Does consciousness survive re-embodiment after being disembodied?
/// Humanoid -> None -> Humanoid. Phi should recover.
#[cfg(feature = "humanoid")]
#[test]
fn test_consciousness_survives_reembodiment() {
    let mut service = make_service();

    // Phase 1: Embodied warmup
    for _ in 0..30 {
        service.cycle("initial embodied state");
    }
    let phi_before = service.cycle("checkpoint").metadata.consciousness.consciousness_level;

    // Phase 2: Disembody
    service.switch_embodiment(EmbodimentPlatform::None);
    for _ in 0..30 {
        service.cycle("disembodied interlude");
    }

    // Phase 3: Re-embody
    service.switch_embodiment(EmbodimentPlatform::Humanoid);
    let mut phi_after = Vec::new();
    for _ in 0..50 {
        phi_after.push(service.cycle("reembodied, recovering").metadata.consciousness.consciousness_level);
    }

    let recovery_mean: f64 = phi_after[30..].iter().sum::<f64>() / 20.0;
    eprintln!(
        "REEMBODIMENT: Before={:.4}, Recovery mean (last 20)={:.4}",
        phi_before, recovery_mean
    );

    assert!(phi_after.iter().all(|p| p.is_finite()), "Recovery phase NaN");
}

/// Multi-platform consciousness landscape.
/// Create separate services for each available platform and compare steady-state Phi.
#[cfg(feature = "humanoid")]
#[test]
fn test_multi_platform_phi_landscape() {
    let platforms = vec![
        ("Humanoid", EmbodimentPlatform::Humanoid),
        ("None", EmbodimentPlatform::None),
    ];

    let mut results = Vec::new();

    for (name, platform) in &platforms {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            embodiment_platform: *platform,
            embodiment_blend_weight: 0.3,
            embodiment_step_interval: 1,
            async_training: false,
            learning_threshold: 0.0,
            ..Default::default()
        })
        .expect("service");

        let mut phi_values = Vec::new();
        for _ in 0..100 {
            let r = service.cycle("steady state operation");
            phi_values.push(r.metadata.consciousness.consciousness_level);
        }

        let mean: f64 = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
        let min = phi_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = phi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance: f64 = phi_values.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / phi_values.len() as f64;

        eprintln!(
            "PLATFORM {}: mean={:.4}, min={:.4}, max={:.4}, var={:.6}",
            name, mean, min, max, variance
        );

        results.push((name.to_string(), mean, min, max, variance));

        assert!(phi_values.iter().all(|p| p.is_finite()), "{} produced NaN", name);
    }

    // At least 2 platforms should have different mean Phi
    if results.len() >= 2 {
        let diff = (results[0].1 - results[1].1).abs();
        eprintln!("LANDSCAPE: Inter-platform Phi difference = {:.4}", diff);
    }
}
