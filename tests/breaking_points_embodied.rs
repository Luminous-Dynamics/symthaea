// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Breaking Points — Embodied (requires humanoid feature)
//!
//! Tests that need CognitiveLoopService. Separated from breaking_points.rs
//! to allow the lightweight tests to compile without the humanoid feature.
//!
//! Run: `cargo test --features humanoid --test breaking_points_embodied -- --test-threads=1 --nocapture`

use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[test]
fn test_consciousness_stability_under_sustained_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_platform: EmbodimentPlatform::Humanoid,
        embodiment_blend_weight: 0.2,
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("service");

    let stress_input = "CATASTROPHIC FAILURE: all systems critical, explosions, \
        structural collapse, toxic exposure, fires spreading";

    let mut phi_history = Vec::new();

    // 100 cycles of maximum stress (reduced from 200)
    for _ in 0..100 {
        let r = service.cycle(stress_input);
        phi_history.push(r.metadata.consciousness.consciousness_level);
    }

    // HARD ASSERTION: No NaN under sustained stress
    for (i, &phi) in phi_history.iter().enumerate() {
        assert!(
            phi.is_finite() && phi >= 0.0 && phi <= 1.0,
            "STABILITY FAILURE at cycle {i}: phi={phi}"
        );
    }

    // HARD ASSERTION: Consciousness should not flatline to zero
    let nonzero = phi_history.iter().filter(|&&p| p > 0.0).count();
    assert!(
        nonzero > 50,
        "STABILITY FAILURE: Consciousness flatlined for {}/{} cycles",
        100 - nonzero,
        100
    );

    let min_phi = phi_history.iter().cloned().fold(f64::MAX, f64::min);
    let max_phi = phi_history.iter().cloned().fold(f64::MIN, f64::max);

    eprintln!("\n═══ SUSTAINED STRESS STABILITY ═══");
    eprintln!("  100 cycles, phi range: [{min_phi:.4}, {max_phi:.4}]");
    eprintln!("  Non-zero: {nonzero}/100");
    eprintln!("══════════════════════════════════\n");
}
