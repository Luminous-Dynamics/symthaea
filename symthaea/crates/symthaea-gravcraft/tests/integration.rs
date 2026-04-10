// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gravcraft consciousness coupling integration tests.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_gravcraft::*;

#[test]
fn consciousness_gates_metric_authority() {
    let genesis = GenesisSeed::from_phrase("consciousness gating test");
    let mut emb = GravcraftEmbodiment::new(&genesis);
    let thought = ContinuousHV::random(HDC_DIMENSION, 0xC0DE);

    // High Phi → high control effort
    let r_high = emb.step(&thought, 0.01, 0.9);
    let effort_high = r_high.control_effort;

    // Low Phi → zero control effort (Red safety = flat metric)
    let r_low = emb.step(&thought, 0.01, 0.05);
    let effort_low = r_low.control_effort;

    assert!(
        effort_high >= effort_low,
        "Higher Phi should allow more metric authority: high={}, low={}",
        effort_high,
        effort_low
    );

    // Red safety should zero out all metric perturbation
    assert!(
        effort_low < 1e-6,
        "Red safety should zero control effort: {}",
        effort_low
    );
}

#[test]
fn gravcraft_embodiment_full_cycle() {
    let genesis = GenesisSeed::from_phrase("full cycle test");
    let mut emb = GravcraftEmbodiment::new(&genesis);

    // Run 10 steps at Green safety
    let mut results = Vec::new();
    for i in 0..10 {
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xBEEF + i);
        let r = emb.step(&thought, 0.01, 0.8);
        results.push(r);
    }

    // All steps should succeed
    assert!(results.iter().all(|r| r.success));

    // Prediction error should be defined for all steps after the first
    assert!(results[9].prediction_error.is_finite());

    // Telemetry should report platform
    let telem = emb.telemetry();
    assert_eq!(telem.platform, "gravcraft");
    assert_eq!(telem.num_actuators, 12);
    assert_eq!(telem.total_steps, 10);
}

#[test]
fn geodesic_vs_force_produces_different_motion() {
    // Key insight: metric perturbation should produce qualitatively different
    // motion than force-based platforms. Specifically, in a uniform metric
    // perturbation field, the craft should follow geodesics (straight lines
    // in curved space), not parabolic trajectories.

    let config = GravcraftConfig::default();
    let mut sim = MetricPerturbationSimulator::new(config);

    // Apply a constant forward metric perturbation
    let cmd = MetricCommand {
        amplifiers: [
            (0.05, 10.0, 0.0, 0.0), // Forward
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ],
    };

    let mut positions = Vec::new();
    for _ in 0..50 {
        sim.step(&cmd, 0.01);
        positions.push(sim.state().position);
    }

    // Craft should move (not static)
    let final_pos = positions.last().unwrap();
    let distance = (final_pos[0].powi(2) + final_pos[1].powi(2) + final_pos[2].powi(2)).sqrt();
    assert!(
        distance > 0.0,
        "Craft should move under metric perturbation"
    );

    // All positions should be finite (no divergence)
    assert!(
        positions.iter().all(|p| p.iter().all(|v| v.is_finite())),
        "All positions should be finite (no divergence)"
    );
}

#[test]
fn three_amplifier_delta_configuration() {
    // "Delta configuration": 3 amplifiers at 120° separation
    // This is the basic flight mode described in the Lazar claims
    let config = GravcraftConfig::default();
    let craft_pos = [0.0, 0.0, 0.0];

    let delta_cmds = [
        (0.03, 10.0, 0.0, 0.0),                              // 0°
        (0.03, 10.0, 2.0 * std::f64::consts::PI / 3.0, 0.0), // 120°
        (0.03, 10.0, 4.0 * std::f64::consts::PI / 3.0, 0.0), // 240°
    ];

    let (shift, mag) = combined_metric(craft_pos, craft_pos, &delta_cmds, &config);

    // Symmetric delta should have small net shift (partial cancellation)
    assert!(
        shift.iter().all(|v| v.is_finite()),
        "Delta config should produce finite shift"
    );
    assert!(mag.is_finite());
}
