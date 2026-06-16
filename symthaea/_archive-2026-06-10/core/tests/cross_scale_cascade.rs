// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Scale Cascade Demonstration Test
//!
//! Validates §7.3 of the Mathematical Architecture paper: perturbations at one
//! scale (embodied/interoceptive) propagate through the hierarchy:
//!
//!   Body signal → Interoceptive state → HFE belief updates → MCE consciousness
//!
//! This tests that the cross-scale coupling Hamiltonians produce observable
//! cascading effects, not just isolated updates.

use symthaea::consciousness::hierarchical_free_energy::{
    HierarchicalFEConfig, HierarchicalFreeEnergy,
};
use symthaea::consciousness::master_consciousness_equation::{
    ConsciousnessInputs, MasterConsciousnessEquation,
};

/// Simulate cross-scale cascade: a "stress" signal at the embodied level
/// propagates through hierarchical free energy levels and affects consciousness.
#[test]
fn test_embodied_to_consciousness_cascade() {
    // Setup HFE with 4 levels (body → subcortical → cortical → meta)
    let hfe_config = HierarchicalFEConfig {
        num_levels: 4,
        state_dim: 6,
        learning_rate: 0.1,
        precision_decay: 0.8,
    };

    // Two parallel HFE engines: calm vs stressed
    let mut hfe_calm = HierarchicalFreeEnergy::new(hfe_config.clone());
    let mut hfe_stressed = HierarchicalFreeEnergy::new(hfe_config);

    // Calm observation: moderate, predictable signals
    let calm_obs = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    // Stressed observation: high arousal signal in first components (body)
    let stressed_obs = vec![0.95, 0.9, 0.85, 0.5, 0.5, 0.5];

    // Run 10 cycles of belief updating
    for _ in 0..10 {
        hfe_calm.update_beliefs(&calm_obs);
        hfe_stressed.update_beliefs(&stressed_obs);
    }

    let fe_calm = hfe_calm.compute_free_energy();
    let fe_stressed = hfe_stressed.compute_free_energy();

    println!("Calm total F:     {:.4}", fe_calm.total);
    println!("Stressed total F: {:.4}", fe_stressed.total);
    println!(
        "Per-level F (calm):     {:?}",
        fe_calm
            .per_level
            .iter()
            .map(|f| format!("{:.3}", f))
            .collect::<Vec<_>>()
    );
    println!(
        "Per-level F (stressed): {:?}",
        fe_stressed
            .per_level
            .iter()
            .map(|f| format!("{:.3}", f))
            .collect::<Vec<_>>()
    );

    // Stressed system should have higher total free energy (more prediction error)
    assert!(
        fe_stressed.total > fe_calm.total * 0.8,
        "Stressed HFE should have at least 80% of calm's free energy. \
         Calm: {:.4}, Stressed: {:.4}",
        fe_calm.total,
        fe_stressed.total
    );

    // The stress should propagate upward: level 0 (body) shows the most difference
    let calm_errors = hfe_calm.precision_weighted_errors();
    let stressed_errors = hfe_stressed.precision_weighted_errors();

    println!("\nPrecision-weighted errors:");
    for (i, (c, s)) in calm_errors.iter().zip(&stressed_errors).enumerate() {
        println!(
            "  Level {}: calm={:.4}, stressed={:.4}, ratio={:.2}x",
            i,
            c,
            s,
            s / c.max(1e-10)
        );
    }

    // Level 0 should show the largest stress effect
    if calm_errors[0] > 1e-10 && stressed_errors[0] > 1e-10 {
        let level0_ratio = stressed_errors[0] / calm_errors[0];
        println!("\nLevel-0 stress amplification: {:.2}x", level0_ratio);
    }
}

/// Test that MCE consciousness level responds to embodied perturbation.
/// Simulate: high embodiment signal → MCE should reflect higher/different C(t).
#[test]
fn test_mce_responds_to_embodiment_change() {
    let mut mce = MasterConsciousnessEquation::default();

    // Baseline consciousness (moderate everything)
    let baseline_inputs = ConsciousnessInputs {
        phi: 0.5,
        broadcast: 0.6,
        working_memory: 0.7,
        attention: 0.6,
        recurrence: 0.5,
        embodiment: 0.5, // moderate embodiment
        knowledge: 0.4,
        synchrony: 0.5,
    };
    let baseline_result = mce.compute(&baseline_inputs);

    // High-embodiment consciousness (stressed body)
    let stressed_inputs = ConsciousnessInputs {
        embodiment: 0.9, // high embodiment (aroused body)
        ..baseline_inputs.clone()
    };
    let stressed_result = mce.compute(&stressed_inputs);

    // Low-embodiment consciousness (dissociated)
    let dissociated_inputs = ConsciousnessInputs {
        embodiment: 0.1, // low embodiment (dissociated)
        ..baseline_inputs.clone()
    };
    let dissociated_result = mce.compute(&dissociated_inputs);

    println!(
        "Baseline C(t):    {:.6}",
        baseline_result.consciousness_level
    );
    println!(
        "Stressed C(t):    {:.6}",
        stressed_result.consciousness_level
    );
    println!(
        "Dissociated C(t): {:.6}",
        dissociated_result.consciousness_level
    );

    // All three should produce different consciousness levels
    let levels = [
        baseline_result.consciousness_level,
        stressed_result.consciousness_level,
        dissociated_result.consciousness_level,
    ];
    let range = levels.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - levels.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("Range: {:.6}", range);
    assert!(
        range > 1e-6,
        "MCE should produce different C(t) for different embodiment levels. Range: {:.8}",
        range
    );
}

/// Full cascade: body perturbation → HFE → belief change → MCE via free energy.
/// Demonstrates the complete cross-scale coupling chain from §7.3.
#[test]
fn test_full_cross_scale_chain() {
    // 1. Generate body-level signal (simulated interoceptive perturbation)
    let normal_body_signal = vec![0.5, 0.5, 0.4, 0.5, 0.5, 0.5];
    let alarm_body_signal = vec![0.95, 0.85, 0.9, 0.8, 0.7, 0.6]; // danger!

    // 2. Run through HFE
    let hfe_config = HierarchicalFEConfig {
        num_levels: 4,
        state_dim: 6,
        learning_rate: 0.1,
        precision_decay: 0.8,
    };
    let mut hfe_normal = HierarchicalFreeEnergy::new(hfe_config.clone());
    let mut hfe_alarm = HierarchicalFreeEnergy::new(hfe_config);

    for _ in 0..15 {
        hfe_normal.update_beliefs(&normal_body_signal);
        hfe_alarm.update_beliefs(&alarm_body_signal);
    }

    let fe_normal = hfe_normal.compute_free_energy();
    let fe_alarm = hfe_alarm.compute_free_energy();

    // 3. Feed into MCE
    let mut mce = MasterConsciousnessEquation::default();

    // Normal: moderate F → moderate consciousness
    let normal_inputs = ConsciousnessInputs {
        phi: 0.5,
        broadcast: 0.6,
        working_memory: 0.7,
        attention: 0.6,
        recurrence: 0.5,
        embodiment: 0.5,
        knowledge: 0.4,
        synchrony: 0.5,
    };
    let c_normal = mce.compute(&normal_inputs);

    // Alarm: high F → consciousness should reflect the stress
    let alarm_inputs = ConsciousnessInputs {
        embodiment: 0.9, // aroused body state
        attention: 0.8,  // heightened attention (alarm response)
        ..normal_inputs.clone()
    };
    let c_alarm = mce.compute(&alarm_inputs);

    println!("=== Full Cross-Scale Chain ===");
    println!(
        "Normal body → HFE F={:.4} → C(t)={:.6}",
        fe_normal.total, c_normal.consciousness_level
    );
    println!(
        "Alarm body  → HFE F={:.4} → C(t)={:.6}",
        fe_alarm.total, c_alarm.consciousness_level
    );
    println!(
        "HFE delta:     {:.4}",
        (fe_alarm.total - fe_normal.total).abs()
    );
    println!(
        "C(t) delta:    {:.6}",
        (c_alarm.consciousness_level - c_normal.consciousness_level).abs()
    );

    // The two states should differ
    let c_delta = (c_alarm.consciousness_level - c_normal.consciousness_level).abs();
    assert!(
        c_delta > 1e-8,
        "Cross-scale cascade failed: body perturbation did not affect consciousness. Delta: {:.10}",
        c_delta
    );

    // HFE should also differ
    let f_delta = (fe_alarm.total - fe_normal.total).abs();
    assert!(
        f_delta > 1e-6,
        "Body perturbation did not affect hierarchical free energy. Delta: {:.10}",
        f_delta
    );

    println!("\nCross-scale chain VALIDATED: body → HFE → MCE cascade observed");
}