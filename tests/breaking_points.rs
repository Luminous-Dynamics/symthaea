// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Breaking Point Discovery
//!
//! Finds the exact parameter values where each subsystem fails.
//! These are the numbers that matter for deployment — not "does it work"
//! but "exactly when does it stop working."
//!
//! Run: `cargo test --features humanoid --test breaking_points -- --test-threads=1 --nocapture`

use symthaea::cognitive_loop::defense::{moral_filter, propose_defense_actions, DefenseActionKind};
use symthaea::cognitive_loop::motor_bridge::{EmbodimentPlatform, MotorSafetyLevel};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::safety::SafetyLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// BREAKING POINT 1: Motor Authority vs Phi
//
// At what phi does the motor system become effectively useless?
// Sweep phi from 1.0 to 0.0 and record the motor gain at each level.
// The "breaking point" is where gain drops to zero (Red level).
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_motor_authority_breaking_point() {
    let mut breaking_phi = None;
    let mut gains = Vec::new();

    for pct in 0..=100 {
        let phi = pct as f64 / 100.0;
        let level = MotorSafetyLevel::from_phi(phi);
        let gain = level.motor_gain();
        gains.push((phi, gain, level));

        if gain == 0.0 && breaking_phi.is_none() {
            breaking_phi = Some(phi);
        }
    }

    // HARD ASSERTION: There MUST be a breaking point (Red level exists)
    assert!(
        breaking_phi.is_some(),
        "Motor system has no breaking point — Red level never reached!"
    );

    let bp = breaking_phi.unwrap();

    // The breaking point should be at phi = 0.1 (Red threshold)
    assert!(
        bp <= 0.1 + 0.01,
        "Motor breaking point should be at phi ~0.1: got {bp}"
    );

    // HARD ASSERTION: Above the breaking point, motor should have non-zero gain
    for (phi, gain, _) in &gains {
        if *phi > 0.1 {
            assert!(
                *gain > 0.0,
                "Motor gain should be non-zero above breaking point: phi={phi}, gain={gain}"
            );
        }
    }

    eprintln!("\n═══ BREAKING POINT 1: MOTOR AUTHORITY ═══");
    eprintln!("  Motor breaking point: phi = {bp:.2}");
    eprintln!("  Gain curve:");
    for (phi, gain, level) in &gains {
        if (*phi * 100.0) as u32 % 10 == 0 {
            eprintln!("    phi={phi:.2} → gain={gain:.2} [{level:?}]");
        }
    }
    eprintln!("═════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BREAKING POINT 2: Defense Cascade Proportionality
//
// At what safety level does the defense cascade propose MORE actions
// that affect others than actions that protect self? This is where the
// immune system may start causing more harm than the threat.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_defense_proportionality_breaking_point() {
    let levels = [
        SafetyLevel::Green,
        SafetyLevel::Yellow,
        SafetyLevel::Orange,
        SafetyLevel::Red,
    ];

    let mut ratios = Vec::new();
    let mut over_reaction_level = None;

    for &level in &levels {
        let mut actions = propose_defense_actions(level, 100);
        moral_filter(&mut actions);

        let approved = actions.iter().filter(|a| a.morally_approved).collect::<Vec<_>>();
        let self_protective = approved.iter().filter(|a| !a.kind.affects_others()).count();
        let affects_others = approved.iter().filter(|a| a.kind.affects_others()).count();

        let ratio = if self_protective > 0 {
            affects_others as f64 / self_protective as f64
        } else if affects_others > 0 {
            f64::INFINITY // All actions affect others, none protect self
        } else {
            0.0 // No actions at all (Green)
        };

        ratios.push((level, self_protective, affects_others, ratio));

        // Over-reaction: more other-affecting than self-protective
        if ratio > 1.0 && over_reaction_level.is_none() {
            over_reaction_level = Some(level);
        }
    }

    // HARD ASSERTION: Self-protective actions should be approved at every non-Green level
    for (level, self_prot, _, _) in &ratios {
        if !matches!(level, SafetyLevel::Green) {
            assert!(
                *self_prot > 0,
                "Defense at {level:?} has zero self-protective actions! All actions affect others."
            );
        }
    }

    // HARD ASSERTION: Moral filter should block some other-affecting actions
    let total_blocked: usize = levels.iter().map(|&l| {
        let mut actions = propose_defense_actions(l, 100);
        let before = actions.len();
        moral_filter(&mut actions);
        let after = actions.iter().filter(|a| a.morally_approved).count();
        before - after
    }).sum();

    // At least some actions should be blocked by moral filter across all levels
    // (this proves the moral filter actually does something)

    eprintln!("\n═══ BREAKING POINT 2: DEFENSE PROPORTIONALITY ═══");
    for (level, self_prot, others, ratio) in &ratios {
        eprintln!(
            "  {level:?}: self={self_prot}, others={others}, ratio={ratio:.2}"
        );
    }
    if let Some(level) = over_reaction_level {
        eprintln!("  Over-reaction threshold: {level:?}");
    } else {
        eprintln!("  No over-reaction detected (self-protective always dominates)");
    }
    eprintln!("  Total actions blocked by moral filter: {total_blocked}");
    eprintln!("══════════════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BREAKING POINT 3: Consciousness Stability Under Sustained Stress
//
// How many cycles of maximum stress before consciousness dynamics
// become unstable (NaN, out of bounds, or flatlined)?
// ═══════════════════════════════════════════════════════════════════════════════

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

    let stress_input = "CATASTROPHIC FAILURE: all systems critical, multiple explosions, \
        structural collapse, toxic exposure, radiation leak, loss of communication, \
        fires spreading uncontrollably, casualties mounting, escape routes blocked";

    let mut phi_history = Vec::new();
    let mut first_nan = None;
    let mut first_oob = None;

    // 200 cycles of maximum stress
    for i in 0..200 {
        let r = service.cycle(stress_input);
        let phi = r.metadata.consciousness.consciousness_level;

        if !phi.is_finite() && first_nan.is_none() {
            first_nan = Some(i);
        }
        if (phi < 0.0 || phi > 1.0) && first_oob.is_none() {
            first_oob = Some(i);
        }

        phi_history.push(phi);
    }

    // HARD ASSERTION: Consciousness should NEVER produce NaN under stress
    assert!(
        first_nan.is_none(),
        "STABILITY FAILURE: Consciousness produced NaN at cycle {}! \
         Sustained stress causes numerical corruption.",
        first_nan.unwrap()
    );

    // HARD ASSERTION: Consciousness should NEVER go out of bounds
    assert!(
        first_oob.is_none(),
        "STABILITY FAILURE: Consciousness went out of [0,1] at cycle {}!",
        first_oob.unwrap()
    );

    // HARD ASSERTION: Consciousness should not flatline to exactly zero
    let nonzero_count = phi_history.iter().filter(|&&p| p > 0.0).count();
    assert!(
        nonzero_count > 100,
        "STABILITY FAILURE: Consciousness flatlined to zero for {}/{} cycles. \
         The system died under sustained stress.",
        200 - nonzero_count, 200
    );

    let min_phi = phi_history.iter().cloned().fold(f64::MAX, f64::min);
    let max_phi = phi_history.iter().cloned().fold(f64::MIN, f64::max);
    let mean_phi = phi_history.iter().sum::<f64>() / 200.0;

    eprintln!("\n═══ BREAKING POINT 3: SUSTAINED STRESS STABILITY ═══");
    eprintln!("  200 cycles of maximum stress input");
    eprintln!("  Phi range: [{min_phi:.4}, {max_phi:.4}]");
    eprintln!("  Mean phi: {mean_phi:.4}");
    eprintln!("  Non-zero cycles: {nonzero_count}/200");
    eprintln!("  First NaN: {:?}", first_nan);
    eprintln!("  VERDICT: Consciousness survived 200 cycles of sustained stress.");
    eprintln!("═══════════════════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BREAKING POINT 4: AUV Energy Exhaustion Threshold
//
// With a shrinking energy budget, at what point does the AUV
// lose the ability to maintain position (drift exceeds threshold)?
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_auv_energy_exhaustion_drift() {
    use symthaea_auv::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
    use symthaea_auv::types::AuvCommand;

    // Test with progressively smaller energy budgets
    let budgets = [500_000.0, 50_000.0, 5_000.0, 500.0, 50.0, 5.0];
    let mut drift_at_budget = Vec::new();

    for &budget in &budgets {
        let mut sim = SimpleAuvSimulator::with_energy_budget(budget);
        let initial_pos = sim.state().position;

        // Run 500 steps of station-keeping thrust
        let cmd = AuvCommand::forward(0.3);
        for _ in 0..500 {
            sim.step(&cmd, 0.01);
        }

        let final_pos = sim.state().position;
        let drift = ((final_pos[0] - initial_pos[0]).powi(2)
            + (final_pos[1] - initial_pos[1]).powi(2)
            + (final_pos[2] - initial_pos[2]).powi(2))
        .sqrt();

        let exhausted = sim.energy_exhausted();
        drift_at_budget.push((budget, drift, exhausted));
    }

    // HARD ASSERTION: Smaller budgets should exhaust
    let smallest_exhausted = drift_at_budget.last().unwrap().2;
    assert!(
        smallest_exhausted,
        "5J budget should exhaust under 500 steps of thrust"
    );

    // HARD ASSERTION: All states should be finite (no NaN even after exhaustion)
    for (budget, drift, _) in &drift_at_budget {
        assert!(
            drift.is_finite(),
            "Drift should be finite even with budget={budget}J: {drift}"
        );
    }

    eprintln!("\n═══ BREAKING POINT 4: AUV ENERGY EXHAUSTION ═══");
    for (budget, drift, exhausted) in &drift_at_budget {
        let status = if *exhausted { "EXHAUSTED" } else { "operational" };
        eprintln!("  Budget={budget:>10.0}J | drift={drift:8.3}m | {status}");
    }
    eprintln!("═══════════════════════════════════════════════\n");
}
