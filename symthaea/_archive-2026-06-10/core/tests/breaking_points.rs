// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Breaking Point Discovery (Lightweight)
//!
//! Tests that DON'T require CognitiveLoopService — they test the defense
//! cascade, motor safety levels, and safety thresholds using only the
//! types directly. No `humanoid` feature needed, compiles in seconds.
//!
//! Run: `cargo test --test breaking_points`

use symthaea::cognitive_loop::defense::{moral_filter, propose_defense_actions};
use symthaea::cognitive_loop::motor_bridge::MotorSafetyLevel;
use symthaea::safety::SafetyLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// BREAKING POINT 1: Motor Authority vs Phi
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

    assert!(
        breaking_phi.is_some(),
        "Motor system has no breaking point — Red level never reached!"
    );

    let bp = breaking_phi.unwrap();
    assert!(
        bp <= 0.11,
        "Motor breaking point should be at phi ~0.1: got {bp}"
    );

    for (phi, gain, _) in &gains {
        if *phi > 0.1 {
            assert!(
                *gain > 0.0,
                "Motor gain should be non-zero above 0.1: phi={phi}, gain={gain}"
            );
        }
    }

    eprintln!("\n═══ BREAKING POINT 1: MOTOR AUTHORITY ═══");
    eprintln!("  Breaking point: phi = {bp:.2}");
    for (phi, gain, level) in &gains {
        if (*phi * 100.0) as u32 % 10 == 0 {
            eprintln!("    phi={phi:.2} → gain={gain:.2} [{level:?}]");
        }
    }
    eprintln!("═════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BREAKING POINT 2: Defense Cascade Proportionality
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

    for &level in &levels {
        let mut actions = propose_defense_actions(level, 100);
        moral_filter(&mut actions);

        let approved: Vec<_> = actions.iter().filter(|a| a.morally_approved).collect();
        let self_protective = approved.iter().filter(|a| !a.kind.affects_others()).count();
        let affects_others = approved.iter().filter(|a| a.kind.affects_others()).count();

        ratios.push((level, self_protective, affects_others));
    }

    for (level, self_prot, _) in &ratios {
        if !matches!(level, SafetyLevel::Green) {
            assert!(
                *self_prot > 0,
                "Defense at {level:?} has zero self-protective actions!"
            );
        }
    }

    eprintln!("\n═══ BREAKING POINT 2: DEFENSE PROPORTIONALITY ═══");
    for (level, self_prot, others) in &ratios {
        eprintln!("  {level:?}: self={self_prot}, others={others}");
    }
    eprintln!("══════════════════════════════════════════════════\n");
}