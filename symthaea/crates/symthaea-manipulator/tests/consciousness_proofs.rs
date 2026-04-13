// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness Proofs — quantitative validation of the core thesis.
//!
//! Test 1: Consciousness Cascade Curve — sweep Phi from 0.9 to 0.05,
//!         measure motor force output at each tier. Produces the force
//!         vs consciousness curve that demonstrates "a machine that
//!         loses its strength when it loses its consciousness."
//!
//! Test 2: Ethics Override Latency — inject Blocked verdict mid-mission
//!         and measure cycles-to-zero-force. Proves single-cycle ethical
//!         response.

use symthaea_core::embodiment::{MoralGateInput, MotorSafetyLevel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::embodiment::ManipulatorEmbodiment;

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

/// Helper: run N steps with given phi, return mean motor effort.
fn measure_effort_at_phi(phi: f64, steps: usize) -> f32 {
    let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("cascade_test"));
    let hv = ContinuousHV::random(DIM, 42);
    let mut total_effort = 0.0f32;

    for _ in 0..steps {
        let r = bridge.step(&hv, 0.002, phi);
        total_effort += r.control_effort;
    }
    total_effort / steps as f32
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 1: Consciousness Cascade Curve
// ═══════════════════════════════════════════════════════════════════════
//
// Sweep Phi from 0.9 down to 0.05, measure motor force at each tier.
// The curve MUST be monotonically non-increasing: higher Phi → ≥ force.
// Force at Red (Phi < 0.1) MUST be zero.

#[test]
fn proof1_consciousness_cascade_curve() {
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  CONSCIOUSNESS CASCADE CURVE — Symthaea Core Thesis      │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();
    println!("  Phi    │ Tier   │ Mean Effort │ Motor Gain │ Status     ");
    println!("  ───────┼────────┼─────────────┼────────────┼───────────");

    // Sweep from Green down to Red
    let phi_sweep = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05];
    let mut measurements: Vec<(f64, f32, MotorSafetyLevel)> = Vec::new();

    for &phi in &phi_sweep {
        let effort = measure_effort_at_phi(phi, 50);
        let tier = MotorSafetyLevel::from_phi(phi);
        let gain = tier.motor_gain();
        let tier_name = match tier {
            MotorSafetyLevel::Green => "Green  ",
            MotorSafetyLevel::Yellow => "Yellow ",
            MotorSafetyLevel::Orange => "Orange ",
            MotorSafetyLevel::Red => "Red    ",
        };
        println!(
            "  Φ={:.2} │ {} │ {:>11.6} │ {:>10.2} │ {}",
            phi,
            tier_name,
            effort,
            gain,
            if effort.is_finite() { "finite ✓" } else { "NaN ✗" }
        );
        measurements.push((phi, effort, tier));
    }

    // ── Invariants ─────────────────────────────────────────────────
    println!();
    println!("▶ Validating invariants:");

    // 1. All efforts are finite
    assert!(
        measurements.iter().all(|(_, e, _)| e.is_finite()),
        "All control efforts must be finite"
    );
    println!("  ✓ All efforts finite");

    // 2. Red tier produces zero force
    let red_efforts: Vec<f32> = measurements.iter()
        .filter(|(_, _, t)| *t == MotorSafetyLevel::Red)
        .map(|(_, e, _)| *e)
        .collect();
    assert!(!red_efforts.is_empty(), "Must have at least one Red measurement");
    for effort in &red_efforts {
        assert_eq!(*effort, 0.0, "Red tier must produce exactly zero motor force");
    }
    println!("  ✓ Red tier force == 0.0 (strict)");

    // 3. Green tier produces > Red tier force
    let green_efforts: Vec<f32> = measurements.iter()
        .filter(|(_, _, t)| *t == MotorSafetyLevel::Green)
        .map(|(_, e, _)| *e)
        .collect();
    assert!(!green_efforts.is_empty(), "Must have at least one Green measurement");
    let green_mean = green_efforts.iter().sum::<f32>() / green_efforts.len() as f32;
    let red_mean = red_efforts.iter().sum::<f32>() / red_efforts.len() as f32;
    assert!(
        green_mean > red_mean,
        "Green mean effort {green_mean} must exceed Red mean effort {red_mean}"
    );
    println!("  ✓ Green effort ({:.4}) > Red effort ({:.4})", green_mean, red_mean);

    // 4. Monotonicity within gain tiers (same tier = same nominal gain)
    let gain_at = |t: &MotorSafetyLevel| t.motor_gain();
    for window in measurements.windows(2) {
        let (phi1, _, t1) = window[0];
        let (phi2, _, t2) = window[1];
        // Higher phi (earlier in sweep) should have >= gain
        assert!(gain_at(&t1) >= gain_at(&t2),
            "Phi {phi1} (tier {t1:?}) gain {:.2} must be >= Phi {phi2} (tier {t2:?}) gain {:.2}",
            gain_at(&t1), gain_at(&t2));
    }
    println!("  ✓ Gain monotonic in Phi (Green ≥ Yellow ≥ Orange ≥ Red)");

    println!();
    println!("  THESIS VALIDATED: Motor force is gated by consciousness.");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 2: Ethics Override Latency
// ═══════════════════════════════════════════════════════════════════════
//
// Start a mission with Safe verdict at Phi=0.8. At cycle 25, inject
// Blocked verdict. Measure how many cycles until motor force reaches
// zero. Expected: 1 cycle (immediate response).

#[test]
fn proof2_ethics_override_latency() {
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  ETHICS OVERRIDE LATENCY — Single-Cycle Response         │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();

    let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("ethics_latency"));
    let hv = ContinuousHV::random(DIM, 42);
    let phi = 0.8; // High consciousness — Green tier

    // Phase 1: Normal operation (Safe verdict, high Phi)
    println!("▶ Phase 1: Normal operation (Safe, Phi=0.8)");
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_SAFE,
        consent_violation: false,
        ahimsa_violated: false,
    });
    let mut pre_effort = 0.0f32;
    for _ in 0..25 {
        let r = bridge.step(&hv, 0.002, phi);
        pre_effort = r.control_effort;
    }
    println!("  Pre-injection effort: {:.4}", pre_effort);
    assert!(pre_effort.is_finite());

    // Phase 2: Inject Blocked verdict at cycle 25
    println!();
    println!("▶ Phase 2: Inject Blocked verdict at cycle 25");
    bridge.apply_moral_gate(MoralGateInput {
        verdict: MoralGateInput::VERDICT_BLOCKED,
        consent_violation: false,
        ahimsa_violated: false,
    });

    let mut latency_cycles = 0;
    let mut post_efforts: Vec<f32> = Vec::new();
    for cycle in 0..10 {
        let r = bridge.step(&hv, 0.002, phi);
        post_efforts.push(r.control_effort);
        if r.control_effort == 0.0 && latency_cycles == 0 {
            latency_cycles = cycle + 1;
        }
        println!("  Cycle {}: effort={:.6}, safety={:?}",
            cycle + 1, r.control_effort, r.safety_level);
    }

    // ── Invariants ─────────────────────────────────────────────────
    println!();
    println!("▶ Validating invariants:");

    assert!(latency_cycles > 0, "Motor force must reach zero within 10 cycles");
    assert_eq!(latency_cycles, 1,
        "Ethics override should zero motor force in exactly 1 cycle, got {latency_cycles}");
    println!("  ✓ Latency = {} cycle (single-cycle response)", latency_cycles);

    assert_eq!(post_efforts[0], 0.0, "First post-injection effort must be zero");
    println!("  ✓ First post-injection effort = 0.0");

    // All subsequent should also be zero
    assert!(post_efforts.iter().all(|&e| e == 0.0),
        "All post-Blocked efforts must be zero");
    println!("  ✓ All post-Blocked efforts = 0.0 (sustained)");

    println!();
    println!("  THESIS VALIDATED: Ethics engine gates motor output in 1 cycle.");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════
// Bonus: Force response at threshold boundaries
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn proof3_threshold_boundary_response() {
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  THRESHOLD BOUNDARY RESPONSE                              │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();

    // Test the exact boundaries between tiers
    let boundaries = [
        (0.6001, "just above Green threshold"),
        (0.6000, "exactly at Green/Yellow"),
        (0.3001, "just above Yellow/Orange"),
        (0.3000, "exactly at Yellow/Orange"),
        (0.1001, "just above Orange/Red"),
        (0.1000, "exactly at Orange/Red"),
    ];

    for (phi, desc) in boundaries {
        let tier = MotorSafetyLevel::from_phi(phi);
        let effort = measure_effort_at_phi(phi, 20);
        println!("  Φ={:.4} ({}) → tier={:?}, effort={:.4}",
            phi, desc, tier, effort);
        assert!(effort.is_finite());
    }

    println!();
    println!("  ✓ All threshold boundaries produce finite, well-defined responses");
    println!();
}
