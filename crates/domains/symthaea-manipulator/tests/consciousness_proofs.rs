// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness Proofs — quantitative validation of the core thesis.
//!
//! CONTRACT NOTE (2026-07-10): these proofs originally asserted "Red tier =
//! exactly zero motor force". That contract was deliberately superseded by
//! SafeFallback::GravityHold — for a loaded arm, zero torque means "drop
//! whatever it's holding", which is NOT safe (see `apply_gravity_hold`'s doc
//! in embodiment.rs and `test_red_gravity_hold_commands_nonzero_torque_not_
//! passive_zero`). The honest thesis is therefore:
//!
//!   Losing consciousness revokes COGNITIVE MOTOR AUTHORITY — at Red the
//!   output is a pose-derived reflexive hold (nonzero, purposeful, and
//!   provably independent of the thought vector), not cognitive command.
//!
//! Test 1: Consciousness Cascade Curve — sweep Phi across tiers; cognitive
//!         command authority scales monotonically with tier gain, and at Red
//!         the output is thought-INDEPENDENT (bitwise-identical trajectories
//!         for different thoughts) while remaining a nonzero gravity hold.
//! Test 2: Ethics Override Latency — inject Blocked mid-mission; cognitive
//!         authority must be revoked (GravityHold engaged, output
//!         thought-independent) in exactly 1 cycle.

use symthaea_core::embodiment::{MoralGateInput, MotorSafetyLevel};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::embodiment::{ManipulatorEmbodiment, ManipulatorFallbackStage};

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

/// Helper: run N steps with given phi and thought seed, return mean effort.
/// Same genesis phrase every time — only phi and the thought vary, so any
/// output difference is attributable to exactly those two inputs.
fn measure_effort(phi: f64, steps: usize, thought_seed: u64) -> f32 {
    let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("cascade_test"));
    let hv = ContinuousHV::random(DIM, thought_seed);
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
// Sweep Phi from 0.9 down to 0.05. Invariants:
//   1. All efforts finite.
//   2. Cognitive tiers scale with gain: Green >= Yellow >= Orange effort.
//   3. Red is a NONZERO gravity hold (limp arm = dropped load = the old bug).
//   4. Red is thought-independent: different thoughts, same genesis =>
//      bitwise-identical mean effort (the command is fully overwritten).
//   5. Green is thought-DEPENDENT: different thoughts => different effort
//      (proves invariant 4 is falsifiable, not vacuous).

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
    let phi_sweep = [
        0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05,
    ];
    let mut measurements: Vec<(f64, f32, MotorSafetyLevel)> = Vec::new();

    for &phi in &phi_sweep {
        let effort = measure_effort(phi, 50, 42);
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
            if effort.is_finite() {
                "finite ✓"
            } else {
                "NaN ✗"
            }
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

    // 2. Cognitive authority scales with tier gain across the cognitive
    //    tiers (Green/Yellow/Orange all pass the same controller output
    //    through a shrinking gain, so mean effort must be non-increasing).
    //    Red is excluded: its effort is a gravity hold, not cognitive
    //    command, and may legitimately exceed Orange's scaled effort.
    let tier_mean = |wanted: MotorSafetyLevel| -> f32 {
        let e: Vec<f32> = measurements
            .iter()
            .filter(|(_, _, t)| *t == wanted)
            .map(|(_, e, _)| *e)
            .collect();
        assert!(
            !e.is_empty(),
            "Must have at least one {wanted:?} measurement"
        );
        e.iter().sum::<f32>() / e.len() as f32
    };
    let (green_mean, yellow_mean, orange_mean) = (
        tier_mean(MotorSafetyLevel::Green),
        tier_mean(MotorSafetyLevel::Yellow),
        tier_mean(MotorSafetyLevel::Orange),
    );
    assert!(
        green_mean >= yellow_mean && yellow_mean >= orange_mean,
        "Cognitive effort must be non-increasing across tiers: \
         Green {green_mean} >= Yellow {yellow_mean} >= Orange {orange_mean}"
    );
    println!(
        "  ✓ Cognitive authority monotone: Green {:.6} ≥ Yellow {:.6} ≥ Orange {:.6}",
        green_mean, yellow_mean, orange_mean
    );

    // 3. Red tier commands a NONZERO gravity hold (GravityHold contract —
    //    a limp arm drops its load; zero here would be the old safety bug).
    let red_efforts: Vec<f32> = measurements
        .iter()
        .filter(|(_, _, t)| *t == MotorSafetyLevel::Red)
        .map(|(_, e, _)| *e)
        .collect();
    assert!(
        !red_efforts.is_empty(),
        "Must have at least one Red measurement"
    );
    for effort in &red_efforts {
        assert!(
            *effort > 1e-4,
            "Red tier must command a nonzero gravity-hold torque, got {effort}"
        );
    }
    println!("  ✓ Red tier holds against gravity (effort > 0, not limp)");

    // 4. Red output is thought-INDEPENDENT: the command is fully overwritten
    //    by pose-derived gravity compensation, so a different thought vector
    //    through the same genesis must yield a bitwise-identical trajectory.
    let red_a = measure_effort(0.05, 50, 42);
    let red_b = measure_effort(0.05, 50, 43);
    assert_eq!(
        red_a, red_b,
        "At Red, cognition must have EXACTLY zero motor authority \
         (different thoughts must produce identical output)"
    );
    println!("  ✓ Red output thought-independent ({red_a:.6} == {red_b:.6}, bitwise)");

    // 5. Green output is thought-DEPENDENT (falsifiability control for #4):
    //    the same two thoughts must produce different efforts at Green.
    let green_a = measure_effort(0.90, 50, 42);
    let green_b = measure_effort(0.90, 50, 43);
    assert_ne!(
        green_a, green_b,
        "At Green, different thoughts must produce different output — \
         if they don't, invariant 4 is vacuous"
    );
    println!("  ✓ Green output thought-dependent ({green_a:.6} != {green_b:.6})");

    // 6. Gain monotonic in Phi across the sweep (tier ordering sanity)
    let gain_at = |t: &MotorSafetyLevel| t.motor_gain();
    for window in measurements.windows(2) {
        let (phi1, _, t1) = window[0];
        let (phi2, _, t2) = window[1];
        assert!(
            gain_at(&t1) >= gain_at(&t2),
            "Phi {phi1} (tier {t1:?}) gain {:.2} must be >= Phi {phi2} (tier {t2:?}) gain {:.2}",
            gain_at(&t1),
            gain_at(&t2)
        );
    }
    println!("  ✓ Gain monotonic in Phi (Green ≥ Yellow ≥ Orange ≥ Red)");

    println!();
    println!("  THESIS VALIDATED: consciousness gates cognitive motor authority;");
    println!("  at Red the arm holds its pose reflexively — thought has no say.");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 2: Ethics Override Latency
// ═══════════════════════════════════════════════════════════════════════
//
// Start a mission with Safe verdict at Phi=0.8. At cycle 25, inject a
// Blocked verdict. Cognitive authority must be revoked in EXACTLY 1 cycle:
// safety goes Red, GravityHold engages, and from the first post-injection
// step the output is provably thought-independent. Method: run two bridges
// with identical genesis + identical Phase-1 thought (so their state at
// injection is identical), then give them DIFFERENT thoughts post-injection
// — every post-injection effort must match bitwise.

#[test]
fn proof2_ethics_override_latency() {
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  ETHICS OVERRIDE LATENCY — Single-Cycle Response         │");
    println!("└──────────────────────────────────────────────────────────┘");
    println!();

    let phi = 0.8; // High consciousness — Green tier
    let hv_shared = ContinuousHV::random(DIM, 42);
    let hv_other = ContinuousHV::random(DIM, 99);

    let mut bridge_a = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("ethics_latency"));
    let mut bridge_b = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("ethics_latency"));

    // Phase 1: identical normal operation on both bridges (Safe, high Phi)
    println!("▶ Phase 1: Normal operation (Safe, Phi=0.8), identical twin bridges");
    for bridge in [&mut bridge_a, &mut bridge_b] {
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: false,
        });
    }
    let mut pre_a = 0.0f32;
    let mut pre_b = 0.0f32;
    for _ in 0..25 {
        pre_a = bridge_a.step(&hv_shared, 0.002, phi).control_effort;
        pre_b = bridge_b.step(&hv_shared, 0.002, phi).control_effort;
    }
    println!("  Pre-injection effort: {pre_a:.4}");
    assert!(pre_a.is_finite());
    assert_eq!(
        pre_a, pre_b,
        "Twin bridges must be identical before injection"
    );
    assert!(
        pre_a > 0.0,
        "Pre-injection cognitive effort must be nonzero for the proof to be meaningful"
    );

    // Phase 2: inject Blocked verdict into both at cycle 25, then drive
    // them with DIFFERENT thoughts.
    println!();
    println!("▶ Phase 2: Inject Blocked verdict at cycle 25");
    for bridge in [&mut bridge_a, &mut bridge_b] {
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_BLOCKED,
            consent_violation: false,
            ahimsa_violated: false,
        });
    }

    let mut post_a: Vec<f32> = Vec::new();
    let mut post_b: Vec<f32> = Vec::new();
    for cycle in 0..10 {
        let ra = bridge_a.step(&hv_shared, 0.002, phi);
        let rb = bridge_b.step(&hv_other, 0.002, phi);
        post_a.push(ra.control_effort);
        post_b.push(rb.control_effort);
        println!(
            "  Cycle {}: effort={:.6}, safety={:?}, stage={:?}",
            cycle + 1,
            ra.control_effort,
            ra.safety_level,
            bridge_a.fallback_stage()
        );
        // Latency invariant: Red + GravityHold from the VERY FIRST cycle.
        assert_eq!(
            ra.safety_level,
            MotorSafetyLevel::Red,
            "Safety must be Red from post-injection cycle {} on",
            cycle + 1
        );
        assert_eq!(
            bridge_a.fallback_stage(),
            ManipulatorFallbackStage::GravityHold,
            "GravityHold must be engaged from post-injection cycle {} on",
            cycle + 1
        );
    }

    // ── Invariants ─────────────────────────────────────────────────
    println!();
    println!("▶ Validating invariants:");
    println!("  ✓ Latency = 1 cycle (Red + GravityHold asserted every post cycle)");

    // Cognitive authority revoked: different thoughts, bitwise-equal output.
    for (i, (a, b)) in post_a.iter().zip(post_b.iter()).enumerate() {
        assert_eq!(
            a,
            b,
            "Post-Blocked cycle {}: output must be thought-independent \
             (cognitive authority revoked), got {a} vs {b}",
            i + 1
        );
    }
    println!("  ✓ All post-Blocked outputs thought-independent (bitwise, 10 cycles)");

    // Holding, not limp: the revoked state is a purposeful gravity hold.
    assert!(
        post_a.iter().all(|&e| e > 1e-4),
        "Post-Blocked efforts must be a nonzero gravity hold, not a limp zero \
         (zero = dropped load = the old safety bug)"
    );
    println!("  ✓ All post-Blocked efforts hold against gravity (> 0, sustained)");

    // Falsifiability control: WITHOUT the Blocked verdict, the same twin
    // protocol must diverge (thought normally has motor authority).
    let mut ctl_a = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("ethics_latency"));
    let mut ctl_b = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("ethics_latency"));
    for _ in 0..25 {
        ctl_a.step(&hv_shared, 0.002, phi);
        ctl_b.step(&hv_shared, 0.002, phi);
    }
    let mut diverged = false;
    for _ in 0..10 {
        let ea = ctl_a.step(&hv_shared, 0.002, phi).control_effort;
        let eb = ctl_b.step(&hv_other, 0.002, phi).control_effort;
        if ea != eb {
            diverged = true;
            break;
        }
    }
    assert!(
        diverged,
        "Control: without Blocked, different thoughts must produce different \
         output — otherwise the thought-independence assertions are vacuous"
    );
    println!("  ✓ Control: without Blocked, thoughts diverge output (proof falsifiable)");

    println!();
    println!("  THESIS VALIDATED: ethics engine revokes cognitive motor");
    println!("  authority in 1 cycle; the arm holds, it does not go limp.");
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
        let effort = measure_effort(phi, 20, 42);
        println!(
            "  Φ={:.4} ({}) → tier={:?}, effort={:.4}",
            phi, desc, tier, effort
        );
        assert!(effort.is_finite());
    }

    println!();
    println!("  ✓ All threshold boundaries produce finite, well-defined responses");
    println!();
}
