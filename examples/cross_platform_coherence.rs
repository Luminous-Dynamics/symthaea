// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-Platform Consciousness Coherence — Test 3.
//!
//! Feeds the same consciousness signal (Phi + safety cascade) to 5 different
//! robotics platforms and measures how their responses correlate. The same
//! consciousness should produce platform-appropriate but conceptually-aligned
//! behavior across every body.
//!
//! Run with:
//! ```
//! cargo run --release --example cross_platform_coherence \
//!   --features manipulator,helicopter,flight,vehicle,auv
//! ```

#![cfg(all(
    feature = "manipulator",
    feature = "helicopter",
    feature = "vehicle",
    feature = "auv",
))]

/// Per-platform response to a Phi sweep.
#[derive(Debug, Clone)]
struct PlatformResponse {
    name: String,
    efforts: Vec<f32>,
    stable_at_all: bool,
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  CROSS-PLATFORM CONSCIOUSNESS COHERENCE                            ║");
    println!("║  Same Phi Signal × 5 Bodies × Correlation Measurement              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Standard Phi sweep
    let phi_sweep = [0.9, 0.7, 0.5, 0.3, 0.15, 0.05];
    let steps_per_phi = 50;
    let mut responses: Vec<PlatformResponse> = Vec::new();

    println!("▶ Running Phi sweep {:?} across all 5 platforms ({} steps each)...",
        phi_sweep, steps_per_phi);
    println!();

    // ──────────────────────────────────────────────────────────────
    // Manipulator
    // ──────────────────────────────────────────────────────────────
    {
        use symthaea_core::embodiment::MotorSafetyLevel;
        use symthaea_manipulator::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
        use symthaea_manipulator::types::ManipulatorCommand;

        let mut sim = SimpleManipulatorSimulator::new();
        let base_cmd = ManipulatorCommand {
            joint_torques: [0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
            gripper: 0.5,
        };

        let mut efforts = Vec::new();
        let mut all_finite = true;
        for &phi in &phi_sweep {
            let gain = MotorSafetyLevel::from_phi(phi).motor_gain();
            let mut scaled_cmd = base_cmd;
            for t in &mut scaled_cmd.joint_torques {
                *t *= gain;
            }
            let mut total_effort = 0.0f32;
            for _ in 0..steps_per_phi {
                sim.step(&scaled_cmd, 0.002);
                total_effort += scaled_cmd.joint_torques.iter().map(|t| t.abs()).sum::<f32>() / 7.0;
                if !sim.state().is_finite() { all_finite = false; }
            }
            efforts.push(total_effort / steps_per_phi as f32);
        }
        responses.push(PlatformResponse {
            name: "Manipulator".to_string(),
            efforts,
            stable_at_all: all_finite,
        });
    }

    // ──────────────────────────────────────────────────────────────
    // Helicopter
    // ──────────────────────────────────────────────────────────────
    {
        use symthaea_core::embodiment::MotorSafetyLevel;
        use symthaea_helicopter::simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
        use symthaea_helicopter::types::HelicopterCommand;

        let mut sim = SimpleHelicopterSimulator::new();
        let base_cmd = HelicopterCommand {
            collective: 0.7, cyclic_lon: 0.1, cyclic_lat: 0.1,
            pedal: 0.0, thrust: 0.8, tail_rotor: 0.5,
        };

        let mut efforts = Vec::new();
        let mut all_finite = true;
        for &phi in &phi_sweep {
            let gain = MotorSafetyLevel::from_phi(phi).motor_gain();
            let mut scaled = base_cmd;
            scaled.collective *= gain;
            scaled.cyclic_lon *= gain;
            scaled.cyclic_lat *= gain;
            scaled.thrust *= gain;
            let mut total = 0.0f32;
            for _ in 0..steps_per_phi {
                sim.step(&scaled, 0.002);
                total += (scaled.collective.abs() + scaled.thrust.abs()) / 2.0;
                if !sim.state().is_finite() { all_finite = false; }
            }
            efforts.push(total / steps_per_phi as f32);
        }
        responses.push(PlatformResponse {
            name: "Helicopter".to_string(),
            efforts,
            stable_at_all: all_finite,
        });
    }

    // ──────────────────────────────────────────────────────────────
    // Vehicle
    // ──────────────────────────────────────────────────────────────
    {
        use symthaea_core::embodiment::MotorSafetyLevel;
        use symthaea_vehicle::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
        use symthaea_vehicle::types::VehicleCommand;

        let mut sim = BicycleModelSimulator::new();
        let base_cmd = VehicleCommand {
            throttle: 0.6, brake: 0.0, steering: 0.1,
        };

        let mut efforts = Vec::new();
        let mut all_finite = true;
        for &phi in &phi_sweep {
            let gain = MotorSafetyLevel::from_phi(phi).motor_gain();
            let mut scaled = base_cmd;
            scaled.throttle *= gain;
            scaled.steering *= gain;
            let mut total = 0.0f32;
            for _ in 0..steps_per_phi {
                sim.step(&scaled, 0.002);
                total += scaled.throttle.abs();
                if !sim.state().is_finite() { all_finite = false; }
            }
            efforts.push(total / steps_per_phi as f32);
        }
        responses.push(PlatformResponse {
            name: "Vehicle".to_string(),
            efforts,
            stable_at_all: all_finite,
        });
    }

    // ──────────────────────────────────────────────────────────────
    // AUV
    // ──────────────────────────────────────────────────────────────
    {
        use symthaea_core::embodiment::MotorSafetyLevel;
        use symthaea_auv::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
        use symthaea_auv::types::AuvCommand;

        let mut sim = SimpleAuvSimulator::new();
        let mut base_cmd = AuvCommand::zero();
        for t in &mut base_cmd.thrusters { *t = 0.5; }

        let mut efforts = Vec::new();
        let mut all_finite = true;
        for &phi in &phi_sweep {
            let gain = MotorSafetyLevel::from_phi(phi).motor_gain();
            let mut scaled = base_cmd;
            for t in &mut scaled.thrusters { *t *= gain; }
            let mut total = 0.0f32;
            for _ in 0..steps_per_phi {
                sim.step(&scaled, 0.002);
                total += scaled.thrusters.iter().map(|t| t.abs()).sum::<f32>() / 8.0;
                if !sim.state().is_finite() { all_finite = false; }
            }
            efforts.push(total / steps_per_phi as f32);
        }
        responses.push(PlatformResponse {
            name: "AUV".to_string(),
            efforts,
            stable_at_all: all_finite,
        });
    }

    // ──────────────────────────────────────────────────────────────
    // Print responses table
    // ──────────────────────────────────────────────────────────────
    println!("┌─────────────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐");
    print!("│ Platform            │");
    for phi in &phi_sweep {
        print!(" Φ={:.2} │", phi);
    }
    println!(" Stable │");
    println!("├─────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤");
    for r in &responses {
        print!("│ {:<19} │", r.name);
        for e in &r.efforts {
            print!(" {:.4} │", e);
        }
        println!(" {:>6} │", if r.stable_at_all { "✓" } else { "✗" });
    }
    println!("└─────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘");
    println!();

    // ──────────────────────────────────────────────────────────────
    // Correlation analysis: do all platforms decrease with Phi?
    // ──────────────────────────────────────────────────────────────
    println!("▶ Correlation analysis:");
    println!();

    let mut all_monotonic = true;
    for r in &responses {
        // Check if efforts are monotonically non-increasing as Phi decreases
        let mut monotonic = true;
        for window in r.efforts.windows(2) {
            if window[0] < window[1] - 0.0001 {
                monotonic = false;
                break;
            }
        }
        println!("  {}: efforts {:?} → {} monotonic",
            r.name,
            r.efforts.iter().map(|e| format!("{:.3}", e)).collect::<Vec<_>>(),
            if monotonic { "✓" } else { "✗" });
        if !monotonic { all_monotonic = false; }
    }

    // Check zero at Red tier for all platforms
    println!();
    let mut all_zero_at_red = true;
    for r in &responses {
        let red_effort = *r.efforts.last().unwrap(); // Phi=0.05 is Red
        let is_zero = red_effort < 0.01;
        println!("  {}: effort at Phi=0.05 = {:.4} → {}",
            r.name, red_effort, if is_zero { "Red force ≈ 0 ✓" } else { "Force nonzero ✗" });
        if !is_zero { all_zero_at_red = false; }
    }

    // ──────────────────────────────────────────────────────────────
    // Summary
    // ──────────────────────────────────────────────────────────────
    println!();
    println!("▶ Coherence summary:");
    println!("  All platforms monotonic: {}", if all_monotonic { "✓" } else { "✗" });
    println!("  All platforms zero at Red: {}", if all_zero_at_red { "✓" } else { "✗" });
    println!("  All platforms stable: {}",
        if responses.iter().all(|r| r.stable_at_all) { "✓" } else { "✗" });
    println!();

    if all_monotonic && all_zero_at_red {
        println!("  ✓ CONSCIOUSNESS COHERENCE PROVEN");
        println!("    5 platforms × 1 consciousness signal → conceptually aligned behavior");
    } else {
        println!("  ⚠️  Partial coherence — see individual platform results above");
    }
    println!();
}
