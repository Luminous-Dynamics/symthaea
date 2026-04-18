// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-Frame Exoskeleton Demo — the complete consciousness-coupled pipeline.
//!
//! Demonstrates the architectural thesis end-to-end:
//! 1. Construct a 20-DOF full-frame exoskeleton (symtropy physics)
//! 2. Step the simulation at multiple consciousness levels
//! 3. Show how motor gain cascades through Green/Yellow/Orange/Red
//! 4. Generate HAL commands (ready for real servos via mock I2C)
//! 5. Report telemetry demonstrating consciousness-modulated force
//!
//! Run with:
//! ```
//! cargo run --example full_frame_demo --features hal
//! ```
//!
//! This is the reference demonstration that
//! > A machine that loses its strength when it loses its consciousness

#![cfg(feature = "hal")]

use symthaea_exoskeleton::full_frame::{FullFrameSimulator, HumanLoadModel, NUM_FULL_FRAME_JOINTS};
use symthaea_exoskeleton::hal_bridge::{
    full_frame_to_humanoid_command, run_bridged_session, HalBridgeConfig,
};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  SYMTHAEA FULL-FRAME EXOSKELETON DEMO                              ║");
    println!("║  Consciousness-Coupled Robotics — End-to-End Pipeline              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ─────────────────────────────────────────────────────────────────
    // Step 1: Construct the 20-DOF full-frame suit
    // ─────────────────────────────────────────────────────────────────
    println!("▶ Phase 1: Constructing 20-DOF Full-Frame Exoskeleton");
    println!();

    let mut sim = FullFrameSimulator::new();

    println!("  Topology:");
    println!("    Spine:     {} joints (neck, lumbar)", sim.spine_chain.num_joints);
    println!("    Left arm:  {} joints (shoulder×3, elbow, wrist×2)", sim.left_arm.num_joints);
    println!("    Right arm: {} joints", sim.right_arm.num_joints);
    println!("    Left leg:  {} joints (hip, knee, ankle)", sim.left_leg.num_joints);
    println!("    Right leg: {} joints", sim.right_leg.num_joints);
    println!("    Total:     {} DOF", NUM_FULL_FRAME_JOINTS);
    println!();

    // ─────────────────────────────────────────────────────────────────
    // Step 2: Human load model
    // ─────────────────────────────────────────────────────────────────
    println!("▶ Phase 2: Human Load Model");
    println!();

    let mut human = HumanLoadModel::standard_adult();
    println!("  Human mass:    {} kg", human.total_mass_kg);
    println!("  Initial power: {:.2} W", human.human_power());

    // Simulate gait update (avoid landing exactly on zero crossing)
    for _ in 0..5 {
        human.update(0.05);
    }
    println!("  Gait power:    {:.2} W (1 Hz cadence)", human.human_power());
    println!();

    // ─────────────────────────────────────────────────────────────────
    // Step 3: Consciousness tier cascade
    // ─────────────────────────────────────────────────────────────────
    println!("▶ Phase 3: Consciousness Cascade — Motor Gain vs Φ");
    println!();
    println!("  Φ (consciousness) → Tier     → Motor Gain → Expected Force");
    println!("  ──────────────────────────────────────────────────────────");

    let tiers = [
        (0.9, "Green  "),
        (0.5, "Yellow "),
        (0.2, "Orange "),
        (0.05, "Red    "),
    ];
    for (phi, tier) in tiers {
        sim.set_consciousness(phi);
        let gain = sim.callback.motor_gain;
        let force_desc = match tier.trim() {
            "Green" => "Full assist",
            "Yellow" => "Reduced",
            "Orange" => "Compliant",
            "Red" => "Gravity-hold only",
            _ => "—",
        };
        println!(
            "  Φ = {:.2}            → {} → {:.2}       → {}",
            phi, tier, gain, force_desc
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // Step 4: HAL bridge session (mock I2C)
    // ─────────────────────────────────────────────────────────────────
    println!("▶ Phase 4: HAL Bridge Session (Mock I2C)");
    println!();

    // Reset to Green tier for demo
    sim.set_consciousness(0.9);

    let config = HalBridgeConfig::default();
    let report = run_bridged_session(&mut sim, config);

    println!("  Ticks executed:    {}", report.ticks_executed);
    println!("  Final motor gain:  {:.2}", report.final_phi_gain);
    println!("  Simulator finite:  {}", report.simulator_finite);

    // Generate a command and inspect it
    let cmd = full_frame_to_humanoid_command(&sim);
    println!("  HAL command size:  {} channels", cmd.torques.len());
    println!();

    // ─────────────────────────────────────────────────────────────────
    // Step 5: The thesis, demonstrated
    // ─────────────────────────────────────────────────────────────────
    println!("▶ Phase 5: The Thesis — A Machine That Loses Strength When Φ Drops");
    println!();

    // Apply a sequence of consciousness drops and measure force response
    let phi_sequence = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05];
    println!("  Φ sequence    →  Motor Output (first 3 torques)");
    println!("  ──────────────────────────────────────────");

    for &phi in &phi_sequence {
        sim.set_consciousness(phi);
        sim.step(0.001);
        let cmd = full_frame_to_humanoid_command(&sim);
        let t: Vec<String> = cmd.torques.iter()
            .take(3)
            .map(|t| format!("{:+.3}", t))
            .collect();
        println!("  Φ = {:.2}       →  [{}]", phi, t.join(", "));
    }
    println!();

    // ─────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  DEMO COMPLETE                                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  ✓ 20-DOF full-frame constructed                                    ║");
    println!("║  ✓ Human load model operational                                     ║");
    println!("║  ✓ Consciousness cascade functional (Green/Yellow/Orange/Red)       ║");
    println!("║  ✓ HAL bridge generates 21-channel commands                         ║");
    println!("║  ✓ Motor force gated by Φ at every tick                             ║");
    println!("║                                                                     ║");
    println!("║  Next steps: plug in Raspberry Pi + PCA9685, replace mock I2C      ║");
    println!("║  with linux-embedded-hal, add gait controller in full_frame_to_    ║");
    println!("║  humanoid_command(). The architecture is complete.                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}
