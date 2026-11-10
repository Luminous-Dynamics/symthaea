// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-frame articulated research-model smoke demonstration.
//!
//! This demonstrates model construction, synthetic load generation, finite
//! simulation stepping, and the current zero-torque command-shape stub.
//! It does not demonstrate a validated exoskeleton controller, HAL transport,
//! physical actuation, Φ-based safety, or human biomechanics.
//!
//! Run with:
//! ```
//! cargo run --example full_frame_demo --features hal
//! ```

#![cfg(feature = "hal")]

use symthaea_exoskeleton::full_frame::{
    FullFrameSimulator, HumanLoadModel, NUM_FULL_FRAME_JOINTS,
};
use symthaea_exoskeleton::hal_bridge::{
    HalBridgeConfig, full_frame_to_humanoid_command, run_bridged_session,
};

fn main() {
    let mut sim = FullFrameSimulator::new();
    println!("Symthaea full-frame articulated research model");
    println!("Research scaffold only; no hardware transport or actuation.");

    println!("Declared generalized joints: {}", NUM_FULL_FRAME_JOINTS);
    println!("Constructed articulated joints: {}", sim.total_joints());

    let mut synthetic_load = HumanLoadModel::reference_75kg_profile();
    for _ in 0..5 {
        synthetic_load.update(0.05);
    }
    println!(
        "Synthetic load proxy (sum absolute joint torques, N·m): {:.3}",
        synthetic_load.sum_abs_joint_torque_nm()
    );
    println!("Note: synthetic load torques are not applied to the articulated plant.");

    for _ in 0..50 {
        sim.step(0.001);
    }

    let config = HalBridgeConfig::default();
    let report = run_bridged_session(&mut sim, config);
    let command = full_frame_to_humanoid_command(&sim);

    println!("Software-loop ticks: {}", report.ticks_executed);
    println!("Research plant finite: {}", report.simulator_finite);
    println!("All loop commands zero torque: {}", report.all_commands_zero_torque);
    println!("Command channels: {}", command.torques.len());
    println!(
        "Command contains only zeros: {}",
        command.torques.iter().all(|torque| *torque == 0.0)
    );
    println!("No I2C transport, HAL runtime, or physical actuator was accessed.");
}
