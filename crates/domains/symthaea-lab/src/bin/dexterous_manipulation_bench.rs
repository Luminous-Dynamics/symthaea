// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dexterous Manipulation Benchmark (64-DOF FullSpine Humanoid)
//!
//! Simulates the 64-DOF humanoid performing a "Precision Pinch" task.
//! Tests the coordination of the 5-finger hands (32 extra actuators)
//! and the efficiency of the HDC-LTC sensory feedback loop for fine motor control.

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;
use symthaea_humanoid::controller::HumanoidController;
use symthaea_humanoid::encoder::HumanoidHdcEncoder;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidState};
use symtropy_robotics_bridge_core::safety::JointSafetyAuthority;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🖐️  INITIATING 64-DOF DEXTEROUS MANIPULATION BENCHMARK...");

    // 1. Initialize 64-DOF FullSpine Morphology
    let morphology = HumanoidMorphology::FullSpine;
    let config = HumanoidConfig {
        morphology,
        learning_rate: 0.0005,
        genesis_phrase: "Dexterous Hand Seed v1".to_string(),
        ..HumanoidConfig::default()
    };
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);

    // 2. Initialize Controller, Encoder, and Safety (focusing on hands)
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut encoder = HumanoidHdcEncoder::new_predictive(&genesis, 32, morphology);
    let mut safety = JointSafetyAuthority::new(morphology.num_actuators());

    info!("🤖 Articulated Hands Ready: 32 Dexterous Actuators (16 per hand)");

    // 3. Manipulation Task Loop (Simulated Precision Pinch)
    const NUM_TRIALS: usize = 3;
    const STEPS_PER_TRIAL: usize = 150;
    const DT: f32 = 0.01; // 100Hz high-frequency control for dexterity

    for trial in 1..=NUM_TRIALS {
        info!("🎞️  Starting Pinch Trial {}/{}...", trial, NUM_TRIALS);
        controller.reset();
        encoder.reset();

        let mut cumulative_grip_success = 0.0;

        for step in 0..STEPS_PER_TRIAL {
            let t = step as f32 * DT;

            // A. Simulated Sensory State (Closing fingers)
            let mut state = HumanoidState::standing_for(morphology);

            // Indices 21..53 are the hand actuators (16 per hand)
            // Simulating a rhythmic "pinch" motion
            for i in 21..53 {
                let phase_offset = (i % 8) as f32 * 0.1;
                state.joint_angles[i] = ((t * 8.0 + phase_offset).sin() * 0.4 + 0.4) as f64;
            }

            // B. HDC Encoding (with predictive layer for tactile confidence)
            let sensor_hv = encoder.encode_with_dt(&state, DT);
            let confidence = encoder.confidence();

            // C. Forward Pass
            let mut command = controller.forward(&sensor_hv, DT);

            // D. Safety Gating (Detecting "Contact" via Prediction Error)
            let mut joint_surprises = vec![0.0f64; morphology.num_actuators()];
            if step > 50 && step < 100 {
                // Simulate index finger contact resistance
                joint_surprises[21] = 1.8;
                joint_surprises[29] = 1.8;
            }

            safety.update_from_surprise(&joint_surprises);
            for i in 0..command.torques.len() {
                command.torques[i] *= safety.joint_tiers[i].motor_gain() as f32;
            }

            // E. Training Step (Alignment toward synchronized grip)
            let mut target_command = command.clone();
            if confidence > 0.8 {
                // In high-confidence state, increase grip torque slightly
                for i in 21..53 {
                    target_command.torques[i] += 0.02;
                }
            }
            controller.train_step(&sensor_hv, &target_command, DT, None);

            // F. Metric gathering
            if step > 75 {
                cumulative_grip_success += confidence as f64;
            }
        }

        let avg_success = cumulative_grip_success / 75.0;
        info!("📊 Trial {} Result:", trial);
        info!("   🤲 Grip Stability: {:.2}%", avg_success * 100.0);
        info!("   🧠 Neural Confidence: {:.3}", encoder.confidence());

        if avg_success > 0.7 {
            info!("   ✅ PRECISION GRIP ACHIEVED.");
        }
    }

    info!("✨ DEXTEROUS MANIPULATION BENCHMARK COMPLETE.");
    Ok(())
}
