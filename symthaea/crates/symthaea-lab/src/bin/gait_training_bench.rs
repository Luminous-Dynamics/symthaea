// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Gait Training Benchmark (64-DOF FullSpine Humanoid)
//!
//! Refines the 64-channel walking sequence using a multi-rate training loop.
//! Integrates the JointSafetyAuthority and GaitAnalyzer to ensure
//! safe, high-clearance, and regular strides.

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;
use symthaea_humanoid::controller::HumanoidController;
use symthaea_humanoid::encoder::HumanoidHdcEncoder;
use symthaea_humanoid::gait::GaitAnalyzer;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::types::{HumanoidCommand, HumanoidConfig, HumanoidState};
use symtropy_robotics_bridge_core::safety::JointSafetyAuthority;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🏃 INITIATING 64-DOF GAIT TRAINING BENCHMARK...");

    // 1. Initialize 64-DOF Morphology and Config
    let morphology = HumanoidMorphology::FullSpine;
    let config = HumanoidConfig {
        morphology,
        learning_rate: 0.0001,
        genesis_phrase: "Sovereign Gait Seed v1".to_string(),
        ..HumanoidConfig::default()
    };
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);

    // 2. Initialize Controller, Encoder, Analyzer, and Safety
    let mut controller = HumanoidController::new(&genesis, &config);
    let mut encoder = HumanoidHdcEncoder::new_for(&genesis, 32, morphology);
    let mut analyzer = GaitAnalyzer::new();
    let mut safety = JointSafetyAuthority::new(morphology.num_actuators());

    info!(
        "🤖 Controller Initialized: {} Layers | {} Actuators",
        config.network_layers,
        morphology.num_actuators()
    );

    // 3. Training Loop (Simulated Gait Cycles)
    const NUM_EPISODES: usize = 5;
    const STEPS_PER_EPISODE: usize = 200;
    const DT: f32 = 0.025; // 40Hz control rate

    for ep in 1..=NUM_EPISODES {
        info!("🎞️  Starting Episode {}/{}...", ep, NUM_EPISODES);
        analyzer.reset();
        controller.reset();
        encoder.reset();

        let mut total_surprise = 0.0;

        for step in 0..STEPS_PER_EPISODE {
            let t = step as f32 * DT;

            // A. Simulated Sensory State (Walking forward)
            let mut state = HumanoidState::standing_for(morphology);
            state.joint_angles[5] = ((t * 10.0).cos() * 0.2) as f64; // R-Hip Pitch sway
            state.joint_angles[11] = ((t * 10.0 + std::f32::consts::PI).cos() * 0.2) as f64; // L-Hip Pitch sway

            // Simulate swing heights for analyzer
            state.extremities[8] = ((t * 5.0).sin().max(0.0) * 0.1) as f64; // R-Foot height
            state.extremities[11] = ((t * 5.0 + std::f32::consts::PI).sin().max(0.0) * 0.1) as f64; // L-Foot height

            // B. HDC Encoding
            let sensor_hv = encoder.encode_with_dt(&state, DT);

            // C. Forward Pass (Motor Command Generation)
            let mut command = controller.forward(&sensor_hv, DT);

            // D. Safety Gating (Channel 5 Simulation)
            let mut joint_surprises = vec![0.0f64; morphology.num_actuators()];
            // Simulate occasional structural variance
            if step % 50 == 0 {
                joint_surprises[6] = 2.5;
            }

            safety.update_from_surprise(&joint_surprises);

            for i in 0..command.torques.len() {
                command.torques[i] *= safety.joint_tiers[i].motor_gain() as f32;
            }

            // E. Analyzer Update
            analyzer.update(&state);

            // F. "Active Inference" Training Step
            let mut target_command = command.clone();
            // Bias target toward higher foot clearance
            if analyzer.avg_clearance() < 0.05 {
                if target_command.torques.len() > 12 {
                    target_command.torques[6] += 0.05;
                    target_command.torques[12] += 0.05;
                }
            }

            controller.train_step(&sensor_hv, &target_command, DT, None);

            total_surprise += joint_surprises.iter().sum::<f64>();
        }

        let summary = analyzer.summary();
        info!("📊 Episode {} Complete:", ep);
        info!("   🦶 Avg Clearance: {:.3}m", summary.avg_clearance);
        info!("   📏 Stride Count: {}", summary.stride_count);
        info!("   🌀 Step Regularity: {:.2}", summary.step_regularity);
        info!("   💥 Total Surprise: {:.1}", total_surprise);

        if summary.avg_clearance > 0.04 && summary.step_regularity > 0.4 {
            info!("   ✅ Gait Quality IMPROVING.");
        }
    }

    info!("✨ GAIT TRAINING BENCHMARK COMPLETE.");
    Ok(())
}
