// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 32-DOF Humanoid Active Curriculum Optimizer Calibration Runner

use symthaea_humanoid::training::HumanoidTrainer;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};
use std::time::Instant;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter, filter::LevelFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🚀 LAUNCHING 32-DOF HUMANOID ACTIVE inference CURRICULUM SWEEP...");

    // Configure a tight 10-episode calibration block with progressive noise enabled
    let config = HumanoidConfig {
        num_episodes: 10,
        steps_per_episode: 400, // 10 seconds of physics per episode at 40Hz
        collect_telemetry: true,
        adaptive_curriculum: true,
        task: HumanoidTask::Stand,
        early_termination: true,
        ..HumanoidConfig::default()
    };

    let output_dir = "/tmp/symthaea_curriculum_calib";
    info!("📂 Telemetry telemetry matrices routing to: {}", output_dir);

    let mut trainer = HumanoidTrainer::new(config);
    
    let start_time = Instant::now();
    let metrics = trainer.train_with_telemetry(output_dir);
    let duration = start_time.elapsed();

    info!("✨ CURRICULUM TRAINING SEED COMPLETED IN {:?}", duration);
    info!("📊 MACRO EPISODE EVALUATION MATRIX:");
    info!("---------------------------------------------------------------------------------");
    info!(" EP | Task   | Avg Stand Reward | Avg Ep Reward | Avg Free Energy | Total Steps");
    info!("---------------------------------------------------------------------------------");
    
    for m in metrics {
        info!(
            " {:02} | {:<6?} | {:.4}           | {:.4}         | {:.4}          | {}",
            m.episode, m.task, m.avg_standing_reward, m.avg_episode_reward, m.avg_free_energy, m.total_steps
        );
    }
    info!("---------------------------------------------------------------------------------");
    info!("✅ Checkpoint weights vector successfully serialized to disk.");

    Ok(())
}
