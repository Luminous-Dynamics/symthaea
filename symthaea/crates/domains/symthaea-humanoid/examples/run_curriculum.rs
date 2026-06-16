// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 53-DOF Humanoid Active Curriculum Calibration Runner

use std::time::Instant;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::training::HumanoidTrainer;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LAUNCHING 53-DOF DEXTEROUS HUMANOID ACTIVE INFERENCE SWEEP...");

    // Scale up configuration parameters to deploy the 53-DOF layout tree
    let config = HumanoidConfig {
        morphology: HumanoidMorphology::Dexterous53,
        num_episodes: 10,
        steps_per_episode: 400, // 10 seconds of physics per episode at 40Hz
        collect_telemetry: true,
        adaptive_curriculum: true,
        task: HumanoidTask::Stand,
        early_termination: true,
        ..HumanoidConfig::default()
    };

    let output_dir = "/tmp/symthaea_curriculum_calib";
    println!("📂 Telemetry matrices routing to: {}", output_dir);

    let mut trainer = HumanoidTrainer::new(config);

    let start_time = Instant::now();
    let metrics = trainer.train_with_telemetry(output_dir);
    let duration = start_time.elapsed();

    println!("✨ CURRICULUM TRAINING SEED COMPLETED IN {:?}", duration);
    println!("📊 MACRO EPISODE EVALUATION MATRIX (53-DOF DEXTEROUS):");
    println!("---------------------------------------------------------------------------------");
    println!(" EP | Task   | Avg Stand Reward | Avg Ep Reward | Avg Free Energy | Total Steps");
    println!("---------------------------------------------------------------------------------");

    for m in metrics {
        let task_string = format!("{:?}", m.task);
        println!(
            " {:02} | {:<6} | {:.4}           | {:.4}         | {:.4}          | {}",
            m.episode,
            task_string,
            m.avg_standing_reward,
            m.avg_episode_reward,
            m.avg_free_energy,
            m.total_steps
        );
    }
    println!("---------------------------------------------------------------------------------");
    println!("✅ Checkpoint weights vector successfully serialized to disk.");

    Ok(())
}
