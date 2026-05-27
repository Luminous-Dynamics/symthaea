// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 53-DOF Humanoid DAgger OOD Error-Correction Calibration Runner

use std::time::Instant;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::training::HumanoidTrainer;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LAUNCHING 53-DOF DEXTEROUS HUMANOID DAGGER CORRECTION SWEEP...");

    // Setup configuration to deploy the 53-DOF Dexterous architecture tree
    let config = HumanoidConfig {
        morphology: HumanoidMorphology::Dexterous53,
        num_episodes: 6,
        steps_per_episode: 400, // 10 seconds of physics per episode at 40Hz
        adaptive_curriculum: true,
        task: HumanoidTask::Stand,
        early_termination: true,
        ..HumanoidConfig::default()
    };

    let mut trainer = HumanoidTrainer::new(config);

    let start_time = Instant::now();
    // Execute DAgger loop: evaluate unassisted exploration every 2 episodes for 200 steps
    let (metrics, _controller, _encoder) = trainer.train_dagger(2, 200);
    let duration = start_time.elapsed();

    println!("✨ DAGGER TRAINING SWEEP COMPLETED IN {:?}", duration);
    println!("📊 MACRO EPISODE EVALUATION MATRIX (DAGGER MODE):");
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
    println!("✅ DAgger unassisted OOD stabilization phase complete.");

    Ok(())
}
