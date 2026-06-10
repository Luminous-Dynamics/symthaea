// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 64-DOF Flagship FullSpine Humanoid Active Inference Curriculum Sweep

use std::time::Instant;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::training::HumanoidTrainer;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 DEPLOYING FLAGSHIP 64-DOF FULLSPINE HUMANOID ARCHITECTURE...");

    // Initialize the 64-DOF layout profile with an adaptive 20-episode gait block
    let config = HumanoidConfig {
        morphology: HumanoidMorphology::FullSpine,
        num_episodes: 20,
        steps_per_episode: 400, // 10 seconds of physics per episode at 40Hz
        adaptive_curriculum: true,
        task: HumanoidTask::Walk,
        early_termination: true,
        ..HumanoidConfig::default()
    };

    let mut trainer = HumanoidTrainer::new(config);

    let start_time = Instant::now();
    // Run DAgger pipeline: assess unassisted torso/gait anomalies every 3 episodes
    let (metrics, _controller, _encoder) = trainer.train_dagger(3, 200);
    let duration = start_time.elapsed();

    println!("✨ FULLSPINE KINEMATIC SWEEP COMPLETED IN {:?}", duration);
    println!("📊 MACRO EPISODE EVALUATION MATRIX (64-DOF FULLSPINE):");
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
    println!("✅ Flagship FullSpine autonomous locomotion sweep complete.");

    Ok(())
}
