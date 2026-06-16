// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 53-DOF Humanoid Active Locomotion Extended Optimizer Curriculum Runner

use std::time::Instant;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::training::HumanoidTrainer;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidTask};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LAUNCHING EXTENDED 50-EPISODE 53-DOF HUMANOID LOCOMOTION SWEEP...");

    // Expand configuration parameters to a 50-episode baseline run
    let config = HumanoidConfig {
        morphology: HumanoidMorphology::Dexterous53,
        num_episodes: 50,
        steps_per_episode: 400, // 10 seconds of physics per episode at 40Hz
        adaptive_curriculum: true,
        task: HumanoidTask::Walk,
        early_termination: true,
        ..HumanoidConfig::default()
    };

    let mut trainer = HumanoidTrainer::new(config);

    let start_time = Instant::now();
    // Execute DAgger pipeline: check unassisted OOD gait deviations every 3 episodes
    let (metrics, _controller, _encoder) = trainer.train_dagger(3, 200);
    let duration = start_time.elapsed();

    println!("✨ EXTENDED LOCOMOTION SWEEP COMPLETED IN {:?}", duration);
    println!("📊 MACRO EPISODE EVALUATION MATRIX (50-DOF WALKING):");
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
    println!("✅ Extended autonomous locomotion calibration block complete.");

    Ok(())
}
