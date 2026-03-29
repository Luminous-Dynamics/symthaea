// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DMC Humanoid Stand training example.
//!
//! Trains an HDC-LTC unified network (16,384D, 3×8 neurons) to control a
//! bipedal humanoid for the DMC Stand task using FEP Active Inference for
//! precision modulation. Curriculum: Stand(PD) → Stand(autonomous) → Walk → Run.
//!
//! Run: `cargo run --features humanoid --example dmc_humanoid_stand --release`

use symthaea::humanoid::{HumanoidConfig, HumanoidTrainer};

fn main() {
    println!("=== Symthaea Humanoid: DMC Stand Training ===");
    println!();

    let config = HumanoidConfig {
        num_episodes: 20,
        steps_per_episode: 400,
        ..HumanoidConfig::default()
    };

    println!("Config:");
    println!(
        "  Physics rate:   {:.0} Hz ({:.3}s timestep)",
        config.physics_hz,
        config.physics_dt()
    );
    println!(
        "  Cognitive rate: {:.0} Hz (every {} physics steps)",
        config.cognitive_hz,
        config.cognitive_interval()
    );
    println!(
        "  Network:        {}×{} neurons (16384D each)",
        config.network_layers, config.neurons_per_layer
    );
    println!("  Episodes:       {}", config.num_episodes);
    println!(
        "  Steps/episode:  {} ({:.1}s)",
        config.steps_per_episode,
        config.steps_per_episode as f64 * config.physics_dt()
    );
    println!("  Learning rate:  {}", config.learning_rate);
    println!("  Task:           {:?}", config.task);
    println!();

    let mut trainer = HumanoidTrainer::new(config);
    let metrics = trainer.train();

    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>8} {:>6} {:>5}",
        "Ep", "Stand R", "Episode R", "Free E", "Head H", "Uprt%", "Steps"
    );
    println!("{}", "-".repeat(62));

    for m in &metrics {
        println!(
            "{:<5} {:>10.4} {:>10.4} {:>10.2} {:>8.3} {:>5.1}% {:>5}",
            m.episode,
            m.avg_standing_reward,
            m.avg_episode_reward,
            m.avg_free_energy,
            m.avg_head_height,
            m.avg_uprightness * 100.0,
            m.total_steps,
        );
    }

    println!();
    if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
        let reward_improvement = if first.avg_standing_reward > 0.0 {
            ((last.avg_standing_reward - first.avg_standing_reward) / first.avg_standing_reward)
                * 100.0
        } else {
            0.0
        };
        println!("Standing reward improvement: {:.1}%", reward_improvement);
        println!(
            "Final standing fraction:     {:.1}% (head > 1.4m, upright > 0.9)",
            last.avg_uprightness * 100.0
        );
    }
}
