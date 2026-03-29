// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 100-episode humanoid training with full telemetry CSV output.
//!
//! Produces:
//! - `output/humanoid_telemetry/summary.csv` — per-episode metrics
//! - `output/humanoid_telemetry/episode_NNNN.csv` — per-step telemetry for each episode
//!
//! Run: `cargo run --features humanoid --example humanoid_100ep_telemetry --release`

use symthaea::humanoid::{HumanoidConfig, HumanoidTrainer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let resume_path = args
        .iter()
        .position(|a| a == "--resume")
        .and_then(|i| args.get(i + 1))
        .cloned();

    println!("=== Symthaea Humanoid: 100-Episode Training with Telemetry ===");
    println!();

    let config = HumanoidConfig {
        num_episodes: 100,
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
        "  Network:        {}x{} neurons (16384D each)",
        config.network_layers, config.neurons_per_layer
    );
    println!("  Episodes:       {}", config.num_episodes);
    println!(
        "  Steps/episode:  {} ({:.1}s sim time)",
        config.steps_per_episode,
        config.steps_per_episode as f64 * config.physics_dt()
    );
    println!("  Learning rate:  {}", config.learning_rate);
    println!("  Output dir:     output/humanoid_telemetry/");

    let mut trainer = if let Some(ref path) = resume_path {
        println!("  Resuming from:  {}", path);
        println!();
        match HumanoidTrainer::with_checkpoint(config, path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to load checkpoint {path}: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!();
        HumanoidTrainer::new(config)
    };
    let start = std::time::Instant::now();

    let metrics = trainer.train_with_telemetry("output/humanoid_telemetry");

    let elapsed = start.elapsed();
    println!();
    println!("Training complete in {:.1}s", elapsed.as_secs_f64());
    println!();

    // Print summary table
    println!(
        "{:<5} {:>7} {:>10} {:>10} {:>8} {:>6} {:>6} {:>7} {:>5}",
        "Ep", "Task", "Stand R", "Episode R", "Free E", "HeadH", "Uprt%", "Speed", "Expl"
    );
    println!("{}", "-".repeat(72));

    for m in &metrics {
        println!(
            "{:<5} {:>7?} {:>10.4} {:>10.4} {:>8.3} {:>6.3} {:>5.1}% {:>6.3} {:>5}",
            m.episode,
            m.task,
            m.avg_standing_reward,
            m.avg_episode_reward,
            m.avg_free_energy,
            m.avg_head_height,
            m.avg_uprightness * 100.0,
            m.avg_horizontal_speed,
            m.exploration_count,
        );
    }

    println!();

    // Compute summary statistics
    let first = &metrics[0];
    let last = metrics.last().unwrap();
    let peak = metrics
        .iter()
        .max_by(|a, b| {
            a.avg_standing_reward
                .partial_cmp(&b.avg_standing_reward)
                .unwrap()
        })
        .unwrap();

    let stand_episodes: Vec<_> = metrics
        .iter()
        .filter(|m| matches!(m.task, symthaea::humanoid::HumanoidTask::Stand))
        .collect();
    let walk_episodes: Vec<_> = metrics
        .iter()
        .filter(|m| matches!(m.task, symthaea::humanoid::HumanoidTask::Walk))
        .collect();
    let run_episodes: Vec<_> = metrics
        .iter()
        .filter(|m| matches!(m.task, symthaea::humanoid::HumanoidTask::Run))
        .collect();

    let avg_reward = |eps: &[&symthaea::humanoid::training::EpisodeMetrics]| -> f64 {
        if eps.is_empty() {
            return 0.0;
        }
        eps.iter().map(|m| m.avg_standing_reward).sum::<f64>() / eps.len() as f64
    };

    println!("Summary:");
    println!("  Total episodes:         {}", metrics.len());
    println!(
        "  Total steps:            {}",
        metrics.iter().map(|m| m.total_steps).sum::<usize>()
    );
    println!("  Training time:          {:.1}s", elapsed.as_secs_f64());
    println!();
    println!("  Initial standing R:     {:.4}", first.avg_standing_reward);
    println!(
        "  Peak standing R:        {:.4} (episode {})",
        peak.avg_standing_reward, peak.episode
    );
    println!("  Final standing R:       {:.4}", last.avg_standing_reward);
    let improvement = if first.avg_standing_reward > 0.0 {
        ((peak.avg_standing_reward - first.avg_standing_reward) / first.avg_standing_reward) * 100.0
    } else {
        0.0
    };
    println!("  Improvement:            +{:.1}%", improvement);
    println!();
    println!(
        "  Stand phase avg:        {:.4} ({} episodes)",
        avg_reward(&stand_episodes),
        stand_episodes.len()
    );
    println!(
        "  Walk phase avg:         {:.4} ({} episodes)",
        avg_reward(&walk_episodes),
        walk_episodes.len()
    );
    println!(
        "  Run phase avg:          {:.4} ({} episodes)",
        avg_reward(&run_episodes),
        run_episodes.len()
    );
    println!();
    println!("  Peak head height:       {:.3}m", peak.avg_head_height);
    println!(
        "  Peak uprightness:       {:.1}%",
        peak.avg_uprightness * 100.0
    );
    println!();
    println!("  CSV output:             output/humanoid_telemetry/summary.csv");
    println!(
        "                          output/humanoid_telemetry/episode_NNNN.csv (x{})",
        metrics.len()
    );
}
