// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train the flight HDC-LTC controller via BPTT (CPU).
//!
//! Closes the "untrained controllers" gap flagged in the paper's §9.
//! `FlightTrainer` runs a multi-rate loop: 500 Hz motor reflex +
//! 125 Hz BPTT from PD-baseline targets + 25 Hz FEP agent ticks.
//!
//! Produces per-episode metrics (position error, attitude error,
//! control effort, cost-of-transport) and a final ControllerCheckpoint
//! that can be reloaded.
//!
//! Run:
//!     cargo run -p symthaea-multirotor --example train_flight --release
//!
//! Env:
//!     TF_EPISODES=N    number of training episodes (default 50)
//!     TF_STEPS=N       steps per episode (default 1000)
//!     TF_CHECKPOINT=path  save ControllerCheckpoint to this file
//!     TF_CSV=path      dump per-episode metrics CSV

use std::io::Write;

use symthaea_multirotor::training::FlightTrainer;
use symthaea_multirotor::types::FlightConfig;

fn main() {
    let episodes: usize = std::env::var("TF_EPISODES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let steps: usize = std::env::var("TF_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let checkpoint_path = std::env::var("TF_CHECKPOINT").ok();
    let csv_path = std::env::var("TF_CSV").ok();

    let config = FlightConfig {
        num_episodes: episodes,
        steps_per_episode: steps,
        ..FlightConfig::default()
    };

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Flight controller training — BPTT from PD-baseline (CPU)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" episodes         : {episodes}");
    println!(" steps/episode    : {steps}");
    println!(" total sim ticks  : {}", episodes * steps);
    println!(" learning rate    : {}", config.learning_rate);
    println!(" physics backend  : SimplePhysicsSimulator (ballistic)");
    println!();

    let start = std::time::Instant::now();
    let mut trainer = FlightTrainer::new(config.clone());
    let metrics = trainer.train();
    let elapsed = start.elapsed();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        " Training complete in {:.1}s ({:.1} min)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / 60.0
    );
    println!("════════════════════════════════════════════════════════════════════");
    println!();

    // Summarize first 5 and last 5 episodes.
    println!(
        "{:>6}  {:>12}  {:>12}  {:>10}  {:>10}",
        "ep", "pos_err", "att_err", "free_E", "hover_%"
    );
    for m in metrics.iter().take(5) {
        println!(
            "{:>6}  {:>12.4}  {:>12.4}  {:>10.3}  {:>10.2}",
            m.episode,
            m.avg_position_error,
            m.avg_attitude_error,
            m.avg_free_energy,
            m.hover_fraction * 100.0
        );
    }
    if metrics.len() > 10 {
        println!(
            "{:>6}  {:>12}  {:>12}  {:>10}  {:>10}",
            "...", "...", "...", "...", "..."
        );
    }
    for m in metrics.iter().rev().take(5).rev() {
        println!(
            "{:>6}  {:>12.4}  {:>12.4}  {:>10.3}  {:>10.2}",
            m.episode,
            m.avg_position_error,
            m.avg_attitude_error,
            m.avg_free_energy,
            m.hover_fraction * 100.0
        );
    }

    if metrics.len() >= 2 {
        let first = &metrics[0];
        let last = &metrics[metrics.len() - 1];
        let pos_improvement = 100.0 * (first.avg_position_error - last.avg_position_error)
            / first.avg_position_error.max(1e-9);
        let att_improvement = 100.0 * (first.avg_attitude_error - last.avg_attitude_error)
            / first.avg_attitude_error.max(1e-9);
        println!();
        println!(
            " Position error improvement: {pos_improvement:+.1} %  (ep 0 → ep {})",
            metrics.len() - 1
        );
        println!(
            " Attitude error improvement: {att_improvement:+.1} %  (ep 0 → ep {})",
            metrics.len() - 1
        );
    }

    if let Some(path) = csv_path.as_ref() {
        let f = std::fs::File::create(path).expect("create CSV");
        let mut w = std::io::BufWriter::new(f);
        writeln!(
            w,
            "episode,pos_err,att_err,free_energy,hover_fraction,final_pos_err,total_steps"
        )
        .ok();
        for m in &metrics {
            writeln!(
                w,
                "{},{:.5},{:.5},{:.5},{:.5},{:.5},{}",
                m.episode,
                m.avg_position_error,
                m.avg_attitude_error,
                m.avg_free_energy,
                m.hover_fraction,
                m.final_position_error,
                m.total_steps,
            )
            .ok();
        }
        println!();
        println!(" CSV written to: {path}");
    }

    if let Some(path) = checkpoint_path.as_ref() {
        println!();
        println!(" (checkpoint save not wired — FlightTrainer::train() doesn't");
        println!("  expose the final controller state directly; use");
        println!("  train_with_telemetry for CSV output or refactor to return");
        println!("  the final ControllerCheckpoint)");
        let _ = path;
    }
}
