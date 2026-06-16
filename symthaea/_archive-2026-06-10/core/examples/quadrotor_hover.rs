// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quadrotor hover training example.
//!
//! Trains an HDC-LTC unified neuron network (16,384D) to hover a simulated
//! Crazyflie 2 quadrotor using FEP Active Inference for precision modulation.
//!
//! Run: `cargo run --features multirotor --example quadrotor_hover --release`

use symthaea::multirotor::{FlightConfig, FlightTrainer};

fn main() {
    println!("=== Symthaea Flight: Quadrotor Hover Training ===");
    println!();

    let config = FlightConfig {
        num_episodes: 20,
        steps_per_episode: 2000, // 4s at 500Hz
        ..FlightConfig::default()
    };

    println!("Config:");
    println!("  Motor rate:    {:.0} Hz", config.motor_hz);
    println!("  Cognitive rate: {:.0} Hz", config.cognitive_hz);
    println!("  Episodes:      {}", config.num_episodes);
    println!(
        "  Steps/episode: {} ({:.1}s)",
        config.steps_per_episode,
        config.steps_per_episode as f64 / config.motor_hz
    );
    println!(
        "  Network:       {}x{} neurons (16384D each)",
        config.network_layers, config.neurons_per_layer
    );
    println!("  Learning rate: {}", config.learning_rate);
    println!();

    let mut trainer = FlightTrainer::new(config);
    let metrics = trainer.train();

    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>8} {:>6}",
        "Ep", "Pos Err", "Att Err", "Free E", "Hover%", "Expl"
    );
    println!("{}", "-".repeat(55));

    for m in &metrics {
        println!(
            "{:<5} {:>10.4} {:>10.4} {:>10.2} {:>7.1}% {:>6}",
            m.episode,
            m.avg_position_error,
            m.avg_attitude_error,
            m.avg_free_energy,
            m.hover_fraction * 100.0,
            m.exploration_count,
        );
    }

    println!();
    if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
        let err_improvement = if first.avg_position_error > 0.0 {
            (1.0 - last.avg_position_error / first.avg_position_error) * 100.0
        } else {
            0.0
        };
        println!("Position error improvement: {:.1}%", err_improvement);
        println!("Final hover fraction: {:.1}%", last.hover_fraction * 100.0);
    }
}
