// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parallel swarm training: N MuJoCo drones with domain randomization.
//!
//! Each drone has randomized mass and wind profile. All drones share a global
//! experience buffer. After training, the swarm's collective knowledge produces
//! lower error than any single drone trained alone.
//!
//! Run: `cargo run --example mujoco_swarm_training --features multirotor-swarm --release`

#[cfg(any(feature = "multirotor-swarm", feature = "flight-swarm"))]
fn main() {
    use symthaea::multirotor::{
        swarm::{train_swarm, SwarmConfig},
        FlightConfig,
    };

    println!("Swarm Training — Parallel MuJoCo Drones");
    println!();

    let config = SwarmConfig {
        num_drones: 128,
        steps_per_drone: 2000,
        episodes: 10,
        buffer_size: 128 * 1024,
        swarm_replay_count: 4,
        mass_range: (0.020, 0.035),
        wind_magnitude_range: (0.0, 0.15),
        flight_config: FlightConfig::default(),
    };

    println!(
        "  Drones: {}, Episodes: {}, Steps/ep: {}",
        config.num_drones, config.episodes, config.steps_per_drone
    );

    let start = std::time::Instant::now();
    let result = train_swarm(&config);
    let elapsed = start.elapsed();

    let total_steps = config.num_drones * config.episodes * config.steps_per_drone;
    println!(
        "  Wall time:       {:.1}s ({:.0} steps/sec)",
        elapsed.as_secs_f64(),
        total_steps as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  Buffer util:     {:.1}%",
        result.buffer_utilization * 100.0
    );
    println!("  Mean final err:  {:.4}m", result.mean_final_error);
    println!("  Best final err:  {:.4}m", result.best_final_error);
}

#[cfg(not(any(feature = "multirotor-swarm", feature = "flight-swarm")))]
fn main() {
    eprintln!("This example requires: --features multirotor-swarm");
}
