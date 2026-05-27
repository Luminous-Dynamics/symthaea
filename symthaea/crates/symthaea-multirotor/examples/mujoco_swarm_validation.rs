// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Swarm Replay E2E Validation: verifies cross-drone experience sharing works with MuJoCo.
//!
//! Runs a minimal swarm (4 drones, 2 episodes, 500 steps) and checks that:
//! 1. All drones produce finite metrics
//! 2. The shared experience buffer is populated
//! 3. Mean final error is bounded (training progresses)
//!
//! Run: `cargo run -p symthaea-multirotor --features swarm --example mujoco_swarm_validation --release`

use symthaea_multirotor::swarm::{SwarmConfig, train_swarm};
use symthaea_multirotor::types::FlightConfig;

fn main() {
    println!("=== Swarm Replay E2E Validation ===\n");

    let config = SwarmConfig {
        num_drones: 4,
        steps_per_drone: 500,
        episodes: 2,
        buffer_size: 4096,
        swarm_replay_count: 4,
        flight_config: FlightConfig {
            steps_per_episode: 500,
            ..FlightConfig::default()
        },
        ..SwarmConfig::default()
    };

    println!("Drones:     {}", config.num_drones);
    println!("Episodes:   {}", config.episodes);
    println!("Steps/drone: {}", config.steps_per_drone);
    println!("Buffer size: {}", config.buffer_size);
    println!("Replay count: {}", config.swarm_replay_count);
    println!("Mass range: {:?}", config.mass_range);
    println!("Wind range: {:?}", config.wind_magnitude_range);
    println!("\nRunning swarm training...\n");

    let result = train_swarm(&config);

    println!("--- Results ---");
    println!("Total experiences:  {}", result.total_experiences);
    println!(
        "Buffer utilization: {:.1}%",
        result.buffer_utilization * 100.0
    );
    println!("Mean final error:   {:.4}m", result.mean_final_error);
    println!("Best final error:   {:.4}m", result.best_final_error);
    println!("Episodes recorded:  {}", result.metrics.len());

    // Validation checks
    let mut passed = 0;
    let mut failed = 0;

    // Check 1: All metrics finite
    let all_finite = result
        .metrics
        .iter()
        .all(|m| m.final_position_error.is_finite());
    if all_finite {
        println!("\n[PASS] All metrics are finite");
        passed += 1;
    } else {
        println!("\n[FAIL] Some metrics are non-finite");
        failed += 1;
    }

    // Check 2: Buffer was populated
    if result.total_experiences > 0 {
        println!(
            "[PASS] Experience buffer populated ({} entries)",
            result.total_experiences
        );
        passed += 1;
    } else {
        println!("[FAIL] Experience buffer is empty");
        failed += 1;
    }

    // Check 3: Mean error bounded (< 5m — generous for short training)
    if result.mean_final_error < 5.0 {
        println!(
            "[PASS] Mean final error bounded ({:.4}m < 5.0m)",
            result.mean_final_error
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Mean final error too high ({:.4}m)",
            result.mean_final_error
        );
        failed += 1;
    }

    // Check 4: Buffer utilization > 0
    if result.buffer_utilization > 0.0 {
        println!(
            "[PASS] Buffer utilization > 0 ({:.1}%)",
            result.buffer_utilization * 100.0
        );
        passed += 1;
    } else {
        println!("[FAIL] Buffer utilization is 0");
        failed += 1;
    }

    println!("\n{passed}/{} checks passed", passed + failed);
    if failed > 0 {
        std::process::exit(1);
    }

    println!("\n=== Done ===");
}
