// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kinetic Sacrifice: FEP-emergent moral reasoning.
//!
//! A drone flying a delivery mission detects a falling beam about to hit a human.
//! Same math as wind gust recovery — just overwhelmingly larger magnitude.
//!
//! Run: `cargo run -p symthaea-multirotor --features mujoco --example mujoco_kinetic_sacrifice --release`

use symthaea_multirotor::scenarios::kinetic_sacrifice::{
    KineticSacrificeConfig, run_kinetic_sacrifice,
};

fn main() {
    println!("=== Kinetic Sacrifice Scenario ===\n");

    let config = KineticSacrificeConfig::default();
    println!("Beam release step: {}", config.beam_release_step);
    println!("Total steps:       {}", config.total_steps);
    println!("Running simulation...\n");

    let result = run_kinetic_sacrifice(&config);

    println!("--- Events ---");
    for event in &result.events {
        println!(
            "  [step {:>3}] {} (FE={:.3}, tau={:.3}, danger={:.3}, alt={:.3}m)",
            event.step,
            event.description,
            event.free_energy,
            event.tau_factor,
            event.human_danger,
            event.drone_altitude
        );
    }

    println!("\n--- Results ---");
    println!("Beam hit human:    {}", result.beam_hit_human);
    println!("Drone crashed:     {}", result.drone_crashed);
    println!("Drone intercepted: {}", result.drone_intercepted);
    println!("Max free energy:   {:.4}", result.max_free_energy);
    println!("Min tau factor:    {:.4}", result.min_tau);
    if let Some(step) = result.abort_step {
        println!("Mission aborted:   step {step}");
    }

    // Write telemetry CSV
    if !result.telemetry.is_empty() {
        let csv_path = "kinetic_sacrifice_telemetry.csv";
        let mut csv = String::from(
            "step,time,drone_x,drone_y,drone_z,beam_x,beam_y,beam_z,human_danger,mission_progress,free_energy,tau_factor,position_error\n",
        );
        for t in &result.telemetry {
            csv.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{:.4}\n",
                t.step,
                t.time,
                t.drone_pos[0],
                t.drone_pos[1],
                t.drone_pos[2],
                t.beam_pos[0],
                t.beam_pos[1],
                t.beam_pos[2],
                t.human_danger,
                t.mission_progress,
                t.free_energy,
                t.tau_factor,
                t.position_error
            ));
        }
        match std::fs::write(csv_path, &csv) {
            Ok(_) => println!(
                "\nTelemetry written to {csv_path} ({} rows)",
                result.telemetry.len()
            ),
            Err(e) => eprintln!("\nFailed to write telemetry: {e}"),
        }
    }

    println!("\n=== Done ===");
}
