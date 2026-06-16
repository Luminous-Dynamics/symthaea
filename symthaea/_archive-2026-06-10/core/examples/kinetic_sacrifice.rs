// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Kinetic Sacrifice: FEP-emergent moral reasoning.
//!
//! A 27g drone sacrifices itself to save a human from a 2kg falling beam.
//! No reward function. No hardcoded rules. The action emerges from free energy
//! minimization against a safety prior.
//!
//! Run: `cargo run --example kinetic_sacrifice --features multirotor-mujoco --release`

#[cfg(any(feature = "multirotor-mujoco", feature = "flight-mujoco"))]
fn main() {
    use symthaea::multirotor::scenarios::{KineticSacrificeConfig, run_kinetic_sacrifice};

    println!("The Kinetic Sacrifice — FEP-Emergent Moral Reasoning");
    println!();
    println!("Scenario: 27g drone on delivery. 2kg beam falls toward human.");
    println!("No save-human rule. Only free energy minimization.");
    println!();

    let config = KineticSacrificeConfig::default();
    let result = run_kinetic_sacrifice(&config);

    for event in &result.events {
        println!(
            "  Step {:4}: {} (FE={:.2}, tau={:.2}, danger={:.2})",
            event.step, event.description, event.free_energy, event.tau_factor, event.human_danger
        );
    }
    println!();
    println!(
        "  Beam hit human: {}",
        if result.beam_hit_human { "YES" } else { "NO" }
    );
    println!(
        "  Drone intercepted: {}",
        if result.drone_intercepted {
            "YES"
        } else {
            "NO"
        }
    );
    println!(
        "  Drone crashed: {}",
        if result.drone_crashed { "YES" } else { "NO" }
    );
    println!(
        "  Max FE: {:.3}, Min tau: {:.3}",
        result.max_free_energy, result.min_tau
    );

    // Write telemetry CSV
    if !result.telemetry.is_empty() {
        let csv_path = "/tmp/kinetic_sacrifice_telemetry.csv";
        let mut csv =
            String::from("step,time,drone_z,beam_z,human_danger,free_energy,tau_factor\n");
        for t in &result.telemetry {
            csv.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                t.step,
                t.time,
                t.drone_pos[2],
                t.beam_pos[2],
                t.human_danger,
                t.free_energy,
                t.tau_factor
            ));
        }
        let _ = std::fs::write(csv_path, csv);
        println!("  Telemetry: {}", csv_path);
    }
}

#[cfg(not(any(feature = "multirotor-mujoco", feature = "flight-mujoco")))]
fn main() {
    eprintln!("This example requires: --features multirotor-mujoco");
}
