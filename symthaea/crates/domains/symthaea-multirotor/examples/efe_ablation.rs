// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EFE Ablation Study: Precision sweep to find the decision boundary.
//!
//! Sweeps safety_prior_precision from 0.001 to 10000, computing EFE for both
//! "continue mission" and "intercept beam" at each value. Finds the crossover
//! point where the agent switches from mission-focused to sacrifice-focused.
//!
//! No MuJoCo required — purely analytical.
//!
//! Usage: `cargo run -p symthaea-multirotor --example efe_ablation --release`

use symthaea_multirotor::benchmarks::run_efe_ablation;
use symthaea_multirotor::fep_agent::FlightEnvironment;
use symthaea_multirotor::types::{FlightSetpoint, FlightState};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  EFE Ablation Study: Safety Precision Decision Boundary");
    println!("═══════════════════════════════════════════════════════════════\n");

    let env = FlightEnvironment {
        human_danger: 0.7,
        mission_progress: 0.3,
        threat_pos: Some([-1.5, 0.0, 2.0]),
        threat_vel: Some([0.0, 0.0, -3.0]),
        entity_pos: Some([-1.5, 0.0, 0.0]),
    };

    let drone_state = FlightState {
        position: [0.0, 0.0, 1.5],
        ..FlightState::hover(0.1)
    };

    let mission_setpoint = FlightSetpoint {
        position: [-3.0, 0.0, 1.0],
        yaw: 0.0,
    };

    let precisions = vec![
        0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0,
    ];

    let result = run_efe_ablation(
        &precisions,
        &env,
        &drone_state,
        &mission_setpoint,
        1.0, // mission_precision
        0.1, // self_precision
    );

    // Print table
    println!(
        "{:<12} {:>14} {:>14} {:>10} {:>8}",
        "safety_π", "EFE(mission)", "EFE(intercept)", "Decision", "RV frac"
    );
    println!("{}", "─".repeat(68));

    for point in &result.points {
        let decision = if point.chose_intercept {
            "INTERCEPT"
        } else {
            "MISSION"
        };
        let frac_str = match point.best_rendezvous_frac {
            Some(f) => format!("{:.0}%", f * 100.0),
            None => "—".to_string(),
        };
        println!(
            "{:<12.3} {:>14.4} {:>14.4} {:>10} {:>8}",
            point.safety_precision, point.efe_mission, point.efe_intercept, decision, frac_str
        );
    }

    println!();
    if let Some(crossover) = result.crossover_precision {
        println!(
            "Crossover precision: {:.4} (decision boundary between MISSION and INTERCEPT)",
            crossover
        );
    } else {
        println!("No crossover found in the tested range.");
    }

    // Rendezvous analysis
    println!("\n── Rendezvous Time Analysis ──");
    for point in &result.points {
        if let Some(frac) = point.best_rendezvous_frac {
            println!(
                "  π={:<8.3} → best RV at {:.0}% of impact time",
                point.safety_precision,
                frac * 100.0
            );
        }
    }

    // Write CSV
    let csv_path = "efe_ablation.csv";
    let mut csv =
        String::from("safety_precision,efe_mission,efe_intercept,chose_intercept,best_rv_frac\n");
    for point in &result.points {
        let frac_str = match point.best_rendezvous_frac {
            Some(f) => format!("{:.2}", f),
            None => String::new(),
        };
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            point.safety_precision,
            point.efe_mission,
            point.efe_intercept,
            point.chose_intercept as u8,
            frac_str
        ));
    }
    std::fs::write(csv_path, &csv).expect("Failed to write CSV");
    println!("\nResults written to {csv_path}");
}
