// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Survival Reflex: PD-CfC-FEP robustness under sudden mass change.
//!
//! Demonstrates extreme robustness of the PD-CfC-FEP architecture under
//! a +30% mass change mid-flight. The PD-CfC blend provides a reactive
//! safety net while the FEP agent modulates tau and learning rate.
//!
//! Runs the same perturbation twice (FEP active vs frozen) for comparison.
//!
//! Run: `cargo run -p symthaea-multirotor --features mujoco --example mujoco_survival_reflex --release`

use symthaea_multirotor::benchmarks::run_perturbation_comparison;
use symthaea_multirotor::perturbations::PerturbationSchedule;
use symthaea_multirotor::types::FlightConfig;

fn main() {
    println!("=== Survival Reflex: FEP Comparison ===");
    println!("PerturbationSchedule::survival_reflex() — +30% mass at step 250");
    println!("Running twice: FEP active vs FEP frozen\n");

    let config = FlightConfig {
        steps_per_episode: 1000,
        collect_telemetry: true,
        ..FlightConfig::default()
    };
    let schedule = PerturbationSchedule::survival_reflex();

    let comparison = run_perturbation_comparison(&config, &schedule);

    let active = &comparison.fep_active;
    let frozen = &comparison.fep_frozen;

    println!("{:<28} {:>12} {:>12}", "", "FEP Active", "FEP Frozen");
    println!("{:-<52}", "");
    println!(
        "{:<28} {:>12.4} {:>12.4}",
        "Pre-perturbation error (m)", active.pre_perturbation_error, frozen.pre_perturbation_error
    );
    println!(
        "{:<28} {:>12.4} {:>12.4}",
        "Peak error (m)", active.peak_error, frozen.peak_error
    );
    println!(
        "{:<28} {:>12} {:>12}",
        "Recovery steps (<5cm)", active.recovery_steps, frozen.recovery_steps
    );
    println!(
        "{:<28} {:>12.4} {:>12.4}",
        "Final error (m)", active.final_error, frozen.final_error
    );
    println!(
        "{:<28} {:>12.3} {:>12.3}",
        "Min tau", active.min_tau, frozen.min_tau
    );
    println!(
        "{:<28} {:>12} {:>12}",
        "Crashed", active.crashed, frozen.crashed
    );

    // Write CSV telemetry (FEP active run)
    let csv_path = "survival_reflex_telemetry.csv";
    let mut csv = String::from("step,position_error,free_energy,tau_factor\n");
    for (i, ((err, fe), tau)) in active
        .error_trace
        .iter()
        .zip(active.free_energy_trace.iter())
        .zip(active.tau_trace.iter())
        .enumerate()
    {
        csv.push_str(&format!("{i},{err:.6},{fe:.6},{tau:.4}\n"));
    }

    match std::fs::write(csv_path, &csv) {
        Ok(_) => println!(
            "\nTelemetry written to {csv_path} ({} rows)",
            active.error_trace.len()
        ),
        Err(e) => eprintln!("\nFailed to write telemetry: {e}"),
    }

    println!("\n=== Done ===");
}
