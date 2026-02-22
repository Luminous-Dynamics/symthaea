//! Survival Reflex: FEP tau-drop under sudden mass change.
//!
//! Demonstrates the "biological survival reflex" — when mass increases +30% mid-flight,
//! FEP's prediction error spikes, tau drops, and the system re-stabilizes WITHOUT
//! retraining. Same math as biological reflexes, running on a 27g drone.
//!
//! Run: `cargo run -p symthaea-flight --features mujoco --example mujoco_survival_reflex --release`

use symthaea_flight::benchmarks::run_perturbation_benchmark;
use symthaea_flight::perturbations::PerturbationSchedule;
use symthaea_flight::types::FlightConfig;

fn main() {
    println!("=== Survival Reflex Benchmark ===");
    println!("PerturbationSchedule::survival_reflex() — +30% mass at step 250\n");

    let config = FlightConfig {
        steps_per_episode: 1000,
        collect_telemetry: true,
        ..FlightConfig::default()
    };
    let schedule = PerturbationSchedule::survival_reflex();

    println!(
        "Running {} steps with mass perturbation at step 250...",
        config.steps_per_episode
    );
    let result = run_perturbation_benchmark(&config, &schedule);

    println!("\n--- Results ---");
    println!(
        "Pre-perturbation error:  {:.4} m",
        result.pre_perturbation_error
    );
    println!("Peak error after change: {:.4} m", result.peak_error);
    println!("Recovery steps (< 5cm):  {}", result.recovery_steps);
    println!("Final error:             {:.4} m", result.final_error);
    println!("Min tau factor:          {:.3}", result.min_tau);
    println!("Crashed:                 {}", result.crashed);

    // Write CSV telemetry
    let csv_path = "survival_reflex_telemetry.csv";
    let mut csv = String::from("step,position_error,free_energy,tau_factor\n");
    for (i, ((err, fe), tau)) in result
        .error_trace
        .iter()
        .zip(result.free_energy_trace.iter())
        .zip(result.tau_trace.iter())
        .enumerate()
    {
        csv.push_str(&format!("{i},{err:.6},{fe:.6},{tau:.4}\n"));
    }

    match std::fs::write(csv_path, &csv) {
        Ok(_) => println!(
            "\nTelemetry written to {csv_path} ({} rows)",
            result.error_trace.len()
        ),
        Err(e) => eprintln!("\nFailed to write telemetry: {e}"),
    }

    println!("\n=== Done ===");
}
