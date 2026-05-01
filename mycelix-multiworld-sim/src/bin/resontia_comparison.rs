// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare 777-year simulation: standard vs Resontia Earth-hardening.
use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    let seed = 42u64;
    eprintln!("=== RESONTIA COMPARISON (seed {seed}, 777 years) ===\n");

    // Standard 777-year
    eprint!("  Running STANDARD...");
    let mut config = SimulationConfig::default_777_year();
    config.seed = seed;
    let mut sim = MultiWorldSimulator::new(config);
    let report_std = sim.run();
    eprintln!(" done (CVS: {:.3})", report_std.final_cvs);

    // Resontia 777-year
    eprint!("  Running RESONTIA...");
    let mut config_r = SimulationConfig::default_777_year();
    config_r.seed = seed;
    let mut sim_r = MultiWorldSimulator::new(config_r);
    sim_r.enable_resontia();
    let report_res = sim_r.run();
    eprintln!(" done (CVS: {:.3})", report_res.final_cvs);

    println!("\n=== RESULTS ===");
    println!(
        "{:<12} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10}",
        "Condition", "CVS", "Pop", "Phi", "Love", "Disasters"
    );
    println!("{}", "-".repeat(70));
    println!(
        "{:<12} | {:>8.3} | {:>8} | {:>8.3} | {:>8.3} | {:>10}",
        "Standard",
        report_std.final_cvs,
        report_std.final_population,
        report_std.final_collective_phi,
        report_std.final_love_coherence,
        report_std.total_disasters
    );
    println!(
        "{:<12} | {:>8.3} | {:>8} | {:>8.3} | {:>8.3} | {:>10}",
        "Resontia",
        report_res.final_cvs,
        report_res.final_population,
        report_res.final_collective_phi,
        report_res.final_love_coherence,
        report_res.total_disasters
    );

    let cvs_delta = ((report_res.final_cvs - report_std.final_cvs) / report_std.final_cvs) * 100.0;
    let pop_delta = report_res.final_population as f64 - report_std.final_population as f64;
    println!(
        "\nDelta: CVS {:+.1}%, Population {:+.0}",
        cvs_delta, pop_delta
    );
}
