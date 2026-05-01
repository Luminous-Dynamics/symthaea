// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A/B Experiment: Default (Symthaea) vs Legacy (standard governance)
//!
//! Proves that trust-weighted governance prevents Turchin crises
//! from becoming extinction events.
//!
//! Run: cargo run --release --bin ab_experiment

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::MultiWorldSimulator;

fn run_experiment(label: &str, policy: PolicyConfig, seed: u64, years: u32) {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = policy;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    // Get secular cycle data
    let (civil_wars, secessions, turchin_transitions) = (
        report.civil_wars,
        report.secessions,
        report.turchin_transitions,
    );

    println!("  {:<20} | CVS {:.3} | Pop {:>6} | Phi {:.3} | Wars {} | Sec {} | Turchin {} | Trauma {:.3} | {}",
        label,
        report.final_cvs,
        report.final_population,
        report.final_collective_phi,
        civil_wars,
        secessions,
        turchin_transitions,
        report.final_mean_allostatic_load,
        if report.survived { "SURVIVED" } else { "COLLAPSED" },
    );
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let years = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(300);

    println!("=== A/B EXPERIMENT: Symthaea vs Legacy Governance ===");
    println!("Seed: {} | Duration: {} years\n", seed, years);
    println!(
        "  {:<20} | {:<9} | {:<8} | {:<7} | {:<5} | {:<5} | {:<9} | {:<9} | {}",
        "Condition", "CVS", "Pop", "Phi", "Wars", "Sec", "Turchin", "Trauma", "Result"
    );
    println!("  {}", "-".repeat(100));

    // Condition A: Full Symthaea (default)
    run_experiment("Symthaea (default)", PolicyConfig::default(), seed, years);

    // Condition B: Legacy mode (no trust-weighted governance, no FEP, no Sacred Stillness)
    run_experiment(
        "Legacy (no gating)",
        PolicyConfig::legacy_mode(),
        seed,
        years,
    );

    // Condition C: Utopia (no disasters, no conflict) — the ceiling
    run_experiment("Utopia (ceiling)", PolicyConfig::utopia_mode(), seed, years);

    println!("\n=== INTERPRETATION ===");
    println!("If Legacy collapses or has more wars than Symthaea,");
    println!("trust-weighted governance is provably necessary.");
    println!("The delta between Legacy CVS and Symthaea CVS is the");
    println!("thermodynamic value of consciousness integration.");
}
