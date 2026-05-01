// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generate reference output tables for reproducibility verification.
//!
//! Runs the 10 canonical seeds at 150yr and prints a CSV table of final metrics.
//! This output can be stored in `data/reference_outputs/` and compared against
//! future runs to detect behavioral regressions.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin generate_reference > data/reference_outputs/canonical_150yr.csv
//! cargo run --release --bin generate_reference -- --ticks 12000 > data/reference_outputs/canonical_1000yr.csv
//! ```

use mycelix_multiworld_sim::config::SimulationConfig;
use mycelix_multiworld_sim::MultiWorldSimulator;

/// The 10 canonical seeds used across all experiments.
const CANONICAL_SEEDS: &[u64] = &[42, 123, 789, 1337, 2718, 3141, 4242, 5555, 7777, 9999];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ticks: u32 = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800); // 150 years

    eprintln!("=== REFERENCE OUTPUT GENERATION ===");
    eprintln!(
        "Seeds: {}, Ticks: {} ({:.0} years)",
        CANONICAL_SEEDS.len(),
        ticks,
        ticks as f64 / 12.0
    );
    eprintln!();

    // CSV header
    println!("seed,ticks,years,final_population,final_cvs,survived,total_disasters,worlds_count,mean_phi,max_phi,mean_gini,governance_authority");

    for &seed in CANONICAL_SEEDS {
        eprint!("Seed {seed}...");
        let mut config = SimulationConfig::default_150_year();
        config.seed = seed;
        config.total_ticks = ticks;

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        let mean_phi: f64 = if sim.worlds.is_empty() {
            0.0
        } else {
            sim.worlds.iter().map(|w| w.mean_phi()).sum::<f64>() / sim.worlds.len() as f64
        };
        let max_phi: f64 = sim
            .worlds
            .iter()
            .map(|w| w.mean_phi())
            .fold(0.0f64, f64::max);
        let mean_gini: f64 = if sim.worlds.is_empty() {
            0.0
        } else {
            sim.worlds
                .iter()
                .map(|w| w.economy.gini_coefficient)
                .sum::<f64>()
                / sim.worlds.len() as f64
        };
        let gov = sim
            .worlds
            .first()
            .map(|w| format!("{}", w.governance.authority_level))
            .unwrap_or_else(|| "None".into());

        println!(
            "{},{},{:.1},{},{:.4},{},{},{},{:.4},{:.4},{:.4},{}",
            seed,
            ticks,
            ticks as f64 / 12.0,
            report.final_population,
            report.final_cvs,
            report.survived,
            report.total_disasters,
            sim.worlds.len(),
            mean_phi,
            max_phi,
            mean_gini,
            gov,
        );
        eprintln!(
            " pop={} cvs={:.4} phi={:.4} gov={}",
            report.final_population, report.final_cvs, mean_phi, gov
        );
    }

    eprintln!();
    eprintln!("Done. Pipe stdout to CSV for reference storage.");
}
