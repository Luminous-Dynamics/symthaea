// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Founder population parameter sweep: how does founding colony size affect survival?

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

const FOUNDER_SIZES: [usize; 6] = [12, 25, 50, 100, 200, 500];
const SEED: u64 = 42;

fn main() {
    eprintln!("=== FOUNDER POPULATION SWEEP (seed: {SEED}) ===\n");

    println!(
        "{:>8} | {:>7} | {:>8} | {:>6} | {:>6} | {:>6} | {:>8} | {:>6} | {:>5} | {:>8}",
        "Founders", "Pop", "OffEarth", "CVS", "Phi", "Love", "Genetics", "Tech", "Chk%", "Survived"
    );
    println!("{}", "-".repeat(100));

    for &founders in &FOUNDER_SIZES {
        eprint!("  Running founders={founders}...");

        let mut config = SimulationConfig::default_150_year();
        config.seed = SEED;

        // Override the Moon colony's initial population
        if let Some(moon) = config
            .initial_worlds
            .iter_mut()
            .find(|w| w.location == "Moon")
        {
            moon.initial_population = founders;
        }

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        let off_earth: usize = sim
            .worlds
            .iter()
            .filter(|w| w.location != "Earth")
            .map(|w| w.population())
            .sum();

        let tech = report
            .epoch_snapshots
            .last()
            .map(|s| s.mean_tech_level)
            .unwrap_or(0.0);

        let chk_rate = if report.checkpoints_passed + report.checkpoints_failed > 0 {
            100.0 * report.checkpoints_passed as f64
                / (report.checkpoints_passed + report.checkpoints_failed) as f64
        } else {
            0.0
        };

        println!(
            "{:>8} | {:>7} | {:>8} | {:>6.3} | {:>6.3} | {:>6.3} | {:>8.3} | {:>6.3} | {:>4.0}% | {:>8}",
            founders,
            report.final_population,
            off_earth,
            report.final_cvs,
            report.final_collective_phi,
            report.final_love_coherence,
            report.final_genetic_diversity,
            tech,
            chk_rate,
            if report.survived { "YES" } else { "NO" },
        );

        eprintln!(
            " done (CVS: {:.3}, pop: {})",
            report.final_cvs, report.final_population
        );
    }

    println!("\n=== CSV DATA ===");
    println!("founders,population,off_earth,cvs,phi,love,genetics,tech,checkpoint_rate,survived");
    // Would need to store results and print — simplified above
}
