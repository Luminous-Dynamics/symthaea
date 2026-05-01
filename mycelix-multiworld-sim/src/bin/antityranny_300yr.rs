// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 300-year anti-tyranny pilot run.
//!
//! Tests whether the 80% veto override, term limits, and emergency council
//! constraints create a stable governance attractor across the Roots-to-Branches
//! transition (years 20-60) and the deep-time Convergence epoch (years 150-200).
//!
//! Key hypothesis: The mere EXISTENCE of the 80% override should deter most
//! Guardian vetoes, meaning veto_count << veto_deterred_count.

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    let seeds: Vec<u64> = std::env::args()
        .nth(1)
        .map(|s| vec![s.parse::<u64>().unwrap_or(42)])
        .unwrap_or_else(|| vec![42, 123, 789, 1337, 2718]);

    println!("=== 300-YEAR ANTI-TYRANNY PILOT RUN ===");
    println!(
        "Testing governance invariants across {} seeds\n",
        seeds.len()
    );
    println!(
        "{:<6} {:>5} {:>5} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6}",
        "Seed", "CVS", "Alive", "Vetoes", "Overrd", "Deter", "Crises", "Amend", "Pop", "Phi"
    );
    println!("{}", "-".repeat(80));

    let mut all_survived = true;
    let mut total_vetoes = 0u32;
    let mut total_overrides = 0u32;
    let mut total_deterred = 0u32;

    for &seed in &seeds {
        let mut config = SimulationConfig::default_300_year();
        config.seed = seed;

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        // Aggregate governance stats from all worlds
        let mut vetoes = 0u32;
        let mut overrides = 0u32;
        let mut deterred = 0u32;
        let mut crises = 0u32;
        let mut amendments = 0u32;

        for world in &sim.worlds {
            vetoes += world.governance.veto_count;
            overrides += world.governance.veto_override_count;
            deterred += world.governance.veto_deterred_count;
            amendments += world.governance.amendment_count;
            if world.governance.constitutional_crisis {
                crises += 1;
            }
        }

        let final_pop = report
            .epoch_snapshots
            .last()
            .map(|s| s.total_population)
            .unwrap_or(0);
        let final_phi = report
            .epoch_snapshots
            .last()
            .map(|s| s.mean_phi)
            .unwrap_or(0.0);

        println!(
            "{:<6} {:>5.3} {:>5} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6.3}",
            seed,
            report.final_cvs,
            if report.survived { "YES" } else { "NO" },
            vetoes,
            overrides,
            deterred,
            crises,
            amendments,
            final_pop,
            final_phi,
        );

        if !report.survived {
            all_survived = false;
        }
        total_vetoes += vetoes;
        total_overrides += overrides;
        total_deterred += deterred;
    }

    println!("{}", "-".repeat(80));
    println!("\n=== ANTI-TYRANNY ANALYSIS ===\n");

    println!("Total vetoes attempted:  {}", total_vetoes);
    println!("Total vetoes overridden: {}", total_overrides);
    println!("Total vetoes deterred:   {}", total_deterred);
    if total_vetoes + total_deterred > 0 {
        let deterrence_rate =
            total_deterred as f64 / (total_vetoes + total_deterred) as f64 * 100.0;
        println!("Deterrence rate:         {:.1}%", deterrence_rate);
        println!(
            "\n  -> {}",
            if deterrence_rate > 50.0 {
                "HYPOTHESIS CONFIRMED: Override existence deters majority of vetoes"
            } else {
                "HYPOTHESIS INCONCLUSIVE: Override mechanism actively needed"
            }
        );
    }

    if total_overrides > 0 {
        let override_rate = total_overrides as f64 / total_vetoes as f64 * 100.0;
        println!(
            "\nOverride success rate:   {:.1}% of attempted vetoes reversed",
            override_rate
        );
    }

    println!(
        "\nSurvival: {} / {} seeds survived 300 years",
        seeds
            .iter()
            .filter(|_| all_survived)
            .count()
            .max(if all_survived { seeds.len() } else { 0 }),
        seeds.len()
    );

    if !all_survived {
        std::process::exit(1);
    }
}
