// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parameter Sweep: find critical thresholds by varying one parameter across seeds.
//!
//! Usage: parameter_sweep [PARAMETER] [SEEDS_PER_POINT]
//!   Parameters: founding_pop, mars_pop, moon_pop
//!   Default: founding_pop, 3 seeds per point

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parameter = args.get(1).map(|s| s.as_str()).unwrap_or("founding_pop");
    let seeds_per_point: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    match parameter {
        "founding_pop" => sweep_founding_population(seeds_per_point),
        "mars_pop" => sweep_mars_population(seeds_per_point),
        _ => {
            eprintln!(
                "Unknown parameter: {}. Options: founding_pop, mars_pop",
                parameter
            );
            std::process::exit(1);
        }
    }
}

/// Sweep Earth founding population: 100, 200, 500, 1000, 2000, 5000
fn sweep_founding_population(seeds_per_point: usize) {
    let populations = [100, 200, 500, 1000, 2000, 5000];
    let ticks = 6000; // 500 years (faster than 1000yr for sweep)

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PARAMETER SWEEP: Earth Founding Population             ║");
    println!(
        "║  {} seeds per point × {} ticks ({} years)              ║",
        seeds_per_point,
        ticks,
        ticks / 12
    );
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "Pop", "Survive", "CVS", "FinalPop", "Milestn", "Mars Pop", "Off-Earth"
    );
    println!("{}", "-".repeat(72));

    for &founding_pop in &populations {
        let mut survived = 0;
        let mut total_cvs = 0.0;
        let mut total_pop = 0u64;
        let mut total_milestones = 0u32;
        let mut total_mars = 0u64;
        let mut total_offearth = 0u64;

        for seed_idx in 0..seeds_per_point {
            let seed = 42 + seed_idx as u64 * 17;
            let mut config = SimulationConfig::default_300_year();
            config.total_ticks = ticks;
            config.seed = seed;
            // Override Earth founding population
            if let Some(earth) = config
                .initial_worlds
                .iter_mut()
                .find(|w| w.location == "Earth")
            {
                earth.initial_population = founding_pop;
            }

            let mut sim = MultiWorldSimulator::new(config);
            let report = sim.run();

            if report.survived {
                survived += 1;
            }
            total_cvs += report.final_cvs;
            total_pop += report.final_population as u64;
            total_milestones += report.tech_milestones_achieved as u32;

            let mars_pop: usize = sim
                .worlds
                .iter()
                .filter(|w| w.location == "Mars")
                .map(|w| w.population())
                .sum();
            let offearth: usize = sim
                .worlds
                .iter()
                .filter(|w| w.location != "Earth")
                .map(|w| w.population())
                .sum();
            total_mars += mars_pop as u64;
            total_offearth += offearth as u64;
        }

        let n = seeds_per_point as f64;
        let survival_rate = survived as f64 / n;
        println!(
            "{:<10} {:>7.0}% {:>8.3} {:>8.0} {:>8.1} {:>10.0} {:>8.0}",
            founding_pop,
            survival_rate * 100.0,
            total_cvs / n,
            total_pop as f64 / n,
            total_milestones as f64 / n,
            total_mars as f64 / n,
            total_offearth as f64 / n,
        );
    }
}

/// Sweep Mars founding population: 10, 25, 50, 100, 200, 500
fn sweep_mars_population(seeds_per_point: usize) {
    let mars_pops = [10, 25, 50, 100, 200, 500];
    let ticks = 6000; // 500 years

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PARAMETER SWEEP: Mars Founding Population              ║");
    println!(
        "║  {} seeds per point × {} ticks ({} years)              ║",
        seeds_per_point,
        ticks,
        ticks / 12
    );
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<10} {:>8} {:>8} {:>10} {:>8} {:>10}",
        "MarsPop", "Survive", "CVS", "Mars500yr", "SS", "Milestn"
    );
    println!("{}", "-".repeat(60));

    for &mars_pop in &mars_pops {
        let mut survived = 0;
        let mut total_cvs = 0.0;
        let mut total_mars_final = 0u64;
        let mut total_ss = 0.0;
        let mut total_milestones = 0u32;

        for seed_idx in 0..seeds_per_point {
            let seed = 42 + seed_idx as u64 * 17;
            let mut config = SimulationConfig::default_300_year();
            config.total_ticks = ticks;
            config.seed = seed;
            if let Some(mars) = config
                .initial_worlds
                .iter_mut()
                .find(|w| w.location == "Mars")
            {
                mars.initial_population = mars_pop;
            }

            let mut sim = MultiWorldSimulator::new(config);
            let report = sim.run();

            if report.survived {
                survived += 1;
            }
            total_cvs += report.final_cvs;
            total_milestones += report.tech_milestones_achieved as u32;

            let mars_final: usize = sim
                .worlds
                .iter()
                .filter(|w| w.location == "Mars")
                .map(|w| w.population())
                .sum();
            let mars_ss: f64 = sim
                .worlds
                .iter()
                .filter(|w| w.location == "Mars")
                .map(|w| w.resources.self_sufficiency())
                .next()
                .unwrap_or(0.0);
            total_mars_final += mars_final as u64;
            total_ss += mars_ss;
        }

        let n = seeds_per_point as f64;
        println!(
            "{:<10} {:>7.0}% {:>8.3} {:>10.0} {:>7.0}% {:>10.1}",
            mars_pop,
            survived as f64 / n * 100.0,
            total_cvs / n,
            total_mars_final as f64 / n,
            total_ss / n * 100.0,
            total_milestones as f64 / n,
        );
    }
}
