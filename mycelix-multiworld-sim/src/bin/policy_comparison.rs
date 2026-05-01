// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Policy Comparison: run 4 governance presets side-by-side.
//!
//! Compares Survival First, Growth First, Science First, and Independence First
//! across N seeds each. Produces the Mycelix governance deliverable: which
//! policy configuration produces the most resilient multi-world civilization?
//!
//! Usage: policy_comparison [SEEDS_PER_POLICY] [TICKS]
//!   Default: 3 seeds per policy, 6000 ticks (500 years)

use mycelix_multiworld_sim::{
    config::{PolicyConfig, SimulationConfig},
    MultiWorldSimulator,
};

struct PolicyResult {
    name: String,
    survival_rate: f64,
    mean_cvs: f64,
    mean_population: f64,
    mean_milestones: f64,
    mean_mars_pop: f64,
    mean_independence: f64,
    mean_projects: f64,
}

fn run_policy(name: &str, policy: PolicyConfig, seeds: usize, ticks: u32) -> PolicyResult {
    let mut survived = 0u32;
    let mut total_cvs = 0.0;
    let mut total_pop = 0.0;
    let mut total_ms = 0.0;
    let mut total_mars = 0.0;
    let mut total_indep = 0.0;
    let mut total_projects = 0.0;

    for i in 0..seeds {
        let seed = 42 + i as u64 * 17;
        eprint!("  {} seed {:>3} ...", name, seed);

        let mut config = SimulationConfig::default_300_year();
        config.total_ticks = ticks;
        config.seed = seed;
        config.policy = policy.clone();

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        if report.survived {
            survived += 1;
        }
        total_cvs += report.final_cvs;
        total_pop += report.final_population as f64;
        total_ms += report.tech_milestones_achieved as f64;

        let mars: f64 = sim
            .worlds
            .iter()
            .filter(|w| w.location == "Mars")
            .map(|w| w.population() as f64)
            .sum();
        let indep = sim
            .events
            .iter()
            .filter(|e| e.description.contains("INDEPENDENCE"))
            .count() as f64;
        let projects = sim
            .events
            .iter()
            .filter(|e| e.description.contains("PROJECT COMPLETE"))
            .count() as f64;

        total_mars += mars;
        total_indep += indep;
        total_projects += projects;

        eprintln!(
            " CVS={:.3} pop={:.0} {}",
            report.final_cvs,
            report.final_population,
            if report.survived { "OK" } else { "FAIL" }
        );
    }

    let n = seeds as f64;
    PolicyResult {
        name: name.to_string(),
        survival_rate: survived as f64 / n,
        mean_cvs: total_cvs / n,
        mean_population: total_pop / n,
        mean_milestones: total_ms / n,
        mean_mars_pop: total_mars / n,
        mean_independence: total_indep / n,
        mean_projects: total_projects / n,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seeds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let ticks: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6000);

    eprintln!(
        "=== POLICY COMPARISON: {} seeds × {} ticks ({} years) ===\n",
        seeds,
        ticks,
        ticks / 12
    );

    let policies: Vec<(&str, PolicyConfig)> = vec![
        ("Balanced", PolicyConfig::default()),
        ("Survival", PolicyConfig::survival_first()),
        ("Growth", PolicyConfig::growth_first()),
        ("Science", PolicyConfig::science_first()),
        ("Independence", PolicyConfig::independence_first()),
    ];

    let mut results = Vec::new();
    for (name, policy) in &policies {
        results.push(run_policy(name, policy.clone(), seeds, ticks));
    }

    // Print comparison table
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  POLICY COMPARISON: Which governance produces the best civilization? ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<14} {:>8} {:>8} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "Policy", "Survive", "CVS", "Population", "Milstn", "Mars Pop", "Indep", "Projects"
    );
    println!("{}", "-".repeat(82));

    for r in &results {
        println!(
            "{:<14} {:>7.0}% {:>8.3} {:>10.0} {:>8.1} {:>10.0} {:>8.1} {:>8.0}",
            r.name,
            r.survival_rate * 100.0,
            r.mean_cvs,
            r.mean_population,
            r.mean_milestones,
            r.mean_mars_pop,
            r.mean_independence,
            r.mean_projects
        );
    }

    // Winner analysis
    println!("\n{}", "=".repeat(82));
    let best_cvs = results
        .iter()
        .max_by(|a, b| a.mean_cvs.partial_cmp(&b.mean_cvs).unwrap())
        .unwrap();
    let best_pop = results
        .iter()
        .max_by(|a, b| a.mean_population.partial_cmp(&b.mean_population).unwrap())
        .unwrap();
    let best_ms = results
        .iter()
        .max_by(|a, b| a.mean_milestones.partial_cmp(&b.mean_milestones).unwrap())
        .unwrap();
    println!(
        "  Best CVS (resilience):  {} ({:.3})",
        best_cvs.name, best_cvs.mean_cvs
    );
    println!(
        "  Best Population:        {} ({:.0})",
        best_pop.name, best_pop.mean_population
    );
    println!(
        "  Best Tech:              {} ({:.1} milestones)",
        best_ms.name, best_ms.mean_milestones
    );

    // Mycelix governance recommendation
    println!("\n  MYCELIX GOVERNANCE RECOMMENDATION:");
    println!(
        "  The {} policy produces the highest civilization viability score.",
        best_cvs.name
    );
    println!(
        "  Validated across {} seeds × {} years of simulation.",
        seeds,
        ticks / 12
    );
}
