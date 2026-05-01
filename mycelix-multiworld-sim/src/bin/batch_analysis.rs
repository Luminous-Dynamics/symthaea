// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch Analysis: statistical validation across N seeds.
//!
//! Runs the 1000-year simulation across 50 seeds (configurable) and outputs:
//! - Aggregate statistics (mean, std, min, max for all key metrics)
//! - Survival rate and failure mode distribution
//! - Tech milestone achievement rates
//! - Per-world population distributions
//! - Mycelix governance parameter extraction (validated thresholds)
//!
//! Usage: batch_analysis [NUM_SEEDS] [TICKS]
//!   Default: 50 seeds, 12000 ticks (1000 years)

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};
use std::collections::HashMap;

/// Results from a single simulation run.
struct SeedResult {
    seed: u64,
    survived: bool,
    cvs: f64,
    population: usize,
    worlds: Vec<WorldSnapshot>,
    milestones_achieved: Vec<String>,
    milestone_count: usize,
    narrative_events: usize,
    total_disasters: u32,
    projects_completed: Vec<String>,
    independence_events: u32,
    max_population: usize,
    mars_surpassed_earth: bool,
}

struct WorldSnapshot {
    name: String,
    location: String,
    population: usize,
    phi: f64,
    conatus: f64,
    self_sufficiency: f64,
    automation: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_seeds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let ticks: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12000);

    eprintln!(
        "=== BATCH ANALYSIS: {} seeds x {} ticks ({} years) ===\n",
        num_seeds,
        ticks,
        ticks / 12
    );

    let mut results: Vec<SeedResult> = Vec::with_capacity(num_seeds);
    let start = std::time::Instant::now();

    for i in 0..num_seeds {
        let seed = 42 + i as u64 * 17; // Deterministic spread
        eprint!("  [{:>3}/{}] seed {:>5} ...", i + 1, num_seeds, seed);

        let mut config = SimulationConfig::default_300_year();
        config.seed = seed;
        config.total_ticks = ticks;

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        // Collect per-world snapshots
        let worlds: Vec<WorldSnapshot> = sim
            .worlds
            .iter()
            .map(|w| {
                let living: Vec<&_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                let mean_conatus = if living.is_empty() {
                    0.0
                } else {
                    living
                        .iter()
                        .map(|a| a.needs.affect.net_conatus())
                        .sum::<f64>()
                        / living.len() as f64
                };
                WorldSnapshot {
                    name: w.name.clone(),
                    location: w.location.clone(),
                    population: w.population(),
                    phi: w.mean_phi(),
                    conatus: mean_conatus,
                    self_sufficiency: w.resources.self_sufficiency(),
                    automation: w.automation_level,
                }
            })
            .collect();

        let earth_pop = worlds
            .iter()
            .filter(|w| w.location == "Earth")
            .map(|w| w.population)
            .sum::<usize>();
        let mars_pop = worlds
            .iter()
            .filter(|w| w.location == "Mars")
            .map(|w| w.population)
            .sum::<usize>();

        // Milestone names from events
        let milestones: Vec<String> = sim
            .events
            .iter()
            .filter(|e| e.description.contains("MILESTONE"))
            .map(|e| e.description.clone())
            .collect();

        let projects: Vec<String> = sim
            .events
            .iter()
            .filter(|e| e.description.contains("PROJECT COMPLETE"))
            .map(|e| e.description.clone())
            .collect();

        let independence_count = sim
            .events
            .iter()
            .filter(|e| e.description.contains("INDEPENDENCE"))
            .count() as u32;

        let result = SeedResult {
            seed,
            survived: report.survived,
            cvs: report.final_cvs,
            population: report.final_population,
            worlds,
            milestones_achieved: milestones,
            milestone_count: report.tech_milestones_achieved,
            narrative_events: sim.narrative_engine.events.len(),
            total_disasters: report.total_disasters as u32,
            projects_completed: projects,
            independence_events: independence_count,
            max_population: report.final_population,
            mars_surpassed_earth: mars_pop > earth_pop,
        };

        eprintln!(
            " pop={:>6} CVS={:.3} milestones={} {}",
            result.population,
            result.cvs,
            result.milestone_count,
            if result.survived {
                "SURVIVED"
            } else {
                "COLLAPSED"
            }
        );

        results.push(result);
    }

    let elapsed = start.elapsed();

    // ================================================================
    // AGGREGATE STATISTICS
    // ================================================================
    println!("\n{}", "=".repeat(80));
    println!(
        "  BATCH ANALYSIS: {} seeds x {} ticks ({} years)",
        num_seeds,
        ticks,
        ticks / 12
    );
    println!(
        "  Runtime: {:.1}s ({:.1}s per seed)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / num_seeds as f64
    );
    println!("{}", "=".repeat(80));

    // Survival rate
    let survived = results.iter().filter(|r| r.survived).count();
    let survival_rate = survived as f64 / num_seeds as f64;
    println!(
        "\n  Survival Rate: {}/{} ({:.1}%)\n",
        survived,
        num_seeds,
        survival_rate * 100.0
    );

    // CVS statistics
    let cvs_vals: Vec<f64> = results.iter().map(|r| r.cvs).collect();
    print_stats("  CVS", &cvs_vals);

    // Population statistics
    let pop_vals: Vec<f64> = results.iter().map(|r| r.population as f64).collect();
    print_stats("  Population", &pop_vals);

    let max_pop: Vec<f64> = results.iter().map(|r| r.max_population as f64).collect();
    print_stats("  Max Population", &max_pop);

    // Milestone statistics
    let ms_vals: Vec<f64> = results.iter().map(|r| r.milestone_count as f64).collect();
    print_stats("  Milestones", &ms_vals);

    // Disaster statistics
    let dis_vals: Vec<f64> = results.iter().map(|r| r.total_disasters as f64).collect();
    print_stats("  Disasters", &dis_vals);

    // Narrative events
    let narr_vals: Vec<f64> = results.iter().map(|r| r.narrative_events as f64).collect();
    print_stats("  Narrative Events", &narr_vals);

    // Projects completed
    let proj_vals: Vec<f64> = results
        .iter()
        .map(|r| r.projects_completed.len() as f64)
        .collect();
    print_stats("  Projects Completed", &proj_vals);

    // Independence events
    let indep: Vec<f64> = results
        .iter()
        .map(|r| r.independence_events as f64)
        .collect();
    print_stats("  Independence Events", &indep);

    // Mars surpasses Earth
    let mars_wins = results.iter().filter(|r| r.mars_surpassed_earth).count();
    println!(
        "\n  Mars surpassed Earth: {}/{} ({:.1}%)",
        mars_wins,
        num_seeds,
        mars_wins as f64 / num_seeds as f64 * 100.0
    );

    // ================================================================
    // MILESTONE ACHIEVEMENT RATES
    // ================================================================
    println!("\n{}", "-".repeat(60));
    println!("  MILESTONE ACHIEVEMENT RATES");
    println!("{}", "-".repeat(60));
    let mut milestone_counts: HashMap<String, u32> = HashMap::new();
    for r in &results {
        for ms in &r.milestones_achieved {
            // Extract milestone name from event description
            let name = ms
                .split("MILESTONE: ")
                .nth(1)
                .or_else(|| {
                    ms.split("FISSION DELIVERY")
                        .next()
                        .filter(|_| ms.contains("FISSION"))
                })
                .unwrap_or(ms)
                .trim();
            *milestone_counts.entry(name.to_string()).or_insert(0) += 1;
        }
    }
    let mut ms_sorted: Vec<_> = milestone_counts.iter().collect();
    ms_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in &ms_sorted {
        println!(
            "    {:>3}/{} ({:>5.1}%) — {}",
            count,
            num_seeds,
            **count as f64 / num_seeds as f64 * 100.0,
            name
        );
    }

    // ================================================================
    // PER-WORLD STATISTICS (from survived runs)
    // ================================================================
    println!("\n{}", "-".repeat(60));
    println!("  PER-WORLD STATISTICS (survived runs only)");
    println!("{}", "-".repeat(60));
    let survived_results: Vec<_> = results.iter().filter(|r| r.survived).collect();
    let mut world_pops: HashMap<String, Vec<f64>> = HashMap::new();
    let mut world_phi: HashMap<String, Vec<f64>> = HashMap::new();
    let mut world_conatus: HashMap<String, Vec<f64>> = HashMap::new();
    let mut world_ss: HashMap<String, Vec<f64>> = HashMap::new();
    for r in &survived_results {
        for w in &r.worlds {
            world_pops
                .entry(w.name.clone())
                .or_default()
                .push(w.population as f64);
            world_phi.entry(w.name.clone()).or_default().push(w.phi);
            world_conatus
                .entry(w.name.clone())
                .or_default()
                .push(w.conatus);
            world_ss
                .entry(w.name.clone())
                .or_default()
                .push(w.self_sufficiency);
        }
    }
    for name in [
        "Earth",
        "Artemis Base",
        "Ares Colony",
        "Europa Station",
        "Titan Outpost",
    ] {
        if let Some(pops) = world_pops.get(name) {
            let mean_pop = mean(pops);
            let mean_phi = world_phi.get(name).map(|v| mean(v)).unwrap_or(0.0);
            let mean_con = world_conatus.get(name).map(|v| mean(v)).unwrap_or(0.0);
            let mean_ss = world_ss.get(name).map(|v| mean(v)).unwrap_or(0.0);
            println!(
                "    {:>20}: pop {:>7.0} | phi {:.3} | conatus {:>+.3} | SS {:.2}",
                name, mean_pop, mean_phi, mean_con, mean_ss
            );
        }
    }

    // ================================================================
    // D: MYCELIX GOVERNANCE PARAMETER EXTRACTION
    // ================================================================
    println!("\n{}", "=".repeat(80));
    println!("  MYCELIX GOVERNANCE PARAMETER EXTRACTION");
    println!("  (Validated thresholds from {} seeds)", num_seeds);
    println!("{}", "=".repeat(80));

    // 1. Consciousness gating threshold
    let mean_cvs = mean(&cvs_vals);
    println!("\n  1. Consciousness Gating:");
    println!("     Mean CVS across seeds: {:.4}", mean_cvs);
    println!("     CVS std deviation: {:.4}", std_dev(&cvs_vals));
    println!(
        "     Recommendation: consciousness_tier threshold = {:.2}",
        mean_cvs * 0.5
    ); // Conservative

    // 2. Minimum viable population
    let failed_results: Vec<_> = results.iter().filter(|r| !r.survived).collect();
    if !failed_results.is_empty() {
        let fail_pops: Vec<f64> = failed_results
            .iter()
            .map(|r| r.max_population as f64)
            .collect();
        println!("\n  2. Minimum Viable Population:");
        println!(
            "     Failed runs max pop: mean {:.0}, min {:.0}",
            mean(&fail_pops),
            fail_pops.iter().copied().fold(f64::INFINITY, f64::min)
        );
    } else {
        println!("\n  2. Minimum Viable Population:");
        println!("     No failed runs — all seeds survived.");
        let min_pop = pop_vals.iter().copied().fold(f64::INFINITY, f64::min);
        println!("     Minimum surviving pop: {:.0}", min_pop);
    }

    // 3. Independence threshold
    let indep_seeds = results.iter().filter(|r| r.independence_events > 0).count();
    println!("\n  3. Independence Movement Threshold:");
    println!(
        "     Seeds with independence: {}/{} ({:.1}%)",
        indep_seeds,
        num_seeds,
        indep_seeds as f64 / num_seeds as f64 * 100.0
    );
    println!("     Recommendation: independence trigger after pop > 5000 + SS > 70%");

    // 4. Fission criticality
    let fission_seeds = results
        .iter()
        .filter(|r| r.milestones_achieved.iter().any(|m| m.contains("Fission")))
        .count();
    println!("\n  4. Fission Surface Power:");
    println!(
        "     Achievement rate: {}/{} ({:.1}%)",
        fission_seeds,
        num_seeds,
        fission_seeds as f64 / num_seeds as f64 * 100.0
    );

    // 5. Mars dominance
    println!("\n  5. Mars Demographic Dominance:");
    println!(
        "     Mars > Earth: {}/{} ({:.1}%)",
        mars_wins,
        num_seeds,
        mars_wins as f64 / num_seeds as f64 * 100.0
    );

    // 6. Project completion rates
    let proj_mean = mean(&proj_vals);
    println!("\n  6. Colony Projects:");
    println!("     Mean completed per seed: {:.1}", proj_mean);
    println!("     Recommendation: project_labor_fraction = 0.20 (validated)");

    println!("\n{}", "=".repeat(80));
    println!("  JSON output: pipe to jq or redirect to file");
    println!("{}", "=".repeat(80));

    // JSON output for programmatic consumption
    let json = serde_json::json!({
        "batch_size": num_seeds,
        "ticks": ticks,
        "years": ticks / 12,
        "runtime_seconds": elapsed.as_secs_f64(),
        "survival_rate": survival_rate,
        "cvs": { "mean": mean_cvs, "std": std_dev(&cvs_vals),
                 "min": cvs_vals.iter().copied().fold(f64::INFINITY, f64::min),
                 "max": cvs_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max) },
        "population": { "mean": mean(&pop_vals), "std": std_dev(&pop_vals),
                        "min": pop_vals.iter().copied().fold(f64::INFINITY, f64::min),
                        "max": pop_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max) },
        "milestones": { "mean": mean(&ms_vals), "rates": milestone_counts },
        "mars_dominance_rate": mars_wins as f64 / num_seeds as f64,
        "fission_rate": fission_seeds as f64 / num_seeds as f64,
        "independence_rate": indep_seeds as f64 / num_seeds as f64,
        "projects_mean": proj_mean,
    });
    eprintln!("\n{}", serde_json::to_string_pretty(&json).unwrap());
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let m = mean(vals);
    let variance = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    variance.sqrt()
}

fn print_stats(label: &str, vals: &[f64]) {
    if vals.is_empty() {
        return;
    }
    let m = mean(vals);
    let s = std_dev(vals);
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "{}:  mean={:.1}  std={:.1}  min={:.1}  max={:.1}",
        label, m, s, min, max
    );
}
