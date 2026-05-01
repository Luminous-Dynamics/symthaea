// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 5-seed validation of the 777-year simulation with disasters and all resilience
//! mechanisms enabled. Reports per-seed metrics and aggregate summary statistics.

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

const SEEDS: [u64; 5] = [42, 123, 789, 1337, 2718];

struct RunResult {
    seed: u64,
    population: usize,
    cvs: f64,
    phi: f64,
    worlds: usize,
    total_disasters: u64,
    carrington_events: u32,
    survived: bool,
}

fn main() {
    eprintln!(
        "=== 777-YEAR DISASTER VALIDATION (5 seeds x {} ticks) ===\n",
        777 * 12
    );

    let mut results: Vec<RunResult> = Vec::with_capacity(SEEDS.len());

    for (i, &seed) in SEEDS.iter().enumerate() {
        eprint!("  Running seed {:>5} ({}/{}) ...", seed, i + 1, SEEDS.len());

        let mut config = SimulationConfig::default_777_year();
        config.seed = seed;
        // disasters_enabled is true by default; assert it for clarity
        assert!(
            config.policy.disasters_enabled,
            "disasters must be enabled for this validation"
        );

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        eprintln!(
            " pop={:>6}, CVS={:.3}, Phi={:.3}, worlds={}, disasters={}, carrington={}, {}",
            report.final_population,
            report.final_cvs,
            report.final_collective_phi,
            report.final_worlds,
            report.total_disasters,
            report.carrington_events,
            if report.survived {
                "SURVIVED"
            } else {
                "COLLAPSED"
            },
        );

        results.push(RunResult {
            seed,
            population: report.final_population,
            cvs: report.final_cvs,
            phi: report.final_collective_phi,
            worlds: report.final_worlds,
            total_disasters: report.total_disasters,
            carrington_events: report.carrington_events,
            survived: report.survived,
        });
    }

    // --- Per-seed table ---
    println!();
    println!("=== 777-YEAR DISASTER VALIDATION (5 seeds) ===");
    println!();
    println!(
        "{:<7}| {:<8}| {:<7}| {:<7}| {:<6}| {:<12}| {:<12}| {}",
        "Seed", "Pop", "CVS", "Phi", "Wrlds", "Disasters", "Carrington", "Survived"
    );
    println!("{}", "-".repeat(80));

    for r in &results {
        println!(
            "{:<7}| {:<8}| {:<7.3}| {:<7.3}| {:<6}| {:<12}| {:<12}| {}",
            r.seed,
            r.population,
            r.cvs,
            r.phi,
            r.worlds,
            r.total_disasters,
            r.carrington_events,
            if r.survived { "YES" } else { "NO" },
        );
    }

    // --- Summary statistics ---
    let n = results.len() as f64;
    let survived_count = results.iter().filter(|r| r.survived).count();

    let mean_pop = results.iter().map(|r| r.population as f64).sum::<f64>() / n;
    let std_pop = (results
        .iter()
        .map(|r| (r.population as f64 - mean_pop).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_cvs = results.iter().map(|r| r.cvs).sum::<f64>() / n;
    let std_cvs = (results
        .iter()
        .map(|r| (r.cvs - mean_cvs).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_phi = results.iter().map(|r| r.phi).sum::<f64>() / n;
    let std_phi = (results
        .iter()
        .map(|r| (r.phi - mean_phi).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_worlds = results.iter().map(|r| r.worlds as f64).sum::<f64>() / n;
    let std_worlds = (results
        .iter()
        .map(|r| (r.worlds as f64 - mean_worlds).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_disasters = results
        .iter()
        .map(|r| r.total_disasters as f64)
        .sum::<f64>()
        / n;
    let std_disasters = (results
        .iter()
        .map(|r| (r.total_disasters as f64 - mean_disasters).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_carrington = results
        .iter()
        .map(|r| r.carrington_events as f64)
        .sum::<f64>()
        / n;
    let std_carrington = (results
        .iter()
        .map(|r| (r.carrington_events as f64 - mean_carrington).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    println!();
    println!("SUMMARY:");
    println!(
        "  Survival rate:      {}/{} ({:.0}%)",
        survived_count,
        results.len(),
        survived_count as f64 / n * 100.0,
    );
    println!("  Mean population:    {:.0} +/- {:.0}", mean_pop, std_pop);
    println!("  Mean CVS:           {:.3} +/- {:.3}", mean_cvs, std_cvs);
    println!("  Mean Phi:           {:.3} +/- {:.3}", mean_phi, std_phi);
    println!(
        "  Mean worlds:        {:.1} +/- {:.1}",
        mean_worlds, std_worlds
    );
    println!(
        "  Mean disasters:     {:.1} +/- {:.1}",
        mean_disasters, std_disasters
    );
    println!(
        "  Mean Carrington:    {:.1} +/- {:.1}",
        mean_carrington, std_carrington
    );
}
