// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-seed analysis: run the 150-year civilization simulation across 10 seeds
//! and report aggregate statistics.

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

const SEEDS: [u64; 10] = [42, 123, 789, 1337, 2718, 3141, 4242, 5555, 7777, 9999];

struct RunResult {
    seed: u64,
    population: usize,
    off_earth_pop: usize,
    worlds: usize,
    cvs: f64,
    phi: f64,
    love: f64,
    genetics: f64,
    tech: f64,
    survived: bool,
    first_birth_year: Option<f64>,
    first_trade_year: Option<f64>,
    checkpoints_passed: usize,
    checkpoints_total: usize,
}

fn main() {
    eprintln!("=== MULTI-SEED ANALYSIS (10 runs x 1800 ticks) ===\n");

    let mut results: Vec<RunResult> = Vec::with_capacity(SEEDS.len());

    for (i, &seed) in SEEDS.iter().enumerate() {
        eprint!(
            "  Running seed {:>5} ({:>2}/{}) ...",
            seed,
            i + 1,
            SEEDS.len()
        );

        let mut config = SimulationConfig::default_150_year();
        config.seed = seed;

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        let off_earth_pop: usize = sim
            .worlds
            .iter()
            .filter(|w| w.location != "Earth")
            .map(|w| w.population())
            .sum();

        // Tech level from knowledge engine (normalized)
        let raw_tech: f64 = if sim.worlds.is_empty() {
            1.0
        } else {
            sim.worlds
                .iter()
                .map(|w| w.knowledge.mean_tech_level())
                .sum::<f64>()
                / sim.worlds.len() as f64
        };
        let tech_norm = ((raw_tech - 1.0) / 9.0).clamp(0.0, 1.0);

        results.push(RunResult {
            seed,
            population: report.final_population,
            off_earth_pop,
            worlds: report.final_worlds,
            cvs: report.final_cvs,
            phi: report.final_collective_phi,
            love: report.final_love_coherence,
            genetics: report.final_genetic_diversity,
            tech: tech_norm,
            survived: report.survived,
            first_birth_year: report.first_birth_tick.map(|t| t as f64 / 12.0),
            first_trade_year: report.first_trade_tick.map(|t| t as f64 / 12.0),
            checkpoints_passed: report.checkpoints_passed,
            checkpoints_total: report.checkpoints_passed + report.checkpoints_failed,
        });

        eprintln!(
            " pop={:>6} (off-Earth {:>5}), CVS={:.3}, {} {}",
            report.final_population,
            off_earth_pop,
            report.final_cvs,
            if report.survived {
                "SURVIVED"
            } else {
                "COLLAPSED"
            },
            format!(
                "[{}/{}]",
                report.checkpoints_passed,
                report.checkpoints_passed + report.checkpoints_failed
            ),
        );
    }

    // Print results table
    println!();
    println!("=== MULTI-SEED ANALYSIS (10 runs) ===");
    println!();
    println!(
        "{:<7}| {:<8}| {:<8}| {:<6}| {:<7}| {:<7}| {:<7}| {:<10}| {:<7}| {:<10}| {:<10}| {}",
        "Seed",
        "Pop",
        "OffErth",
        "Wrlds",
        "CVS",
        "Phi",
        "Love",
        "Genetics",
        "Tech",
        "Birth Yr",
        "Trade Yr",
        "Survived"
    );
    println!("{}", "-".repeat(115));

    for r in &results {
        let birth_str = r
            .first_birth_year
            .map(|y| format!("{:.1}", y))
            .unwrap_or_else(|| "Never".into());
        let trade_str = r
            .first_trade_year
            .map(|y| format!("{:.1}", y))
            .unwrap_or_else(|| "Never".into());
        println!(
            "{:<7}| {:<8}| {:<8}| {:<6}| {:<7.3}| {:<7.3}| {:<7.3}| {:<10.4}| {:<7.3}| {:<10}| {:<10}| {}",
            r.seed,
            r.population,
            r.off_earth_pop,
            r.worlds,
            r.cvs,
            r.phi,
            r.love,
            r.genetics,
            r.tech,
            birth_str,
            trade_str,
            if r.survived { "YES" } else { "NO" },
        );
    }

    // Summary statistics
    let n = results.len() as f64;
    let survived_count = results.iter().filter(|r| r.survived).count();

    let mean_cvs = results.iter().map(|r| r.cvs).sum::<f64>() / n;
    let std_cvs = (results
        .iter()
        .map(|r| (r.cvs - mean_cvs).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_pop = results.iter().map(|r| r.population as f64).sum::<f64>() / n;
    let std_pop = (results
        .iter()
        .map(|r| (r.population as f64 - mean_pop).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_off = results.iter().map(|r| r.off_earth_pop as f64).sum::<f64>() / n;
    let std_off = (results
        .iter()
        .map(|r| (r.off_earth_pop as f64 - mean_off).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_love = results.iter().map(|r| r.love).sum::<f64>() / n;
    let std_love = (results
        .iter()
        .map(|r| (r.love - mean_love).powi(2))
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

    let mean_genetics = results.iter().map(|r| r.genetics).sum::<f64>() / n;
    let std_genetics = (results
        .iter()
        .map(|r| (r.genetics - mean_genetics).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_tech = results.iter().map(|r| r.tech).sum::<f64>() / n;
    let std_tech = (results
        .iter()
        .map(|r| (r.tech - mean_tech).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let mean_checkpoints: f64 = results
        .iter()
        .map(|r| r.checkpoints_passed as f64 / r.checkpoints_total.max(1) as f64)
        .sum::<f64>()
        / n;

    println!();
    println!("SUMMARY:");
    println!(
        "  Survival rate:          {}/{} ({:.0}%)",
        survived_count,
        results.len(),
        survived_count as f64 / n * 100.0,
    );
    println!(
        "  Mean CVS:               {:.3} +/- {:.3}",
        mean_cvs, std_cvs
    );
    println!(
        "  Mean final population:  {:.0} +/- {:.0}",
        mean_pop, std_pop,
    );
    println!(
        "  Mean off-Earth pop:     {:.0} +/- {:.0}",
        mean_off, std_off,
    );
    println!(
        "  Mean love coherence:    {:.3} +/- {:.3}",
        mean_love, std_love
    );
    println!(
        "  Mean collective phi:    {:.3} +/- {:.3}",
        mean_phi, std_phi
    );
    println!(
        "  Mean genetic diversity: {:.4} +/- {:.4}",
        mean_genetics, std_genetics,
    );
    println!(
        "  Mean tech level:        {:.3} +/- {:.3}",
        mean_tech, std_tech
    );
    println!("  Mean checkpoint rate:   {:.1}%", mean_checkpoints * 100.0,);
}
