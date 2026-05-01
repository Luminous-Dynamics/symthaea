// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario comparison: test how different care/bonding/isolation parameters
//! affect civilization viability across 150 years.
//!
//! Runs 6 scenarios with the same seed and compares outcomes.

use mycelix_multiworld_sim::{
    config::{PolicyConfig, SimulationConfig},
    MultiWorldSimulator,
};

struct ScenarioResult {
    name: String,
    cvs: f64,
    population: usize,
    phi: f64,
    love_coherence: f64,
    allostatic_load: f64,
    engagement: f64,
    thrill_incidents: usize,
    biohack_incidents: usize,
    genetics: f64,
    harmony_scores: [f64; 8],
    checkpoints_passed: usize,
    checkpoints_total: usize,
    survived: bool,
}

fn run_scenario(name: &str, seed: u64, ticks: u32, policy: PolicyConfig) -> ScenarioResult {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.total_ticks = ticks;
    config.policy = policy;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    ScenarioResult {
        name: name.into(),
        cvs: report.final_cvs,
        population: report.final_population,
        phi: report.final_collective_phi,
        love_coherence: report.final_love_coherence,
        allostatic_load: report.final_mean_allostatic_load,
        engagement: report.final_mean_engagement,
        thrill_incidents: report.thrill_incidents_total,
        biohack_incidents: report.biohack_incidents_total,
        genetics: report.final_genetic_diversity,
        harmony_scores: report.final_harmony_scores,
        checkpoints_passed: report.checkpoints_passed,
        checkpoints_total: report.checkpoints_passed + report.checkpoints_failed,
        survived: report.survived,
    }
}

fn main() {
    let seed = 42u64;
    let ticks = 1800u32;

    eprintln!(
        "=== POLICY VARIANT COMPARISON (seed: {}, {} ticks) ===\n",
        seed, ticks
    );

    // Define policy variants that test the core thesis
    let scenarios: Vec<(&str, PolicyConfig)> = vec![
        ("baseline", PolicyConfig::default()),
        (
            "no-care",
            PolicyConfig {
                care_effectiveness: 0.0,
                ..PolicyConfig::default()
            },
        ),
        (
            "no-bonds",
            PolicyConfig {
                pair_bond_rate: 0.0,
                ..PolicyConfig::default()
            },
        ),
        (
            "no-pharma",
            PolicyConfig {
                pharma_boost: 0.0,
                ..PolicyConfig::default()
            },
        ),
        (
            "isolationist",
            PolicyConfig {
                migration_enabled: false,
                deep_space_isolation_mult: 2.5,
                ..PolicyConfig::default()
            },
        ),
        (
            "max-care",
            PolicyConfig {
                care_effectiveness: 3.0,
                pair_bond_rate: 0.05,
                pharma_boost: 0.5,
                ..PolicyConfig::default()
            },
        ),
        (
            "1602-punitive",
            PolicyConfig {
                // The 1602 model: no care economy, no bonding support,
                // no astropharmacy, isolationist, high punishment, no education guild
                care_effectiveness: 0.0,
                pair_bond_rate: 0.005,
                pharma_boost: 0.0,
                migration_enabled: false,
                deep_space_isolation_mult: 2.0,
                migration_max_per_cycle: 0,
                education_enabled: false,
                trust_weighted_governance: false,
                faction_enabled: true,
                amendment_enabled: true,
                outer_system_fission_enabled: true,
                speciation_friction_enabled: true,
                hostile_guardian: false,
                disasters_enabled: true,
                hybrid_earth: false,
                ..PolicyConfig::default()
            },
        ),
    ];

    let mut results: Vec<ScenarioResult> = Vec::new();

    for (name, policy) in &scenarios {
        eprint!("  {:<15} ... ", name);
        let r = run_scenario(name, seed, ticks, policy.clone());
        eprintln!(
            "pop={:>5}, CVS={:.3}, load={:.3}, eng={:.3}, love={:.3}, phi={:.3} {}",
            r.population,
            r.cvs,
            r.allostatic_load,
            r.engagement,
            r.love_coherence,
            r.phi,
            if r.survived { "SURVIVED" } else { "COLLAPSED" }
        );
        results.push(r);
    }

    // Print detailed comparison
    println!("\n{}", "=".repeat(120));
    println!("SCENARIO COMPARISON");
    println!("{}", "=".repeat(120));
    println!();

    // Header
    println!(
        "{:<12} | {:>6} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} | {:>6} | {:>6} | {:>5} | {:>5} | {}",
        "Scenario", "Pop", "CVS", "Phi", "Love", "Load", "Eng", "Thrill", "Biohk", "Genet", "CP", "Alive"
    );
    println!("{}", "-".repeat(120));

    for r in &results {
        println!(
            "{:<12} | {:>6} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>6} | {:>6} | {:>5.3} | {:>2}/{:<2} | {}",
            r.name, r.population, r.cvs, r.phi, r.love_coherence,
            r.allostatic_load, r.engagement,
            r.thrill_incidents, r.biohack_incidents,
            r.genetics,
            r.checkpoints_passed, r.checkpoints_total,
            if r.survived { "YES" } else { "NO" }
        );
    }

    // Harmony detail
    println!("\nHARMONY SCORES BY SCENARIO:");
    let harmony_names = [
        "Resonant Coherence",
        "Pan-Sentient Flourishing",
        "Integral Wisdom",
        "Infinite Play",
        "Sacred Reciprocity",
        "Sacred Stillness",
        "Evolutionary Progression",
        "Sacred Stillness",
    ];
    println!(
        "{:<12} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Scenario", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"
    );
    println!("{}", "-".repeat(100));
    for r in &results {
        println!(
            "{:<12} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3}",
            r.name,
            r.harmony_scores[0], r.harmony_scores[1], r.harmony_scores[2], r.harmony_scores[3],
            r.harmony_scores[4], r.harmony_scores[5], r.harmony_scores[6], r.harmony_scores[7],
        );
    }

    // Summary statistics
    let n = results.len() as f64;
    let survived = results.iter().filter(|r| r.survived).count();
    let mean =
        |f: &dyn Fn(&ScenarioResult) -> f64| -> f64 { results.iter().map(f).sum::<f64>() / n };
    let std = |f: &dyn Fn(&ScenarioResult) -> f64| -> f64 {
        let m = mean(f);
        (results.iter().map(|r| (f(r) - m).powi(2)).sum::<f64>() / n).sqrt()
    };

    println!("\nAGGREGATES ({} runs):", results.len());
    println!("  Survival rate:     {}/{}", survived, results.len());
    println!(
        "  CVS:               {:.3} +/- {:.3}",
        mean(&|r| r.cvs),
        std(&|r| r.cvs)
    );
    println!(
        "  Population:        {:.0} +/- {:.0}",
        mean(&|r| r.population as f64),
        std(&|r| r.population as f64)
    );
    println!(
        "  Collective Phi:    {:.3} +/- {:.3}",
        mean(&|r| r.phi),
        std(&|r| r.phi)
    );
    println!(
        "  Love Coherence:    {:.3} +/- {:.3}",
        mean(&|r| r.love_coherence),
        std(&|r| r.love_coherence)
    );
    println!(
        "  Allostatic Load:   {:.3} +/- {:.3}",
        mean(&|r| r.allostatic_load),
        std(&|r| r.allostatic_load)
    );
    println!(
        "  Engagement:        {:.3} +/- {:.3}",
        mean(&|r| r.engagement),
        std(&|r| r.engagement)
    );
    println!(
        "  Genetic Diversity: {:.3} +/- {:.3}",
        mean(&|r| r.genetics),
        std(&|r| r.genetics)
    );
    println!(
        "  Thrill Incidents:  {:.0} +/- {:.0}",
        mean(&|r| r.thrill_incidents as f64),
        std(&|r| r.thrill_incidents as f64)
    );

    // Key diagnostic
    println!("\nDIAGNOSTICS:");
    if mean(&|r| r.allostatic_load) < 0.01 {
        println!("  WARNING: Allostatic load near zero — care workers over-compensate. CARE_LOAD_REDUCTION may be too high.");
    }
    if mean(&|r| r.engagement) > 0.95 {
        println!("  WARNING: Engagement near 1.0 — escapism loop never fires. Load threshold or social threshold may be too strict.");
    }
    if mean(&|r| r.genetics) < 0.01 {
        println!("  WARNING: Genetic diversity collapsed — small off-world populations inbreed. Need immigration or larger founding groups.");
    }
    if mean(&|r| r.love_coherence) < 0.1 {
        println!("  WARNING: Love coherence very low — harmony scores are dominated by tension/low diversity.");
    }
    let mean_thrill = mean(&|r| r.thrill_incidents as f64);
    if mean_thrill > 1000.0 {
        println!(
            "  WARNING: {} mean thrill incidents — condition may be too loose.",
            mean_thrill as usize
        );
    }
}
