// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hostile Guardian adversarial scenario.
//!
//! Injects a Guardian that vetoes EVERY proposal regardless of community
//! support. Proves the 80% override mechanism actively fires under attack.
//!
//! Compares:
//! - Baseline (no hostile guardian) — should have 0 vetoes (deterrence)
//! - Hostile (hostile guardian injected) — should have vetoes + overrides
//!
//! Key metric: Does civilization survive DESPITE a hostile Guardian?

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn run_scenario(name: &str, hostile: bool, seed: u64) -> (f64, bool, u32, u32, u32) {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.policy.hostile_guardian = hostile;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    let mut vetoes = 0u32;
    let mut overrides = 0u32;
    let mut deterred = 0u32;
    for world in &sim.worlds {
        vetoes += world.governance.veto_count;
        overrides += world.governance.veto_override_count;
        deterred += world.governance.veto_deterred_count;
    }

    println!(
        "  [{:<12}] seed={:<5} CVS={:.3}  survived={}  vetoes={:<3} overrides={:<3} deterred={:<3}",
        name,
        seed,
        report.final_cvs,
        if report.survived { "YES" } else { "NO " },
        vetoes,
        overrides,
        deterred,
    );

    (
        report.final_cvs,
        report.survived,
        vetoes,
        overrides,
        deterred,
    )
}

fn main() {
    println!("=== HOSTILE GUARDIAN ADVERSARIAL SCENARIO ===\n");
    println!("Testing: Does the 80% veto override protect against a malicious Guardian?\n");

    let seeds = [42, 123, 789];

    println!("--- BASELINE (no hostile guardian) ---");
    let mut baseline_cvs = Vec::new();
    let mut baseline_all_survived = true;
    for &seed in &seeds {
        let (cvs, survived, _, _, _) = run_scenario("baseline", false, seed);
        baseline_cvs.push(cvs);
        if !survived {
            baseline_all_survived = false;
        }
    }

    println!("\n--- HOSTILE GUARDIAN (vetoes every proposal) ---");
    let mut hostile_cvs = Vec::new();
    let mut hostile_all_survived = true;
    let mut total_vetoes = 0u32;
    let mut total_overrides = 0u32;
    for &seed in &seeds {
        let (cvs, survived, vetoes, overrides, _) = run_scenario("hostile", true, seed);
        hostile_cvs.push(cvs);
        total_vetoes += vetoes;
        total_overrides += overrides;
        if !survived {
            hostile_all_survived = false;
        }
    }

    // Analysis
    println!("\n{}", "=".repeat(60));
    println!("=== ADVERSARIAL ANALYSIS ===\n");

    let baseline_mean = baseline_cvs.iter().sum::<f64>() / baseline_cvs.len() as f64;
    let hostile_mean = hostile_cvs.iter().sum::<f64>() / hostile_cvs.len() as f64;
    let cvs_delta = hostile_mean - baseline_mean;

    println!("Baseline mean CVS:  {:.3}", baseline_mean);
    println!("Hostile mean CVS:   {:.3}", hostile_mean);
    println!(
        "CVS delta:          {:.3} ({:.1}%)",
        cvs_delta,
        cvs_delta / baseline_mean * 100.0
    );
    println!();
    println!(
        "Baseline survival:  {}/{}",
        if baseline_all_survived {
            seeds.len()
        } else {
            0
        },
        seeds.len()
    );
    println!(
        "Hostile survival:   {}/{}",
        if hostile_all_survived { seeds.len() } else { 0 },
        seeds.len()
    );
    println!();
    println!("Total hostile vetoes:    {}", total_vetoes);
    println!("Total vetoes overridden: {}", total_overrides);

    if total_vetoes > 0 {
        let override_rate = total_overrides as f64 / total_vetoes as f64 * 100.0;
        println!("Override success rate:   {:.1}%", override_rate);
    }

    println!();
    if hostile_all_survived && total_overrides > 0 {
        println!("RESULT: Override mechanism PROVEN — civilization survives hostile Guardian");
        println!(
            "        The 80% supermajority actively reversed {} vetoes",
            total_overrides
        );
    } else if hostile_all_survived && total_vetoes == 0 {
        println!("RESULT: Rate limiting prevented hostile Guardian from exercising any vetoes");
        println!("        (7-tick cooldown limits damage to 1 veto per 7 months)");
    } else if !hostile_all_survived {
        println!("WARNING: Hostile Guardian caused civilization collapse!");
        println!("         Anti-tyranny measures insufficient against sustained attack");
        std::process::exit(1);
    }
}
