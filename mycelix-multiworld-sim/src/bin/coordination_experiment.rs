// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 1: Coordination Science Understanding vs General Education
//!
//! Three arms × 10 seeds = 30 simulations:
//!   A) Baseline: cu=0, edu_boost=0
//!   B) Education: cu=0, edu_boost=0.2
//!   C) Coordination: cu=0.5, edu_boost=0
//!
//! Measures: CVS, Phi, population, civil wars, faction count

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

struct ArmConfig {
    name: &'static str,
    coordination_understanding: f64,
    education_boost: f64,
}

struct RunResult {
    seed: u64,
    arm: String,
    cvs: f64,
    population: usize,
    phi: f64,
    civil_wars: usize,
    survived: bool,
}

fn run_arm(arm: &ArmConfig, seed: u64) -> RunResult {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.policy.coordination_understanding_initial = arm.coordination_understanding;
    config.policy.education_boost = arm.education_boost;

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    RunResult {
        seed,
        arm: arm.name.to_string(),
        cvs: report.final_cvs,
        population: report.final_population,
        phi: report.final_collective_phi,
        civil_wars: report.civil_wars,
        survived: report.survived,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let var = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

fn main() {
    let arms = [
        ArmConfig {
            name: "A_Baseline",
            coordination_understanding: 0.0,
            education_boost: 0.0,
        },
        ArmConfig {
            name: "B_Education",
            coordination_understanding: 0.0,
            education_boost: 0.2,
        },
        ArmConfig {
            name: "C_Coordination",
            coordination_understanding: 0.5,
            education_boost: 0.0,
        },
    ];

    let seeds: Vec<u64> = (42..52).collect(); // 10 seeds
    let n_seeds = seeds.len();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 1: Coordination Science vs General Education    ║");
    println!(
        "║  3 arms × {} seeds = {} simulations (150 years each)        ║",
        n_seeds,
        arms.len() * n_seeds
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut all_results: Vec<RunResult> = Vec::new();

    for arm in &arms {
        println!("── Running {} ──", arm.name);
        for &seed in &seeds {
            let result = run_arm(arm, seed);
            print!(
                "  seed={}: CVS={:.4}, pop={}, phi={:.4}, wars={}, {}",
                seed,
                result.cvs,
                result.population,
                result.phi,
                result.civil_wars,
                if result.survived { "OK" } else { "COLLAPSED" }
            );
            println!();
            all_results.push(result);
        }
        println!();
    }

    // ── Summary statistics ──
    println!("══════════════════════════════════════════════════════════════");
    println!("  RESULTS SUMMARY");
    println!("══════════════════════════════════════════════════════════════\n");

    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Arm", "CVS", "±", "Phi", "Pop", "Wars"
    );
    println!("{}", "-".repeat(72));

    for arm in &arms {
        let arm_results: Vec<&RunResult> =
            all_results.iter().filter(|r| r.arm == arm.name).collect();

        let cvs_vals: Vec<f64> = arm_results.iter().map(|r| r.cvs).collect();
        let phi_vals: Vec<f64> = arm_results.iter().map(|r| r.phi).collect();
        let pop_vals: Vec<f64> = arm_results.iter().map(|r| r.population as f64).collect();
        let war_vals: Vec<f64> = arm_results.iter().map(|r| r.civil_wars as f64).collect();

        println!(
            "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.0} {:>10.1}",
            arm.name,
            mean(&cvs_vals),
            std_dev(&cvs_vals),
            mean(&phi_vals),
            mean(&pop_vals),
            mean(&war_vals),
        );
    }

    // ── Effect sizes ──
    let baseline_cvs: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "A_Baseline")
        .map(|r| r.cvs)
        .collect();
    let education_cvs: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "B_Education")
        .map(|r| r.cvs)
        .collect();
    let coordination_cvs: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "C_Coordination")
        .map(|r| r.cvs)
        .collect();

    let delta_edu = mean(&education_cvs) - mean(&baseline_cvs);
    let delta_coord = mean(&coordination_cvs) - mean(&baseline_cvs);
    let delta_coord_vs_edu = mean(&coordination_cvs) - mean(&education_cvs);

    println!("\n── Effect Sizes (CVS) ──");
    println!("  Education vs Baseline:     {:+.4}", delta_edu);
    println!("  Coordination vs Baseline:  {:+.4}", delta_coord);
    println!("  Coordination vs Education: {:+.4}", delta_coord_vs_edu);

    println!("\n── Verdict ──");
    if delta_coord > delta_edu + 0.005 {
        println!("  COORDINATION SCIENCE OUTPERFORMS GENERAL EDUCATION");
        println!("  Effect is specific to coordination understanding, not general intelligence.");
    } else if delta_coord > delta_edu - 0.005 {
        println!("  COORDINATION AND EDUCATION ARE COMPARABLE");
        println!("  Coordination understanding may be one component of general education.");
    } else {
        println!("  GENERAL EDUCATION OUTPERFORMS COORDINATION SCIENCE");
        println!("  Coordination understanding alone is less effective than broad education.");
    }

    // Phi comparison
    let baseline_phi: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "A_Baseline")
        .map(|r| r.phi)
        .collect();
    let coordination_phi: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "C_Coordination")
        .map(|r| r.phi)
        .collect();
    let phi_delta = mean(&coordination_phi) - mean(&baseline_phi);
    println!(
        "\n  Phi effect (Coordination vs Baseline): {:+.4}",
        phi_delta
    );

    // Conflict reduction
    let baseline_wars: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "A_Baseline")
        .map(|r| r.civil_wars as f64)
        .collect();
    let coordination_wars: Vec<f64> = all_results
        .iter()
        .filter(|r| r.arm == "C_Coordination")
        .map(|r| r.civil_wars as f64)
        .collect();
    let war_reduction = mean(&baseline_wars) - mean(&coordination_wars);
    println!(
        "  Conflict reduction (wars):             {:+.1}",
        war_reduction
    );
}
