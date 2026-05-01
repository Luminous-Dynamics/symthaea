// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stress tests for the zero-wars finding:
//! 1. High-power replication (50 seeds, Cell A vs Cell D)
//! 2. Harsh conditions (all disasters + Turchin pressure)
//! 3. Adversarial robustness (5 red team agents per world)

use mycelix_multiworld_sim::{
    config::SimulationConfig, red_team::AdversarialStrategy, MultiWorldSimulator,
};

struct ArmResult {
    cvs: f64,
    phi: f64,
    population: usize,
    civil_wars: usize,
    survived: bool,
}

fn run_cell_d(seed: u64, harsh: bool) -> ArmResult {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.policy.coordination_understanding_initial = 0.5;
    config.policy.consciousness_boost = 0.4;
    if harsh {
        config.policy.trade_openness = 0.3; // Restricted trade
        config.policy.defense_spending = 0.15; // High military
    }
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    ArmResult {
        cvs: report.final_cvs,
        phi: report.final_collective_phi,
        population: report.final_population,
        civil_wars: report.civil_wars,
        survived: report.survived,
    }
}

fn run_baseline(seed: u64, harsh: bool) -> ArmResult {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    config.policy.coordination_understanding_initial = 0.0;
    config.policy.consciousness_boost = 0.0;
    if harsh {
        config.policy.trade_openness = 0.3;
        config.policy.defense_spending = 0.15;
    }
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    ArmResult {
        cvs: report.final_cvs,
        phi: report.final_collective_phi,
        population: report.final_population,
        civil_wars: report.civil_wars,
        survived: report.survived,
    }
}

fn run_adversarial(seed: u64, with_coordination: bool) -> ArmResult {
    let mut config = SimulationConfig::default_150_year();
    config.seed = seed;
    if with_coordination {
        config.policy.coordination_understanding_initial = 0.5;
        config.policy.consciousness_boost = 0.4;
    }
    let mut sim = MultiWorldSimulator::new(config);
    sim.initialize();
    sim.inject_adversaries(AdversarialStrategy::ProfileMaximizer, 5);
    let report = sim.run();
    ArmResult {
        cvs: report.final_cvs,
        phi: report.final_collective_phi,
        population: report.final_population,
        civil_wars: report.civil_wars,
        survived: report.survived,
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn print_summary(name: &str, results: &[ArmResult]) {
    let cvs: Vec<f64> = results.iter().map(|r| r.cvs).collect();
    let wars: Vec<f64> = results.iter().map(|r| r.civil_wars as f64).collect();
    let pop: Vec<f64> = results.iter().map(|r| r.population as f64).collect();
    let phi: Vec<f64> = results.iter().map(|r| r.phi).collect();
    let zero_war_runs = results.iter().filter(|r| r.civil_wars == 0).count();
    let collapsed = results.iter().filter(|r| !r.survived).count();
    println!(
        "  {:<30} CVS={:.4} Phi={:.4} Pop={:>7.0} Wars={:.2} ZeroWar={}/{} Collapsed={}",
        name,
        mean(&cvs),
        mean(&phi),
        mean(&pop),
        mean(&wars),
        zero_war_runs,
        results.len(),
        collapsed
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Stress Tests: Is the Zero-Wars Finding Robust?            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Test 1: High-power replication (50 seeds) ──
    println!("── Test 1: High-Power Replication (50 seeds) ──");
    let seeds_50: Vec<u64> = (42..92).collect();

    let mut baseline_50 = Vec::new();
    let mut cell_d_50 = Vec::new();

    for (i, &seed) in seeds_50.iter().enumerate() {
        let b = run_baseline(seed, false);
        let d = run_cell_d(seed, false);
        if (i + 1) % 10 == 0 {
            println!("  ...completed {}/{} seed pairs", i + 1, seeds_50.len());
        }
        baseline_50.push(b);
        cell_d_50.push(d);
    }

    print_summary("Baseline (50 seeds)", &baseline_50);
    print_summary("Cell D: C+CU (50 seeds)", &cell_d_50);

    let baseline_war_rate = mean(
        &baseline_50
            .iter()
            .map(|r| r.civil_wars as f64)
            .collect::<Vec<_>>(),
    );
    let cell_d_war_rate = mean(
        &cell_d_50
            .iter()
            .map(|r| r.civil_wars as f64)
            .collect::<Vec<_>>(),
    );
    let baseline_zero = baseline_50.iter().filter(|r| r.civil_wars == 0).count();
    let cell_d_zero = cell_d_50.iter().filter(|r| r.civil_wars == 0).count();

    println!(
        "\n  War rate: Baseline {:.2}/run vs Cell D {:.2}/run",
        baseline_war_rate, cell_d_war_rate
    );
    println!(
        "  Zero-war runs: Baseline {}/50 vs Cell D {}/50",
        baseline_zero, cell_d_zero
    );

    if cell_d_war_rate < baseline_war_rate * 0.5 {
        println!("  CONFIRMED: Cell D war rate < 50% of baseline");
    } else {
        println!("  NOT CONFIRMED: Cell D war rate not significantly lower");
    }

    // ── Test 2: Harsh conditions ──
    println!("\n── Test 2: Harsh Conditions (restricted trade + high military) ──");
    let seeds_20: Vec<u64> = (42..62).collect();

    let mut baseline_harsh = Vec::new();
    let mut cell_d_harsh = Vec::new();

    for &seed in &seeds_20 {
        baseline_harsh.push(run_baseline(seed, true));
        cell_d_harsh.push(run_cell_d(seed, true));
    }

    print_summary("Baseline (harsh, 20 seeds)", &baseline_harsh);
    print_summary("Cell D (harsh, 20 seeds)", &cell_d_harsh);

    let harsh_baseline_wars = mean(
        &baseline_harsh
            .iter()
            .map(|r| r.civil_wars as f64)
            .collect::<Vec<_>>(),
    );
    let harsh_cell_d_wars = mean(
        &cell_d_harsh
            .iter()
            .map(|r| r.civil_wars as f64)
            .collect::<Vec<_>>(),
    );
    println!(
        "\n  Harsh war rate: Baseline {:.2}/run vs Cell D {:.2}/run",
        harsh_baseline_wars, harsh_cell_d_wars
    );

    if harsh_cell_d_wars < harsh_baseline_wars * 0.5 {
        println!("  ROBUST: Zero-wars finding SURVIVES harsh conditions");
    } else {
        println!("  FRAGILE: Zero-wars finding does NOT survive harsh conditions");
    }

    // ── Test 3: Adversarial robustness ──
    println!("\n── Test 3: Adversarial Robustness (5 red team agents/world) ──");

    let mut baseline_adv = Vec::new();
    let mut cell_d_adv = Vec::new();

    for &seed in &seeds_20 {
        baseline_adv.push(run_adversarial(seed, false));
        cell_d_adv.push(run_adversarial(seed, true));
    }

    print_summary("Baseline + adversaries (20)", &baseline_adv);
    print_summary("Cell D + adversaries (20)", &cell_d_adv);

    let adv_baseline_wars = mean(
        &baseline_adv
            .iter()
            .map(|r| r.civil_wars as f64)
            .collect::<Vec<_>>(),
    );
    let adv_cell_d_wars = mean(
        &cell_d_adv
            .iter()
            .map(|r| r.civil_wars as f64)
            .collect::<Vec<_>>(),
    );
    let adv_baseline_cvs = mean(&baseline_adv.iter().map(|r| r.cvs).collect::<Vec<_>>());
    let adv_cell_d_cvs = mean(&cell_d_adv.iter().map(|r| r.cvs).collect::<Vec<_>>());
    println!(
        "\n  Adversarial wars: Baseline {:.2} vs Cell D {:.2}",
        adv_baseline_wars, adv_cell_d_wars
    );
    println!(
        "  Adversarial CVS:  Baseline {:.4} vs Cell D {:.4} (Δ={:+.4})",
        adv_baseline_cvs,
        adv_cell_d_cvs,
        adv_cell_d_cvs - adv_baseline_cvs
    );

    // ── Final verdict ──
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  FINAL VERDICT");
    println!("══════════════════════════════════════════════════════════════");

    let replication_holds = cell_d_war_rate < baseline_war_rate * 0.5;
    let harsh_holds = harsh_cell_d_wars < harsh_baseline_wars * 0.5;
    let adversarial_holds = adv_cell_d_wars < adv_baseline_wars * 0.75;

    let tests_passed = [replication_holds, harsh_holds, adversarial_holds]
        .iter()
        .filter(|&&x| x)
        .count();

    println!(
        "  Replication (50 seeds):  {}",
        if replication_holds {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "  Harsh conditions:       {}",
        if harsh_holds { "PASSED" } else { "FAILED" }
    );
    println!(
        "  Adversarial robustness: {}",
        if adversarial_holds {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!("  Overall: {}/3 stress tests passed", tests_passed);

    if tests_passed == 3 {
        println!("\n  THE ZERO-WARS FINDING IS ROBUST.");
    } else if tests_passed >= 2 {
        println!("\n  The finding is PARTIALLY ROBUST — holds under most but not all conditions.");
    } else {
        println!("\n  The finding is FRAGILE — does not survive stress testing.");
    }
}
