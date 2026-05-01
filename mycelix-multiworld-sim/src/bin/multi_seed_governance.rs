// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-seed governance comparison: statistical test of consciousness gating.
//! Runs N seeds × 2 models (consciousness-gated vs equal weight) and reports
//! mean, std, confidence interval, and p-value approximation.

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    let n_seeds = 10;
    let years = 150u32;

    println!("=== MULTI-SEED GOVERNANCE COMPARISON ===");
    println!("Seeds: {n_seeds}, Duration: {years} years per run");
    println!("Models: ConsciousnessGated vs EqualWeight\n");

    let mut cg_cvs = Vec::new();
    let mut ew_cvs = Vec::new();

    for seed in 0..n_seeds {
        let actual_seed = seed * 137 + 42; // spread seeds

        // Consciousness-Gated
        let mut config = SimulationConfig::default_150_year();
        config.seed = actual_seed;
        config.total_ticks = years * 12;
        config.policy.trust_weighted_governance = true;
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        cg_cvs.push(report.final_cvs);

        // Equal Weight
        let mut config = SimulationConfig::default_150_year();
        config.seed = actual_seed;
        config.total_ticks = years * 12;
        config.policy.trust_weighted_governance = false;
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        ew_cvs.push(report.final_cvs);

        eprint!("."); // progress
    }
    eprintln!();

    // Statistics
    let cg_mean = mean(&cg_cvs);
    let ew_mean = mean(&ew_cvs);
    let cg_std = std_dev(&cg_cvs);
    let ew_std = std_dev(&ew_cvs);

    let delta_mean = cg_mean - ew_mean;
    let delta_pct = if ew_mean > 0.0 {
        delta_mean / ew_mean * 100.0
    } else {
        0.0
    };

    // Paired differences for t-test
    let diffs: Vec<f64> = cg_cvs
        .iter()
        .zip(ew_cvs.iter())
        .map(|(a, b)| a - b)
        .collect();
    let diff_mean = mean(&diffs);
    let diff_std = std_dev(&diffs);
    let t_stat = if diff_std > 0.0 {
        diff_mean / (diff_std / (n_seeds as f64).sqrt())
    } else {
        0.0
    };
    // Rough p-value approximation (two-tailed, df=n-1)
    let p_approx = 2.0 * t_to_p(t_stat.abs(), n_seeds - 1);

    println!("\n=== RESULTS ===\n");
    println!("{:<28} {:>10} {:>10}", "Model", "Mean CVS", "Std Dev");
    println!("{}", "-".repeat(50));
    println!(
        "{:<28} {:>10.4} {:>10.4}",
        "Consciousness-Gated", cg_mean, cg_std
    );
    println!("{:<28} {:>10.4} {:>10.4}", "Equal Weight", ew_mean, ew_std);
    println!();
    println!("Delta (CG - EW): {:+.4} ({:+.1}%)", delta_mean, delta_pct);
    println!("Paired t-stat: {:.3}", t_stat);
    println!("Approx p-value: {:.4}", p_approx);

    println!("\n=== PER-SEED DETAIL ===\n");
    println!(
        "{:>6} {:>10} {:>10} {:>10}",
        "Seed", "CG CVS", "EW CVS", "Delta"
    );
    for i in 0..n_seeds as usize {
        println!(
            "{:>6} {:>10.4} {:>10.4} {:>+10.4}",
            i * 137 + 42,
            cg_cvs[i],
            ew_cvs[i],
            cg_cvs[i] - ew_cvs[i]
        );
    }

    println!("\n=== VERDICT ===");
    if p_approx < 0.05 && delta_mean > 0.0 {
        println!(
            "Consciousness gating SIGNIFICANTLY improves CVS (p < 0.05, Δ = {:+.1}%)",
            delta_pct
        );
    } else if delta_mean > 0.0 {
        println!("Consciousness gating shows TREND toward improvement ({:+.1}%) but NOT statistically significant (p = {:.3})", delta_pct, p_approx);
    } else {
        println!(
            "Consciousness gating does NOT improve CVS (Δ = {:+.1}%)",
            delta_pct
        );
    }
    println!(
        "Published claim: +6.9%. Observed: {:+.1}% ± {:.1}%",
        delta_pct,
        diff_std * 100.0 / ew_mean
    );

    println!("\nHONEST CAVEAT: This is a simulated comparison, not a real-world experiment.");
    println!(
        "The consciousness-gating model was designed by the same team that built the simulator."
    );
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    let variance = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    variance.sqrt()
}

/// Rough t-distribution CDF approximation for p-value.
/// Uses normal approximation for df > 5.
fn t_to_p(t: f64, df: u64) -> f64 {
    if df == 0 {
        return 1.0;
    }
    // Normal approximation (good for df > 5)
    let z = t * (1.0 - 1.0 / (4.0 * df as f64));
    // Standard normal CDF approximation
    let p = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    1.0 - p
}

/// Error function approximation (Abramowitz and Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        -result
    }
}
