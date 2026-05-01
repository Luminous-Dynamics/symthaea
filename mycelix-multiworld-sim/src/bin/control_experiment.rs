// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Control experiment: trust-weighted governance enabled vs disabled.
//!
//! Proves that trust-weighted governance improves civilization outcomes
//! by comparing the full system against a control where gating is disabled.

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

const SEEDS: [u64; 5] = [42, 123, 789, 1337, 2718];

fn main() {
    eprintln!("=== CONTROL EXPERIMENT: Consciousness Gating ON vs OFF ===\n");

    println!(
        "{:<6} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10}",
        "Seed",
        "CVS_ON",
        "CVS_OFF",
        "Phi_ON",
        "Phi_OFF",
        "Love_ON",
        "Love_OFF",
        "Pop_ON",
        "Pop_OFF"
    );
    println!("{}", "-".repeat(110));

    let mut cvs_on_sum = 0.0f64;
    let mut cvs_off_sum = 0.0f64;
    let mut phi_on_sum = 0.0f64;
    let mut phi_off_sum = 0.0f64;

    for &seed in &SEEDS {
        // --- WITH trust-weighted governance (treatment) ---
        let mut config_on = SimulationConfig::default_150_year();
        config_on.seed = seed;
        config_on.policy.trust_weighted_governance = true;
        let mut sim_on = MultiWorldSimulator::new(config_on);
        let report_on = sim_on.run();

        // --- WITHOUT trust-weighted governance (control) ---
        let mut config_off = SimulationConfig::default_150_year();
        config_off.seed = seed;
        config_off.policy.trust_weighted_governance = false;
        let mut sim_off = MultiWorldSimulator::new(config_off);
        let report_off = sim_off.run();

        println!(
            "{:<6} | {:>10.3} {:>10.3} | {:>10.3} {:>10.3} | {:>10.3} {:>10.3} | {:>10} {:>10}",
            seed,
            report_on.final_cvs,
            report_off.final_cvs,
            report_on.final_collective_phi,
            report_off.final_collective_phi,
            report_on.final_love_coherence,
            report_off.final_love_coherence,
            report_on.final_population,
            report_off.final_population,
        );

        cvs_on_sum += report_on.final_cvs;
        cvs_off_sum += report_off.final_cvs;
        phi_on_sum += report_on.final_collective_phi;
        phi_off_sum += report_off.final_collective_phi;

        eprintln!(
            "  Seed {seed}: ON={:.3} OFF={:.3} (delta={:+.3})",
            report_on.final_cvs,
            report_off.final_cvs,
            report_on.final_cvs - report_off.final_cvs
        );
    }

    let n = SEEDS.len() as f64;
    let mean_on = cvs_on_sum / n;
    let mean_off = cvs_off_sum / n;
    let improvement = ((mean_on - mean_off) / mean_off) * 100.0;

    println!("\n{}", "=".repeat(110));
    println!("RESULT:");
    println!("  Mean CVS with gating:    {:.3}", mean_on);
    println!("  Mean CVS without gating: {:.3}", mean_off);
    println!("  Improvement:             {:+.1}%", improvement);
    println!("  Mean Phi with gating:    {:.3}", phi_on_sum / n);
    println!("  Mean Phi without gating: {:.3}", phi_off_sum / n);
    println!("{}", "=".repeat(110));

    if mean_on > mean_off {
        println!(
            "\n  CONCLUSION: Consciousness gating IMPROVES civilization viability by {:.1}%",
            improvement
        );
    } else {
        println!(
            "\n  CONCLUSION: Consciousness gating shows no improvement (delta: {:.3})",
            mean_on - mean_off
        );
    }
}
