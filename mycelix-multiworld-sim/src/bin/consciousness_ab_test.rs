// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A/B Test: consciousness gating ON vs OFF.
//!
//! The definitive test of whether consciousness-metric-gated governance
//! produces better civilizational outcomes than ungated governance.
//!
//! Usage: consciousness_ab_test [SEEDS] [TICKS]

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seeds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let ticks: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6000);

    eprintln!("=== CONSCIOUSNESS GATING A/B TEST: {} seeds × {} ticks ({} years) ===\n",
        seeds, ticks, ticks / 12);

    let mut gated_cvs = Vec::new();
    let mut gated_pop = Vec::new();
    let mut gated_phi = Vec::new();
    let mut ungated_cvs = Vec::new();
    let mut ungated_pop = Vec::new();
    let mut ungated_phi = Vec::new();

    for i in 0..seeds {
        let seed = 42 + i as u64 * 17;

        // A: Consciousness gating ON (default)
        eprint!("  Gated   seed {:>3} ...", seed);
        let mut config_a = SimulationConfig::default_300_year();
        config_a.total_ticks = ticks;
        config_a.seed = seed;
        config_a.policy.consciousness_gating_enabled = true;
        let mut sim_a = MultiWorldSimulator::new(config_a);
        let report_a = sim_a.run();
        let phi_a = sim_a.worlds.iter()
            .map(|w| w.collective_phi() * w.population() as f64)
            .sum::<f64>() / report_a.final_population.max(1) as f64;
        eprintln!(" CVS={:.3} pop={:>6} phi={:.3}", report_a.final_cvs, report_a.final_population, phi_a);
        gated_cvs.push(report_a.final_cvs);
        gated_pop.push(report_a.final_population as f64);
        gated_phi.push(phi_a);

        // B: Consciousness gating OFF
        eprint!("  Ungated seed {:>3} ...", seed);
        let mut config_b = SimulationConfig::default_300_year();
        config_b.total_ticks = ticks;
        config_b.seed = seed;
        config_b.policy.consciousness_gating_enabled = false;
        let mut sim_b = MultiWorldSimulator::new(config_b);
        let report_b = sim_b.run();
        let phi_b = sim_b.worlds.iter()
            .map(|w| w.collective_phi() * w.population() as f64)
            .sum::<f64>() / report_b.final_population.max(1) as f64;
        eprintln!(" CVS={:.3} pop={:>6} phi={:.3}", report_b.final_cvs, report_b.final_population, phi_b);
        ungated_cvs.push(report_b.final_cvs);
        ungated_pop.push(report_b.final_population as f64);
        ungated_phi.push(phi_b);
    }

    // Results
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  CONSCIOUSNESS GATING A/B TEST RESULTS                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("{:<12} {:>10} {:>10} {:>10}", "", "CVS", "Population", "Phi");
    println!("{}", "-".repeat(44));
    println!("{:<12} {:>10.4} {:>10.0} {:>10.4}", "GATED", mean(&gated_cvs), mean(&gated_pop), mean(&gated_phi));
    println!("{:<12} {:>10.4} {:>10.0} {:>10.4}", "UNGATED", mean(&ungated_cvs), mean(&ungated_pop), mean(&ungated_phi));
    println!("{}", "-".repeat(44));

    let cvs_diff = mean(&gated_cvs) - mean(&ungated_cvs);
    let phi_diff = mean(&gated_phi) - mean(&ungated_phi);
    let pop_diff = mean(&gated_pop) - mean(&ungated_pop);

    println!("\n  CVS difference:  {:>+.4} ({:>+.1}%)", cvs_diff, cvs_diff / mean(&ungated_cvs) * 100.0);
    println!("  Phi difference:  {:>+.4} ({:>+.1}%)", phi_diff, phi_diff / mean(&ungated_phi).max(0.001) * 100.0);
    println!("  Pop difference:  {:>+.0} ({:>+.1}%)", pop_diff, pop_diff / mean(&ungated_pop) * 100.0);

    println!("\n  VERDICT:");
    if cvs_diff > 0.02 {
        println!("  CONSCIOUSNESS GATING SIGNIFICANTLY IMPROVES OUTCOMES.");
        println!("  {:.1}% CVS improvement validates consciousness-metric-gated governance.", cvs_diff / mean(&ungated_cvs) * 100.0);
    } else if cvs_diff > 0.005 {
        println!("  Consciousness gating provides a small but measurable benefit.");
    } else if cvs_diff > -0.005 {
        println!("  Consciousness gating has no significant effect on outcomes.");
    } else {
        println!("  Consciousness gating REDUCES outcomes — the penalty model may be miscalibrated.");
    }
}
