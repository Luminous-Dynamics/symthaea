// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Run all 5 governance models with the same seed and compare CVS outcomes.
//! This is the honest test of the consciousness-gating hypothesis.

use mycelix_multiworld_sim::{
    config::SimulationConfig, governance_models::GovernanceModel, MultiWorldSimulator,
};

fn main() {
    let seed = 42u64;
    let years = 150u32;

    println!("=== GOVERNANCE MODEL COMPARISON ===");
    println!("Seed: {seed}, Duration: {years} years, 5 models\n");
    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Model", "CVS", "Pop", "Worlds", "Disast", "Phi"
    );
    println!("{}", "-".repeat(80));

    let mut results = Vec::new();

    for model in GovernanceModel::all() {
        let mut config = SimulationConfig::default_150_year();
        config.seed = seed;
        config.total_ticks = years * 12;

        // Set governance model via trust_weighted_governance flag
        match model {
            GovernanceModel::ConsciousnessGated => {
                config.policy.trust_weighted_governance = true;
            }
            GovernanceModel::EqualWeight => {
                config.policy.trust_weighted_governance = false;
            }
            GovernanceModel::Sortition
            | GovernanceModel::Meritocratic
            | GovernanceModel::ElderCouncil => {
                // These models aren't wired into the tick loop yet —
                // for now, approximate by disabling trust weighting
                // (the voting_weight() function exists but isn't called by governance.rs)
                config.policy.trust_weighted_governance = false;
                // TODO: Wire GovernanceModel::voting_weight() into governance.rs
            }
        }

        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        let mean_phi: f64 = sim
            .worlds
            .iter()
            .filter(|w| w.population() > 0)
            .map(|w| w.mean_phi())
            .sum::<f64>()
            / sim
                .worlds
                .iter()
                .filter(|w| w.population() > 0)
                .count()
                .max(1) as f64;

        println!(
            "{:<30} {:>8.4} {:>8} {:>8} {:>8} {:>8.3}",
            model.name(),
            report.final_cvs,
            report.final_population,
            report.final_worlds,
            report.total_disasters,
            mean_phi,
        );

        results.push((model.name(), report.final_cvs, report.final_population));
    }

    // Analysis
    println!("\n=== ANALYSIS ===");
    let consciousness_cvs = results
        .iter()
        .find(|(n, _, _)| *n == "Consciousness-Gated")
        .map(|(_, cvs, _)| *cvs)
        .unwrap_or(0.0);
    let equal_cvs = results
        .iter()
        .find(|(n, _, _)| *n == "Equal Weight (Democracy)")
        .map(|(_, cvs, _)| *cvs)
        .unwrap_or(0.0);

    let delta = consciousness_cvs - equal_cvs;
    let pct = if equal_cvs > 0.0 {
        delta / equal_cvs * 100.0
    } else {
        0.0
    };

    println!(
        "Consciousness-Gated vs Equal Weight: ΔCVS = {:+.4} ({:+.1}%)",
        delta, pct
    );
    println!("Published claim: +6.9%. Observed: {:+.1}%", pct);

    if pct > 5.0 {
        println!("STATUS: Claim SUPPORTED (>{:.0}% improvement)", pct);
    } else if pct > 0.0 {
        println!("STATUS: Marginal improvement — claim WEAKLY SUPPORTED");
    } else {
        println!("STATUS: Claim NOT SUPPORTED — equal weight performs as well or better");
    }

    println!("\nHONEST CAVEAT: Sortition, Meritocratic, and Elder Council models are");
    println!("approximated by disabling trust weighting. Full implementation requires");
    println!("wiring GovernanceModel::voting_weight() into governance.rs.");
    println!("The comparison between ConsciousnessGated and EqualWeight is accurate.");
}
