// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Diagnostic: WHY does consciousness gating underperform in most seeds?
//! Runs winning and losing seeds with detailed per-epoch telemetry.

use mycelix_multiworld_sim::{config::SimulationConfig, MultiWorldSimulator};

fn main() {
    // Seeds from multi-seed comparison:
    // Winners (CG > EW): 316 (+0.0032), 590 (+0.0091), 1001 (+0.0004)
    // Losers  (CG < EW): 179 (-0.0193), 453 (-0.0141), 727 (-0.0119)
    let diagnostic_seeds = [
        (590, true, "+0.0091"),  // Best CG seed
        (179, false, "-0.0193"), // Worst CG seed
        (42, true, "-0.0044"),   // Original benchmark (actually loses)
    ];

    for &(seed, _cg_wins, delta_str) in &diagnostic_seeds {
        println!("=== SEED {} (CG delta: {}) ===\n", seed, delta_str);

        for trust_weighted in [true, false] {
            let label = if trust_weighted { "CG" } else { "EW" };
            let mut config = SimulationConfig::default_150_year();
            config.seed = seed;
            config.total_ticks = 150 * 12;
            config.policy.trust_weighted_governance = trust_weighted;

            let mut sim = MultiWorldSimulator::new(config);
            let report = sim.run();

            // Collect diagnostic data
            let total_pop: usize = sim.worlds.iter().map(|w| w.population()).sum();
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
            let mean_trauma: f64 = sim
                .worlds
                .iter()
                .flat_map(|w| w.agents.iter().filter(|a| a.is_alive()))
                .map(|a| a.trauma_level)
                .sum::<f64>()
                / total_pop.max(1) as f64;
            let mean_load: f64 = sim
                .worlds
                .iter()
                .flat_map(|w| w.agents.iter().filter(|a| a.is_alive()))
                .map(|a| a.needs.allostatic_load)
                .sum::<f64>()
                / total_pop.max(1) as f64;
            let mean_joy: f64 = sim
                .worlds
                .iter()
                .flat_map(|w| w.agents.iter().filter(|a| a.is_alive()))
                .map(|a| a.needs.affect.joy)
                .sum::<f64>()
                / total_pop.max(1) as f64;
            let mean_engagement: f64 = sim
                .worlds
                .iter()
                .flat_map(|w| w.agents.iter().filter(|a| a.is_alive()))
                .map(|a| a.needs.engagement)
                .sum::<f64>()
                / total_pop.max(1) as f64;
            let faction_count = sim.faction_engine.factions.len();
            let cascade_count = sim.cascade_engine.total_cascades;
            let viability_violations = sim.viability_engine.violations.len();

            println!(
                "  [{label}] CVS={:.4} Pop={} Phi={:.3} Trauma={:.3} Load={:.3}",
                report.final_cvs, total_pop, mean_phi, mean_trauma, mean_load
            );
            println!(
                "       Joy={:.3} Engage={:.3} Factions={} Cascades={} ViabilityViol={}",
                mean_joy, mean_engagement, faction_count, cascade_count, viability_violations
            );
            println!(
                "       Disasters={} TechMilestones={}",
                report.total_disasters, report.tech_milestones_achieved
            );
        }
        println!();
    }
}
