// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Red team test: can adversarial agents game consciousness-gated governance?
//! Injects ProfileMaximizer agents (5× faster phi growth) and measures impact.

use mycelix_multiworld_sim::{
    config::SimulationConfig,
    red_team::{evaluate_resilience, AdversarialStrategy},
    MultiWorldSimulator,
};

fn main() {
    println!("=== RED TEAM: ADVERSARIAL AGENT INJECTION TEST ===\n");

    let seeds = [42, 179, 590, 727, 1001];
    let years = 150u32;
    let adversaries_per_world = 10; // 10 out of ~200 agents = 5%

    println!(
        "{:<8} {:>8} {:>8} {:>+8} {:>8} {:>8} {:>10}",
        "Seed", "Adv CVS", "Base CVS", "ΔCVS", "Adv φ", "Pop φ", "Status"
    );
    println!("{}", "-".repeat(70));

    let mut deltas = Vec::new();

    for &seed in &seeds {
        // Baseline: consciousness-gated, no adversaries
        let mut config = SimulationConfig::default_150_year();
        config.seed = seed;
        config.total_ticks = years * 12;
        config.policy.trust_weighted_governance = true;
        let mut base_sim = MultiWorldSimulator::new(config);
        let base_report = base_sim.run();
        let base_cvs = base_report.final_cvs;

        // Adversarial: inject ProfileMaximizers after initialization
        let mut config = SimulationConfig::default_150_year();
        config.seed = seed;
        config.total_ticks = years * 12;
        config.policy.trust_weighted_governance = true;
        let mut adv_sim = MultiWorldSimulator::new(config);
        adv_sim.initialize(); // create worlds + agents
        adv_sim.inject_adversaries(AdversarialStrategy::ProfileMaximizer, adversaries_per_world);
        let adv_report = adv_sim.run(); // run with adversaries active

        // Measure adversarial agent phi vs population phi
        let mut adv_phi_sum = 0.0;
        let mut adv_count = 0;
        let mut pop_phi_sum = 0.0;
        let mut pop_count = 0;
        for world in &adv_sim.worlds {
            for agent in world.agents.iter().filter(|a| a.is_alive()) {
                let phi = agent.consciousness.phi();
                if agent.adversarial.is_some() {
                    adv_phi_sum += phi;
                    adv_count += 1;
                } else {
                    pop_phi_sum += phi;
                    pop_count += 1;
                }
            }
        }
        let adv_phi = if adv_count > 0 {
            adv_phi_sum / adv_count as f64
        } else {
            0.0
        };
        let pop_phi = if pop_count > 0 {
            pop_phi_sum / pop_count as f64
        } else {
            0.0
        };
        let delta = adv_report.final_cvs - base_cvs;
        deltas.push(delta);

        let status = if delta < -0.02 {
            "VULNERABLE"
        } else if delta < -0.005 {
            "MINOR HIT"
        } else {
            "RESILIENT"
        };

        println!(
            "{:<8} {:>8.4} {:>8.4} {:>+8.4} {:>8.3} {:>8.3} {:>10}",
            seed, adv_report.final_cvs, base_cvs, delta, adv_phi, pop_phi, status
        );
    }

    // Statistics
    let mean_delta: f64 = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let max_damage = deltas.iter().cloned().fold(0.0f64, f64::min);

    println!("\n=== VERDICT ===");
    println!("Mean ΔCVS from adversaries: {:+.4}", mean_delta);
    println!("Worst-case ΔCVS: {:+.4}", max_damage);

    if max_damage > -0.005 {
        println!("STATUS: Governance system ROBUST — adversaries have negligible impact");
    } else if max_damage > -0.02 {
        println!("STATUS: MINOR vulnerability — adversaries cause small CVS degradation");
    } else {
        println!("STATUS: VULNERABLE — adversaries significantly degrade civilizational outcomes");
    }

    println!(
        "\nAdversary details: {} ProfileMaximizer agents per world (5× phi growth rate)",
        adversaries_per_world
    );
    println!("These agents game the consciousness profile to gain disproportionate voting power.");
}
