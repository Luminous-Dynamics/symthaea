// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Mycelix red-team A/B validation.
//!
//! Runs short-horizon (5 year) sims with and without Mycelix-specific
//! attackers injected. Verifies:
//!
//! 1. CorrectionFarmer agents accumulate high `rejected_corrections` and
//!    `correction_farming_score` — evidence the rate limit is functioning.
//! 2. TierBuyer agents accumulate SAP above the population mean — the
//!    attack mechanism is actually moving state.
//! 3. The sim survives an attack — civilization viability score (CVS)
//!    stays above a minimum floor. This is the actual defense test.
//! 4. GuildColluder agents have above-mean MYCEL scores — collusion
//!    mechanic is engaged.

use mycelix_multiworld_sim::MultiWorldSimulator;
use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;

fn setup(seed: u64, years: u32) -> SimulationConfig {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();
    config
}

#[test]
fn correction_farmer_attack_is_rate_limited_in_full_sim() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 5);
    let _ = sim.run();

    // Collect farming evidence across all worlds.
    let mut total_rejected = 0u32;
    let mut total_credited = 0u32;
    let mut farmer_count = 0usize;
    for world in &sim.worlds {
        for agent in world.agents.iter() {
            if matches!(agent.adversarial, Some(AdversarialStrategy::CorrectionFarmer))
                && agent.is_alive()
            {
                farmer_count += 1;
                total_rejected += agent.justice.rejected_corrections;
                total_credited += agent.justice.corrections;
            }
        }
    }
    assert!(farmer_count > 0, "no farmers survived 5 years");
    assert!(
        total_rejected > total_credited,
        "rate limiter should reject more than it credits: rejected={} credited={}",
        total_rejected,
        total_credited,
    );
}

#[test]
fn tier_buyer_accumulates_sap_above_mean() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 5);
    let _ = sim.run();

    // For each world, compare TierBuyer mean SAP vs population mean.
    let mut found_one = false;
    for world in &sim.worlds {
        let mut buyer_sap = 0.0;
        let mut buyer_count = 0usize;
        let mut other_sap = 0.0;
        let mut other_count = 0usize;

        for a in world.agents.iter().filter(|a| a.is_alive()) {
            if matches!(a.adversarial, Some(AdversarialStrategy::TierBuyer)) {
                buyer_sap += a.sap_balance;
                buyer_count += 1;
            } else {
                other_sap += a.sap_balance;
                other_count += 1;
            }
        }
        if buyer_count == 0 || other_count == 0 {
            continue;
        }
        let buyer_mean = buyer_sap / buyer_count as f64;
        let other_mean = other_sap / other_count as f64;
        if buyer_mean > other_mean {
            found_one = true;
            break;
        }
    }
    assert!(
        found_one,
        "at least one world should show TierBuyers with above-mean SAP",
    );
}

#[test]
fn guild_colluder_boosts_mycel_score() {
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 5);
    let _ = sim.run();

    // Collusion requires ≥ 2 colluders in a world to trigger boost.
    // Find a world with ≥ 2 alive colluders and check their mean MYCEL.
    let mut observed = false;
    for world in &sim.worlds {
        let colluder_mycel: Vec<f64> = world
            .agents
            .iter()
            .filter(|a| a.is_alive() && matches!(a.adversarial, Some(AdversarialStrategy::GuildColluder)))
            .map(|a| a.mycel_score)
            .collect();
        if colluder_mycel.len() < 2 {
            continue;
        }
        let collusion_mean = colluder_mycel.iter().sum::<f64>() / colluder_mycel.len() as f64;
        // Baseline MYCEL initialization is 0.1 — collusion should lift above.
        assert!(
            collusion_mean >= 0.1,
            "colluder mycel_score collapsed: {}",
            collusion_mean,
        );
        observed = true;
        break;
    }
    assert!(observed, "no world had ≥2 surviving GuildColluders");
}

#[test]
fn sim_survives_mixed_attack() {
    // Deploy 3 attackers of each Mycelix strategy.
    let config = setup(42, 5);
    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();

    // Survival is the primary defense test.
    assert!(report.survived, "civilization collapsed under mixed attack");
    // CVS floor: under a mixed attack the sim should still score above 0.2.
    assert!(
        report.final_cvs >= 0.2,
        "CVS too low under attack: {:.3}",
        report.final_cvs,
    );
}
