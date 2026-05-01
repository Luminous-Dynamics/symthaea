// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Phase 2 dose-sensitivity smoke (SIMULATOR_ROADMAP A3).
//!
//! Fast guard: Phase 2 machinery should still produce valid output under
//! a high attacker dose (30 per strategy = 150 total adversaries). The
//! broader 4-dose × 3-seed sweep lives in
//! `examples/dose_sensitivity_sweep.rs`.

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

fn run(
    seed: u64,
    years: u32,
    phase2: bool,
    per_strategy: usize,
) -> mycelix_multiworld_sim::report::CivilizationReport {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();
    config.policy.phase2_enabled = phase2;

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, per_strategy);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, per_strategy);
    sim.run()
}

#[test]
fn sim_handles_high_attacker_dose() {
    // 30 per strategy × 5 = 150 adversaries. Short horizon for test speed.
    let report = run(42, 10, true, 30);

    // The sim should still produce a valid report — not crash under the
    // load. Resilience should still be computable.
    assert!(report.final_cvs > 0.0, "CVS should be positive");
    assert!(report.mycelix_resilience.is_some(), "resilience available");
    let r = report.mycelix_resilience.unwrap();
    for v in [
        r.tier_buy_resilience,
        r.demurrage_resilience,
        r.correction_farm_resilience,
        r.cross_cluster_resilience,
        r.guild_collusion_resilience,
    ] {
        assert!((0.0..=1.0).contains(&v), "resilience out of bounds: {}", v);
    }
}

#[test]
fn phase2_does_not_break_under_stress() {
    // Sanity: at high dose, both conditions still survive.
    let r_on = run(42, 10, true, 30);
    let r_off = run(42, 10, false, 30);
    assert!(r_on.survived, "Phase 2 should survive high-dose 10yr");
    assert!(r_off.survived, "Baseline should survive high-dose 10yr");
}
