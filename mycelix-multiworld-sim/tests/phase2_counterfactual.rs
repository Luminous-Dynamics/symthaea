// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Phase 2 counterfactual (SIMULATOR_ROADMAP A2).
//!
//! Answers the scientific question underlying Phase 2a-2c:
//! *Does Phase 2 machinery (8D sovereign profile + restorative justice +
//! correction rate limit) actually outperform baseline 4D Phi + MYCEL
//! governance under the same adversarial environment?*
//!
//! `PolicyConfig::phase2_enabled = false` disables:
//! - `World::refresh_sovereign_profiles` call in the engine tick
//! - The 8D civic blend in `tick_governance_full` (reverts to Phi+MYCEL 50/50)
//! - Violation recording in `sanctions::apply_sanctions`
//! - The per-tick `apply_restorative_corrections` pass
//!
//! Attackers are injected identically in both conditions. Only the
//! DEFENSE differs.

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::red_team::AdversarialStrategy;
use mycelix_multiworld_sim::MultiWorldSimulator;

fn run_condition(seed: u64, years: u32, phase2: bool) -> (bool, f64, usize) {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = PolicyConfig::default();
    config.policy.phase2_enabled = phase2;

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::TierBuyer, 3);
    sim.inject_adversaries(AdversarialStrategy::DemurrageEvader, 3);
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    sim.inject_adversaries(AdversarialStrategy::CrossClusterAmplifier, 3);
    sim.inject_adversaries(AdversarialStrategy::GuildColluder, 3);
    let report = sim.run();
    (report.survived, report.final_cvs, report.final_population)
}

#[test]
fn counterfactual_30yr_seed42() {
    // A single-seed smoke test — the 10-seed sweep lives in
    // `examples/phase2_counterfactual_sweep.rs`.
    let (surv_on, cvs_on, pop_on) = run_condition(42, 30, true);
    let (surv_off, cvs_off, pop_off) = run_condition(42, 30, false);

    // Both conditions should survive 30 years — Phase 2 is not supposed
    // to be the difference between collapse and survival at this horizon.
    assert!(surv_on, "Phase 2 enabled should survive");
    assert!(surv_off, "Phase 2 disabled should also survive at 30yr");

    // Report both CVS values so test output surfaces the actual delta.
    println!(
        "counterfactual 30yr seed=42: phase2=on CVS={:.3} pop={}, \
         phase2=off CVS={:.3} pop={}, delta={:+.3}",
        cvs_on,
        pop_on,
        cvs_off,
        pop_off,
        cvs_on - cvs_off,
    );
}

#[test]
fn counterfactual_disables_restorative_justice() {
    // Confirm that Phase 2 disabled path actually zeroes out the
    // restorative-justice bookkeeping. With no violation recording,
    // every farmer's `rejected_corrections` should be zero (rate limit
    // still active at the primitive level, but no violations → no tier
    // penalty → the attack has no point to aim at).
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = 3 * 12;
    config.seed = 42;
    config.policy = PolicyConfig::default();
    config.policy.phase2_enabled = false;

    let mut sim = MultiWorldSimulator::new(config);
    sim.run_initialization();
    sim.inject_adversaries(AdversarialStrategy::CorrectionFarmer, 3);
    let report = sim.run();

    // The report should still expose resilience (adversaries are tagged).
    assert!(report.mycelix_resilience.is_some());

    // No agent's justice should have recorded any violations (the sanctions
    // hook is gated off). Agents may have SOME corrections from the
    // attack itself; what should be zero is violations.
    let mut total_violations = 0u32;
    for world in &sim.worlds {
        for a in world.agents.iter() {
            total_violations += a.justice.violations;
        }
    }
    assert_eq!(
        total_violations, 0,
        "violation hook should be skipped in counterfactual mode",
    );
}
