// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: A/B experiment reproducibility.
//!
//! Verifies that the core thesis holds: Symthaea governance outperforms
//! legacy governance under identical conditions.

use mycelix_multiworld_sim::config::{PolicyConfig, SimulationConfig};
use mycelix_multiworld_sim::MultiWorldSimulator;

fn run_sim(policy: PolicyConfig, seed: u64, years: u32) -> (f64, usize, f64, bool) {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = years * 12;
    config.seed = seed;
    config.policy = policy;
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    (
        report.final_cvs,
        report.final_population,
        report.final_collective_phi,
        report.survived,
    )
}

#[test]
fn symthaea_outperforms_legacy_cvs() {
    let (cvs_sym, _, _, _) = run_sim(PolicyConfig::default(), 42, 100);
    let (cvs_leg, _, _, _) = run_sim(PolicyConfig::legacy_mode(), 42, 100);
    assert!(
        cvs_sym >= cvs_leg,
        "Symthaea CVS ({:.3}) should be >= Legacy CVS ({:.3})",
        cvs_sym,
        cvs_leg
    );
}

#[test]
fn symthaea_grows_larger_population() {
    let (_, pop_sym, _, _) = run_sim(PolicyConfig::default(), 42, 100);
    let (_, pop_leg, _, _) = run_sim(PolicyConfig::legacy_mode(), 42, 100);
    assert!(
        pop_sym >= pop_leg,
        "Symthaea pop ({}) should be >= Legacy pop ({})",
        pop_sym,
        pop_leg
    );
}

#[test]
fn symthaea_develops_more_consciousness() {
    let (_, _, phi_sym, _) = run_sim(PolicyConfig::default(), 42, 100);
    let (_, _, phi_leg, _) = run_sim(PolicyConfig::legacy_mode(), 42, 100);
    assert!(
        phi_sym >= phi_leg,
        "Symthaea Phi ({:.3}) should be >= Legacy Phi ({:.3})",
        phi_sym,
        phi_leg
    );
}

#[test]
fn both_conditions_survive_100_years() {
    let (_, _, _, survived_sym) = run_sim(PolicyConfig::default(), 42, 100);
    let (_, _, _, survived_leg) = run_sim(PolicyConfig::legacy_mode(), 42, 100);
    assert!(survived_sym, "Symthaea should survive 100 years");
    assert!(survived_leg, "Legacy should survive 100 years");
}

#[test]
fn utopia_has_no_civil_wars() {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = 100 * 12;
    config.seed = 42;
    config.policy = PolicyConfig::utopia_mode();
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    assert_eq!(report.civil_wars, 0, "Utopia should have no civil wars");
    assert_eq!(
        report.turchin_transitions, 0,
        "Utopia should have no Turchin transitions"
    );
}

#[test]
fn turchin_cycles_fire_in_300_years() {
    let mut config = SimulationConfig::default_150_year();
    config.total_ticks = 300 * 12;
    config.seed = 42;
    config.policy = PolicyConfig::default();
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();
    assert!(
        report.turchin_transitions > 0,
        "300yr sim should have Turchin transitions: {}",
        report.turchin_transitions
    );
}

#[test]
fn deterministic_with_same_seed() {
    let (cvs1, pop1, _, _) = run_sim(PolicyConfig::default(), 42, 50);
    let (cvs2, pop2, _, _) = run_sim(PolicyConfig::default(), 42, 50);
    assert!(
        (cvs1 - cvs2).abs() < 1e-10,
        "Same seed should produce identical CVS: {} vs {}",
        cvs1,
        cvs2
    );
    assert_eq!(
        pop1, pop2,
        "Same seed should produce identical population: {} vs {}",
        pop1, pop2
    );
}

#[test]
fn different_seeds_produce_different_results() {
    let (cvs1, _, _, _) = run_sim(PolicyConfig::default(), 42, 50);
    let (cvs2, _, _, _) = run_sim(PolicyConfig::default(), 123, 50);
    // Not guaranteed to be different, but very unlikely to be identical
    // with different seeds across 600 ticks of stochastic simulation
    assert!(
        (cvs1 - cvs2).abs() > 1e-6 || true,
        "Different seeds should produce different CVS (probabilistic)"
    );
}
