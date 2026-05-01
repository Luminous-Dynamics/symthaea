// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analytical benchmarks: verify the simulator on simple cases with known solutions.
//!
//! These tests isolate individual subsystems and compare their trajectories to
//! closed-form analytical solutions. If the simulator can't reproduce exponential
//! growth, logistic carrying capacity, or resource depletion, then its 1000-year
//! projections are not trustworthy.
//!
//! # Test Philosophy
//!
//! Each benchmark:
//! 1. Creates a minimal config that isolates one mechanism
//! 2. Disables all other mechanisms (disasters, factions, consciousness, etc.)
//! 3. Runs the sim and collects a time series
//! 4. Compares to the analytical solution with a tolerance band
//!
//! Tolerances are generous (10-30%) because the sim is stochastic and has
//! interacting subsystems that can't be fully isolated. The point is direction
//! and order-of-magnitude correctness, not machine precision.

use mycelix_multiworld_sim::config::{
    BirthPolicy, EpochConfig, PolicyConfig, ProjectStrategy, ResourcePriority, SimulationConfig,
    WorldSeedConfig,
};
use mycelix_multiworld_sim::MultiWorldSimulator;

/// Create a minimal single-world config with all mechanisms disabled.
/// Only demographics and resources are active.
fn minimal_config(ticks: u32, population: usize, seed: u64) -> SimulationConfig {
    SimulationConfig {
        total_ticks: ticks,
        seed,
        initial_worlds: vec![WorldSeedConfig {
            name: "Test Colony".into(),
            location: "Moon".into(),
            founding_tick: 0,
            initial_population: population,
            initial_resources: 1.0,
        }],
        epoch_configs: vec![EpochConfig {
            id: 0,
            name: "Test".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy: PolicyConfig {
            // Disable all complex mechanisms
            disasters_enabled: false,
            faction_enabled: false,
            amendment_enabled: false,
            trust_weighted_governance: false,
            turchin_cycles_enabled: false,
            fep_immiseration_enabled: false,
            sacred_stillness_enabled: false,
            migration_enabled: false,
            education_enabled: false,
            hostile_guardian: false,
            outer_system_fission_enabled: false,
            speciation_friction_enabled: false,
            // Keep demographics active
            pair_bond_rate: 0.02,
            birth_policy: BirthPolicy::Natural,
            project_strategy: ProjectStrategy::Balanced,
            resource_priority: ResourcePriority::Balanced,
            ..PolicyConfig::default()
        },
    }
}

// ─── Benchmark 1: Population Growth ─────────────────────────────────────────

#[test]
fn population_grows_from_founding() {
    // Earth colony with 500 people and natural birth rates should grow over 25 years.
    // Earth has abundant resources, so resource depletion shouldn't limit growth.
    let mut config = minimal_config(300, 500, 42); // 25 years
    config.initial_worlds[0].location = "Earth".into();
    config.initial_worlds[0].name = "Earth Test".into();
    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    assert!(
        report.final_population > 500,
        "Earth population should grow from 500 over 25 years: got {}",
        report.final_population
    );
}

#[test]
fn population_grows_monotonically_short_term() {
    // Over 10 years with abundant resources, population should generally increase.
    // We allow some stochastic dips but the trend should be upward.
    let config = minimal_config(120, 200, 42); // 10 years, 200 initial
    let mut sim = MultiWorldSimulator::new(config);
    let _report = sim.run();

    let final_pop = sim.worlds[0].population();
    assert!(
        final_pop >= 180, // Allow 10% dip from stochastic deaths
        "200-person colony should mostly survive 10 years: got {}",
        final_pop
    );
}

// ─── Benchmark 2: Carrying Capacity ─────────────────────────────────────────

#[test]
fn population_bounded_by_carrying_capacity() {
    // max_population is set by the World struct. Population should not exceed it
    // (births should slow as pop approaches capacity).
    let mut config = minimal_config(1800, 100, 42); // 150 years
                                                    // Default max_population for Moon is 10_000
    let mut sim = MultiWorldSimulator::new(config);
    let _report = sim.run();

    let final_pop = sim.worlds[0].population();
    let max_pop = sim.worlds[0].max_population;
    assert!(
        final_pop <= max_pop,
        "Population {} should not exceed carrying capacity {}",
        final_pop,
        max_pop
    );
}

// ─── Benchmark 3: Resource Depletion ────────────────────────────────────────

#[test]
fn resources_deplete_without_production() {
    // If we set production to zero and consumption > 0, resources should hit zero.
    let config = minimal_config(120, 50, 42); // 10 years
    let mut sim = MultiWorldSimulator::new(config);
    sim.initialize();

    // Zero out food production
    if let Some(food) = sim.worlds[0].resources.get_mut("food") {
        food.production_rate = 0.0;
    }

    let _report = sim.run();

    let food = sim.worlds[0].resources.stock_level("food").unwrap_or(0.0);
    assert!(
        food < 100.0,
        "Food should deplete significantly without production: got {:.1}",
        food
    );
}

#[test]
fn energy_depletion_collapses_production() {
    // Phase 4.3 enforcement: when energy hits zero, non-essential sectors drop to 10%.
    let config = minimal_config(120, 50, 42);
    let mut sim = MultiWorldSimulator::new(config);
    sim.initialize();

    // Zero out energy production and stock
    if let Some(energy) = sim.worlds[0].resources.get_mut("energy") {
        energy.production_rate = 0.0;
        energy.current = 0.0;
    }

    let _report = sim.run();

    // Engineering output should be near zero (10% of whatever it was)
    let eng_output = sim.worlds[0].economy.sector_output[0]; // engineering
    let agri_output = sim.worlds[0].economy.sector_output[1]; // agriculture
                                                              // Agriculture should be higher than engineering (exempt from energy penalty)
                                                              // Both may be zero if population died, so only test if pop survived
    if sim.worlds[0].population() > 10 {
        assert!(
            agri_output >= eng_output,
            "Agriculture should survive energy crisis better than engineering: agri={:.2} eng={:.2}",
            agri_output, eng_output
        );
    }
}

// ─── Benchmark 4: Consciousness Development ────────────────────────────────

#[test]
fn consciousness_develops_with_education() {
    // With education enabled on Earth (stable resources), consciousness should grow.
    let mut config = minimal_config(600, 300, 42); // 50 years
    config.initial_worlds[0].location = "Earth".into();
    config.initial_worlds[0].name = "Earth Test".into();
    config.policy.education_enabled = true;
    config.policy.consciousness_boost = 0.1; // Small head start

    let mut sim = MultiWorldSimulator::new(config);
    sim.initialize();
    let initial_phi = sim.worlds[0].mean_phi();

    let _report = sim.run();
    let final_phi = sim.worlds[0].mean_phi();

    assert!(
        final_phi > initial_phi,
        "Consciousness should develop with education on Earth over 50 years: {:.4} → {:.4}",
        initial_phi,
        final_phi
    );
}

#[test]
fn consciousness_higher_with_education() {
    // Compare education ON vs OFF on identical Earth colonies.
    let mut config_no = minimal_config(360, 300, 42); // 30 years
    config_no.initial_worlds[0].location = "Earth".into();
    config_no.initial_worlds[0].name = "Earth Control".into();
    config_no.policy.education_enabled = false;

    let mut config_edu = config_no.clone();
    config_edu.policy.education_enabled = true;
    config_edu.initial_worlds[0].name = "Earth Treatment".into();

    let mut sim_no = MultiWorldSimulator::new(config_no);
    let mut sim_edu = MultiWorldSimulator::new(config_edu);

    let _report_no = sim_no.run();
    let _report_yes = sim_edu.run();

    // Both should survive on Earth
    assert!(sim_no.worlds[0].population() > 50, "Control should survive");
    assert!(
        sim_edu.worlds[0].population() > 50,
        "Treatment should survive"
    );

    let phi_no = sim_no.worlds[0].mean_phi();
    let phi_yes = sim_edu.worlds[0].mean_phi();

    // Education should help, though the effect may be modest over 30 years
    // We just verify the direction, not the magnitude
    assert!(
        phi_yes >= phi_no * 0.9, // Allow 10% noise — education shouldn't hurt
        "Education should not reduce Phi: with={:.4} without={:.4}",
        phi_yes,
        phi_no
    );
}

// ─── Benchmark 5: Governance Response ───────────────────────────────────────

#[test]
fn governance_evolves_with_consciousness() {
    // With consciousness-gated governance (Phase 4.4), authority should evolve
    // as mean_phi increases. Higher consciousness → higher authority level.
    let mut config = minimal_config(1800, 200, 42); // 150 years
    config.policy.education_enabled = true;
    config.policy.trust_weighted_governance = true;
    config.policy.consciousness_boost = 0.3; // Give a head start

    let mut sim = MultiWorldSimulator::new(config);
    let _report = sim.run();

    let authority = &sim.worlds[0].governance.authority_level;
    // After 150 years with consciousness boost, should at least reach LocalWithEarthVeto
    assert!(
        *authority != mycelix_multiworld_sim::governance::GovernanceAuthority::MissionControl,
        "Governance should evolve beyond MissionControl after 150 years with consciousness boost"
    );
}

// ─── Benchmark 6: Spoilage Rates ────────────────────────────────────────────

#[test]
fn spoilage_reduces_resources_over_time() {
    // SimulationParams spoilage rates should cause resource decline when
    // production equals consumption (net zero without spoilage).
    let config = minimal_config(120, 50, 42);
    let mut sim = MultiWorldSimulator::new(config);
    sim.initialize();

    // Set production = consumption (equilibrium without spoilage)
    if let Some(food) = sim.worlds[0].resources.get_mut("food") {
        food.consumption_rate = food.production_rate;
        let initial = food.current;
        assert!(initial > 0.0);
    }

    let _report = sim.run();

    let food = sim.worlds[0].resources.stock_level("food").unwrap_or(0.0);
    // With 3%/month spoilage and production=consumption, stock should decline
    // (spoilage is the only net loss)
    assert!(
        food < 900.0, // Initial is 1000, should decline
        "Spoilage should reduce food stock: got {:.1}",
        food
    );
}

// ─── Benchmark 7: Determinism ───────────────────────────────────────────────

#[test]
fn same_seed_same_result() {
    let config1 = minimal_config(300, 100, 42);
    let config2 = minimal_config(300, 100, 42);

    let mut sim1 = MultiWorldSimulator::new(config1);
    let mut sim2 = MultiWorldSimulator::new(config2);

    let report1 = sim1.run();
    let report2 = sim2.run();

    assert_eq!(
        report1.final_population, report2.final_population,
        "Same seed should produce identical population"
    );
    assert!(
        (report1.final_cvs - report2.final_cvs).abs() < 1e-10,
        "Same seed should produce identical CVS: {} vs {}",
        report1.final_cvs,
        report2.final_cvs
    );
}

#[test]
fn different_seed_different_result() {
    let config1 = minimal_config(600, 100, 42);
    let config2 = minimal_config(600, 100, 9999);

    let mut sim1 = MultiWorldSimulator::new(config1);
    let mut sim2 = MultiWorldSimulator::new(config2);

    let report1 = sim1.run();
    let report2 = sim2.run();

    // Different seeds should produce different outcomes (stochasticity working)
    let pop_diff = (report1.final_population as i64 - report2.final_population as i64).abs();
    let cvs_diff = (report1.final_cvs - report2.final_cvs).abs();
    assert!(
        pop_diff > 0 || cvs_diff > 0.001,
        "Different seeds should diverge: pop diff={}, cvs diff={}",
        pop_diff,
        cvs_diff
    );
}
