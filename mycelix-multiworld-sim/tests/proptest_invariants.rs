// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Property-based tests for multiworld-sim invariants.
//!
//! These tests verify that core simulation invariants hold across
//! randomized configurations and tick counts.

use proptest::prelude::{any, prop_assert, proptest, ProptestConfig, Strategy};

use mycelix_multiworld_sim::config::{
    EpochConfig, PolicyConfig, SimulationConfig, WorldSeedConfig,
};
use mycelix_multiworld_sim::MultiWorldSimulator;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a PolicyConfig with randomized knobs within valid ranges.
fn arb_policy() -> impl Strategy<Value = PolicyConfig> {
    (
        0.0..0.05f64,  // pair_bond_rate
        0.0..2.0f64,   // care_effectiveness
        0.0..1.0f64,   // pharma_boost
        1.0..3.0f64,   // deep_space_isolation_mult
        any::<bool>(), // migration_enabled
        1..6u32,       // migration_max_per_cycle
        any::<bool>(), // education_enabled
        any::<bool>(), // trust_weighted_governance
        any::<bool>(), // faction_enabled
        any::<bool>(), // amendment_enabled
    )
        .prop_map(
            |(
                pair_bond_rate,
                care_effectiveness,
                pharma_boost,
                deep_space_isolation_mult,
                migration_enabled,
                migration_max_per_cycle,
                education_enabled,
                trust_weighted_governance,
                faction_enabled,
                amendment_enabled,
            )| {
                PolicyConfig {
                    pair_bond_rate,
                    care_effectiveness,
                    pharma_boost,
                    deep_space_isolation_mult,
                    migration_enabled,
                    migration_max_per_cycle,
                    education_enabled,
                    trust_weighted_governance,
                    faction_enabled,
                    amendment_enabled,
                    // Keep these fixed to reduce state space while still covering
                    // the core policy knobs above.
                    outer_system_fission_enabled: false,
                    speciation_friction_enabled: false,
                    hostile_guardian: false,
                    disasters_enabled: true,
                    hybrid_earth: false,
                    project_strategy: mycelix_multiworld_sim::config::ProjectStrategy::Balanced,
                    birth_policy: mycelix_multiworld_sim::config::BirthPolicy::Natural,
                    resource_priority: mycelix_multiworld_sim::config::ResourcePriority::Balanced,
                    trade_openness: 0.5,
                    defense_spending: 0.05,
                    exploration_investment: 0.05,
                    ..PolicyConfig::default()
                }
            },
        )
}

/// Build a short simulation config: Earth + Moon, randomized policy, few ticks.
fn arb_short_sim(max_ticks: u32) -> impl Strategy<Value = SimulationConfig> {
    (1..max_ticks, 0..10_000u64, arb_policy()).prop_map(|(ticks, seed, policy)| SimulationConfig {
        total_ticks: ticks,
        seed,
        initial_worlds: vec![
            WorldSeedConfig {
                name: "Earth".into(),
                location: "Earth".into(),
                founding_tick: 0,
                initial_population: 100,
                initial_resources: 1.0,
            },
            WorldSeedConfig {
                name: "Artemis Base".into(),
                location: "Moon".into(),
                founding_tick: 0,
                initial_population: 12,
                initial_resources: 0.3,
            },
        ],
        epoch_configs: vec![EpochConfig {
            id: 0,
            name: "Foundation".into(),
            start_tick: 0,
            end_tick: ticks,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy,
    })
}

// ---------------------------------------------------------------------------
// Helper: run a sim and return final state for inspection
// ---------------------------------------------------------------------------

fn run_sim(config: SimulationConfig) -> MultiWorldSimulator {
    let mut sim = MultiWorldSimulator::new(config);
    let _report = sim.run();
    sim
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Population must never be negative (it's usize, so this checks that
    /// the simulation doesn't panic from underflow and worlds still exist).
    #[test]
    fn population_never_negative(config in arb_short_sim(60)) {
        let sim = run_sim(config);
        for world in &sim.worlds {
            // population() counts living agents, which is always >= 0
            let _pop = world.population();
            // Also verify no agent has impossible state
            for agent in &world.agents {
                prop_assert!(agent.health >= 0.0,
                    "Agent {} health negative: {}", agent.id, agent.health);
            }
        }
    }

    /// Resource values should never be NaN or negative.
    #[test]
    fn resources_bounded(config in arb_short_sim(60)) {
        let sim = run_sim(config);
        for world in &sim.worlds {
            for name in world.resources.resource_names() {
                let stock = world.resources.get(name).unwrap();
                prop_assert!(!stock.current.is_nan(),
                    "World {} resource {} current is NaN", world.name, name);
                prop_assert!(stock.current >= 0.0,
                    "World {} resource {} current negative: {}",
                    world.name, name, stock.current);
                prop_assert!(!stock.capacity.is_nan(),
                    "World {} resource {} capacity is NaN", world.name, name);
                prop_assert!(stock.capacity >= 0.0,
                    "World {} resource {} capacity negative: {}",
                    world.name, name, stock.capacity);
            }
        }
    }

    /// Community Vitality Score (CVS) should always be in [0.0, 1.0].
    #[test]
    fn cvs_bounded(config in arb_short_sim(60)) {
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        let cvs = report.final_cvs;
        prop_assert!(cvs >= 0.0 && cvs <= 1.0,
            "CVS out of bounds: {}", cvs);
        // Also check epoch snapshots
        for snap in &report.epoch_snapshots {
            let snap_cvs = snap.civilization_viability_score;
            prop_assert!(snap_cvs >= 0.0 && snap_cvs <= 1.0,
                "Snapshot CVS out of bounds at tick {}: {}", snap.tick, snap_cvs);
        }
    }

    /// Consciousness (phi) values should be in [0.0, 1.0] for all agents.
    #[test]
    fn phi_bounded(config in arb_short_sim(60)) {
        let sim = run_sim(config);
        for world in &sim.worlds {
            let mean = world.mean_phi();
            prop_assert!(!mean.is_nan(),
                "World {} mean_phi is NaN", world.name);
            prop_assert!(mean >= 0.0 && mean <= 1.0,
                "World {} mean_phi out of [0,1]: {}", world.name, mean);
            // Check individual agents
            for agent in world.agents.iter().filter(|a| a.is_alive()) {
                let phi = agent.consciousness.phi();
                prop_assert!(phi >= 0.0 && phi <= 1.0,
                    "Agent {} phi out of [0,1]: {}", agent.id, phi);
            }
        }
    }

    /// Total energy across all worlds should be non-negative.
    #[test]
    fn energy_non_negative(config in arb_short_sim(60)) {
        let sim = run_sim(config);
        let total_energy: f64 = sim.worlds.iter()
            .filter_map(|w| w.resources.get("energy"))
            .map(|s| s.current)
            .sum();
        prop_assert!(!total_energy.is_nan(),
            "Total energy is NaN");
        prop_assert!(total_energy >= 0.0,
            "Total energy negative: {}", total_energy);
    }

    /// Total population across all worlds should not spontaneously multiply.
    /// After simulation, total living population should not exceed initial total
    /// + a generous upper bound on births (no duplication bugs).
    /// We use initial_pop * ticks as a very generous birth ceiling.
    #[test]
    fn population_no_duplication(config in arb_short_sim(60)) {
        let initial_total: usize = config.initial_worlds.iter()
            .map(|w| w.initial_population)
            .sum();
        let ticks = config.total_ticks;
        let sim = run_sim(config);
        let final_total: usize = sim.worlds.iter().map(|w| w.population()).sum();
        // Very generous bound: each initial person could have at most 1 child per tick
        let max_possible = initial_total + (initial_total * ticks as usize);
        prop_assert!(final_total <= max_possible,
            "Population {} exceeds generous maximum {} (initial {} x {} ticks)",
            final_total, max_possible, initial_total, ticks);
    }

    /// Epoch index should never decrease over time.
    #[test]
    fn epoch_monotonic(config in arb_short_sim(60)) {
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();
        let mut prev_epoch: u8 = 0;
        for snap in &report.epoch_snapshots {
            prop_assert!(snap.epoch_id >= prev_epoch,
                "Epoch decreased from {} to {} at tick {}",
                prev_epoch, snap.epoch_id, snap.tick);
            prev_epoch = snap.epoch_id;
        }
    }

    /// Governance scores should be bounded and not NaN after any tick sequence.
    #[test]
    fn governance_stable(config in arb_short_sim(60)) {
        let sim = run_sim(config);
        for world in &sim.worlds {
            let gov = &world.governance;
            prop_assert!(!gov.oppression_index.is_nan(),
                "World {} governance oppression_index is NaN", world.name);
            prop_assert!(gov.oppression_index >= 0.0 && gov.oppression_index <= 1.0,
                "World {} oppression_index out of [0,1]: {}",
                world.name, gov.oppression_index);
            for (i, threshold) in gov.trust_depth_threshold.iter().enumerate() {
                prop_assert!(!threshold.is_nan(),
                    "World {} trust_depth_threshold[{}] is NaN", world.name, i);
                prop_assert!(*threshold >= 0.0 && *threshold <= 1.0,
                    "World {} trust_depth_threshold[{}] out of [0,1]: {}",
                    world.name, i, threshold);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Long-duration property tests (50-year runs)
// Catches late-stage emergent bugs invisible to the 5-year short tests.
// Fewer cases (8) since each run takes longer.
// ---------------------------------------------------------------------------

/// Build a 50-year simulation config for long-duration testing.
fn arb_long_sim() -> impl Strategy<Value = SimulationConfig> {
    (0..10_000u64, arb_policy()).prop_map(|(seed, policy)| {
        SimulationConfig {
            total_ticks: 600, // 50 years
            seed,
            initial_worlds: vec![
                WorldSeedConfig {
                    name: "Earth".into(),
                    location: "Earth".into(),
                    founding_tick: 0,
                    initial_population: 100,
                    initial_resources: 1.0,
                },
                WorldSeedConfig {
                    name: "Artemis Base".into(),
                    location: "Moon".into(),
                    founding_tick: 0,
                    initial_population: 20,
                    initial_resources: 0.3,
                },
            ],
            epoch_configs: vec![EpochConfig {
                id: 0,
                name: "Foundation".into(),
                start_tick: 0,
                end_tick: 600,
                population_trigger: None,
                self_sufficiency_trigger: None,
            }],
            policy,
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// All short-run invariants must hold over 50 years.
    #[test]
    fn long_run_invariants_hold(config in arb_long_sim()) {
        let mut sim = MultiWorldSimulator::new(config);
        let report = sim.run();

        // Population non-negative
        for world in &sim.worlds {
            for agent in &world.agents {
                prop_assert!(agent.health >= 0.0,
                    "Agent {} health negative after 50yr: {}", agent.id, agent.health);
            }
        }

        // Resources bounded
        for world in &sim.worlds {
            for name in world.resources.resource_names() {
                let stock = world.resources.get(name).unwrap();
                prop_assert!(!stock.current.is_nan(),
                    "World {} resource {} NaN after 50yr", world.name, name);
                prop_assert!(stock.current >= 0.0,
                    "World {} resource {} negative after 50yr: {}",
                    world.name, name, stock.current);
            }
        }

        // CVS bounded
        prop_assert!(report.final_cvs >= 0.0 && report.final_cvs <= 1.0,
            "CVS out of bounds after 50yr: {}", report.final_cvs);

        // Phi bounded
        for world in &sim.worlds {
            let mean = world.mean_phi();
            prop_assert!(!mean.is_nan() && mean >= 0.0 && mean <= 1.0,
                "World {} mean_phi out of bounds after 50yr: {}", world.name, mean);
        }

        // Governance stable
        for world in &sim.worlds {
            prop_assert!(!world.governance.oppression_index.is_nan(),
                "Oppression NaN after 50yr in {}", world.name);
            prop_assert!(world.governance.oppression_index >= 0.0
                && world.governance.oppression_index <= 1.0,
                "Oppression out of [0,1] after 50yr in {}: {}",
                world.name, world.governance.oppression_index);
        }
    }
}

// ---------------------------------------------------------------------------
// Outcome plausibility tests (deterministic, 100-year)
// Verify the simulation produces reasonable outcomes, not just valid ones.
// ---------------------------------------------------------------------------

#[test]
fn outcome_plausibility_100_years() {
    let config = SimulationConfig {
        total_ticks: 1200, // 100 years
        seed: 42,
        initial_worlds: vec![
            WorldSeedConfig {
                name: "Earth".into(),
                location: "Earth".into(),
                founding_tick: 0,
                initial_population: 200,
                initial_resources: 1.0,
            },
            WorldSeedConfig {
                name: "Artemis Base".into(),
                location: "Moon".into(),
                founding_tick: 0,
                initial_population: 30,
                initial_resources: 0.3,
            },
        ],
        epoch_configs: vec![EpochConfig {
            id: 0,
            name: "Foundation".into(),
            start_tick: 0,
            end_tick: 1200,
            population_trigger: None,
            self_sufficiency_trigger: None,
        }],
        policy: PolicyConfig::default(),
    };

    let mut sim = MultiWorldSimulator::new(config);
    let report = sim.run();

    // Population should grow from initial 230 but stay bounded
    assert!(
        report.final_population >= 100,
        "Population should survive 100yr: {}",
        report.final_population
    );
    assert!(
        report.final_population <= 100_000,
        "Population should be bounded after 100yr: {}",
        report.final_population
    );

    // Consciousness should develop
    let max_phi: f64 = sim
        .worlds
        .iter()
        .map(|w| w.mean_phi())
        .fold(0.0f64, f64::max);
    assert!(
        max_phi > 0.05,
        "Some consciousness should develop over 100yr: max_phi={max_phi}"
    );

    // CVS should be computed and positive
    assert!(
        report.final_cvs > 0.0 && report.final_cvs <= 1.0,
        "CVS should be positive and bounded: {}",
        report.final_cvs
    );

    // Simulation should have experienced some disasters
    assert!(
        report.total_disasters > 0,
        "100yr sim should have disasters: {}",
        report.total_disasters
    );
}
