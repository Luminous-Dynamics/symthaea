// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Production regression for the independent evolution-RNG wiring.
//!
//! `Population::step` and `Population::step_social` contain separate authoritative bookkeeping
//! loops. This test makes them traverse several forced-birth generations and requires the
//! heritable population to remain identical under both `FromParent` and `RandomPeer` inheritance.
//! The social scheduler has its own RNG and must not perturb evolutionary source/mutation streams.

use symthaea_alife::{
    EncounterScheduler, Genome, InheritanceMode, OrganismConfig, PairingMode, Population,
    PopulationConfig,
};

fn forced_birth_cfg(inheritance: InheritanceMode) -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: -100.0,
        reproduction_energy_threshold: 0.0,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(),
        mutation_rate: 0.37,
        mutation_std: 0.05,
        inheritance,
    }
}

fn genome_population(population: &Population) -> Vec<Genome> {
    population
        .organisms
        .iter()
        .map(|organism| Genome::from_config(&organism.cfg))
        .collect()
}

#[test]
fn ordinary_and_social_steps_share_exact_evolution_birth_semantics() {
    for inheritance in [InheritanceMode::FromParent, InheritanceMode::RandomPeer] {
        let seed = 0x51a7_5eed_u64;
        let initial_count = 3usize;
        let cfg = forced_birth_cfg(inheritance);
        let mut ordinary = Population::new(cfg, initial_count, seed);
        let mut social = Population::new(cfg, initial_count, seed);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, seed.wrapping_add(100_003));

        for generation_step in 0..4 {
            let ordinary_before = ordinary.len();
            let social_before = social.len();
            assert_eq!(ordinary_before, social_before);

            let ordinary_summary = ordinary.step(|_| 1.0);
            let social_summary = social.step_social(|_| 1.0, &mut scheduler);

            assert_eq!(
                ordinary_summary.births_this_tick, ordinary_before as u64,
                "ordinary path did not force exactly one birth per current organism at step {generation_step}"
            );
            assert_eq!(
                social_summary.births_this_tick, social_before as u64,
                "social path did not force exactly one birth per current organism at step {generation_step}"
            );
            assert_eq!(ordinary_summary.deaths_this_tick, 0);
            assert_eq!(social_summary.deaths_this_tick, 0);
            assert_eq!(ordinary.len(), social.len());
            assert_eq!(
                genome_population(&ordinary),
                genome_population(&social),
                "evolutionary birth semantics diverged between step paths for {inheritance:?} at generation step {generation_step}"
            );
        }

        assert_eq!(ordinary.total_births, social.total_births);
        assert_eq!(ordinary.total_deaths, 0);
        assert_eq!(social.total_deaths, 0);
    }
}
