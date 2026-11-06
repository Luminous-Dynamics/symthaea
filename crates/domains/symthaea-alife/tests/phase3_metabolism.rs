// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 3 model-cost regression tests, per ALife plan §3c.
//!
//! The first test checks internal arithmetic: the configured per-tick model
//! cost includes at least its Landauer-inspired component. It does not validate
//! a physical lower bound, because both the information quantity and the
//! Boltzmann-like coefficient are model proxies in dimensionless ALife units.
//!
//! The second test checks an outcome of the configured resource and cost
//! policies: an organism forced to rest declines while one forced to forage
//! survives in the specified environment. This is a simulation result, not a
//! physical prediction or evidence about biological metabolism.
//!
//! The exploratory comparisons below retain their original negative findings
//! and parameter-sensitivity controls; they are evidence about this simulation
//! configuration only.

use symthaea_alife::{Action, Environment, Organism, OrganismConfig};

#[test]
fn modeled_cost_includes_erasure_inspired_component() {
    let mut organism = Organism::new(OrganismConfig::default(), 7);
    let env = Environment::default();

    for t in 0..1000u64 {
        let tick = organism.tick(env.resource_at(t), None);
        let erasure_inspired_component = symthaea_alife::landauer_minimum(
            tick.bits_processed,
            organism.cfg.effective_temperature,
        );
        assert!(
            tick.physical_cost >= erasure_inspired_component - 1e-12,
            "t={t}: model_cost={:.6} was below its erasure-inspired component={:.6} for \
             modelled_information={:.4} -- the configured cost component was not included",
            tick.physical_cost,
            erasure_inspired_component,
            tick.bits_processed
        );
    }
}

#[test]
fn never_foraging_dies_while_always_foraging_survives() {
    const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
    const MAX_TICKS: u64 = 5_000;

    for &seed in SEEDS {
        let cfg = OrganismConfig::default();
        let env = Environment::default();

        let mut never_forages = Organism::new(cfg, seed);
        let mut always_forages = Organism::new(cfg, seed);

        let mut never_forages_died = false;

        for t in 0..MAX_TICKS {
            let resource = env.resource_at(t);
            let rest_tick = never_forages.tick(resource, Some(Action::Rest.index()));
            let forage_tick = always_forages.tick(resource, Some(Action::Forage.index()));

            if rest_tick.is_dead {
                never_forages_died = true;
            }
            assert!(
                !forage_tick.is_dead,
                "seed={seed}: an organism that forages every tick should survive under \
                 Environment::default(), but died at t={t}"
            );
        }

        assert!(
            never_forages_died,
            "seed={seed}: an organism that never forages (guaranteed zero income, but still \
             pays metabolic_cost and Phase 3's physical_cost every tick) should have died \
             within {MAX_TICKS} ticks -- if it didn't, energy accounting isn't doing real work"
        );
    }
}
