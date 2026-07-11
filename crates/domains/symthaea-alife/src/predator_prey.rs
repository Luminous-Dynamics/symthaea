// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two [`Population`]s coupled predator/prey, per `ALIFE_PLAN_2026-07-08.md` Phase 1b.
//!
//! Both directions of coupling are real consequences of real per-organism decisions, not scripted
//! numbers, but this is deliberately **not a mass-conserving system** — say so plainly rather than
//! imply more physical rigor than exists:
//!
//! - **Prey → predator**: each predator's `resource_level` observation this tick is prey
//!   availability (last tick's prey count per predator, scaled). This drives the predator's own
//!   `Organism::tick()` exactly as any other resource signal would (Phase 0/1a) — its energy gain
//!   is a real consequence of its own `perceive`/`select_action` loop reacting to that signal.
//! - **Predator → prey**: separately, however many predators *actually chose to forage* this tick
//!   (`select_action()`'s real output, not a rate we pick) determines how many prey are actually
//!   removed from the population (weakest first). This is the real mortality consequence.
//!
//! These two channels aren't forced to conserve energy/biomass between them — a predator's
//! resource-driven energy gain and the number of prey it removes are correlated (both derive from
//! the same prey-density signal and the same forage decision) but not identical quantities. A more
//! physically unified model is future work, not required for Phase 1b's actual claim: that
//! coupling two populations this way produces a phase-shifted oscillation, not scripted curves.

use crate::population::{Population, PopulationConfig};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredatorPreyConfig {
    pub prey_cfg: PopulationConfig,
    pub predator_cfg: PopulationConfig,
    /// Total shared plant resource available to prey each tick (Phase 1a's `R_total`).
    pub plant_resource_total: f64,
    /// Converts "prey per predator" into a `[0, 1]`-ish resource signal for the predator's own
    /// `Organism::tick()`.
    pub predation_scale: f64,
    /// Expected prey caught per predator that actually forages this tick.
    pub predation_efficiency: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredatorPreyStep {
    pub tick: u64,
    pub prey_count: usize,
    pub predator_count: usize,
    pub prey_killed: usize,
}

pub struct PredatorPreySim {
    pub prey: Population,
    pub predator: Population,
    pub cfg: PredatorPreyConfig,
    pub t: u64,
    rng_state: u64,
}

impl PredatorPreySim {
    pub fn new(
        cfg: PredatorPreyConfig,
        initial_prey: usize,
        initial_predators: usize,
        seed_base: u64,
    ) -> Self {
        let prey = Population::new(cfg.prey_cfg, initial_prey, seed_base);
        let predator = Population::new(
            cfg.predator_cfg,
            initial_predators,
            seed_base.wrapping_add(1_000_003),
        );
        Self {
            prey,
            predator,
            cfg,
            t: 0,
            rng_state: seed_base.wrapping_add(1_000_007).max(1),
        }
    }

    /// Stream of pseudo-random `[0, 1)` values, independent of the `Organism`-internal and
    /// `Environment` RNGs -- this is only for the stochastic predation-success trials below.
    /// An iterated xorshift64 stream (same algorithm `ActiveInferenceAgent::select_action` uses
    /// internally) -- NOT `environment::xorshift_unit`, which single-pass-hashes a *key* and
    /// degenerates badly when fed small sequential keys (found via a traced diagnostic: it made
    /// "5% catch probability" fire on nearly every attempt).
    fn next_unit(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    pub fn step(&mut self) -> PredatorPreyStep {
        self.t += 1;
        let prey_count_before = self.prey.len();

        // 1. Predators perceive last tick's prey density and act on it for real.
        let predation_scale = self.cfg.predation_scale;
        let predator_summary = self.predator.step(|n| {
            let n = (n.max(1)) as f64;
            (prey_count_before as f64 / n * predation_scale).min(1.0)
        });

        // 2. Real consequence: each predator that actually foraged this tick independently
        // succeeds at catching prey with probability `predation_efficiency` (a Bernoulli trial,
        // not `round(forage_count * efficiency)` -- the deterministic version rounded every
        // small-but-real catch probability down to exactly zero forever, silently decoupling
        // predators from ever actually removing prey; found via a traced diagnostic run where
        // `prey_killed` was 0 at literally every sampled tick despite predators clearly not
        // starving).
        let mut attempted = 0usize;
        for _ in 0..predator_summary.forage_count {
            if self.next_unit() < self.cfg.predation_efficiency {
                attempted += 1;
            }
        }
        let prey_killed = self.prey.cull_weakest(attempted);

        // 3. Surviving prey graze the shared plant pool.
        let plant_total = self.cfg.plant_resource_total;
        let prey_summary = self.prey.step(|n| plant_total / (n.max(1) as f64));

        PredatorPreyStep {
            tick: self.t,
            prey_count: prey_summary.population,
            predator_count: predator_summary.population,
            prey_killed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::OrganismConfig;

    fn small_cfg() -> PredatorPreyConfig {
        PredatorPreyConfig {
            prey_cfg: PopulationConfig {
                death_energy_threshold: 0.05,
                reproduction_energy_threshold: 0.8,
                reproduction_energy_cost: 0.4,
                organism_cfg: OrganismConfig::default(),
                ..Default::default()
            },
            predator_cfg: PopulationConfig {
                death_energy_threshold: 0.05,
                reproduction_energy_threshold: 0.8,
                reproduction_energy_cost: 0.4,
                organism_cfg: OrganismConfig::default(),
                ..Default::default()
            },
            plant_resource_total: 0.3,
            predation_scale: 1.0,
            predation_efficiency: 0.5,
        }
    }

    #[test]
    fn neither_population_goes_negative_and_counts_stay_consistent() {
        let mut sim = PredatorPreySim::new(small_cfg(), 4, 3, 1);
        for _ in 0..300u64 {
            let step = sim.step();
            assert_eq!(step.prey_count, sim.prey.len());
            assert_eq!(step.predator_count, sim.predator.len());
            assert!(step.prey_killed <= 10_000); // sanity: no runaway value
        }
    }

    #[test]
    fn predators_can_starve_to_extinction_without_prey() {
        // No prey, no plant resource for prey to regrow from -- predators should not be able to
        // sustain themselves on a resource signal of ~0 forever.
        let mut cfg = small_cfg();
        cfg.plant_resource_total = 0.0;
        let mut sim = PredatorPreySim::new(cfg, 0, 3, 2);
        for _ in 0..2000u64 {
            sim.step();
            if sim.predator.is_empty() {
                break;
            }
        }
        assert!(
            sim.predator.is_empty(),
            "predators should starve without any prey ever appearing"
        );
    }
}
