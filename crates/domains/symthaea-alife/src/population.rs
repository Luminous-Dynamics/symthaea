// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A population of [`Organism`]s sharing a resource pool, per `ALIFE_PLAN_2026-07-08.md` Phase 1.
//!
//! `Population` is deliberately content-agnostic about *what* the resource is (plants, prey,
//! anything else): the caller supplies a `resource_for(current_count) -> f64` closure each tick,
//! so density-dependence (a smaller per-capita share as population grows) is whatever the caller
//! wires in. This lets the same type drive both Phase 1a (one species, a shared plant pool) and
//! Phase 1b (two coupled species, `crate::predator_prey`) without duplicating birth/death logic.
//!
//! Reproduction and death are driven entirely by each [`Organism`]'s own real energy trajectory
//! (itself the product of its own `select_action`/`perceive` loop) — this module adds no FEP
//! machinery of its own, only the population-level bookkeeping around it.

use crate::genome::Genome;
use crate::organism::{Action, Organism, OrganismConfig};

/// How an offspring's genome is chosen at reproduction, per `ALIFE_PLAN_2026-07-08.md` Phase 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InheritanceMode {
    /// Offspring inherit (mutated) from the parent that actually reproduced -- real selection:
    /// a genome only propagates by way of an organism that stayed alive long enough to reach
    /// `reproduction_energy_threshold`.
    #[default]
    FromParent,
    /// Offspring inherit (mutated) from a uniformly-random *current* population member,
    /// independent of who actually reproduced -- deliberately breaks the fitness→inheritance
    /// link. Exists only as Phase 4c's falsifiable control, never a "better" mode: real
    /// selection pressure (death, reproduction thresholds) still applies, but a successful
    /// reproduction event no longer preferentially propagates *that individual's* genome.
    RandomPeer,
}

/// Birth/death thresholds and the per-organism config newborns inherit.
///
/// `Default` exists so existing call sites can add `..Default::default()` to pick up Phase 4's
/// new `mutation_rate`/`mutation_std`/`inheritance` fields without re-specifying the rest --
/// **do not rely on it for the threshold fields themselves** (they default to `0.0`, which
/// would make every organism reproduce on its very first tick).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PopulationConfig {
    /// An organism with energy at or below this dies and is removed.
    pub death_energy_threshold: f64,
    /// An organism with energy at or above this reproduces this tick.
    pub reproduction_energy_threshold: f64,
    /// Energy the parent is left with (and the offspring starts with) after reproducing — a
    /// real cost, not a free clone, so reproduction can't happen every tick indefinitely.
    pub reproduction_energy_cost: f64,
    /// Config the *initial* population is seeded with. Offspring no longer simply clone this
    /// (Phase 4) — they inherit a mutated `Genome` from a parent chosen per `inheritance`; only
    /// the non-heritable fields (costs, thermodynamic constants, thresholds) come from here.
    pub organism_cfg: OrganismConfig,
    /// Probability each heritable trait mutates at a birth event. `0.0` (Phase 0-3's implicit
    /// behavior) means offspring are exact genome copies of their chosen parent.
    pub mutation_rate: f64,
    /// Magnitude of a mutation, when one occurs (`Genome::mutate`'s `mutation_std`).
    pub mutation_std: f64,
    pub inheritance: InheritanceMode,
}

/// What happened during one [`Population::step`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepSummary {
    /// Population size after this tick's births/deaths.
    pub population: usize,
    /// How many organisms actually chose to forage this tick (their own `select_action()`
    /// result, not a scripted rate) — exposed so callers like `predator_prey` can turn real
    /// foraging decisions into a real consequence for a coupled species.
    pub forage_count: usize,
    pub births_this_tick: u64,
    pub deaths_this_tick: u64,
}

pub struct Population {
    pub organisms: Vec<Organism>,
    pub cfg: PopulationConfig,
    next_seed: u64,
    /// Stream for mutation and (in `InheritanceMode::RandomPeer`) parent-selection randomness --
    /// deliberately independent of any `Organism`'s own RNG, same precedent as
    /// `predator_prey.rs`'s `PredatorPreySim::rng_state`.
    rng_state: u64,
    pub total_births: u64,
    pub total_deaths: u64,
}

impl Population {
    pub fn new(cfg: PopulationConfig, initial_count: usize, seed_base: u64) -> Self {
        let organisms = (0..initial_count)
            .map(|i| Organism::new(cfg.organism_cfg, seed_base.wrapping_add(i as u64).max(1)))
            .collect();
        Self {
            organisms,
            cfg,
            next_seed: seed_base.wrapping_add(initial_count as u64).max(1),
            rng_state: seed_base.wrapping_add(1_000_011).max(1),
            total_births: 0,
            total_deaths: 0,
        }
    }

    fn next_unit(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Remove up to `n` of the lowest-energy organisms (predation picks off the weak first —
    /// a real, if simplified, ecological detail). Returns how many were actually removed,
    /// capped by the current population size.
    pub fn cull_weakest(&mut self, n: usize) -> usize {
        let n = n.min(self.organisms.len());
        if n == 0 {
            return 0;
        }
        self.organisms
            .sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        self.organisms.drain(0..n);
        self.total_deaths += n as u64;
        n
    }

    /// One tick for every living organism: real `perceive`/`select_action`/`act`/
    /// `learn_from_outcome`, then birth/death from the resulting energy.
    ///
    /// `resource_for(n)` is called once with the population size *before* this tick's births/
    /// deaths, and its result is given to every organism as their `resource_level` observation
    /// this tick — the shared-pool coupling that produces density-dependence.
    ///
    /// `FnMut`, not `Fn` -- lets a caller drive a genuinely stateful resource source (e.g.
    /// Phase 5's `EarthForcedEnvironment`, which integrates real physics forward one tick per
    /// call) rather than requiring every resource source to be a pure function of `n`. Every
    /// existing caller passes a pure closure, which trivially also satisfies `FnMut`, so this is
    /// a non-breaking relaxation.
    pub fn step(&mut self, mut resource_for: impl FnMut(usize) -> f64) -> StepSummary {
        let n = self.organisms.len();
        let resource = resource_for(n);

        let mut forage_count = 0u64;
        let mut births_this_tick = 0u64;
        let mut deaths_this_tick = 0u64;
        let mut newborns = Vec::new();

        let mut i = 0;
        while i < self.organisms.len() {
            let tick = self.organisms[i].tick(resource, None);
            if tick.action == Action::Forage.index() {
                forage_count += 1;
            }

            if tick.energy <= self.cfg.death_energy_threshold {
                self.organisms.remove(i);
                self.total_deaths += 1;
                deaths_this_tick += 1;
                continue; // next element has shifted into position i
            }

            if tick.energy >= self.cfg.reproduction_energy_threshold {
                self.organisms[i].energy = self.cfg.reproduction_energy_cost;

                // Phase 4: the offspring's genome comes from a parent chosen per
                // `self.cfg.inheritance`, mutated, then reapplied to the population's shared
                // non-heritable constants (costs, thermodynamic constants, thresholds -- see
                // `Genome`'s module docs for exactly which fields are heritable).
                let parent_genome = match self.cfg.inheritance {
                    InheritanceMode::FromParent => Genome::from_config(&self.organisms[i].cfg),
                    InheritanceMode::RandomPeer => {
                        let r = self.next_unit();
                        let peer_idx = ((r * self.organisms.len() as f64) as usize)
                            .min(self.organisms.len() - 1);
                        Genome::from_config(&self.organisms[peer_idx].cfg)
                    }
                };
                let offspring_genome = parent_genome.mutate(
                    &mut self.rng_state,
                    self.cfg.mutation_rate,
                    self.cfg.mutation_std,
                );
                let offspring_cfg = offspring_genome.apply_to(self.cfg.organism_cfg);

                let mut offspring = Organism::new(offspring_cfg, self.next_seed);
                self.next_seed = self.next_seed.wrapping_add(1).max(1);
                offspring.energy = self.cfg.reproduction_energy_cost;
                newborns.push(offspring);
                self.total_births += 1;
                births_this_tick += 1;
            }

            i += 1;
        }

        self.organisms.extend(newborns);

        StepSummary {
            population: self.organisms.len(),
            forage_count: forage_count as usize,
            births_this_tick,
            deaths_this_tick,
        }
    }

    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        }
    }

    /// A traced diagnostic run (see `ALIFE_PLAN_2026-07-08.md` Phase 1 dev notes / commit history)
    /// found `OrganismConfig::default()`'s `forage_efficiency: 0.15` puts an organism's
    /// break-even resource share close to 0.9 -- because the untrained/lightly-trained policy
    /// forages only ~50% of the time, not ~100% as a naive mean-field estimate assumes. That's
    /// fine for Phase 0 (no reproduction/death was in play), but leaves Phase 1 populations on a
    /// knife-edge where even a flat, undivided resource of 0.5 was sub-breakeven and the
    /// population went extinct. Phase 1 tests use this more sustainable efficiency instead, so
    /// growth is a robust, unambiguous outcome under genuinely abundant resources rather than a
    /// coin flip -- Phase 0's already-verified tests are untouched (they use `cfg()` above).
    fn sustainable_cfg() -> PopulationConfig {
        PopulationConfig {
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..cfg()
        }
    }

    #[test]
    fn population_never_goes_negative_and_reports_consistent_counts() {
        let mut pop = Population::new(cfg(), 3, 1);
        for _ in 0..500u64 {
            let summary = pop.step(|n| 0.3 / (n.max(1) as f64));
            assert_eq!(summary.population, pop.len());
            assert!(summary.forage_count <= summary.population + summary.deaths_this_tick as usize);
        }
    }

    #[test]
    fn cull_weakest_never_removes_more_than_present() {
        let mut pop = Population::new(cfg(), 4, 7);
        let removed = pop.cull_weakest(100);
        assert_eq!(removed, 4);
        assert!(pop.is_empty());
        assert_eq!(pop.cull_weakest(5), 0);
    }

    #[test]
    fn abundant_resources_grow_the_population() {
        // A generous, constant (NOT shared/divided by population size) per-capita resource is
        // deliberately unbounded growth -- and it compounds fast (~15-20 ticks per doubling, so
        // even a few hundred ticks means millions of organisms and a test that never finishes).
        // A couple of doubling times is enough to prove births can outpace deaths at all;
        // density-dependence is exactly what Phase 1a's shared-pool test covers instead.
        let mut pop = Population::new(sustainable_cfg(), 2, 42);
        for _ in 0..60u64 {
            pop.step(|_n| 0.5); // fixed generous per-capita resource, not shared/divided
        }
        assert!(
            pop.len() > 2,
            "population should have grown, got {}",
            pop.len()
        );
    }
}
