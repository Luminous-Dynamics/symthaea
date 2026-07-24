// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolution across generations, per `ALIFE_PLAN_2026-07-08.md` Phase 4.
//!
//! A [`Genome`] is the heritable subset of an [`crate::OrganismConfig`]: traits that plausibly
//! matter for fitness and can be inherited (with mutation) from parent to offspring. Deliberately
//! small and scoped to traits with a real, defensible fitness story, rather than making every
//! config field heritable for its own sake:
//!
//! - `set_point`: how much energy the organism aims to maintain -- aiming higher risks more
//!   activity cost to get there; aiming lower is a smaller, safer margin against death.
//! - `forage_efficiency`: how much energy a unit of gathered resource actually yields --
//!   directly, monotonically fitness-relevant (Phase 1/3 already established default 0.15 is a
//!   marginal, knife-edge value; a population that evolves it upward should genuinely do better).
//! - `action_temperature`: `select_action`'s softmax exploration/exploitation balance --
//!   previously hardcoded to `symthaea-fep`'s own default, now a real, mutable trait.
//! - `perceptual_grain`: `Option<f64>`, `None` unless an organism was explicitly constructed
//!   as part of the Hoffman Fitness-Beats-Truth experiment
//!   (`tests/hoffman_fitness_beats_truth.rs`, per `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`).
//!   `None` reproduces this crate's exact pre-existing behavior (raw, unquantized resource
//!   observation, no added perceptual-resolution energy cost) and mutation never activates it --
//!   `Genome::mutate` only perturbs an already-`Some` value, so every organism/population that
//!   doesn't explicitly opt in is provably unaffected by this trait's existence. `Some(grain)`
//!   coarse-grains the resource observation to bucket width `grain` and charges a real Landauer
//!   cost for the resolved detail (see `metabolism::{quantize_to_grain,
//!   perceptual_resolution_bits}`) -- smaller `grain` is finer/costlier/closer to truth-tracking,
//!   larger `grain` is coarser/cheaper/closer to Hoffman's "interface" strategies.
//!
//! All other `OrganismConfig` fields (metabolic/activity costs, thermodynamic constants, death
//! threshold, `goal_precision`) are treated as fixed environmental/physiological constants
//! shared by the whole population, not individual traits -- a deliberate scope boundary, not an
//! oversight.

use crate::organism::OrganismConfig;

/// Bounds mutation can never push a trait outside of -- prevents pathological values (e.g.
/// negative efficiency, zero temperature) from a run of bad luck ending a lineage's viability
/// entirely via undefined behavior rather than via the real fitness landscape.
const SET_POINT_BOUNDS: (f64, f64) = (0.2, 0.95);
const FORAGE_EFFICIENCY_BOUNDS: (f64, f64) = (0.01, 2.0);
const ACTION_TEMPERATURE_BOUNDS: (f64, f64) = (0.1, 5.0);
/// Only applies once `perceptual_grain` is `Some` -- see the module docs above. `0.02` is ~50
/// distinguishable buckets across `[0,1]` (~5.6 bits); `0.5` is ~2 buckets (~1 bit).
const GRAIN_BOUNDS: (f64, f64) = (0.02, 0.5);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Genome {
    pub set_point: f64,
    pub forage_efficiency: f64,
    pub action_temperature: f64,
    pub perceptual_grain: Option<f64>,
}

impl Genome {
    pub fn from_config(cfg: &OrganismConfig) -> Self {
        Self {
            set_point: cfg.set_point,
            forage_efficiency: cfg.forage_efficiency,
            action_temperature: cfg.action_temperature,
            perceptual_grain: cfg.perceptual_grain,
        }
    }

    /// Returns a copy of `cfg` with this genome's trait values written in; every other field
    /// (costs, thermodynamic constants, thresholds) is preserved from `cfg` unchanged.
    pub fn apply_to(&self, mut cfg: OrganismConfig) -> OrganismConfig {
        cfg.set_point = self.set_point;
        cfg.forage_efficiency = self.forage_efficiency;
        cfg.action_temperature = self.action_temperature;
        cfg.perceptual_grain = self.perceptual_grain;
        cfg
    }

    /// Each trait independently mutates with probability `mutation_rate`, by a uniform
    /// perturbation in `[-mutation_std, +mutation_std]`, clamped to that trait's bounds.
    /// `rng_state` is advanced (iterated xorshift64, same algorithm used elsewhere in this
    /// crate) -- deliberately a plain `&mut u64`, not a shared RNG type, matching
    /// `predator_prey.rs`'s precedent of each concern owning an independent stream.
    ///
    /// `perceptual_grain` is the one exception to "always mutates": a `None` value is left
    /// `None` (never spontaneously activated by mutation), while a `Some(v)` perturbs `v` within
    /// [`GRAIN_BOUNDS`] exactly like the other traits. See the module docs for why.
    pub fn mutate(&self, rng_state: &mut u64, mutation_rate: f64, mutation_std: f64) -> Self {
        let mut maybe_mutate = |value: f64, bounds: (f64, f64)| -> f64 {
            if next_unit(rng_state) < mutation_rate {
                let perturbation = (next_unit(rng_state) - 0.5) * 2.0 * mutation_std;
                (value + perturbation).clamp(bounds.0, bounds.1)
            } else {
                value
            }
        };

        let set_point = maybe_mutate(self.set_point, SET_POINT_BOUNDS);
        let forage_efficiency = maybe_mutate(self.forage_efficiency, FORAGE_EFFICIENCY_BOUNDS);
        let action_temperature = maybe_mutate(self.action_temperature, ACTION_TEMPERATURE_BOUNDS);
        let perceptual_grain = self.perceptual_grain.map(|v| maybe_mutate(v, GRAIN_BOUNDS));

        Self {
            set_point,
            forage_efficiency,
            action_temperature,
            perceptual_grain,
        }
    }
}

fn next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_config_preserves_genome() {
        let cfg = OrganismConfig::default();
        let genome = Genome::from_config(&cfg);
        let cfg2 = genome.apply_to(cfg);
        assert_eq!(genome, Genome::from_config(&cfg2));
    }

    #[test]
    fn apply_to_preserves_non_genome_fields() {
        let cfg = OrganismConfig::default();
        let mutant = Genome {
            set_point: 0.5,
            forage_efficiency: 0.9,
            action_temperature: 2.0,
            perceptual_grain: Some(0.1),
        };
        let mutated_cfg = mutant.apply_to(cfg);
        assert_eq!(mutated_cfg.metabolic_cost, cfg.metabolic_cost);
        assert_eq!(
            mutated_cfg.death_energy_threshold,
            cfg.death_energy_threshold
        );
        assert_eq!(mutated_cfg.set_point, 0.5);
        assert_eq!(mutated_cfg.perceptual_grain, Some(0.1));
    }

    #[test]
    fn mutation_never_leaves_bounds() {
        let genome = Genome {
            set_point: 0.5,
            forage_efficiency: 0.5,
            action_temperature: 1.0,
            perceptual_grain: Some(0.2),
        };
        let mut rng_state = 12345u64;
        for _ in 0..10_000 {
            let mutated = genome.mutate(&mut rng_state, 1.0, 10.0); // always-mutate, huge perturbation
            assert!((SET_POINT_BOUNDS.0..=SET_POINT_BOUNDS.1).contains(&mutated.set_point));
            assert!(
                (FORAGE_EFFICIENCY_BOUNDS.0..=FORAGE_EFFICIENCY_BOUNDS.1)
                    .contains(&mutated.forage_efficiency)
            );
            assert!(
                (ACTION_TEMPERATURE_BOUNDS.0..=ACTION_TEMPERATURE_BOUNDS.1)
                    .contains(&mutated.action_temperature)
            );
            let grain = mutated
                .perceptual_grain
                .expect("Some stays Some under mutation");
            assert!((GRAIN_BOUNDS.0..=GRAIN_BOUNDS.1).contains(&grain));
        }
    }

    #[test]
    fn zero_mutation_rate_never_changes_genome() {
        let genome = Genome {
            set_point: 0.5,
            forage_efficiency: 0.5,
            action_temperature: 1.0,
            perceptual_grain: Some(0.2),
        };
        let mut rng_state = 999u64;
        for _ in 0..1000 {
            let mutated = genome.mutate(&mut rng_state, 0.0, 1.0);
            assert_eq!(mutated, genome);
        }
    }

    #[test]
    fn perceptual_grain_none_is_never_activated_by_mutation() {
        // The load-bearing safety property: every pre-existing phase/test constructs organisms
        // with `perceptual_grain: None` (via `OrganismConfig::default()`) and never touches this
        // trait -- if mutation could spontaneously turn `None` into `Some`, those populations
        // would silently start paying a perceptual-resolution cost they were never designed
        // around. `always-mutate, huge perturbation` settings are deliberately hostile here.
        let genome = Genome {
            set_point: 0.5,
            forage_efficiency: 0.5,
            action_temperature: 1.0,
            perceptual_grain: None,
        };
        let mut rng_state = 42u64;
        for _ in 0..10_000 {
            let mutated = genome.mutate(&mut rng_state, 1.0, 10.0);
            assert_eq!(
                mutated.perceptual_grain, None,
                "None must never be activated by mutation"
            );
        }
    }
}
