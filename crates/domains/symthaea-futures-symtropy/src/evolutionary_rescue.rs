// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Futures Laboratory's third scenario family, per the Phase 2 addendum's item 1 checklist
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`): hunt for the external-forcing lever
//! *before* building a rung hierarchy. Here the lever was already found and validated —
//! `symthaea-alife`'s own `tests/phase7_evolutionary_rescue.rs` ("evolving genome survives longer
//! under a real worsening climate") unifies `EarthForcedEnvironment::with_secular_drift` (a
//! permanent solar-constant decline layered on the seasonal cycle) with a heritable, mutable
//! `Genome` — a real bifurcation collapse is guaranteed eventually (an intrinsic property of the
//! ice-albedo model, not something any genome can prevent), but an evolving population survives
//! measurably longer than a frozen one (73-226 ticks longer, confirmed in 8/8 seeds).
//!
//! ## Why this is a genuinely new forecasting target, not `ecological` relabeled
//!
//! `ecological`/`predator_prey` both ask "does extinction happen within horizon" from a
//! population-count signal alone. Here the ground truth also carries a heritable trait
//! (`forage_efficiency`) whose trajectory is informative about *how much longer* the population
//! will survive — a forecaster that could observe the trait's rise would have a genuinely
//! different, richer signal than population count alone. **This first version deliberately does
//! not expose that signal**: `PopulationCensusObservationPolicy` here reveals only a capped
//! population count, exactly like `ecological`'s policy of the same name — the trait/genome state
//! is oracle/evaluation-only (`EvolutionaryRescueGroundTruth::true_mean_forage_efficiency`),
//! never crossing the observation firewall. A future policy that reveals a noised trait signal is
//! the natural next step for this family specifically, not built here.
//!
//! ## Scope note
//!
//! Matches Phase 1's own precedent: this is the ground-truth adapter + observation firewall only
//! (mirroring how `ecological`/`predator_prey` each started) — no `TrajectoryGenerator`/rung
//! implementations exist for this family yet.

use symthaea_alife::{EarthForcedEnvironment, Population};

/// Hidden ground truth for one evolutionary-rescue scenario instance. No `Clone`/`Debug`/
/// `PartialEq` derive, matching `ecological::EcologicalGroundTruth`'s reasoning: `Population`
/// itself has none of these (it owns per-organism `ActiveInferenceAgent`s with no such derives
/// either).
pub struct EvolutionaryRescueGroundTruth {
    pub environment: EarthForcedEnvironment,
    pub population: Population,
    pub plant_resource_total: f64,
    tick: u64,
}

impl EvolutionaryRescueGroundTruth {
    pub fn new(
        environment: EarthForcedEnvironment,
        population: Population,
        plant_resource_total: f64,
    ) -> Self {
        Self {
            environment,
            population,
            plant_resource_total,
            tick: 0,
        }
    }

    /// Advances one tick, reproducing `tests/phase7_evolutionary_rescue.rs`'s exact coupling:
    /// `population.step(|n| environment.step() * plant_resource_total / n.max(1))`. Field-borrow-
    /// split (same pattern `ecological::EcologicalGroundTruth::step` uses) because a closure
    /// capturing `self` directly would collide with the second mutable borrow.
    pub fn step(&mut self) {
        self.tick += 1;
        let environment = &mut self.environment;
        let plant_resource_total = self.plant_resource_total;
        self.population
            .step(|n| environment.step() * plant_resource_total / (n.max(1) as f64));
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn is_extinct(&self) -> bool {
        self.population.organisms.is_empty()
    }

    pub fn true_population_count(&self) -> usize {
        self.population.organisms.len()
    }

    /// Oracle/evaluation-only accessor — the whole point of this scenario family, but
    /// deliberately never exposed through [`ObservationPolicy`](crate::ObservationPolicy) in this
    /// first version (see module docs).
    pub fn true_mean_forage_efficiency(&self) -> f64 {
        let n = self.population.organisms.len();
        if n == 0 {
            return 0.0;
        }
        self.population
            .organisms
            .iter()
            .map(|o| o.cfg.forage_efficiency)
            .sum::<f64>()
            / n as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvolutionaryRescueSample {
    pub sampled_alive_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvolutionaryRescueObservation {
    pub tick: u64,
    pub sample: Option<EvolutionaryRescueSample>,
}

/// Same design as `ecological::PopulationCensusObservationPolicy`: reports
/// `min(true population count, sample_size)` — a capped census, not identity-based tracking of
/// specific individuals. Leakage-safe because `sample_size` is a fixed construction-time
/// constant: any composition of individuals above the threshold yields an identical output, and
/// the exact count is only ever revealed once the population has genuinely dropped below
/// `sample_size`. Trait/genome state is never read here at all (see module docs).
pub struct PopulationCensusObservationPolicy {
    observation_frequency_ticks: u64,
    sample_size: usize,
}

impl PopulationCensusObservationPolicy {
    pub fn new(sample_size: usize, observation_frequency_ticks: u64) -> Self {
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            sample_size,
        }
    }
}

impl crate::ObservationPolicy for PopulationCensusObservationPolicy {
    type GroundTruth = EvolutionaryRescueGroundTruth;
    type Observation = EvolutionaryRescueObservation;

    fn observe(
        &mut self,
        truth: &EvolutionaryRescueGroundTruth,
        tick: u64,
    ) -> EvolutionaryRescueObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return EvolutionaryRescueObservation { tick, sample: None };
        }

        let count = truth.true_population_count().min(self.sample_size);
        EvolutionaryRescueObservation {
            tick,
            sample: Some(EvolutionaryRescueSample {
                sampled_alive_count: count,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObservationPolicy;
    use symthaea_alife::{InheritanceMode, OrganismConfig, PopulationConfig};

    const SEASONAL_PERIOD_TICKS: f64 = 200.0;
    const SECULAR_DRIFT_PER_TICK: f64 = -0.01;
    const PLANT_RESOURCE_TOTAL: f64 = 3.0;

    fn population_config(mutation_rate: f64) -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            mutation_rate,
            mutation_std: 0.05,
            inheritance: InheritanceMode::FromParent,
        }
    }

    fn truth_with(
        mutation_rate: f64,
        initial_count: usize,
        seed: u64,
    ) -> EvolutionaryRescueGroundTruth {
        let environment = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS)
            .with_secular_drift(SECULAR_DRIFT_PER_TICK);
        let population = Population::new(population_config(mutation_rate), initial_count, seed);
        EvolutionaryRescueGroundTruth::new(environment, population, PLANT_RESOURCE_TOTAL)
    }

    #[test]
    fn population_census_never_leaks_trait_state() {
        // Same population count (12), same seed -- only mutation_rate (hence trait trajectory)
        // differs. If trait state ever leaked, these observations would diverge.
        let frozen = truth_with(0.0, 12, 7);
        let evolving = truth_with(0.1, 12, 7);

        let mut policy_a = PopulationCensusObservationPolicy::new(20, 1);
        let mut policy_b = PopulationCensusObservationPolicy::new(20, 1);

        assert_eq!(policy_a.observe(&frozen, 0), policy_b.observe(&evolving, 0));
    }

    #[test]
    fn population_census_caps_regardless_of_true_count_above_threshold() {
        let small = truth_with(0.1, 15, 3);
        let large = truth_with(0.1, 30, 3);

        let mut policy_small = PopulationCensusObservationPolicy::new(10, 1);
        let mut policy_large = PopulationCensusObservationPolicy::new(10, 1);

        let obs_small = policy_small.observe(&small, 0);
        let obs_large = policy_large.observe(&large, 0);
        assert_eq!(obs_small, obs_large);
        assert_eq!(
            obs_small.sample.unwrap().sampled_alive_count,
            10,
            "both should be capped at sample_size, not the true (differing) counts"
        );
    }

    #[test]
    fn off_frequency_tick_reveals_nothing_regardless_of_ground_truth() {
        let alive_truth = truth_with(0.1, 12, 1);
        let mut extinct_truth = truth_with(0.1, 12, 1);
        // Force one to extinction, leave the other alive -- the policy still must not reveal
        // anything different for an off-frequency tick.
        extinct_truth.population.organisms.clear();

        let mut policy = PopulationCensusObservationPolicy::new(5, 10);
        let obs_alive = policy.observe(&alive_truth, 3);
        let obs_extinct = policy.observe(&extinct_truth, 3);

        assert_eq!(
            obs_alive,
            EvolutionaryRescueObservation {
                tick: 3,
                sample: None
            }
        );
        assert_eq!(obs_alive, obs_extinct);
    }
}
