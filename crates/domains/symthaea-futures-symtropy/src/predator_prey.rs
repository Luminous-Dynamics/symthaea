// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Futures Laboratory's second scenario family: predator extinction forecasting on
//! `symthaea-alife`'s coupled predator/prey scenario
//! (`ALIFE_PLAN_2026-07-08.md` Phase 1b, `symthaea_alife::predator_prey`), per
//! `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s "pick 2, not all" scenario-family
//! requirement.
//!
//! ## Scope — deliberately smaller than `ecological`, and that's disclosed, not hidden
//!
//! This module gives the second family its ground truth type and observation firewall only —
//! `symthaea_alife::PredatorPreySim` needs no new wrapper (unlike `ecological`'s
//! `EcologicalGroundTruth`, `PredatorPreySim` already combines both populations into one owned
//! type with a `step()` method), and [`PredatorExtinctionObservationPolicy`] follows the same
//! fixed-cohort-by-`AgentId` pattern already proven for `ecological`. **No baseline-rung
//! (`TrajectoryGenerator`) implementations exist for this family yet** — the plan's Phase 1
//! deliverables list commits to "adapters for the 2 chosen scenario families + the leakage test
//! suite," which this satisfies; re-deriving all six rungs against a second target is real,
//! separate follow-up work, not attempted in this pass.
//!
//! ## Why predator extinction, not prey extinction or something else
//!
//! `PredatorPreySim`'s own module docs describe a real (if not mass-conserving) coupled
//! oscillation: predators perceive prey density as their resource signal, and however many
//! predators actually forage removes that many prey. Predators are the side genuinely at risk of
//! boom-bust extinction in this dynamic (a resource-dependent population one step removed from
//! the driving signal), making "does the predator population go extinct within horizon" a
//! meaningfully different ground-truth story than `ecological`'s direct climate-forced
//! single-species collapse — not just the same question asked of a differently-named species.
//!
//! ## A new leakage boundary this family introduces: the *other* species
//!
//! `ecological` only had one population to keep out of the observation. Here, the ground truth
//! has **two** coupled populations — [`PredatorExtinctionObservationPolicy::observe`] must never
//! let prey state leak into a predator-extinction forecast, in addition to never letting
//! untracked predators leak. The leakage test below covers exactly this: a different prey
//! population entirely, same predator population, must yield an identical observation.

use symthaea_alife::PredatorPreySim;

use crate::ObservationPolicy;

/// The observation for this scenario family: the tracked predator cohort's alive count, or
/// `None` on ticks the policy doesn't observe. **Disclosed simplification**: no sensor noise
/// modeled (unlike `ecological::EcologicalSample`) and prey state is never observed at all in
/// this first version — both are legitimate future extensions, not built here.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PredatorPreyObservation {
    pub tick: u64,
    pub sampled_predator_alive_count: Option<usize>,
}

/// The observation firewall for the predator/prey scenario family. Fixed cohort by `AgentId`,
/// same reasoning as `ecological::ExtinctionObservationPolicy`: `predator_sample_size` is
/// computed once from the known `initial_predator_count` (a legitimate experiment-setup
/// parameter), never recomputed from the live predator count — recomputing per-tick would leak
/// that count back out arithmetically.
pub struct PredatorExtinctionObservationPolicy {
    observation_frequency_ticks: u64,
    predator_sample_size: u64,
}

impl PredatorExtinctionObservationPolicy {
    pub fn new(
        initial_predator_count: usize,
        sample_fraction: f64,
        observation_frequency_ticks: u64,
    ) -> Self {
        let predator_sample_size =
            (sample_fraction.clamp(0.0, 1.0) * initial_predator_count as f64).round() as u64;
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            predator_sample_size,
        }
    }
}

impl ObservationPolicy for PredatorExtinctionObservationPolicy {
    type GroundTruth = PredatorPreySim;
    type Observation = PredatorPreyObservation;

    fn observe(&mut self, truth: &PredatorPreySim, tick: u64) -> PredatorPreyObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return PredatorPreyObservation {
                tick,
                sampled_predator_alive_count: None,
            };
        }

        // truth.prey is never read anywhere in this function -- see the leakage test below,
        // which exists specifically to catch a future change that violates this.
        let count = truth
            .predator
            .organisms
            .iter()
            .filter(|o| o.id.raw() < self.predator_sample_size)
            .count();

        PredatorPreyObservation {
            tick,
            sampled_predator_alive_count: Some(count),
        }
    }
}

/// A second observation policy for the same ground truth, fixing the cohort/population mismatch
/// [`PredatorExtinctionObservationPolicy`] was found to share with `ecological`'s original
/// `ExtinctionObservationPolicy` (see `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s
/// predator/prey backtest finding, 2026-07-26): fixed-original-cohort tracking reports a signal
/// that can decline to zero even while the whole predator population thrives via untracked
/// offspring.
///
/// Same fix as `ecological::PopulationCensusObservationPolicy`, same reasoning: report
/// `min(true predator population count, sample_size)` — a capped census, not identity-based
/// tracking of specific individuals. `sample_size` is a fixed construction-time constant, never
/// derived from live population count, so this can never reveal "population is exactly N" for
/// N > sample_size. Reuses [`PredatorPreyObservation`] unchanged, so every existing rung built
/// against `PredatorExtinctionObservationPolicy` can be re-tested against this policy with zero
/// code changes.
pub struct PredatorPopulationCensusObservationPolicy {
    observation_frequency_ticks: u64,
    sample_size: usize,
}

impl PredatorPopulationCensusObservationPolicy {
    pub fn new(sample_size: usize, observation_frequency_ticks: u64) -> Self {
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            sample_size,
        }
    }
}

impl ObservationPolicy for PredatorPopulationCensusObservationPolicy {
    type GroundTruth = PredatorPreySim;
    type Observation = PredatorPreyObservation;

    fn observe(&mut self, truth: &PredatorPreySim, tick: u64) -> PredatorPreyObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return PredatorPreyObservation {
                tick,
                sampled_predator_alive_count: None,
            };
        }

        // truth.prey is never read here either -- same leakage boundary as the fixed-cohort
        // policy above.
        let count = truth.predator.len().min(self.sample_size);

        PredatorPreyObservation {
            tick,
            sampled_predator_alive_count: Some(count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig};

    fn scenario_config() -> PredatorPreyConfig {
        let prey_cfg = PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..Default::default()
        };
        let predator_cfg = PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..Default::default()
        };
        PredatorPreyConfig {
            prey_cfg,
            predator_cfg,
            plant_resource_total: 3.0,
            predation_scale: 1.0,
            predation_efficiency: 0.3,
        }
    }

    #[test]
    fn prey_population_never_leaks_into_a_predator_observation() {
        // Same predator population and seed on both sides; wildly different prey population
        // (different initial count) -- if the policy ever reads `truth.prey`, this test would
        // catch it via a differing observation.
        let truth_a = PredatorPreySim::new(scenario_config(), 6, 6, 11);
        let truth_b = PredatorPreySim::new(scenario_config(), 40, 6, 11);

        let mut policy_a = PredatorExtinctionObservationPolicy::new(6, 0.5, 1);
        let mut policy_b = PredatorExtinctionObservationPolicy::new(6, 0.5, 1);

        assert_eq!(policy_a.observe(&truth_a, 0), policy_b.observe(&truth_b, 0));
    }

    #[test]
    fn tick_outside_observation_frequency_is_none_regardless_of_ground_truth() {
        let truth_a = PredatorPreySim::new(scenario_config(), 6, 6, 11);
        let truth_b = PredatorPreySim::new(scenario_config(), 6, 0, 11); // predators already extinct

        let mut policy_a = PredatorExtinctionObservationPolicy::new(6, 0.5, 5);
        let mut policy_b = PredatorExtinctionObservationPolicy::new(6, 0.5, 5);

        let obs_a = policy_a.observe(&truth_a, 3);
        let obs_b = policy_b.observe(&truth_b, 3);
        assert_eq!(obs_a, obs_b);
        assert_eq!(obs_a.sampled_predator_alive_count, None);
    }
}

#[cfg(test)]
mod population_census_tests {
    use super::*;
    use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig};

    fn scenario_config() -> PredatorPreyConfig {
        let organism_cfg = OrganismConfig {
            forage_efficiency: 0.6,
            ..OrganismConfig::default()
        };
        let pop_cfg = PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg,
            ..Default::default()
        };
        PredatorPreyConfig {
            prey_cfg: pop_cfg,
            predator_cfg: pop_cfg,
            plant_resource_total: 3.0,
            predation_scale: 1.0,
            predation_efficiency: 0.3,
        }
    }

    #[test]
    fn predator_population_above_threshold_reports_only_the_cap() {
        let truth_a = PredatorPreySim::new(scenario_config(), 6, 8, 11);
        let truth_b = PredatorPreySim::new(scenario_config(), 6, 20, 11);

        let mut policy_a = PredatorPopulationCensusObservationPolicy::new(5, 1);
        let mut policy_b = PredatorPopulationCensusObservationPolicy::new(5, 1);

        let obs_a = policy_a.observe(&truth_a, 0);
        let obs_b = policy_b.observe(&truth_b, 0);
        assert_eq!(obs_a, obs_b);
        assert_eq!(obs_a.sampled_predator_alive_count, Some(5));
    }

    #[test]
    fn prey_population_never_leaks_into_the_census_observation() {
        let truth_a = PredatorPreySim::new(scenario_config(), 6, 8, 11);
        let truth_b = PredatorPreySim::new(scenario_config(), 40, 8, 11);

        let mut policy_a = PredatorPopulationCensusObservationPolicy::new(5, 1);
        let mut policy_b = PredatorPopulationCensusObservationPolicy::new(5, 1);

        assert_eq!(policy_a.observe(&truth_a, 0), policy_b.observe(&truth_b, 0));
    }

    #[test]
    fn below_threshold_reveals_the_true_remaining_count_intentionally() {
        let truth = PredatorPreySim::new(scenario_config(), 6, 3, 11);
        let mut policy = PredatorPopulationCensusObservationPolicy::new(5, 1);

        let obs = policy.observe(&truth, 0);
        assert_eq!(obs.sampled_predator_alive_count, Some(3));
    }

    #[test]
    fn tick_outside_observation_frequency_is_none_regardless_of_ground_truth() {
        let truth_a = PredatorPreySim::new(scenario_config(), 6, 8, 11);
        let truth_b = PredatorPreySim::new(scenario_config(), 6, 0, 11);

        let mut policy_a = PredatorPopulationCensusObservationPolicy::new(5, 5);
        let mut policy_b = PredatorPopulationCensusObservationPolicy::new(5, 5);

        let obs_a = policy_a.observe(&truth_a, 3);
        let obs_b = policy_b.observe(&truth_b, 3);
        assert_eq!(obs_a, obs_b);
        assert_eq!(obs_a.sampled_predator_alive_count, None);
    }
}
