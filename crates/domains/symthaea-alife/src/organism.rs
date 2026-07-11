// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A single organism: a Markov blanket wrapping one [`ActiveInferenceAgent`].
//!
//! Every tick genuinely calls `perceive` → `select_action` → `act` → `learn_from_outcome`
//! and consumes every result (`ALIFE_PLAN_2026-07-08.md` Phase 0 §0b — the anti-theater rule).
//! There is no flag to disable perception here: `Organism` always perceives for real. The
//! "does perception matter" comparison in Phase 0's ground-truth tests uses a separate,
//! non-perceiving *constant* belief as the baseline, not a crippled `Organism` — production
//! code shouldn't carry a theater on/off switch even for testing purposes.

use serde::{Deserialize, Serialize};
use symthaea_fep::markov_blanket::{BlanketPermeability, MarkovBoundaryOperator, MarkovPartition};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

use crate::metabolism::{landauer_minimum, prigogine_dissipation_cost, shannon_entropy_bits};
use crate::types::BoundaryModulators;

/// The two behaviors an [`Organism`] can select between.
///
/// Kept intentionally small and biology-plain rather than reusing `MotorCommandType`
/// (`AttentionShift`, `ReflectionInitiate`, ...) — those are cognition-flavored labels from a
/// different context; an organism's action space here is literally "spend effort gathering
/// resources" vs "conserve."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Forage,
    Rest,
}

impl Action {
    pub const COUNT: usize = 2;

    pub fn from_index(i: usize) -> Self {
        if i == 0 { Action::Forage } else { Action::Rest }
    }

    pub fn index(self) -> usize {
        match self {
            Action::Forage => 0,
            Action::Rest => 1,
        }
    }
}

/// Metabolic/behavioral parameters for an [`Organism`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrganismConfig {
    /// Preferred (homeostatic set-point) energy level, in `[0, 1]`.
    pub set_point: f64,
    /// Baseline energy cost paid every tick regardless of action.
    pub metabolic_cost: f64,
    /// Additional energy cost paid only when foraging.
    pub forage_activity_cost: f64,
    /// How efficiently a gated resource observation converts into energy gain when foraging.
    pub forage_efficiency: f64,
    /// Precision passed to `ActiveInferenceAgent::set_goals` for the energy preference.
    pub goal_precision: f64,
    /// Effective temperature (alife energy units) used by the Landauer/Prigogine physics --
    /// see `metabolism` module docs. A fixed constant here, deliberately: Phase 3 ports the
    /// physics *formulas*, not the main crate's adaptive-temperature machinery.
    pub effective_temperature: f64,
    /// Transport-coefficient-like constant scaling Prigogine dissipation cost with how much
    /// homeostatic order is being maintained. See `metabolism::prigogine_dissipation_cost`.
    pub dissipation_rate: f64,
    /// Energy at or below which the organism is dead (`Organism::is_dead`) -- a hard,
    /// intrinsic property of the organism itself (Phase 3b), not only an emergent side effect
    /// of `Population`'s own bookkeeping (which has its own, possibly different, threshold).
    pub death_energy_threshold: f64,
    /// Softmax temperature for `select_action`'s expected-free-energy-weighted action choice
    /// (`symthaea_fep::ActiveInferenceAgentConfig::action_temperature`) -- higher means more
    /// random/exploratory, lower means more greedily exploitative. Previously hardcoded to
    /// `symthaea-fep`'s own default (1.0) via `..Default::default()`; exposed here so Phase 4
    /// can treat it as a heritable, mutable trait (see `genome`).
    pub action_temperature: f64,
}

impl Default for OrganismConfig {
    fn default() -> Self {
        Self {
            set_point: 0.8,
            metabolic_cost: 0.01,
            forage_activity_cost: 0.01,
            forage_efficiency: 0.15,
            goal_precision: 2.0,
            effective_temperature: 1.0,
            // See metabolism::K_ALIFE_BOLTZMANN's doc comment -- scaled down ~10x from an
            // initial guess that was measured (not assumed) to roughly double the baseline
            // metabolic_cost outright.
            dissipation_rate: 0.001,
            death_energy_threshold: 0.05,
            action_temperature: 1.0,
        }
    }
}

/// Telemetry from one [`Organism::tick`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismTick {
    /// The true exogenous resource level this tick (before any gating).
    pub resource_observed: f64,
    /// The organism's current belief about the resource level.
    pub belief_resource: f64,
    /// The organism's current belief about its own energy level.
    pub belief_energy: f64,
    /// Actual physiological energy after this tick's action.
    pub energy: f64,
    /// Action actually executed (0 = forage, 1 = rest).
    pub action: usize,
    /// Current variational free energy.
    pub free_energy: f64,
    /// Current blanket permeability.
    pub permeability: BlanketPermeability,
    /// |predicted resource observation from `act()` − actual resource level| — consumes
    /// `act()`'s return value instead of discarding it.
    pub act_prediction_error: f64,
    /// Real bits of information processed this tick (perception's `belief_change` +
    /// action-selection's Shannon entropy) -- what `physical_cost`'s Landauer term is charged
    /// against. Phase 3 telemetry, not decorative.
    pub bits_processed: f64,
    /// Landauer + Prigogine energy cost actually charged this tick (Phase 3b: a real,
    /// unavoidable thermodynamic floor added on top of `metabolic_cost`/`forage_activity_cost`,
    /// not an alternative to them).
    pub physical_cost: f64,
    /// Whether the organism is dead after this tick (`energy <= death_energy_threshold`).
    pub is_dead: bool,
}

/// A living system modeled as a Markov blanket minimizing free energy.
pub struct Organism {
    pub agent: ActiveInferenceAgent,
    pub boundary: MarkovBoundaryOperator,
    pub energy: f64,
    pub cfg: OrganismConfig,
    /// The true resource level from this organism's most recent `tick()` -- kept so
    /// Phase 2 coalition analysis can evaluate a pooled belief against a real observation
    /// without needing the caller to separately track it.
    pub last_resource_observed: f64,
}

impl Organism {
    /// Observation layout: `[resource_level, energy_level]` (exteroceptive + interoceptive).
    /// Phase 0 gates both uniformly through the blanket for simplicity; a real
    /// exteroceptive/interoceptive permeability asymmetry is left to a later phase.
    const OBS_DIM: usize = 2;
    const STATE_DIM: usize = 2;

    pub fn new(cfg: OrganismConfig, seed: u64) -> Self {
        let agent_cfg = ActiveInferenceAgentConfig {
            state_dim: Self::STATE_DIM,
            obs_dim: Self::OBS_DIM,
            num_actions: Action::COUNT,
            action_temperature: cfg.action_temperature,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_cfg);
        agent.set_rng_seed(seed);
        // Prefer a moderate resource reading and a high (near-set-point) energy reading.
        agent.set_goals(vec![0.5, cfg.set_point], cfg.goal_precision);

        let boundary = MarkovBoundaryOperator::new(MarkovPartition {
            internal_dim: Self::STATE_DIM,
            sensory_dim: 1,
            active_dim: 1,
        });

        Self {
            agent,
            boundary,
            energy: cfg.set_point,
            cfg,
            last_resource_observed: 0.5,
        }
    }

    /// A hard, checkable constraint (Phase 3b) -- not an FEP preference that can be talked out
    /// of. True once energy has fallen to or below `cfg.death_energy_threshold`.
    pub fn is_dead(&self) -> bool {
        self.energy <= self.cfg.death_energy_threshold
    }

    /// One full perceive → select_action → act → learn_from_outcome cycle.
    ///
    /// `forced_action`, when `Some`, overrides the agent's own `select_action()` choice for the
    /// real-world consequence and the learning step — used only by Phase 0's ground-truth test
    /// comparing FEP-guided vs. uniform-random action selection. Perception and learning always
    /// run for real regardless of `forced_action`.
    pub fn tick(&mut self, resource_level: f64, forced_action: Option<usize>) -> OrganismTick {
        // 1. Boundary permeability from the organism's own physiological deficit.
        let deficit = (self.cfg.set_point - self.energy).max(0.0) / self.cfg.set_point.max(1e-6);
        let modulators = BoundaryModulators::from_energy_deficit(deficit);
        let permeability = self
            .boundary
            .compute_permeability(&modulators.to_permeability_inputs())
            .clone();

        // 2. Gate the raw observation through the blanket toward the current prior.
        let raw_obs = Observation::new(vec![resource_level, self.energy], 1.0, "resource+energy");
        let gated_obs = self.boundary.gate_observation(&raw_obs, &self.agent.belief);

        // 3. Perceive: update belief from the gated observation. Consumed via belief_resource
        //    below, and via belief_change feeding Phase 3's real Landauer bit-count (not
        //    discarded).
        let perception = self.agent.perceive(&gated_obs);

        // 4. Select action (always run for real; forced_action only overrides which one is
        //    actually executed on the environment, for the Phase-0 baseline comparison).
        let selection = self.agent.select_action();
        let action_index = forced_action.unwrap_or(selection.action);

        // 5. Act: predicted outcome, consumed as a real prediction-error metric.
        let predicted = self.agent.act(action_index);
        let predicted_resource = predicted
            .expected_observation
            .first()
            .copied()
            .unwrap_or(0.5);

        // 6. Real-world consequence: only foraging can gain energy, and it costs more to do.
        // Deliberately uses the true `resource_level`, not `gated_obs` -- the blanket gates what
        // the organism *believes*, not what it can physically metabolize. Using the belief-gated
        // value here would let a slowly-adapting belief "hallucinate" real energy from a resource
        // that has actually crashed to near zero, for as long as belief lags the true world state
        // (found via Phase 1's breakeven calibration collapsing toward ~0 -- an organism was
        // living off its own stale optimism, not the environment).
        let foraging = action_index == Action::Forage.index();
        let gained = if foraging {
            self.cfg.forage_efficiency * resource_level * permeability.active
        } else {
            0.0
        };
        let activity_cost = if foraging {
            self.cfg.forage_activity_cost
        } else {
            0.0
        };

        // 6b. Phase 3: real thermodynamic floor, charged on top of the config-driven costs
        // above, not instead of them. `bits_processed` is real, not decorative -- perception's
        // own `belief_change` (magnitude of belief actually written this tick) plus the Shannon
        // entropy of the action-selection distribution actually used (how many bits of
        // uncertainty were resolved by committing to one action). `order_maintained` reuses
        // `deficit` from step 1: staying close to the homeostatic set-point *is* the order this
        // organism is dissipating energy to maintain (Prigogine).
        let bits_processed =
            perception.belief_change + shannon_entropy_bits(&selection.action_probabilities);
        let order_maintained = 1.0 - deficit;
        let physical_cost = landauer_minimum(bits_processed, self.cfg.effective_temperature)
            + prigogine_dissipation_cost(
                order_maintained,
                self.cfg.effective_temperature,
                self.cfg.dissipation_rate,
            );

        self.energy =
            (self.energy + gained - self.cfg.metabolic_cost - activity_cost - physical_cost)
                .clamp(0.0, 1.0);

        // 7. Learn from the actual post-action outcome (internally perceives again with the
        //    real consequence, per ActiveInferenceAgent::learn_from_outcome's own contract).
        let actual_obs =
            Observation::new(vec![resource_level, self.energy], 1.0, "resource+energy");
        self.agent.learn_from_outcome(action_index, &actual_obs);

        self.last_resource_observed = resource_level;

        OrganismTick {
            resource_observed: resource_level,
            belief_resource: self.agent.belief.mean[0],
            belief_energy: self.agent.belief.mean[1],
            energy: self.energy,
            action: action_index,
            free_energy: self.agent.current_free_energy(),
            permeability,
            act_prediction_error: (predicted_resource - resource_level).abs(),
            bits_processed,
            physical_cost,
            is_dead: self.is_dead(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;

    #[test]
    fn energy_stays_in_unit_range_over_many_ticks() {
        let mut organism = Organism::new(OrganismConfig::default(), 1);
        let env = Environment::default();
        for t in 0..2000u64 {
            let tick = organism.tick(env.resource_at(t), None);
            assert!(
                (0.0..=1.0).contains(&tick.energy),
                "energy out of range at t={t}: {}",
                tick.energy
            );
        }
    }

    #[test]
    fn action_index_is_always_valid() {
        let mut organism = Organism::new(OrganismConfig::default(), 2);
        let env = Environment::default();
        for t in 0..500u64 {
            let tick = organism.tick(env.resource_at(t), None);
            assert!(tick.action < Action::COUNT);
        }
    }

    #[test]
    fn perceiving_updates_belief_away_from_default() {
        // A genuinely inert agent starts at belief mean 0.5 in every dimension. If perceive()
        // were theater (called but not actually load-bearing), belief would stay pinned there.
        let mut organism = Organism::new(OrganismConfig::default(), 3);
        let env = Environment::default();
        for t in 0..50u64 {
            organism.tick(env.resource_at(t), None);
        }
        assert!(
            (organism.agent.belief.mean[0] - 0.5).abs() > 0.01,
            "belief_resource didn't move from its default: {}",
            organism.agent.belief.mean[0]
        );
    }
}
