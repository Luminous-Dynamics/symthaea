// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! All six baseline rungs for the second scenario family — predator extinction forecasting on
//! `symthaea-futures-symtropy::predator_prey` — mirroring `ecological`'s design decisions
//! directly rather than re-deriving them:
//!
//! - Rungs 1/2/4 operate on a single [`PredatorPreyObservation`] snapshot; rung 3 needs a
//!   history (`type Observation = Vec<PredatorPreyObservation>`); rung 5 also needs a history
//!   (to replay through a fresh `ActiveInferenceAgent`); rung 6 needs the whole recorded
//!   trajectory (`type Observation = u64`, the tick being forecast from), same as `ecological`.
//! - Rung 5 uses the same Stage B `mask_observation`-based adapter and the same `num_actions: 1`
//!   + explicit `observe_transition()` calls `ecological::FepDrivenGenerator` needed to avoid the
//!   "perceive-only teaches nothing" trap (see that module's docs for the full explanation — not
//!   repeated here since the mechanism is identical, only the observation channel changed).
//! - Rung 4's closed-form independent-survival equation is identical in shape to
//!   `ecological::ScenarioMechanisticGenerator`; the disclosed simplification (member
//!   independence, ignoring this scenario's real density-dependent coupling through the shared
//!   prey-density signal) applies here too.
//!
//! **Not built here**: this family has no sensor noise (per
//! `symthaea-futures-symtropy::predator_prey`'s own disclosed scope) and no
//! `observed_mean_energy`/`observed_temperature` fields at all — `PredatorPreyObservation` only
//! ever carries `sampled_predator_alive_count`, so rung 5's observation channel is simpler by
//! construction, not by a further simplifying choice made here.

use symthaea_futures_core::{AbstentionReason, ForecastOutput, Horizon, TrajectoryGenerator};
use symthaea_futures_state::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, mask_observation,
};
use symthaea_futures_symtropy::predator_prey::PredatorPreyObservation;

use crate::BaselineRung;

const PREDATOR_EXTINCTION_WITHIN_HORIZON: &str = "predator_extinction_within_horizon";

fn boolean_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    p_true: f64,
) -> symthaea_futures_core::ForecastDistribution {
    crate::boolean_distribution(
        issued_at_tick,
        horizon,
        p_true,
        PREDATOR_EXTINCTION_WITHIN_HORIZON,
    )
}

/// Rung 1: "whatever's true now continues," on the tracked predator cohort's observed status.
pub struct PersistenceGenerator;

impl PersistenceGenerator {
    pub const RUNG: BaselineRung = BaselineRung::Persistence;
}

impl TrajectoryGenerator for PersistenceGenerator {
    type Observation = PredatorPreyObservation;

    fn generate(&self, observation: &PredatorPreyObservation, horizon: Horizon) -> ForecastOutput {
        let Some(count) = observation.sampled_predator_alive_count else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };
        let p_true = if count == 0 { 0.9 } else { 0.05 };
        ForecastOutput::Distribution(boolean_distribution(observation.tick, horizon, p_true))
    }
}

/// Rung 2: base rate across training seeds, ignoring the observation entirely.
pub struct HistoricalFrequencyGenerator {
    pub base_rate: f64,
}

impl HistoricalFrequencyGenerator {
    pub const RUNG: BaselineRung = BaselineRung::HistoricalFrequency;
}

impl TrajectoryGenerator for HistoricalFrequencyGenerator {
    type Observation = PredatorPreyObservation;

    fn generate(&self, observation: &PredatorPreyObservation, horizon: Horizon) -> ForecastOutput {
        ForecastOutput::Distribution(boolean_distribution(
            observation.tick,
            horizon,
            self.base_rate,
        ))
    }
}

/// Rung 3: a real OLS linear-trend fit on the tracked predator cohort's observed history.
/// **Disclosed simplification**: same linear-proportion probability mapping as
/// `ecological::SimpleStatisticalGenerator` — not a full prediction-interval treatment.
pub struct SimpleStatisticalGenerator;

impl SimpleStatisticalGenerator {
    pub const RUNG: BaselineRung = BaselineRung::SimpleStatistical;
}

impl TrajectoryGenerator for SimpleStatisticalGenerator {
    type Observation = Vec<PredatorPreyObservation>;

    fn generate(&self, history: &Vec<PredatorPreyObservation>, horizon: Horizon) -> ForecastOutput {
        let points: Vec<(f64, f64)> = history
            .iter()
            .filter_map(|obs| {
                obs.sampled_predator_alive_count
                    .map(|c| (obs.tick as f64, c as f64))
            })
            .collect();

        if points.len() < 2 {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        }

        let issued_at_tick = history.last().map(|o| o.tick).unwrap_or(0);
        let reference = points[0].1;
        if reference <= 0.0 {
            return ForecastOutput::Distribution(boolean_distribution(
                issued_at_tick,
                horizon,
                0.9,
            ));
        }

        let n = points.len() as f64;
        let x_mean = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
        let y_mean = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
        let denom: f64 = points.iter().map(|&(x, _)| (x - x_mean).powi(2)).sum();

        let slope = if denom > 0.0 {
            points
                .iter()
                .map(|&(x, y)| (x - x_mean) * (y - y_mean))
                .sum::<f64>()
                / denom
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        let target_tick = issued_at_tick as f64 + horizon.0 as f64;
        let projected = (intercept + slope * target_tick).max(0.0);
        let p_true = (1.0 - projected / reference).clamp(0.0, 1.0);

        ForecastOutput::Distribution(boolean_distribution(issued_at_tick, horizon, p_true))
    }
}

/// Rung 4: a real closed-form equation (independent per-member survival), not a fit. Same
/// disclosed simplification as `ecological::ScenarioMechanisticGenerator`: assumes member
/// independence, ignoring this scenario's real density-dependent predator/prey coupling.
pub struct ScenarioMechanisticGenerator {
    pub per_member_death_probability: f64,
}

impl ScenarioMechanisticGenerator {
    pub const RUNG: BaselineRung = BaselineRung::ScenarioMechanistic;
}

impl TrajectoryGenerator for ScenarioMechanisticGenerator {
    type Observation = PredatorPreyObservation;

    fn generate(&self, observation: &PredatorPreyObservation, horizon: Horizon) -> ForecastOutput {
        let Some(count) = observation.sampled_predator_alive_count else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };
        if count == 0 {
            return ForecastOutput::Distribution(boolean_distribution(
                observation.tick,
                horizon,
                0.9,
            ));
        }

        let p_survive_one_tick = 1.0 - self.per_member_death_probability.clamp(0.0, 1.0);
        let p_survives_horizon = p_survive_one_tick.powi(horizon.0 as i32);
        let p_dies_within_horizon = 1.0 - p_survives_horizon;
        let p_true = p_dies_within_horizon.powi(count as i32);

        ForecastOutput::Distribution(boolean_distribution(observation.tick, horizon, p_true))
    }
}

/// Rung 5: the FEP-driven ensemble. See `ecological::FepDrivenGenerator`'s module docs for the
/// full "perceive-only teaches nothing" trap and why `num_actions: 1` + explicit
/// `observe_transition` calls are required — identical mechanism here, only the observation
/// channel (predator cohort survival fraction, no energy/temperature channels at all) differs.
pub struct FepDrivenGenerator {
    pub agent_config: ActiveInferenceAgentConfig,
}

impl Default for FepDrivenGenerator {
    fn default() -> Self {
        Self {
            agent_config: ActiveInferenceAgentConfig {
                state_dim: 1,
                obs_dim: 1,
                num_actions: 1,
                ..ActiveInferenceAgentConfig::default()
            },
        }
    }
}

impl FepDrivenGenerator {
    pub const RUNG: BaselineRung = BaselineRung::FepDriven;
}

impl TrajectoryGenerator for FepDrivenGenerator {
    type Observation = Vec<PredatorPreyObservation>;

    fn generate(&self, history: &Vec<PredatorPreyObservation>, horizon: Horizon) -> ForecastOutput {
        let issued_at_tick = match history.last() {
            Some(o) => o.tick,
            None => {
                return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
            }
        };

        let Some(reference) = history
            .first()
            .and_then(|o| o.sampled_predator_alive_count)
            .map(|c| c as f64)
        else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };

        if reference <= 0.0 {
            return ForecastOutput::Distribution(boolean_distribution(
                issued_at_tick,
                horizon,
                0.9,
            ));
        }

        let mut agent = ActiveInferenceAgent::new(self.agent_config.clone());
        let mut prev_belief = agent.belief.clone();

        for obs in history {
            let raw_value = obs
                .sampled_predator_alive_count
                .map(|c| (c as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            let visibility = if obs.sampled_predator_alive_count.is_some() {
                1.0
            } else {
                0.0
            };

            let raw_obs = Observation::new(vec![raw_value], 1.0, "predator_survival_fraction");
            let masked = mask_observation(&raw_obs, &agent.belief, &[visibility]);

            agent.perceive(&masked);
            let new_belief = agent.belief.clone();
            agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
            prev_belief = new_belief;
        }

        let mut projected = agent.belief.clone();
        for _ in 0..horizon.0 {
            projected = agent.model.predict_next_state(&projected, 0);
        }
        let projected_fraction = projected
            .mean
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let p_true = (1.0 - projected_fraction).clamp(0.0, 1.0);

        ForecastOutput::Distribution(boolean_distribution(issued_at_tick, horizon, p_true))
    }
}

/// Rung 6: the oracle upper bound. Same design as `ecological::OracleGenerator` — the whole run
/// recorded once into a `Vec<bool>` (predator-population extinct at each tick), looked up by
/// direct index rather than live re-simulation.
pub struct OracleGenerator {
    trajectory: Vec<bool>,
}

impl OracleGenerator {
    pub const RUNG: BaselineRung = BaselineRung::OracleUpperBound;

    pub fn from_trajectory(trajectory: Vec<bool>) -> Self {
        Self { trajectory }
    }
}

impl TrajectoryGenerator for OracleGenerator {
    type Observation = u64;

    fn generate(&self, issued_at_tick: &u64, horizon: Horizon) -> ForecastOutput {
        let target_tick = issued_at_tick + horizon.0;
        let Some(&actually_extinct) = self.trajectory.get(target_tick as usize) else {
            return ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange);
        };
        let p_true = if actually_extinct { 1.0 } else { 0.0 };
        ForecastOutput::Distribution(boolean_distribution(*issued_at_tick, horizon, p_true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::OutcomeRegion;

    fn observation_with(tick: u64, count: Option<usize>) -> PredatorPreyObservation {
        PredatorPreyObservation {
            tick,
            sampled_predator_alive_count: count,
        }
    }

    fn extract_p_true(output: &ForecastOutput) -> f64 {
        let ForecastOutput::Distribution(dist) = output else {
            panic!("expected a distribution, got an abstention");
        };
        dist.branches()
            .iter()
            .find(|b| b.outcome == OutcomeRegion::Boolean(true))
            .map(|b| b.probability.get())
            .unwrap_or(0.0)
    }

    #[test]
    fn persistence_abstains_with_no_reading() {
        let generator = PersistenceGenerator;
        let out = generator.generate(&observation_with(0, None), Horizon(10));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn persistence_predicts_high_when_cohort_extinct() {
        let generator = PersistenceGenerator;
        let out = generator.generate(&observation_with(0, Some(0)), Horizon(10));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn historical_frequency_ignores_the_observation_entirely() {
        let generator = HistoricalFrequencyGenerator { base_rate: 0.42 };
        for obs in [
            observation_with(0, None),
            observation_with(0, Some(0)),
            observation_with(0, Some(5)),
        ] {
            let out = generator.generate(&obs, Horizon(10));
            assert_eq!(extract_p_true(&out), 0.42);
        }
    }

    #[test]
    fn simple_statistical_predicts_high_on_a_clear_declining_trend() {
        let generator = SimpleStatisticalGenerator;
        let history: Vec<_> = (0..6u64)
            .map(|t| observation_with(t, Some((6 - t) as usize)))
            .collect();
        let out = generator.generate(&history, Horizon(5));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn scenario_mechanistic_matches_hand_computed_value() {
        let generator = ScenarioMechanisticGenerator {
            per_member_death_probability: 0.5,
        };
        let out = generator.generate(&observation_with(0, Some(1)), Horizon(1));
        assert!((extract_p_true(&out) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fep_driven_predicts_high_on_a_clear_declining_trend_given_enough_data() {
        // Mirrors ecological's own finding: FEP-driven forecasting needs substantially more
        // replayed history than the closed-form rungs to reach a comparable confidence level.
        let generator = FepDrivenGenerator::default();
        let history: Vec<_> = (0..60u64)
            .map(|t| observation_with(t, Some((60 - t) as usize)))
            .collect();
        let out = generator.generate(&history, Horizon(6));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn fep_driven_transition_matrix_actually_learns_from_replayed_history() {
        let config = FepDrivenGenerator::default().agent_config;
        let mut agent = ActiveInferenceAgent::new(config);
        let initial_self_transition = agent.model.transition_matrices[0][0][0];

        let mut prev_belief = agent.belief.clone();
        let reference = 8.0;
        for t in 0..8u64 {
            let count = 8 - t;
            let raw_value = (count as f64 / reference).clamp(0.0, 1.0);
            let raw_obs = Observation::new(vec![raw_value], 1.0, "predator_survival_fraction");
            let masked = mask_observation(&raw_obs, &agent.belief, &[1.0]);
            agent.perceive(&masked);
            let new_belief = agent.belief.clone();
            agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
            prev_belief = new_belief;
        }

        assert_ne!(
            agent.model.transition_matrices[0][0][0], initial_self_transition,
            "transition_matrices[0] never changed -- observe_transition isn't teaching anything"
        );
    }

    #[test]
    fn oracle_looks_up_the_recorded_trajectory_exactly() {
        let trajectory = vec![false, false, false, false, false, true, true];
        let oracle = OracleGenerator::from_trajectory(trajectory);
        assert_eq!(extract_p_true(&oracle.generate(&0, Horizon(5))), 1.0);
        assert_eq!(extract_p_true(&oracle.generate(&0, Horizon(3))), 0.0);
    }

    #[test]
    fn oracle_abstains_beyond_the_recorded_trajectory() {
        let oracle = OracleGenerator::from_trajectory(vec![false, false, false]);
        let out = oracle.generate(&0, Horizon(10));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange)
        ));
    }
}
