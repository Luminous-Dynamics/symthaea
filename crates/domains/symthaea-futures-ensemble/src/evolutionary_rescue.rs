// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The seven-rung hierarchy for the third scenario family — evolutionary-rescue collapse
//! forecasting on `symthaea-futures-symtropy::evolutionary_rescue` — per
//! `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s Phase 2.2B design, which explicitly asks
//! for a rung set that isn't a mechanical copy of `ecological`/`predator_prey`'s six:
//!
//! 1. [`HistoricalFrequencyGenerator`] — historical survival-extension baseline (ignores the
//!    observation entirely, same design as the other two families' rung 1/2).
//! 2. [`CensusOnlyStatisticalGenerator`] — a census-only time-to-collapse predictor: OLS trend
//!    on `sampled_alive_count` alone. Reads `observed_mean_forage_efficiency` *never*, by
//!    construction — feeding it any of the four Phase 2.2B observation conditions must produce
//!    an identical forecast (see the module's own test for this invariant), which is exactly
//!    what makes it a fair "census-only" control for the acceptance gate.
//! 3. [`TraitTrendStatisticalGenerator`] — the plan's own new rung: an OLS trend on
//!    `observed_mean_forage_efficiency` alone (ignores population count entirely), mapped
//!    through a disclosed floor/ceiling normalization (see its own docs for the constants and
//!    why they were chosen from `tests/phase7_evolutionary_rescue.rs`'s own measured range
//!    rather than invented).
//! 4. [`AdaptationVsForcingGenerator`] — a simplified closed-form mechanistic model combining
//!    the known secular climate-drift rate (a legitimate setup parameter, not a truth-read —
//!    same status as `ecological::ScenarioMechanisticGenerator`'s per-member death probability)
//!    with the observed trait level as a stress-mitigation factor.
//! 5. [`FepCensusOnlyGenerator`] — the FEP-driven ensemble, 1D belief state, reading
//!    `sampled_alive_count` only. This is the acceptance gate's baseline arm.
//! 6. [`FepCensusPlusTraitGenerator`] — the same FEP mechanism, 2D belief state, reading
//!    `sampled_alive_count` *and* `observed_mean_forage_efficiency` jointly. This is the
//!    acceptance gate's trait-augmented arm — evaluated once on real noisy-trait trajectories
//!    (condition 2) and once on `shuffle_trait_readings`-permuted trajectories (condition 3, the
//!    load-bearing control).
//! 7. [`OracleGenerator`] — privileged hindsight reference, same design as the other two
//!    families' final rung.
//!
//! **Not built here**: the actual backtest harness (train/test seed generation, scoring, and
//! the acceptance-gate evaluation) — that's a separate example binary, per the plan's own
//! precedent of keeping rung definitions and backtest scripts in different files.

use symthaea_futures_core::{AbstentionReason, ForecastOutput, Horizon, TrajectoryGenerator};
use symthaea_futures_state::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, mask_observation,
};
use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueObservation;

use crate::BaselineRung;

const POPULATION_COLLAPSE_WITHIN_HORIZON: &str = "evolutionary_rescue_collapse_within_horizon";

/// Same measured range `tests/phase7_evolutionary_rescue.rs` documents: frozen populations hold
/// `forage_efficiency` exactly at `OrganismConfig::default()`'s 0.15, evolving populations rise
/// to 0.37-0.49 by t=9,000. `TRAIT_CEILING` is set comfortably above the observed range (not
/// exactly at it) so a population that adapts even further than any seed in that 5-seed sweep
/// doesn't saturate the normalization at 1.0 immediately.
const TRAIT_FLOOR: f64 = 0.15;
const TRAIT_CEILING: f64 = 0.6;

/// Same value `symthaea-futures-symtropy::evolutionary_rescue`'s own test module and
/// `tests/phase7_evolutionary_rescue.rs` both use — a legitimate setup parameter (known to
/// whoever configured the scenario), not a truth-read.
const SECULAR_DRIFT_PER_TICK_MAGNITUDE: f64 = 0.01;

fn boolean_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    p_true: f64,
) -> symthaea_futures_core::ForecastDistribution {
    crate::boolean_distribution(
        issued_at_tick,
        horizon,
        p_true,
        POPULATION_COLLAPSE_WITHIN_HORIZON,
    )
}

fn normalize_trait(trait_value: f64) -> f64 {
    ((trait_value - TRAIT_FLOOR) / (TRAIT_CEILING - TRAIT_FLOOR)).clamp(0.0, 1.0)
}

/// Rung 1: base rate across training seeds, ignoring the observation entirely.
pub struct HistoricalFrequencyGenerator {
    pub base_rate: f64,
}

impl HistoricalFrequencyGenerator {
    pub const RUNG: BaselineRung = BaselineRung::HistoricalFrequency;
}

impl TrajectoryGenerator for HistoricalFrequencyGenerator {
    type Observation = EvolutionaryRescueObservation;

    fn generate(
        &self,
        observation: &EvolutionaryRescueObservation,
        horizon: Horizon,
    ) -> ForecastOutput {
        ForecastOutput::Distribution(boolean_distribution(
            observation.tick,
            horizon,
            self.base_rate,
        ))
    }
}

/// Rung 2: a real OLS linear-trend fit on `sampled_alive_count` alone. **Never reads
/// `observed_mean_forage_efficiency`** — see module docs on why that's the point.
pub struct CensusOnlyStatisticalGenerator;

impl CensusOnlyStatisticalGenerator {
    pub const RUNG: BaselineRung = BaselineRung::SimpleStatistical;
}

impl TrajectoryGenerator for CensusOnlyStatisticalGenerator {
    type Observation = Vec<EvolutionaryRescueObservation>;

    fn generate(
        &self,
        history: &Vec<EvolutionaryRescueObservation>,
        horizon: Horizon,
    ) -> ForecastOutput {
        let points: Vec<(f64, f64)> = history
            .iter()
            .filter_map(|obs| {
                obs.sample
                    .map(|s| (obs.tick as f64, s.sampled_alive_count as f64))
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

/// Rung 3: a real OLS linear-trend fit on `observed_mean_forage_efficiency` alone — never reads
/// population count. Projected trait level is mapped through [`normalize_trait`]'s disclosed
/// floor/ceiling normalization: `p_true = 1 - normalize_trait(projected)`, i.e. a
/// well-adapted-and-improving trend predicts low collapse probability, a frozen-or-declining
/// one predicts high. **Disclosed simplification**: ignores the actual forcing/collapse
/// mechanism entirely, exactly the complementary blind spot to
/// [`CensusOnlyStatisticalGenerator`]'s.
pub struct TraitTrendStatisticalGenerator;

impl TraitTrendStatisticalGenerator {
    pub const RUNG: BaselineRung = BaselineRung::TraitTrend;
}

impl TrajectoryGenerator for TraitTrendStatisticalGenerator {
    type Observation = Vec<EvolutionaryRescueObservation>;

    fn generate(
        &self,
        history: &Vec<EvolutionaryRescueObservation>,
        horizon: Horizon,
    ) -> ForecastOutput {
        let points: Vec<(f64, f64)> = history
            .iter()
            .filter_map(|obs| {
                obs.sample
                    .and_then(|s| s.observed_mean_forage_efficiency)
                    .map(|trait_value| (obs.tick as f64, trait_value))
            })
            .collect();

        if points.len() < 2 {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        }

        let issued_at_tick = history.last().map(|o| o.tick).unwrap_or(0);
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
        let projected_trait = intercept + slope * target_tick;
        let p_true = (1.0 - normalize_trait(projected_trait)).clamp(0.0, 1.0);

        ForecastOutput::Distribution(boolean_distribution(issued_at_tick, horizon, p_true))
    }
}

/// Rung 4: a real closed-form equation (not a fit), combining the known secular-drift magnitude
/// with the observed trait level as a stress-mitigation factor. **Disclosed simplification**:
/// treats accumulated drift-exposure (`SECULAR_DRIFT_PER_TICK_MAGNITUDE * target_tick`) as a
/// linear proxy for environmental stress and divides it by `1 + adaptation_sensitivity *
/// trait_level` — the real ice-albedo bifurcation is a genuinely nonlinear collapse, not this
/// equation's smooth ratio; this rung is deliberately the "back of the envelope" model, not a
/// refit of the real physics. Falls back to `TRAIT_FLOOR` (the frozen baseline) when no trait
/// signal is available — a conservative default, not an optimistic one.
pub struct AdaptationVsForcingGenerator {
    pub adaptation_sensitivity: f64,
    pub collapse_threshold: f64,
}

impl AdaptationVsForcingGenerator {
    pub const RUNG: BaselineRung = BaselineRung::ScenarioMechanistic;
}

impl TrajectoryGenerator for AdaptationVsForcingGenerator {
    type Observation = EvolutionaryRescueObservation;

    fn generate(
        &self,
        observation: &EvolutionaryRescueObservation,
        horizon: Horizon,
    ) -> ForecastOutput {
        let Some(sample) = observation.sample else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };
        if sample.sampled_alive_count == 0 {
            return ForecastOutput::Distribution(boolean_distribution(
                observation.tick,
                horizon,
                0.9,
            ));
        }

        let trait_level = sample
            .observed_mean_forage_efficiency
            .unwrap_or(TRAIT_FLOOR);
        let target_tick = (observation.tick + horizon.0) as f64;
        let cumulative_stress = SECULAR_DRIFT_PER_TICK_MAGNITUDE * target_tick;
        let effective_stress =
            cumulative_stress / (1.0 + self.adaptation_sensitivity * trait_level);
        let p_true = (effective_stress / self.collapse_threshold).clamp(0.0, 1.0);

        ForecastOutput::Distribution(boolean_distribution(observation.tick, horizon, p_true))
    }
}

/// Rung 5: the FEP-driven ensemble, 1D belief state, reading `sampled_alive_count` only — the
/// acceptance gate's baseline arm. Same "perceive-only teaches nothing" mechanism as
/// `ecological::FepDrivenGenerator` (`num_actions: 1` + explicit `observe_transition` calls);
/// not re-explained here.
pub struct FepCensusOnlyGenerator {
    pub agent_config: ActiveInferenceAgentConfig,
}

impl Default for FepCensusOnlyGenerator {
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

impl FepCensusOnlyGenerator {
    pub const RUNG: BaselineRung = BaselineRung::FepDriven;
}

impl TrajectoryGenerator for FepCensusOnlyGenerator {
    type Observation = Vec<EvolutionaryRescueObservation>;

    fn generate(
        &self,
        history: &Vec<EvolutionaryRescueObservation>,
        horizon: Horizon,
    ) -> ForecastOutput {
        let issued_at_tick = match history.last() {
            Some(o) => o.tick,
            None => {
                return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
            }
        };

        let Some(reference) = history
            .first()
            .and_then(|o| o.sample)
            .map(|s| s.sampled_alive_count as f64)
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
                .sample
                .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            let visibility = if obs.sample.is_some() { 1.0 } else { 0.0 };

            let raw_obs = Observation::new(vec![raw_value], 1.0, "population_survival_fraction");
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

/// Rung 6: the same FEP mechanism, 2D belief state — `[population_survival_fraction,
/// normalized_trait_level]` — reading `sampled_alive_count` *and*
/// `observed_mean_forage_efficiency` jointly. The acceptance gate's trait-augmented arm.
///
/// **A tick with no trait reading is marked invisible on that channel alone**, not on the whole
/// observation — `mask_observation`'s per-element visibility vector lets the population channel
/// stay perceived even on a tick where the trait channel wasn't sampled (e.g. a real-world
/// budget/frequency mismatch between the two signals), rather than discarding the whole tick.
pub struct FepCensusPlusTraitGenerator {
    pub agent_config: ActiveInferenceAgentConfig,
}

impl Default for FepCensusPlusTraitGenerator {
    fn default() -> Self {
        Self {
            agent_config: ActiveInferenceAgentConfig {
                state_dim: 2,
                obs_dim: 2,
                num_actions: 1,
                ..ActiveInferenceAgentConfig::default()
            },
        }
    }
}

impl FepCensusPlusTraitGenerator {
    pub const RUNG: BaselineRung = BaselineRung::FepCensusPlusTrait;
}

impl TrajectoryGenerator for FepCensusPlusTraitGenerator {
    type Observation = Vec<EvolutionaryRescueObservation>;

    fn generate(
        &self,
        history: &Vec<EvolutionaryRescueObservation>,
        horizon: Horizon,
    ) -> ForecastOutput {
        let issued_at_tick = match history.last() {
            Some(o) => o.tick,
            None => {
                return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
            }
        };

        let Some(reference) = history
            .first()
            .and_then(|o| o.sample)
            .map(|s| s.sampled_alive_count as f64)
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
            let pop_value = obs
                .sample
                .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            let pop_visible = if obs.sample.is_some() { 1.0 } else { 0.0 };

            let trait_reading = obs.sample.and_then(|s| s.observed_mean_forage_efficiency);
            let trait_value = trait_reading.map(normalize_trait).unwrap_or(0.5);
            let trait_visible = if trait_reading.is_some() { 1.0 } else { 0.0 };

            let raw_obs = Observation::new(
                vec![pop_value, trait_value],
                1.0,
                "population_and_trait_fraction",
            );
            let masked = mask_observation(&raw_obs, &agent.belief, &[pop_visible, trait_visible]);

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

/// Rung 7: the oracle upper bound. Same design as `ecological`/`predator_prey`'s own
/// `OracleGenerator` — the whole run recorded once into a `Vec<bool>` (population collapsed at
/// each tick), looked up by direct index.
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
        let Some(&actually_collapsed) = self.trajectory.get(target_tick as usize) else {
            return ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange);
        };
        let p_true = if actually_collapsed { 1.0 } else { 0.0 };
        ForecastOutput::Distribution(boolean_distribution(*issued_at_tick, horizon, p_true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::OutcomeRegion;
    use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueSample;

    /// `None` for `count` means "no observation at all this tick" (mirrors an off-frequency
    /// tick) — it must also blank `trait_value`, since a real observation firewall reveals
    /// nothing at all on a skipped tick, not a trait reading with no census. Tests that want a
    /// trait-only reading (no useful census) should pass `count: Some(0)` deliberately, not
    /// `None`.
    fn observation_with(
        tick: u64,
        count: Option<usize>,
        trait_value: Option<f64>,
    ) -> EvolutionaryRescueObservation {
        EvolutionaryRescueObservation {
            tick,
            sample: count.map(|c| EvolutionaryRescueSample {
                sampled_alive_count: c,
                observed_mean_forage_efficiency: trait_value,
            }),
        }
    }

    fn extract_p_true(output: &ForecastOutput) -> f64 {
        let ForecastOutput::Distribution(dist) = output else {
            panic!("expected a distribution, got an abstention");
        };
        dist.branches
            .iter()
            .find(|b| b.outcome == OutcomeRegion::Boolean(true))
            .map(|b| b.probability)
            .unwrap_or(0.0)
    }

    #[test]
    fn historical_frequency_ignores_the_observation_entirely() {
        let generator = HistoricalFrequencyGenerator { base_rate: 0.37 };
        for obs in [
            observation_with(0, None, None),
            observation_with(0, Some(0), None),
            observation_with(0, Some(20), Some(0.5)),
        ] {
            let out = generator.generate(&obs, Horizon(10));
            assert_eq!(extract_p_true(&out), 0.37);
        }
    }

    #[test]
    fn census_only_predicts_high_on_a_clear_declining_population_trend() {
        let generator = CensusOnlyStatisticalGenerator;
        let history: Vec<_> = (0..6u64)
            .map(|t| observation_with(t, Some((6 - t) as usize), None))
            .collect();
        let out = generator.generate(&history, Horizon(5));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn census_only_is_invariant_to_the_trait_field_by_construction() {
        // Identical population-count history, but one side also carries a rich trait signal --
        // this rung must never read it, so both forecasts must be bit-identical.
        let no_trait: Vec<_> = (0..6u64)
            .map(|t| observation_with(t, Some(10 + t as usize), None))
            .collect();
        let with_trait: Vec<_> = (0..6u64)
            .map(|t| observation_with(t, Some(10 + t as usize), Some(0.5 - 0.01 * t as f64)))
            .collect();

        let generator = CensusOnlyStatisticalGenerator;
        let p_no_trait = extract_p_true(&generator.generate(&no_trait, Horizon(5)));
        let p_with_trait = extract_p_true(&generator.generate(&with_trait, Horizon(5)));
        assert_eq!(p_no_trait, p_with_trait);
    }

    #[test]
    fn trait_trend_predicts_low_when_trait_is_rising_toward_ceiling() {
        let generator = TraitTrendStatisticalGenerator;
        let history: Vec<_> = (0..6u64)
            .map(|t| observation_with(t, Some(0), Some(0.15 + 0.05 * t as f64)))
            .collect();
        let out = generator.generate(&history, Horizon(5));
        assert!(extract_p_true(&out) < 0.5);
    }

    #[test]
    fn trait_trend_predicts_high_when_trait_stays_frozen_at_floor() {
        let generator = TraitTrendStatisticalGenerator;
        let history: Vec<_> = (0..6u64)
            .map(|t| observation_with(t, Some(0), Some(0.15)))
            .collect();
        let out = generator.generate(&history, Horizon(5));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn adaptation_vs_forcing_gives_a_better_survival_chance_to_a_higher_trait_level() {
        let generator = AdaptationVsForcingGenerator {
            adaptation_sensitivity: 5.0,
            collapse_threshold: 50.0,
        };
        let low_trait = observation_with(9000, Some(12), Some(0.15));
        let high_trait = observation_with(9000, Some(12), Some(0.45));
        let p_low = extract_p_true(&generator.generate(&low_trait, Horizon(300)));
        let p_high = extract_p_true(&generator.generate(&high_trait, Horizon(300)));
        assert!(
            p_high < p_low,
            "a higher trait level should predict a lower collapse probability: p_low={p_low} \
             p_high={p_high}"
        );
    }

    #[test]
    fn adaptation_vs_forcing_falls_back_to_the_conservative_floor_with_no_trait_signal() {
        let generator = AdaptationVsForcingGenerator {
            adaptation_sensitivity: 5.0,
            collapse_threshold: 50.0,
        };
        let no_trait = observation_with(9000, Some(12), None);
        let floor_trait = observation_with(9000, Some(12), Some(TRAIT_FLOOR));
        let p_no_trait = extract_p_true(&generator.generate(&no_trait, Horizon(300)));
        let p_floor_trait = extract_p_true(&generator.generate(&floor_trait, Horizon(300)));
        assert!((p_no_trait - p_floor_trait).abs() < 1e-12);
    }

    #[test]
    fn fep_census_only_predicts_high_on_a_clear_declining_trend_given_enough_data() {
        let generator = FepCensusOnlyGenerator::default();
        let history: Vec<_> = (0..60u64)
            .map(|t| observation_with(t, Some((60 - t) as usize), None))
            .collect();
        let out = generator.generate(&history, Horizon(6));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn fep_census_plus_trait_transition_matrix_learns_from_replayed_history() {
        let config = FepCensusPlusTraitGenerator::default().agent_config;
        let mut agent = ActiveInferenceAgent::new(config);
        let initial_self_transition = agent.model.transition_matrices[0][0][0];

        let mut prev_belief = agent.belief.clone();
        let reference = 20.0;
        for t in 0..20u64 {
            let count = 20 - t;
            let pop_value = (count as f64 / reference).clamp(0.0, 1.0);
            let trait_value = normalize_trait(0.5 - 0.01 * t as f64);
            let raw_obs = Observation::new(
                vec![pop_value, trait_value],
                1.0,
                "population_and_trait_fraction",
            );
            let masked = mask_observation(&raw_obs, &agent.belief, &[1.0, 1.0]);
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
