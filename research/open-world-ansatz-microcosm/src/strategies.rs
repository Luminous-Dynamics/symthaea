//! Comparable one-step experimental-design baselines over the blind episode API.
//!
//! These are policy controls, not claims of optimality. Every strategy sees the
//! same AgentView and scores only the interventions legal in the public task.

use crate::{self, interaction, numeric};
use crate::protocol::{
    AgentIntervention, AgentMeasurement, AgentView, EpisodeStatus, StructureAccess,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStrategy {
    /// Maximize the number of distinct predicted outcome partitions.
    MaximumPartition,
    /// Minimize the largest posterior candidate set under a uniform prior.
    Minimax,
    /// Maximize expected posterior entropy under a uniform prior.
    MaximumEntropy,
    /// Minimize expected remaining candidate count under a uniform prior.
    MinimumExpectedRemaining,
    /// Choose the first action in the declared canonical domain ordering.
    FixedGrid,
    /// Choose a reproducible pseudo-random legal action.
    Random { seed: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMetrics {
    pub intervention: AgentIntervention,
    pub outcome_partition_count: usize,
    pub worst_case_remaining_models: usize,
    /// Sum of squared partition sizes. Divide by denominator for expected
    /// remaining model count under a uniform prior.
    pub expected_remaining_numerator: u64,
    pub expected_remaining_denominator: usize,
    pub entropy_bits: f64,
    pub probe_cost_units: u64,
}

impl ProbeMetrics {
    pub fn expected_remaining_models(&self) -> f64 {
        if self.expected_remaining_denominator == 0 {
            return 0.0;
        }
        self.expected_remaining_numerator as f64
            / self.expected_remaining_denominator as f64
    }
}

/// Return exact partition statistics for every currently legal one-step action.
pub fn score_available_probes(view: &AgentView) -> Result<Vec<ProbeMetrics>, StrategyError> {
    if view.status != EpisodeStatus::Open {
        return Err(StrategyError::EpisodeNotOpen);
    }
    if view.remaining_probe_actions == 0 {
        return Err(StrategyError::ProbeBudgetExhausted);
    }

    match view.task.world_family {
        crate::protocol::WorldFamily::BooleanMechanism => score_boolean(view),
        crate::protocol::WorldFamily::NumericMechanism => score_numeric(view),
        crate::protocol::WorldFamily::TwoInputResponse => score_interaction(view),
    }
}

/// Select a legal one-step action using one of the explicit comparison strategies.
pub fn choose_probe(
    view: &AgentView,
    strategy: ProbeStrategy,
) -> Result<AgentIntervention, StrategyError> {
    let metrics = score_available_probes(view)?;
    if metrics.is_empty() {
        return Err(StrategyError::NoCompatibleModels);
    }

    let selected = match strategy {
        ProbeStrategy::FixedGrid => metrics.first(),
        ProbeStrategy::Random { seed } => {
            let index = (mix_seed(seed) as usize) % metrics.len();
            metrics.get(index)
        }
        _ => metrics.iter().max_by(|left, right| compare_metrics(left, right, strategy)),
    };
    selected
        .map(|value| value.intervention)
        .ok_or(StrategyError::NoLegalProbe)
}

fn compare_metrics(
    left: &ProbeMetrics,
    right: &ProbeMetrics,
    strategy: ProbeStrategy,
) -> Ordering {
    let primary = match strategy {
        ProbeStrategy::MaximumPartition => left
            .outcome_partition_count
            .cmp(&right.outcome_partition_count),
        ProbeStrategy::Minimax => right
            .worst_case_remaining_models
            .cmp(&left.worst_case_remaining_models),
        ProbeStrategy::MaximumEntropy => left.entropy_bits.total_cmp(&right.entropy_bits),
        ProbeStrategy::MinimumExpectedRemaining => right
            .expected_remaining_numerator
            .cmp(&left.expected_remaining_numerator),
        ProbeStrategy::FixedGrid | ProbeStrategy::Random { .. } => Ordering::Equal,
    };

    primary
        .then_with(|| {
            right
                .worst_case_remaining_models
                .cmp(&left.worst_case_remaining_models)
        })
        .then_with(|| left.entropy_bits.total_cmp(&right.entropy_bits))
        .then_with(|| {
            right
                .expected_remaining_numerator
                .cmp(&left.expected_remaining_numerator)
        })
        .then_with(|| right.intervention.cmp(&left.intervention))
}

fn mix_seed(mut value: u64) -> u64 {
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    value.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn metrics<T: Ord>(
    intervention: AgentIntervention,
    predictions: impl IntoIterator<Item = T>,
    denominator: usize,
) -> Result<ProbeMetrics, StrategyError> {
    if denominator == 0 {
        return Err(StrategyError::NoCompatibleModels);
    }
    let mut partitions: BTreeMap<T, u64> = BTreeMap::new();
    let mut prediction_count = 0usize;
    for prediction in predictions {
        *partitions.entry(prediction).or_default() += 1;
        prediction_count += 1;
    }
    if prediction_count != denominator {
        return Err(StrategyError::PredictionCardinalityMismatch);
    }

    let mut expected_remaining_numerator = 0u64;
    let mut worst_case_remaining_models = 0usize;
    let mut entropy_bits = 0.0;
    for count in partitions.values() {
        expected_remaining_numerator += count * count;
        worst_case_remaining_models = worst_case_remaining_models.max(*count as usize);
        let probability = *count as f64 / denominator as f64;
        entropy_bits -= probability * probability.log2();
    }

    Ok(ProbeMetrics {
        intervention,
        outcome_partition_count: partitions.len(),
        worst_case_remaining_models,
        expected_remaining_numerator,
        expected_remaining_denominator: denominator,
        entropy_bits,
        probe_cost_units: 1,
    })
}

fn score_boolean(view: &AgentView) -> Result<Vec<ProbeMetrics>, StrategyError> {
    let access = match view.task.structure_access {
        StructureAccess::BooleanWithheld => crate::StructureAccess::Withheld,
        StructureAccess::BooleanSupplied(mechanism) => {
            crate::StructureAccess::Supplied(mechanism)
        }
        _ => return Err(StrategyError::ManifestFamilyMismatch),
    };
    let log = boolean_log(view)?;
    let hypotheses = if log.interventions.is_empty() {
        crate::candidate_hypotheses(access)
    } else {
        crate::discover(&log, access)
            .map_err(|error| StrategyError::DiscoveryFailure(error.to_string()))?
            .compatible_hypotheses
    };
    if hypotheses.is_empty() {
        return Err(StrategyError::NoCompatibleModels);
    }

    [false, true]
        .into_iter()
        .map(|input| {
            let predictions = hypotheses.iter().map(|hypothesis| {
                hypothesis
                    .predict_continuation(&log.interventions, &[input])
                    .map(|output| output[0])
                    .map_err(|error| StrategyError::PredictionFailure(error.to_string()))
            });
            let values = predictions.collect::<Result<Vec<_>, _>>()?;
            metrics(
                AgentIntervention::Boolean(input),
                values,
                hypotheses.len(),
            )
        })
        .collect()
}

fn score_numeric(view: &AgentView) -> Result<Vec<ProbeMetrics>, StrategyError> {
    let access = match view.task.structure_access {
        StructureAccess::NumericWithheld => numeric::StructureAccess::Withheld,
        StructureAccess::NumericSupplied(mechanism) => {
            numeric::StructureAccess::Supplied(mechanism)
        }
        _ => return Err(StrategyError::ManifestFamilyMismatch),
    };
    let log = numeric_log(view)?;
    let hypotheses = if log.interventions.is_empty() {
        numeric::candidate_models(access)
    } else {
        numeric::discover(&log, access)
            .map_err(|error| StrategyError::DiscoveryFailure(error.to_string()))?
            .compatible_models
    };
    if hypotheses.is_empty() {
        return Err(StrategyError::NoCompatibleModels);
    }

    [-1_i64, 0, 1]
        .into_iter()
        .map(|input| {
            let predictions = hypotheses.iter().map(|hypothesis| {
                hypothesis
                    .predict_continuation(&log.interventions, &[input])
                    .map(|output| output[0])
                    .map_err(|error| StrategyError::PredictionFailure(error.to_string()))
            });
            let values = predictions.collect::<Result<Vec<_>, _>>()?;
            metrics(
                AgentIntervention::Numeric(input),
                values,
                hypotheses.len(),
            )
        })
        .collect()
}

fn score_interaction(view: &AgentView) -> Result<Vec<ProbeMetrics>, StrategyError> {
    let access = match view.task.structure_access {
        StructureAccess::InteractionWithheld => interaction::StructureAccess::Withheld,
        StructureAccess::InteractionSupplied(structure) => {
            interaction::StructureAccess::Supplied(structure)
        }
        _ => return Err(StrategyError::ManifestFamilyMismatch),
    };
    let log = interaction_log(view)?;
    let hypotheses = if log.interventions.is_empty() {
        interaction::candidate_models(access)
    } else {
        interaction::discover(&log, access)
            .map_err(|error| StrategyError::DiscoveryFailure(error.to_string()))?
            .compatible_models
    };
    if hypotheses.is_empty() {
        return Err(StrategyError::NoCompatibleModels);
    }

    interaction::FACTORIAL_DESIGN
        .into_iter()
        .map(|input| {
            let predictions = hypotheses.iter().map(|hypothesis| {
                hypothesis
                    .predict(input)
                    .map_err(|error| StrategyError::PredictionFailure(error.to_string()))
            });
            let values = predictions.collect::<Result<Vec<_>, _>>()?;
            metrics(
                AgentIntervention::Pair(input),
                values,
                hypotheses.len(),
            )
        })
        .collect()
}

fn boolean_log(view: &AgentView) -> Result<crate::ObservationLog, StrategyError> {
    let mut interventions = Vec::with_capacity(view.observations.len());
    let mut measurements = Vec::with_capacity(view.observations.len());
    for record in &view.observations {
        match (record.intervention, record.measurement) {
            (AgentIntervention::Boolean(input), AgentMeasurement::Boolean(output)) => {
                interventions.push(input);
                measurements.push(output);
            }
            _ => return Err(StrategyError::ObservationFamilyMismatch),
        }
    }
    Ok(crate::ObservationLog {
        interventions,
        measurements,
    })
}

fn numeric_log(view: &AgentView) -> Result<numeric::ObservationLog, StrategyError> {
    let mut interventions = Vec::with_capacity(view.observations.len());
    let mut measurements = Vec::with_capacity(view.observations.len());
    for record in &view.observations {
        match (record.intervention, record.measurement) {
            (AgentIntervention::Numeric(input), AgentMeasurement::Numeric(output)) => {
                interventions.push(input);
                measurements.push(output);
            }
            _ => return Err(StrategyError::ObservationFamilyMismatch),
        }
    }
    Ok(numeric::ObservationLog {
        interventions,
        measurements,
    })
}

fn interaction_log(view: &AgentView) -> Result<interaction::ObservationLog, StrategyError> {
    let mut interventions = Vec::with_capacity(view.observations.len());
    let mut measurements = Vec::with_capacity(view.observations.len());
    for record in &view.observations {
        match (record.intervention, record.measurement) {
            (AgentIntervention::Pair(input), AgentMeasurement::Numeric(output)) => {
                interventions.push(input);
                measurements.push(output);
            }
            _ => return Err(StrategyError::ObservationFamilyMismatch),
        }
    }
    Ok(interaction::ObservationLog {
        interventions,
        measurements,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyError {
    EpisodeNotOpen,
    ProbeBudgetExhausted,
    ManifestFamilyMismatch,
    ObservationFamilyMismatch,
    NoCompatibleModels,
    NoLegalProbe,
    PredictionCardinalityMismatch,
    DiscoveryFailure(String),
    PredictionFailure(String),
}

impl std::fmt::Display for StrategyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for StrategyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{BlindEpisode, AgentAction};

    #[test]
    fn all_strategies_return_a_legal_probe_in_each_world_family() {
        let strategies = [
            ProbeStrategy::MaximumPartition,
            ProbeStrategy::Minimax,
            ProbeStrategy::MaximumEntropy,
            ProbeStrategy::MinimumExpectedRemaining,
            ProbeStrategy::FixedGrid,
            ProbeStrategy::Random { seed: 17 },
        ];

        for strategy in strategies {
            let mut boolean = BlindEpisode::new_boolean(
                "strategy-bool-01",
                crate::Mechanism::HiddenLag,
                crate::MeasurementMap::Identity,
                Some(false),
                StructureAccess::BooleanWithheld,
                4,
            ).unwrap();
            let view = boolean.view().unwrap();
            let intervention = choose_probe(&view, strategy).unwrap();
            let response = boolean.apply(AgentAction::Probe { intervention }).unwrap();
            assert_eq!(response.view.observations.len(), 1);

            let mut numeric_episode = BlindEpisode::new_numeric(
                "strategy-numeric-01",
                numeric::ModelHypothesis {
                    mechanism: numeric::Mechanism::LeakyMemory,
                    gain: 2,
                    decay: Some(1),
                    initial_state: Some(1),
                    sensor: numeric::SensorMap::Identity,
                },
                StructureAccess::NumericWithheld,
                4,
            ).unwrap();
            let view = numeric_episode.view().unwrap();
            let intervention = choose_probe(&view, strategy).unwrap();
            let response = numeric_episode.apply(AgentAction::Probe { intervention }).unwrap();
            assert_eq!(response.view.observations.len(), 1);

            let mut interaction_episode = BlindEpisode::new_interaction(
                "strategy-interaction-01",
                interaction::ModelHypothesis {
                    structure: interaction::Structure::BilinearInteraction,
                    bias: 0,
                    weight_u: 1,
                    weight_v: 1,
                    interaction_weight: 1,
                },
                StructureAccess::InteractionWithheld,
                4,
            ).unwrap();
            let view = interaction_episode.view().unwrap();
            let intervention = choose_probe(&view, strategy).unwrap();
            let response = interaction_episode.apply(AgentAction::Probe { intervention }).unwrap();
            assert_eq!(response.view.observations.len(), 1);
        }
    }

    #[test]
    fn expected_remaining_count_is_exact_for_uniform_partition_weights() {
        let result = metrics(
            AgentIntervention::Numeric(0),
            [0, 0, 1, 1, 1, 2],
            6,
        ).unwrap();
        assert_eq!(result.outcome_partition_count, 3);
        assert_eq!(result.worst_case_remaining_models, 3);
        assert_eq!(result.expected_remaining_numerator, 14);
        assert_eq!(result.expected_remaining_denominator, 6);
        assert!((result.expected_remaining_models() - (14.0 / 6.0)).abs() < 1e-12);
        assert!((result.entropy_bits - (1.4591479170272448)).abs() < 1e-12);
    }

    #[test]
    fn fixed_grid_is_stable_and_random_strategy_is_seed_reproducible() {
        let episode = BlindEpisode::new_interaction(
            "strategy-determinism-01",
            interaction::ModelHypothesis {
                structure: interaction::Structure::BilinearInteraction,
                bias: 0,
                weight_u: 1,
                weight_v: 1,
                interaction_weight: 1,
            },
            StructureAccess::InteractionWithheld,
            4,
        ).unwrap();
        let view = episode.view().unwrap();
        assert_eq!(
            choose_probe(&view, ProbeStrategy::FixedGrid).unwrap(),
            AgentIntervention::Pair([-1, -1])
        );
        assert_eq!(
            choose_probe(&view, ProbeStrategy::Random { seed: 99 }).unwrap(),
            choose_probe(&view, ProbeStrategy::Random { seed: 99 }).unwrap()
        );
    }
}
