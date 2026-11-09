//! Comparable one-step experimental-design baselines over the blind episode API.
//!
//! These are policy controls, not claims of optimality. Every strategy sees the
//! same AgentView and scores only the interventions legal in the public task.

use crate::protocol::{
    AgentIntervention, AgentMeasurement, AgentView, EpisodeStatus, StructureAccess,
};
use crate::{interaction, numeric};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

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
    /// Maximize information gain over structural families with equal family prior.
    /// Parameterizations are weighted uniformly only within each family.
    MaximumStructureInformationGain,
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
    /// Mutual information between the next observation and structural family,
    /// using equal prior mass per surviving family and a uniform conditional
    /// prior over its surviving parameterizations.
    pub structure_information_gain_bits: f64,
    /// Maximum number of structural families compatible with any one outcome.
    pub worst_case_remaining_structures: usize,
    pub probe_cost_units: u64,
}

impl ProbeMetrics {
    pub fn expected_remaining_models(&self) -> f64 {
        if self.expected_remaining_denominator == 0 {
            return 0.0;
        }
        self.expected_remaining_numerator as f64 / self.expected_remaining_denominator as f64
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

/// Deterministic algorithmic-work proxy for one probe-selection decision.
/// These counts describe enumerated candidate predictions, not CPU time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeScoringWork {
    pub legal_actions_scored: u64,
    pub model_prediction_evaluations: u64,
}

/// Select a legal one-step action and return the exact number of model/action
/// predictions consumed by the finite partition scorer.
pub fn choose_probe_with_work(
    view: &AgentView,
    strategy: ProbeStrategy,
) -> Result<(AgentIntervention, ProbeScoringWork), StrategyError> {
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
        _ => metrics
            .iter()
            .max_by(|left, right| compare_metrics(left, right, strategy)),
    };
    let selected = selected.ok_or(StrategyError::NoLegalProbe)?;
    let work = ProbeScoringWork {
        legal_actions_scored: metrics.len() as u64,
        model_prediction_evaluations: metrics
            .iter()
            .map(|metric| metric.expected_remaining_denominator as u64)
            .sum(),
    };
    Ok((selected.intervention, work))
}

/// Compatibility helper for callers that need only the selected intervention.
pub fn choose_probe(
    view: &AgentView,
    strategy: ProbeStrategy,
) -> Result<AgentIntervention, StrategyError> {
    choose_probe_with_work(view, strategy).map(|(intervention, _)| intervention)
}

fn compare_metrics(left: &ProbeMetrics, right: &ProbeMetrics, strategy: ProbeStrategy) -> Ordering {
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
        ProbeStrategy::MaximumStructureInformationGain => left
            .structure_information_gain_bits
            .total_cmp(&right.structure_information_gain_bits),
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
        structure_information_gain_bits: 0.0,
        worst_case_remaining_structures: 0,
        probe_cost_units: 1,
    })
}

/// Attach structure-balanced one-step mutual information to ordinary
/// parameterized-model partition metrics. The prior is uniform over the
/// structural families that still have support, then uniform over the
/// surviving parameterizations within each family. This avoids letting a
/// larger syntactic family dominate merely because it has more parameters.
fn attach_structure_metrics<T, S>(
    mut score: ProbeMetrics,
    predictions: impl IntoIterator<Item = (T, S)>,
) -> Result<ProbeMetrics, StrategyError>
where
    T: Ord + Clone,
    S: Ord,
{
    let mut by_structure: BTreeMap<S, BTreeMap<T, usize>> = BTreeMap::new();
    let mut total_predictions = 0usize;
    for (outcome, structure) in predictions {
        *by_structure
            .entry(structure)
            .or_default()
            .entry(outcome)
            .or_default() += 1;
        total_predictions += 1;
    }
    if total_predictions == 0 {
        return Err(StrategyError::NoCompatibleModels);
    }
    if total_predictions != score.expected_remaining_denominator {
        return Err(StrategyError::PredictionCardinalityMismatch);
    }

    let family_count = by_structure.len();
    let family_prior = 1.0 / family_count as f64;
    let mut outcome_probability: BTreeMap<T, f64> = BTreeMap::new();
    let mut family_sizes = BTreeMap::new();

    for (structure, outcomes) in &by_structure {
        let family_size: usize = outcomes.values().sum();
        if family_size == 0 {
            return Err(StrategyError::PredictionCardinalityMismatch);
        }
        family_sizes.insert(structure, family_size);
        for (outcome, count) in outcomes {
            *outcome_probability.entry(outcome.clone()).or_default() +=
                family_prior * (*count as f64 / family_size as f64);
        }
    }

    let prior_entropy = (family_count as f64).log2();
    let mut expected_posterior_entropy = 0.0;
    let mut worst_case_remaining_structures = 0usize;

    for (outcome, probability) in &outcome_probability {
        if *probability <= 0.0 {
            return Err(StrategyError::InvalidOutcomeProbability);
        }
        let mut posterior_entropy = 0.0;
        let mut compatible_families = 0usize;

        for (structure, outcomes) in &by_structure {
            let count = outcomes.get(outcome).copied().unwrap_or(0);
            if count == 0 {
                continue;
            }
            compatible_families += 1;
            let family_size = family_sizes[structure];
            let posterior = family_prior * (count as f64 / family_size as f64) / probability;
            posterior_entropy -= posterior * posterior.log2();
        }

        worst_case_remaining_structures = worst_case_remaining_structures.max(compatible_families);
        expected_posterior_entropy += probability * posterior_entropy;
    }

    score.structure_information_gain_bits = (prior_entropy - expected_posterior_entropy).max(0.0);
    score.worst_case_remaining_structures = worst_case_remaining_structures;
    Ok(score)
}

fn score_boolean(view: &AgentView) -> Result<Vec<ProbeMetrics>, StrategyError> {
    let access = match view.task.structure_access {
        StructureAccess::BooleanWithheld => crate::StructureAccess::Withheld,
        StructureAccess::BooleanSupplied(mechanism) => crate::StructureAccess::Supplied(mechanism),
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
            let score = metrics(
                AgentIntervention::Boolean(input),
                values.iter().copied(),
                hypotheses.len(),
            )?;
            attach_structure_metrics(
                score,
                hypotheses
                    .iter()
                    .zip(values)
                    .map(|(hypothesis, output)| (output, hypothesis.mechanism)),
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
            let score = metrics(
                AgentIntervention::Numeric(input),
                values.iter().copied(),
                hypotheses.len(),
            )?;
            attach_structure_metrics(
                score,
                hypotheses
                    .iter()
                    .zip(values)
                    .map(|(hypothesis, output)| (output, hypothesis.mechanism)),
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

    let observed: BTreeSet<[i64; 2]> = log.interventions.iter().copied().collect();
    interaction::FACTORIAL_DESIGN
        .into_iter()
        .filter(|input| !observed.contains(input))
        .map(|input| {
            let predictions = hypotheses.iter().map(|hypothesis| {
                hypothesis
                    .predict(input)
                    .map_err(|error| StrategyError::PredictionFailure(error.to_string()))
            });
            let values = predictions.collect::<Result<Vec<_>, _>>()?;
            let score = metrics(
                AgentIntervention::Pair(input),
                values.iter().copied(),
                hypotheses.len(),
            )?;
            attach_structure_metrics(
                score,
                hypotheses
                    .iter()
                    .zip(values)
                    .map(|(hypothesis, output)| (output, hypothesis.structure)),
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
    InvalidOutcomeProbability,
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
    use crate::protocol::{AgentAction, BlindEpisode};

    #[test]
    fn all_strategies_return_a_legal_probe_in_each_world_family() {
        let strategies = [
            ProbeStrategy::MaximumPartition,
            ProbeStrategy::Minimax,
            ProbeStrategy::MaximumEntropy,
            ProbeStrategy::MinimumExpectedRemaining,
            ProbeStrategy::MaximumStructureInformationGain,
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
            )
            .unwrap();
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
            )
            .unwrap();
            let view = numeric_episode.view().unwrap();
            let intervention = choose_probe(&view, strategy).unwrap();
            let response = numeric_episode
                .apply(AgentAction::Probe { intervention })
                .unwrap();
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
            )
            .unwrap();
            let view = interaction_episode.view().unwrap();
            let intervention = choose_probe(&view, strategy).unwrap();
            let response = interaction_episode
                .apply(AgentAction::Probe { intervention })
                .unwrap();
            assert_eq!(response.view.observations.len(), 1);
        }
    }

    #[test]
    fn interaction_strategy_never_recommends_an_already_observed_design_point() {
        let mut episode = BlindEpisode::new_interaction(
            "strategy-no-repeat-01",
            interaction::ModelHypothesis {
                structure: interaction::Structure::BilinearInteraction,
                bias: 0,
                weight_u: 1,
                weight_v: 1,
                interaction_weight: 1,
            },
            StructureAccess::InteractionWithheld,
            9,
        )
        .unwrap();
        let first_view = episode.view().unwrap();
        let first = choose_probe(&first_view, ProbeStrategy::FixedGrid).unwrap();
        episode
            .apply(AgentAction::Probe {
                intervention: first,
            })
            .unwrap();
        let second_view = episode.view().unwrap();
        let remaining = score_available_probes(&second_view).unwrap();
        assert_eq!(remaining.len(), 8);
        assert!(!remaining.iter().any(|metric| metric.intervention == first));
    }

    #[test]
    fn expected_remaining_count_is_exact_for_uniform_partition_weights() {
        let result = metrics(AgentIntervention::Numeric(0), [0, 0, 1, 1, 1, 2], 6).unwrap();
        assert_eq!(result.outcome_partition_count, 3);
        assert_eq!(result.worst_case_remaining_models, 3);
        assert_eq!(result.expected_remaining_numerator, 14);
        assert_eq!(result.expected_remaining_denominator, 6);
        assert!((result.expected_remaining_models() - (14.0 / 6.0)).abs() < 1e-12);
        assert!((result.entropy_bits - (1.4591479170272448)).abs() < 1e-12);
    }

    #[test]
    fn structure_information_gain_is_stable_to_family_cardinality_when_outcome_mix_is_unchanged() {
        let balanced = vec![
            (0_i64, 0_usize),
            (0_i64, 0_usize),
            (0_i64, 1_usize),
            (1_i64, 1_usize),
        ];
        // Expand one family without changing its within-family outcome distribution.
        let overparameterized = std::iter::once((0_i64, 1_usize))
            .chain(std::iter::once((1_i64, 1_usize)))
            .chain(std::iter::repeat_n((0_i64, 0_usize), 100))
            .collect::<Vec<_>>();

        let score = metrics(
            AgentIntervention::Numeric(0),
            balanced.iter().map(|(outcome, _)| *outcome),
            balanced.len(),
        )
        .unwrap();
        let balanced_score = attach_structure_metrics(score, balanced).unwrap();

        let expanded = metrics(
            AgentIntervention::Numeric(0),
            overparameterized.iter().map(|(outcome, _)| *outcome),
            overparameterized.len(),
        )
        .unwrap();
        let expanded_score = attach_structure_metrics(expanded, overparameterized).unwrap();

        assert!(
            (balanced_score.structure_information_gain_bits - 0.31127812445913283).abs() < 1e-12
        );
        assert!(
            (balanced_score.structure_information_gain_bits
                - expanded_score.structure_information_gain_bits)
                .abs()
                < 1e-12
        );
        assert_eq!(balanced_score.worst_case_remaining_structures, 2);
        assert_eq!(expanded_score.worst_case_remaining_structures, 2);
        assert!(expanded_score.structure_information_gain_bits > expanded_score.entropy_bits);
    }

    #[test]
    fn structure_balanced_policy_prefers_a_mechanism_probe_over_model_entropy() {
        let episode = BlindEpisode::new_numeric(
            "strategy-structure-balanced-01",
            numeric::ModelHypothesis {
                mechanism: numeric::Mechanism::DirectResponse,
                gain: 2,
                decay: None,
                initial_state: None,
                sensor: numeric::SensorMap::Identity,
            },
            StructureAccess::NumericWithheld,
            4,
        )
        .unwrap();
        let view = episode.view().unwrap();

        let structure_probe =
            choose_probe(&view, ProbeStrategy::MaximumStructureInformationGain).unwrap();
        let model_entropy_probe = choose_probe(&view, ProbeStrategy::MaximumEntropy).unwrap();
        let scores = score_available_probes(&view).unwrap();
        let zero_probe = scores
            .iter()
            .find(|score| score.intervention == AgentIntervention::Numeric(0))
            .unwrap();
        let positive_probe = scores
            .iter()
            .find(|score| score.intervention == AgentIntervention::Numeric(1))
            .unwrap();

        assert_eq!(structure_probe, AgentIntervention::Numeric(0));
        assert_ne!(model_entropy_probe, AgentIntervention::Numeric(0));
        assert!(
            zero_probe.structure_information_gain_bits
                > positive_probe.structure_information_gain_bits
        );
        assert!(zero_probe.entropy_bits < positive_probe.entropy_bits);
    }

    #[test]
    fn structure_information_gain_is_zero_when_only_one_family_remains() {
        let score = metrics(AgentIntervention::Numeric(1), [0_i64, 1, 1, 2], 4).unwrap();
        let score =
            attach_structure_metrics(score, [(0_i64, 7_usize), (1, 7), (1, 7), (2, 7)]).unwrap();
        assert_eq!(score.structure_information_gain_bits, 0.0);
        assert_eq!(score.worst_case_remaining_structures, 1);
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
        )
        .unwrap();
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
