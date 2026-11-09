//! Finite exact-enumeration reference policy for blind episode protocol v0.
//!
//! This is a transparent algorithmic baseline, not a learned Symthaea policy.
//! It consumes AgentView only, maintains a finite compatible-model set, and
//! selects receding-horizon partition probes. The evaluator truth is never read.

use crate::protocol::{
    AgentAction, AgentClaim, AgentIntervention, AgentSubmission, AgentView, ClaimDisposition,
    StructureAccess,
};
use crate::{interaction, numeric};

/// Baseline controller over the declared finite hypothesis family.
pub struct FiniteEnumeratorBaseline;

impl FiniteEnumeratorBaseline {
    /// Choose the next probe or submit the current compatible version space.
    /// Every decision is derived solely from the public AgentView.
    pub fn next_action(view: &AgentView) -> Result<AgentAction, PolicyError> {
        if view.status != crate::protocol::EpisodeStatus::Open {
            return Err(PolicyError::EpisodeNotOpen);
        }

        match view.task.world_family {
            crate::protocol::WorldFamily::BooleanMechanism => Self::boolean_action(view),
            crate::protocol::WorldFamily::NumericMechanism => Self::numeric_action(view),
            crate::protocol::WorldFamily::TwoInputResponse => Self::interaction_action(view),
        }
    }

    fn boolean_action(view: &AgentView) -> Result<AgentAction, PolicyError> {
        let access = match view.task.structure_access {
            StructureAccess::BooleanWithheld => crate::StructureAccess::Withheld,
            StructureAccess::BooleanSupplied(mechanism) => {
                crate::StructureAccess::Supplied(mechanism)
            }
            _ => return Err(PolicyError::ManifestFamilyMismatch),
        };
        let log = boolean_log(view)?;
        if log.interventions.is_empty() {
            return Ok(probe(AgentIntervention::Boolean(true)));
        }
        let report = crate::discover(&log, access)
            .map_err(|error| PolicyError::DiscoveryFailure(error.to_string()))?;
        if view.remaining_probe_actions == 0 || report.compatible_hypotheses.len() <= 1 {
            return Ok(submit(
                view,
                AgentClaim::Boolean {
                    disposition: disposition(report.compatible_hypotheses.len()),
                    compatible_models: report.compatible_hypotheses,
                },
            ));
        }
        let horizon = view.remaining_probe_actions.clamp(1, 3);
        let recommendation = crate::recommend_discriminating_probe(&log, &report, horizon)
            .map_err(|error| PolicyError::ProbeFailure(error.to_string()))?;
        match recommendation.and_then(|result| result.interventions.first().copied()) {
            Some(value) => Ok(probe(AgentIntervention::Boolean(value))),
            None => Ok(submit(
                view,
                AgentClaim::Boolean {
                    disposition: disposition(report.compatible_hypotheses.len()),
                    compatible_models: report.compatible_hypotheses,
                },
            )),
        }
    }

    fn numeric_action(view: &AgentView) -> Result<AgentAction, PolicyError> {
        let access = match view.task.structure_access {
            StructureAccess::NumericWithheld => numeric::StructureAccess::Withheld,
            StructureAccess::NumericSupplied(mechanism) => {
                numeric::StructureAccess::Supplied(mechanism)
            }
            _ => return Err(PolicyError::ManifestFamilyMismatch),
        };
        let log = numeric_log(view)?;
        if log.interventions.is_empty() {
            return Ok(probe(AgentIntervention::Numeric(1)));
        }
        let report = numeric::discover(&log, access)
            .map_err(|error| PolicyError::DiscoveryFailure(error.to_string()))?;
        if view.remaining_probe_actions == 0 || report.compatible_models.len() <= 1 {
            return Ok(submit(
                view,
                AgentClaim::Numeric {
                    disposition: disposition(report.compatible_models.len()),
                    compatible_models: report.compatible_models,
                },
            ));
        }
        let horizon = view.remaining_probe_actions.clamp(1, 3);
        let recommendation = numeric::recommend_discriminating_probe(&log, &report, horizon)
            .map_err(|error| PolicyError::ProbeFailure(error.to_string()))?;
        match recommendation.and_then(|result| result.interventions.first().copied()) {
            Some(value) => Ok(probe(AgentIntervention::Numeric(value))),
            None => Ok(submit(
                view,
                AgentClaim::Numeric {
                    disposition: disposition(report.compatible_models.len()),
                    compatible_models: report.compatible_models,
                },
            )),
        }
    }

    fn interaction_action(view: &AgentView) -> Result<AgentAction, PolicyError> {
        let access = match view.task.structure_access {
            StructureAccess::InteractionWithheld => interaction::StructureAccess::Withheld,
            StructureAccess::InteractionSupplied(structure) => {
                interaction::StructureAccess::Supplied(structure)
            }
            _ => return Err(PolicyError::ManifestFamilyMismatch),
        };
        let log = interaction_log(view)?;
        if log.interventions.is_empty() {
            return Ok(probe(AgentIntervention::Pair([1, 1])));
        }
        let report = interaction::discover(&log, access)
            .map_err(|error| PolicyError::DiscoveryFailure(error.to_string()))?;
        if view.remaining_probe_actions == 0 || report.compatible_models.len() <= 1 {
            return Ok(submit(
                view,
                AgentClaim::Interaction {
                    disposition: disposition(report.compatible_models.len()),
                    compatible_models: report.compatible_models,
                },
            ));
        }
        let recommendation = interaction::recommend_discriminating_probe(&log, &report)
            .map_err(|error| PolicyError::ProbeFailure(error.to_string()))?;
        match recommendation.map(|result| result.intervention) {
            Some(value) => Ok(probe(AgentIntervention::Pair(value))),
            None => Ok(submit(
                view,
                AgentClaim::Interaction {
                    disposition: disposition(report.compatible_models.len()),
                    compatible_models: report.compatible_models,
                },
            )),
        }
    }
}

/// Runs the same exact finite inference baseline with interchangeable
/// one-step experiment-selection strategies, enabling controlled policy arms.
pub struct StrategyEnumeratorBaseline;

impl StrategyEnumeratorBaseline {
    pub fn next_action(
        view: &AgentView,
        strategy: crate::strategies::ProbeStrategy,
    ) -> Result<AgentAction, PolicyError> {
        Self::next_action_with_work(view, strategy).map(|(action, _)| action)
    }

    /// Select an action and return deterministic probe-scoring work counts.
    /// The counters cover model predictions used by the one-step probe scorer;
    /// they do not claim to measure discovery filtering, wall time or CPU cost.
    pub fn next_action_with_work(
        view: &AgentView,
        strategy: crate::strategies::ProbeStrategy,
    ) -> Result<(AgentAction, crate::strategies::ProbeScoringWork), PolicyError> {
        use crate::strategies::ProbeScoringWork;

        if view.status != crate::protocol::EpisodeStatus::Open {
            return Err(PolicyError::EpisodeNotOpen);
        }
        match view.task.world_family {
            crate::protocol::WorldFamily::BooleanMechanism => {
                let access = match view.task.structure_access {
                    StructureAccess::BooleanWithheld => crate::StructureAccess::Withheld,
                    StructureAccess::BooleanSupplied(mechanism) => {
                        crate::StructureAccess::Supplied(mechanism)
                    }
                    _ => return Err(PolicyError::ManifestFamilyMismatch),
                };
                let log = boolean_log(view)?;
                let models = if log.interventions.is_empty() {
                    crate::candidate_hypotheses(access)
                } else {
                    crate::discover(&log, access)
                        .map_err(|error| PolicyError::DiscoveryFailure(error.to_string()))?
                        .compatible_hypotheses
                };
                if !log.interventions.is_empty()
                    && (view.remaining_probe_actions == 0 || models.len() <= 1)
                {
                    let count = models.len();
                    return Ok((
                        submit(
                            view,
                            AgentClaim::Boolean {
                                compatible_models: models,
                                disposition: disposition(count),
                            },
                        ),
                        ProbeScoringWork::default(),
                    ));
                }
            }
            crate::protocol::WorldFamily::NumericMechanism => {
                let access = match view.task.structure_access {
                    StructureAccess::NumericWithheld => numeric::StructureAccess::Withheld,
                    StructureAccess::NumericSupplied(mechanism) => {
                        numeric::StructureAccess::Supplied(mechanism)
                    }
                    _ => return Err(PolicyError::ManifestFamilyMismatch),
                };
                let log = numeric_log(view)?;
                let models = if log.interventions.is_empty() {
                    numeric::candidate_models(access)
                } else {
                    numeric::discover(&log, access)
                        .map_err(|error| PolicyError::DiscoveryFailure(error.to_string()))?
                        .compatible_models
                };
                if !log.interventions.is_empty()
                    && (view.remaining_probe_actions == 0 || models.len() <= 1)
                {
                    let count = models.len();
                    return Ok((
                        submit(
                            view,
                            AgentClaim::Numeric {
                                compatible_models: models,
                                disposition: disposition(count),
                            },
                        ),
                        ProbeScoringWork::default(),
                    ));
                }
            }
            crate::protocol::WorldFamily::TwoInputResponse => {
                let access = match view.task.structure_access {
                    StructureAccess::InteractionWithheld => interaction::StructureAccess::Withheld,
                    StructureAccess::InteractionSupplied(structure) => {
                        interaction::StructureAccess::Supplied(structure)
                    }
                    _ => return Err(PolicyError::ManifestFamilyMismatch),
                };
                let log = interaction_log(view)?;
                let models = if log.interventions.is_empty() {
                    interaction::candidate_models(access)
                } else {
                    interaction::discover(&log, access)
                        .map_err(|error| PolicyError::DiscoveryFailure(error.to_string()))?
                        .compatible_models
                };
                if !log.interventions.is_empty()
                    && (view.remaining_probe_actions == 0 || models.len() <= 1)
                {
                    let count = models.len();
                    return Ok((
                        submit(
                            view,
                            AgentClaim::Interaction {
                                compatible_models: models,
                                disposition: disposition(count),
                            },
                        ),
                        ProbeScoringWork::default(),
                    ));
                }
            }
        }
        let action_strategy = strategy_for_decision_step(strategy, view.observations.len());
        let (intervention, work) = crate::strategies::choose_probe_with_work(view, action_strategy)
            .map_err(|error| PolicyError::StrategyFailure(error.to_string()))?;
        Ok((probe(intervention), work))
    }
}

fn strategy_for_decision_step(
    strategy: crate::strategies::ProbeStrategy,
    observed_steps: usize,
) -> crate::strategies::ProbeStrategy {
    match strategy {
        crate::strategies::ProbeStrategy::Random { seed } => {
            let step = (observed_steps as u64).wrapping_add(1);
            crate::strategies::ProbeStrategy::Random {
                seed: seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            }
        }
        other => other,
    }
}

fn probe(intervention: AgentIntervention) -> AgentAction {
    AgentAction::Probe { intervention }
}

fn submit(view: &AgentView, claim: AgentClaim) -> AgentAction {
    AgentAction::Submit {
        submission: AgentSubmission {
            claim,
            evidence_prefix_sha256: view.evidence_prefix_sha256.clone(),
            cited_evidence_ids: view
                .observations
                .iter()
                .map(|record| record.evidence_id.clone())
                .collect(),
            rationale: "Finite enumerator baseline: exact consistency filtering over the declared candidate family; probe selection uses finite outcome partitions.".to_string(),
        },
    }
}

fn disposition(model_count: usize) -> ClaimDisposition {
    match model_count {
        0 => ClaimDisposition::NoCompatibleModel,
        1 => ClaimDisposition::Identified,
        _ => ClaimDisposition::Ambiguous,
    }
}

fn boolean_log(view: &AgentView) -> Result<crate::ObservationLog, PolicyError> {
    let mut interventions = Vec::with_capacity(view.observations.len());
    let mut measurements = Vec::with_capacity(view.observations.len());
    for record in &view.observations {
        match (record.intervention, record.measurement) {
            (
                AgentIntervention::Boolean(input),
                crate::protocol::AgentMeasurement::Boolean(output),
            ) => {
                interventions.push(input);
                measurements.push(output);
            }
            _ => return Err(PolicyError::ObservationFamilyMismatch),
        }
    }
    Ok(crate::ObservationLog {
        interventions,
        measurements,
    })
}

fn numeric_log(view: &AgentView) -> Result<numeric::ObservationLog, PolicyError> {
    let mut interventions = Vec::with_capacity(view.observations.len());
    let mut measurements = Vec::with_capacity(view.observations.len());
    for record in &view.observations {
        match (record.intervention, record.measurement) {
            (
                AgentIntervention::Numeric(input),
                crate::protocol::AgentMeasurement::Numeric(output),
            ) => {
                interventions.push(input);
                measurements.push(output);
            }
            _ => return Err(PolicyError::ObservationFamilyMismatch),
        }
    }
    Ok(numeric::ObservationLog {
        interventions,
        measurements,
    })
}

fn interaction_log(view: &AgentView) -> Result<interaction::ObservationLog, PolicyError> {
    let mut interventions = Vec::with_capacity(view.observations.len());
    let mut measurements = Vec::with_capacity(view.observations.len());
    for record in &view.observations {
        match (record.intervention, record.measurement) {
            (
                AgentIntervention::Pair(input),
                crate::protocol::AgentMeasurement::Numeric(output),
            ) => {
                interventions.push(input);
                measurements.push(output);
            }
            _ => return Err(PolicyError::ObservationFamilyMismatch),
        }
    }
    Ok(interaction::ObservationLog {
        interventions,
        measurements,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    EpisodeNotOpen,
    ManifestFamilyMismatch,
    ObservationFamilyMismatch,
    DiscoveryFailure(String),
    ProbeFailure(String),
    StrategyFailure(String),
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PolicyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{BlindEpisode, EpisodeStatus, ProtocolError};

    fn run_to_submission(episode: &mut BlindEpisode, budget: usize) -> Result<(), ProtocolError> {
        for _ in 0..=budget {
            let view = episode.view()?;
            let action = FiniteEnumeratorBaseline::next_action(&view)
                .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
            let response = episode.apply(action)?;
            if response.status == EpisodeStatus::Submitted {
                return Ok(());
            }
        }
        Err(ProtocolError::NotSubmitted)
    }

    fn run_strategy_to_submission(
        episode: &mut BlindEpisode,
        budget: usize,
        strategy: crate::strategies::ProbeStrategy,
    ) -> Result<(), ProtocolError> {
        for _ in 0..=budget {
            let view = episode.view()?;
            let action = StrategyEnumeratorBaseline::next_action(&view, strategy)
                .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
            let response = episode.apply(action)?;
            if response.status == EpisodeStatus::Submitted {
                return Ok(());
            }
        }
        Err(ProtocolError::NotSubmitted)
    }

    #[test]
    fn matched_numeric_a0_and_a2_use_identical_truth_and_budget() {
        let truth = numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::LeakyMemory,
            gain: 2,
            decay: Some(1),
            initial_state: Some(1),
            sensor: numeric::SensorMap::Identity,
        };
        let mut a0 = BlindEpisode::new_numeric(
            "baseline-numeric-a0",
            truth,
            StructureAccess::NumericSupplied(numeric::Mechanism::LeakyMemory),
            12,
        )
        .unwrap();
        let mut a2 = BlindEpisode::new_numeric(
            "baseline-numeric-a2",
            truth,
            StructureAccess::NumericWithheld,
            12,
        )
        .unwrap();

        run_to_submission(&mut a0, 12).unwrap();
        run_to_submission(&mut a2, 12).unwrap();
        let score_a0 = a0.evaluate_after_submission().unwrap();
        let score_a2 = a2.evaluate_after_submission().unwrap();
        assert!(score_a0.set_score.exact_version_space_match);
        assert!(score_a2.set_score.exact_version_space_match);
        assert!(score_a0.decision_correct);
        assert!(score_a2.decision_correct);
        assert!(score_a0.evidence_references_valid);
        assert!(score_a2.evidence_references_valid);
        assert!(score_a0.generating_model_in_version_space);
        assert!(score_a2.generating_model_in_version_space);
        assert!(score_a0.probe_count <= 12);
        assert!(score_a2.probe_count <= 12);
    }

    #[test]
    fn boolean_baseline_discovers_hidden_lag_and_preserves_generator_model() {
        let mut episode = BlindEpisode::new_boolean(
            "baseline-boolean-a2",
            crate::Mechanism::HiddenLag,
            crate::MeasurementMap::Identity,
            Some(false),
            StructureAccess::BooleanWithheld,
            6,
        )
        .unwrap();
        run_to_submission(&mut episode, 6).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
        assert!(score.generating_model_in_version_space);
        assert!(score.evidence_references_valid);
    }

    #[test]
    fn interaction_baseline_discovers_variable_roles_and_cross_term() {
        let truth = interaction::ModelHypothesis {
            structure: interaction::Structure::BilinearInteraction,
            bias: 0,
            weight_u: 1,
            weight_v: 1,
            interaction_weight: 1,
        };
        let mut episode = BlindEpisode::new_interaction(
            "baseline-interaction-a2",
            truth,
            StructureAccess::InteractionWithheld,
            9,
        )
        .unwrap();
        run_to_submission(&mut episode, 9).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
        assert!(score.generating_model_in_version_space);
    }

    #[test]
    fn strategy_baseline_completes_matched_world_episodes_with_exact_claims() {
        let strategies = [
            crate::strategies::ProbeStrategy::MaximumPartition,
            crate::strategies::ProbeStrategy::Minimax,
            crate::strategies::ProbeStrategy::MaximumEntropy,
            crate::strategies::ProbeStrategy::MinimumExpectedRemaining,
            crate::strategies::ProbeStrategy::FixedGrid,
            crate::strategies::ProbeStrategy::Random { seed: 23 },
        ];

        for strategy in strategies {
            let mut boolean = BlindEpisode::new_boolean(
                "strategy-policy-boolean",
                crate::Mechanism::HiddenLag,
                crate::MeasurementMap::Identity,
                Some(false),
                StructureAccess::BooleanWithheld,
                6,
            )
            .unwrap();
            run_strategy_to_submission(&mut boolean, 6, strategy).unwrap();
            let boolean_score = boolean.evaluate_after_submission().unwrap();
            assert!(boolean_score.set_score.exact_version_space_match);
            assert!(boolean_score.generating_model_in_version_space);

            let numeric_truth = numeric::ModelHypothesis {
                mechanism: numeric::Mechanism::LeakyMemory,
                gain: 2,
                decay: Some(1),
                initial_state: Some(1),
                sensor: numeric::SensorMap::Identity,
            };
            let mut numeric_a0 = BlindEpisode::new_numeric(
                "strategy-policy-numeric-a0",
                numeric_truth,
                StructureAccess::NumericSupplied(numeric::Mechanism::LeakyMemory),
                12,
            )
            .unwrap();
            let mut numeric_a2 = BlindEpisode::new_numeric(
                "strategy-policy-numeric-a2",
                numeric_truth,
                StructureAccess::NumericWithheld,
                12,
            )
            .unwrap();
            run_strategy_to_submission(&mut numeric_a0, 12, strategy).unwrap();
            run_strategy_to_submission(&mut numeric_a2, 12, strategy).unwrap();
            let a0_score = numeric_a0.evaluate_after_submission().unwrap();
            let a2_score = numeric_a2.evaluate_after_submission().unwrap();
            assert!(a0_score.set_score.exact_version_space_match);
            assert!(a2_score.set_score.exact_version_space_match);
            assert!(a0_score.generating_model_in_version_space);
            assert!(a2_score.generating_model_in_version_space);

            let mut interaction_episode = BlindEpisode::new_interaction(
                "strategy-policy-interaction",
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
            run_strategy_to_submission(&mut interaction_episode, 9, strategy).unwrap();
            let interaction_score = interaction_episode.evaluate_after_submission().unwrap();
            assert!(interaction_score.set_score.exact_version_space_match);
            assert!(interaction_score.generating_model_in_version_space);
        }
    }

    #[test]
    fn random_strategy_uses_a_deterministic_distinct_seed_per_decision_step() {
        let policy = crate::strategies::ProbeStrategy::Random { seed: 23 };
        let first = strategy_for_decision_step(policy, 0);
        let second = strategy_for_decision_step(policy, 1);
        assert_eq!(first, strategy_for_decision_step(policy, 0));
        assert_ne!(first, second);
        assert_eq!(
            strategy_for_decision_step(crate::strategies::ProbeStrategy::FixedGrid, 1),
            crate::strategies::ProbeStrategy::FixedGrid
        );
    }

    #[test]
    fn baseline_refuses_to_act_on_a_submitted_view() {
        let mut episode = BlindEpisode::new_boolean(
            "baseline-terminal-01",
            crate::Mechanism::Direct,
            crate::MeasurementMap::Identity,
            None,
            StructureAccess::BooleanWithheld,
            2,
        )
        .unwrap();
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            })
            .unwrap();
        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Boolean {
                compatible_models: crate::candidate_hypotheses(crate::StructureAccess::Withheld),
                disposition: ClaimDisposition::Ambiguous,
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec!["E0001".to_string()],
            rationale: "Terminal-state control.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let terminal_view = episode.view().unwrap();
        assert_eq!(
            FiniteEnumeratorBaseline::next_action(&terminal_view),
            Err(PolicyError::EpisodeNotOpen)
        );
    }
}
