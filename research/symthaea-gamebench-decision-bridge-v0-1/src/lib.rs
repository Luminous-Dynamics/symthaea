//! Game-specific bridge to the shared decision-relevant predictability evaluator.
//!
//! This crate does not implement a second EVSI/robust/sequential oracle. It
//! maps read-only game mechanics probes onto the shared evaluator's
//! CandidateEpisode / EvaluatorWorld / CandidateSubmission contract and adds
//! a proper score for candidate-authored pre-probe outcome forecasts.
//!
//! EvaluatorWorld is evaluator-only. Passing it to a candidate is a data leak.
//! The shared evaluator is deterministic and does not execute probes.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use decision_relevant_predictability::{
    evaluate_candidate, CandidateEpisode, CandidateSubmission, EvaluatorWorld,
    EvaluationError, ProbeAdmission, ProbeValue,
};

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;

/// First bridge version accepts observations that do not mutate game state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeSideEffect {
    ReadOnly,
    WorldMutating,
}

/// Public outcome vocabulary. Hidden likelihoods and task losses remain
/// evaluator-only in EvaluatorWorld.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeOutcomeContract {
    pub probe_id: String,
    pub outcome_ids: Vec<String>,
    pub side_effect: ProbeSideEffect,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeProbability {
    pub outcome_id: String,
    pub probability: f64,
}

/// Candidate-authored distribution emitted before the probe executes.
#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeForecast {
    pub probe_id: String,
    pub probabilities: Vec<OutcomeProbability>,
}

/// Evaluator-only summary passed through from the shared EVSI evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct GameProbeAssessment {
    pub probe_id: String,
    pub admission: ProbeAdmission,
    pub current_bayes_risk: f64,
    pub expected_posterior_bayes_risk: f64,
    pub gross_expected_value_of_sample_information: f64,
    pub acquisition_cost: f64,
    pub net_expected_value_of_sample_information: f64,
    pub state_information_gain_bits: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BridgeError {
    EmptyProbeId,
    EmptyOutcomeSet,
    EmptyOutcomeId,
    DuplicateOutcomeId(String),
    UnknownProbe(String),
    ProbeNotPermitted(String),
    ProbeContractMismatch(String),
    UnsupportedWorldMutatingProbe(String),
    ForecastProbeMismatch { expected: String, actual: String },
    ForecastOutcomeSetMismatch,
    DuplicateForecastOutcome(String),
    InvalidForecastProbability { outcome_id: String, value: f64 },
    ForecastProbabilityMass(f64),
    UnknownObservedOutcome(String),
    Reference(EvaluationError),
}

impl Display for BridgeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProbeId => write!(formatter, "probe ID must not be empty"),
            Self::EmptyOutcomeSet => write!(formatter, "probe outcome set must not be empty"),
            Self::EmptyOutcomeId => write!(formatter, "outcome ID must not be empty"),
            Self::DuplicateOutcomeId(id) => write!(formatter, "duplicate outcome ID: {id}"),
            Self::UnknownProbe(id) => write!(formatter, "unknown probe: {id}"),
            Self::ProbeNotPermitted(id) => write!(formatter, "probe is not permitted: {id}"),
            Self::ProbeContractMismatch(id) => {
                write!(formatter, "public outcome contract mismatches evaluator probe: {id}")
            }
            Self::UnsupportedWorldMutatingProbe(id) => write!(
                formatter,
                "probe {id} changes the world; read-only EVSI bridge refuses to score it"
            ),
            Self::ForecastProbeMismatch { expected, actual } => write!(
                formatter,
                "forecast probe mismatch: expected {expected}, got {actual}"
            ),
            Self::ForecastOutcomeSetMismatch => {
                write!(formatter, "forecast must contain exactly the declared outcome IDs")
            }
            Self::DuplicateForecastOutcome(id) => {
                write!(formatter, "forecast repeats outcome ID: {id}")
            }
            Self::InvalidForecastProbability { outcome_id, value } => {
                write!(formatter, "forecast probability for {outcome_id} is invalid: {value}")
            }
            Self::ForecastProbabilityMass(sum) => {
                write!(formatter, "forecast probability mass is {sum}, expected 1")
            }
            Self::UnknownObservedOutcome(id) => {
                write!(formatter, "observed outcome is not declared: {id}")
            }
            Self::Reference(error) => write!(formatter, "shared reference evaluator: {error}"),
        }
    }
}

impl Error for BridgeError {}

impl From<EvaluationError> for BridgeError {
    fn from(value: EvaluationError) -> Self {
        Self::Reference(value)
    }
}

/// Evaluate one candidate-selected, read-only game probe through the existing
/// shared Bayesian EVSI evaluator. No new EVSI calculation is implemented here.
///
/// The provisional decision is the candidate's current terminal choice before
/// observing the probe. Returned posterior and EVSI values are evaluator-side
/// diagnostics only and must not be fed back to the candidate.
pub fn assess_read_only_probe(
    episode: &CandidateEpisode,
    world: &EvaluatorWorld,
    provisional_decision_id: &str,
    contract: &ProbeOutcomeContract,
    forecast: &OutcomeForecast,
) -> Result<GameProbeAssessment, BridgeError> {
    validate_probe_contract(episode, world, contract)?;
    validate_forecast(contract, forecast)?;

    let evaluation = evaluate_candidate(
        episode,
        world,
        &CandidateSubmission {
            decision_id: provisional_decision_id.to_owned(),
            requested_probe_id: Some(contract.probe_id.clone()),
        },
    )?;

    let probe: ProbeValue = evaluation
        .requested_probe
        .ok_or_else(|| BridgeError::UnknownProbe(contract.probe_id.clone()))?;

    Ok(GameProbeAssessment {
        probe_id: probe.probe_id,
        admission: probe.admission,
        current_bayes_risk: probe.current_bayes_risk,
        expected_posterior_bayes_risk: probe.expected_posterior_bayes_risk,
        gross_expected_value_of_sample_information:
            probe.gross_expected_value_of_sample_information,
        acquisition_cost: probe.acquisition_cost,
        net_expected_value_of_sample_information:
            probe.net_expected_value_of_sample_information,
        state_information_gain_bits: probe.state_information_gain_bits,
    })
}

/// Score a candidate's pre-intervention forecast against the actual
/// externally observed outcome using the multiclass Brier score.
pub fn score_outcome_forecast(
    contract: &ProbeOutcomeContract,
    forecast: &OutcomeForecast,
    observed_outcome_id: &str,
) -> Result<f64, BridgeError> {
    validate_contract_shape(contract)?;
    validate_forecast(contract, forecast)?;

    if !contract.outcome_ids.iter().any(|id| id == observed_outcome_id) {
        return Err(BridgeError::UnknownObservedOutcome(observed_outcome_id.to_owned()));
    }

    let mut score = 0.0;
    for expected_id in &contract.outcome_ids {
        let probability = forecast
            .probabilities
            .iter()
            .find(|entry| &entry.outcome_id == expected_id)
            .expect("forecast validation ensures every contract outcome exists")
            .probability;
        let target = if expected_id == observed_outcome_id { 1.0 } else { 0.0 };
        score += (probability - target).powi(2);
    }
    Ok(score)
}

fn validate_probe_contract(
    episode: &CandidateEpisode,
    world: &EvaluatorWorld,
    contract: &ProbeOutcomeContract,
) -> Result<(), BridgeError> {
    validate_contract_shape(contract)?;

    if contract.side_effect == ProbeSideEffect::WorldMutating {
        return Err(BridgeError::UnsupportedWorldMutatingProbe(contract.probe_id.clone()));
    }
    if !episode.permitted_probes.iter().any(|option| option.probe_id == contract.probe_id) {
        return Err(BridgeError::ProbeNotPermitted(contract.probe_id.clone()));
    }

    let evaluator_probe = world
        .probes
        .iter()
        .find(|probe| probe.probe_id == contract.probe_id)
        .ok_or_else(|| BridgeError::UnknownProbe(contract.probe_id.clone()))?;

    let public_ids: BTreeSet<&str> =
        contract.outcome_ids.iter().map(String::as_str).collect();
    let evaluator_ids: BTreeSet<&str> = evaluator_probe
        .outcomes
        .iter()
        .map(|outcome| outcome.outcome_id.as_str())
        .collect();
    if public_ids != evaluator_ids {
        return Err(BridgeError::ProbeContractMismatch(contract.probe_id.clone()));
    }
    Ok(())
}

fn validate_contract_shape(contract: &ProbeOutcomeContract) -> Result<(), BridgeError> {
    if contract.probe_id.trim().is_empty() {
        return Err(BridgeError::EmptyProbeId);
    }
    if contract.outcome_ids.is_empty() {
        return Err(BridgeError::EmptyOutcomeSet);
    }
    let mut ids = BTreeSet::new();
    for id in &contract.outcome_ids {
        if id.trim().is_empty() {
            return Err(BridgeError::EmptyOutcomeId);
        }
        if !ids.insert(id.as_str()) {
            return Err(BridgeError::DuplicateOutcomeId(id.clone()));
        }
    }
    Ok(())
}

fn validate_forecast(
    contract: &ProbeOutcomeContract,
    forecast: &OutcomeForecast,
) -> Result<(), BridgeError> {
    if forecast.probe_id != contract.probe_id {
        return Err(BridgeError::ForecastProbeMismatch {
            expected: contract.probe_id.clone(),
            actual: forecast.probe_id.clone(),
        });
    }

    let mut seen = BTreeSet::new();
    let mut total = 0.0;
    for entry in &forecast.probabilities {
        if !contract.outcome_ids.iter().any(|id| id == &entry.outcome_id) {
            return Err(BridgeError::ForecastOutcomeSetMismatch);
        }
        if !seen.insert(entry.outcome_id.as_str()) {
            return Err(BridgeError::DuplicateForecastOutcome(entry.outcome_id.clone()));
        }
        if !entry.probability.is_finite() || !(0.0..=1.0).contains(&entry.probability) {
            return Err(BridgeError::InvalidForecastProbability {
                outcome_id: entry.outcome_id.clone(),
                value: entry.probability,
            });
        }
        total += entry.probability;
    }

    if seen.len() != contract.outcome_ids.len()
        || contract.outcome_ids.iter().any(|id| !seen.contains(id.as_str()))
    {
        return Err(BridgeError::ForecastOutcomeSetMismatch);
    }
    if !total.is_finite() || (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(BridgeError::ForecastProbabilityMass(total));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use decision_relevant_predictability::{
        EvidenceChannelModel, OutcomeLikelihood, ProbeModel, PublicProbeOption, StateModel,
    };
    use std::collections::BTreeMap;

    fn state(id: &str, mechanism_a: bool) -> StateModel {
        let choose_a_loss = if mechanism_a { 0.0 } else { 1.0 };
        let choose_b_loss = if mechanism_a { 1.0 } else { 0.0 };
        StateModel {
            state_id: id.to_owned(),
            prior_probability: 0.25,
            decision_loss: BTreeMap::from([
                ("choose_A".into(), choose_a_loss),
                ("choose_B".into(), choose_b_loss),
                ("abstain".into(), 0.4),
            ]),
        }
    }

    fn likelihood(id: &str, entries: &[(&str, f64)]) -> OutcomeLikelihood {
        OutcomeLikelihood {
            outcome_id: id.to_owned(),
            probability_by_state: entries
                .iter()
                .map(|(state, probability)| ((*state).to_owned(), *probability))
                .collect(),
        }
    }

    fn fixture() -> (
        CandidateEpisode,
        EvaluatorWorld,
        ProbeOutcomeContract,
        ProbeOutcomeContract,
    ) {
        let states = vec![
            state("A_N0", true),
            state("A_N1", true),
            state("B_N0", false),
            state("B_N1", false),
        ];

        let nuisance = ProbeModel {
            probe_id: "inspect_lamp".into(),
            acquisition_cost: 0.04,
            result_delay_steps: 1,
            outcomes: vec![
                likelihood("lamp_dark", &[
                    ("A_N0", 1.0), ("A_N1", 0.0),
                    ("B_N0", 1.0), ("B_N1", 0.0),
                ]),
                likelihood("lamp_lit", &[
                    ("A_N0", 0.0), ("A_N1", 1.0),
                    ("B_N0", 0.0), ("B_N1", 1.0),
                ]),
            ],
        };

        let switch = ProbeModel {
            probe_id: "inspect_switch_trace".into(),
            acquisition_cost: 0.04,
            result_delay_steps: 1,
            outcomes: vec![
                likelihood("trace_A", &[
                    ("A_N0", 0.75), ("A_N1", 0.75),
                    ("B_N0", 0.25), ("B_N1", 0.25),
                ]),
                likelihood("trace_B", &[
                    ("A_N0", 0.25), ("A_N1", 0.25),
                    ("B_N0", 0.75), ("B_N1", 0.75),
                ]),
            ],
        };

        let episode = CandidateEpisode {
            episode_id: "gamebench-f6-development-fixture".into(),
            current_step: 0,
            decision_deadline_step: 2,
            remaining_probe_budget: 0.5,
            maximum_worst_case_loss: 1.0,
            allowed_decision_ids: vec![
                "choose_A".into(),
                "choose_B".into(),
                "abstain".into(),
            ],
            evidence: Vec::new(),
            permitted_probes: vec![
                PublicProbeOption {
                    probe_id: "inspect_lamp".into(),
                    acquisition_cost: 0.04,
                    result_delay_steps: 1,
                },
                PublicProbeOption {
                    probe_id: "inspect_switch_trace".into(),
                    acquisition_cost: 0.04,
                    result_delay_steps: 1,
                },
            ],
        };

        let world = EvaluatorWorld {
            states,
            evidence_channels: vec![EvidenceChannelModel {
                channel_id: "unused_development_channel".into(),
                outcomes: vec![likelihood("noop", &[
                    ("A_N0", 1.0), ("A_N1", 1.0),
                    ("B_N0", 1.0), ("B_N1", 1.0),
                ])],
            }],
            probes: vec![nuisance, switch],
            evidence_channels_conditionally_independent: true,
        };

        let nuisance_contract = ProbeOutcomeContract {
            probe_id: "inspect_lamp".into(),
            outcome_ids: vec!["lamp_dark".into(), "lamp_lit".into()],
            side_effect: ProbeSideEffect::ReadOnly,
        };
        let switch_contract = ProbeOutcomeContract {
            probe_id: "inspect_switch_trace".into(),
            outcome_ids: vec!["trace_A".into(), "trace_B".into()],
            side_effect: ProbeSideEffect::ReadOnly,
        };
        (episode, world, nuisance_contract, switch_contract)
    }

    fn forecast(probe_id: &str, first: &str, second: &str, p_first: f64) -> OutcomeForecast {
        OutcomeForecast {
            probe_id: probe_id.to_owned(),
            probabilities: vec![
                OutcomeProbability {
                    outcome_id: first.to_owned(),
                    probability: p_first,
                },
                OutcomeProbability {
                    outcome_id: second.to_owned(),
                    probability: 1.0 - p_first,
                },
            ],
        }
    }

    #[test]
    fn lower_information_probe_has_higher_decision_value_than_nuisance_probe() {
        let (episode, world, nuisance, switch) = fixture();
        let nuisance_result = assess_read_only_probe(
            &episode,
            &world,
            "abstain",
            &nuisance,
            &forecast("inspect_lamp", "lamp_dark", "lamp_lit", 0.5),
        ).expect("nuisance assessment");
        let switch_result = assess_read_only_probe(
            &episode,
            &world,
            "abstain",
            &switch,
            &forecast("inspect_switch_trace", "trace_A", "trace_B", 0.5),
        ).expect("switch assessment");

        assert_eq!(nuisance_result.admission, ProbeAdmission::Eligible);
        assert_eq!(switch_result.admission, ProbeAdmission::Eligible);
        assert!(nuisance_result.state_information_gain_bits > switch_result.state_information_gain_bits);
        assert!(nuisance_result.gross_expected_value_of_sample_information.abs() < 1.0e-9);
        assert!(nuisance_result.net_expected_value_of_sample_information < 0.0);
        assert!((switch_result.gross_expected_value_of_sample_information - 0.15).abs() < 1.0e-9);
        assert!((switch_result.net_expected_value_of_sample_information - 0.11).abs() < 1.0e-9);
    }

    #[test]
    fn perfect_forecast_scores_zero_brier() {
        let (_episode, _world, _nuisance, switch) = fixture();
        let score = score_outcome_forecast(
            &switch,
            &forecast("inspect_switch_trace", "trace_A", "trace_B", 1.0),
            "trace_A",
        ).expect("score");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn probabilistic_forecast_scores_multiclass_brier() {
        let (_episode, _world, _nuisance, switch) = fixture();
        let score = score_outcome_forecast(
            &switch,
            &forecast("inspect_switch_trace", "trace_A", "trace_B", 0.75),
            "trace_A",
        ).expect("score");
        assert!((score - 0.125).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_mutating_probe_in_the_read_only_evsi_adapter() {
        let (episode, world, _nuisance, mut switch) = fixture();
        switch.side_effect = ProbeSideEffect::WorldMutating;
        let result = assess_read_only_probe(
            &episode,
            &world,
            "abstain",
            &switch,
            &forecast("inspect_switch_trace", "trace_A", "trace_B", 0.5),
        );
        assert!(matches!(result, Err(BridgeError::UnsupportedWorldMutatingProbe(_))));
    }

    #[test]
    fn rejects_forecast_with_malformed_probability_mass() {
        let (_episode, _world, _nuisance, switch) = fixture();
        let malformed = OutcomeForecast {
            probe_id: "inspect_switch_trace".into(),
            probabilities: vec![
                OutcomeProbability { outcome_id: "trace_A".into(), probability: 0.7 },
                OutcomeProbability { outcome_id: "trace_B".into(), probability: 0.7 },
            ],
        };
        assert!(matches!(
            score_outcome_forecast(&switch, &malformed, "trace_A"),
            Err(BridgeError::ForecastProbabilityMass(_))
        ));
    }

    #[test]
    fn rejects_duplicate_or_unknown_forecast_outcomes() {
        let (_episode, _world, _nuisance, switch) = fixture();
        let duplicate = OutcomeForecast {
            probe_id: "inspect_switch_trace".into(),
            probabilities: vec![
                OutcomeProbability { outcome_id: "trace_A".into(), probability: 0.5 },
                OutcomeProbability { outcome_id: "trace_A".into(), probability: 0.5 },
            ],
        };
        assert!(matches!(
            score_outcome_forecast(&switch, &duplicate, "trace_A"),
            Err(BridgeError::DuplicateForecastOutcome(_))
        ));

        let unknown = OutcomeForecast {
            probe_id: "inspect_switch_trace".into(),
            probabilities: vec![
                OutcomeProbability { outcome_id: "trace_A".into(), probability: 0.5 },
                OutcomeProbability { outcome_id: "hidden_rule_label".into(), probability: 0.5 },
            ],
        };
        assert!(matches!(
            score_outcome_forecast(&switch, &unknown, "trace_A"),
            Err(BridgeError::ForecastOutcomeSetMismatch)
        ));
    }

    #[test]
    fn rejects_public_outcome_vocabulary_that_does_not_match_evaluator() {
        let (episode, world, _nuisance, _switch) = fixture();
        let wrong = ProbeOutcomeContract {
            probe_id: "inspect_switch_trace".into(),
            outcome_ids: vec!["H_A".into(), "H_B".into()],
            side_effect: ProbeSideEffect::ReadOnly,
        };
        let result = assess_read_only_probe(
            &episode,
            &world,
            "abstain",
            &wrong,
            &forecast("inspect_switch_trace", "H_A", "H_B", 0.5),
        );
        assert!(matches!(result, Err(BridgeError::ProbeContractMismatch(_))));
    }
}
