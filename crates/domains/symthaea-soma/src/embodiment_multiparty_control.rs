// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Multi-party addressee, silence-preservation and human-resolution benchmark.
//!
//! The evaluator consumes scenario annotations and independently graded
//! predictions. It contains no audio, transcripts, speaker biometrics or media.
//! It does not detect speakers or determine addressees by itself.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const MULTIPARTY_CONTROL_SCHEMA_V1: &str = "symthaea.multiparty-control.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultipartyAddressing {
    DirectToAgent,
    DirectToOther,
    ImplicitToOther,
    GroupBroadcast,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultipartyDecision {
    Respond,
    ContinueSpeaking,
    StaySilent,
    StopSpeaking,
    AskClarifying,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultipartyScenarioV1 {
    pub schema: String,
    pub scenario_id: String,
    /// Opaque participant tokens only; no names, voices or biometric IDs.
    pub participant_ids: Vec<String>,
    pub agent_participant_id: String,
    pub current_speaker_id: String,
    pub addressee_id: Option<String>,
    pub addressing: MultipartyAddressing,
    pub human_resolved_request: bool,
    pub agent_was_speaking: bool,
    pub source_event_digest: String,
    pub response_deadline_ms: u32,
}

impl MultipartyScenarioV1 {
    pub fn expected_decision(&self) -> MultipartyDecision {
        if self.human_resolved_request {
            return if self.agent_was_speaking {
                MultipartyDecision::StopSpeaking
            } else {
                MultipartyDecision::StaySilent
            };
        }
        match self.addressing {
            MultipartyAddressing::DirectToAgent | MultipartyAddressing::GroupBroadcast => {
                if self.agent_was_speaking {
                    MultipartyDecision::ContinueSpeaking
                } else {
                    MultipartyDecision::Respond
                }
            }
            MultipartyAddressing::DirectToOther | MultipartyAddressing::ImplicitToOther => {
                if self.agent_was_speaking {
                    MultipartyDecision::StopSpeaking
                } else {
                    MultipartyDecision::StaySilent
                }
            }
            MultipartyAddressing::Ambiguous => {
                if self.agent_was_speaking {
                    MultipartyDecision::StopSpeaking
                } else {
                    MultipartyDecision::AskClarifying
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultipartyPredictionV1 {
    pub scenario_id: String,
    pub decision: MultipartyDecision,
    pub response_target_id: Option<String>,
    /// Set by an independent answer grader for Respond/ContinueSpeaking only.
    pub answer_correct: Option<bool>,
    pub decision_latency_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultipartyControlError {
    WrongSchema,
    EmptyEvaluationSet,
    InvalidIdentifier,
    InvalidDigest,
    InvalidParticipantCount,
    DuplicateParticipant,
    UnknownAgent,
    UnknownSpeaker,
    InvalidAddressee,
    MissingAddressee,
    UnexpectedAddressee,
    InvalidDeadline,
    DuplicateScenario,
    DuplicatePrediction,
    MissingPrediction,
    UnexpectedPrediction,
    MissingAnswerGrade,
    UnexpectedAnswerGrade,
    MissingResponseTarget,
    UnexpectedResponseTarget,
    InvalidResponseTarget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultipartyControlMetrics {
    pub item_count: usize,
    pub action_correct_count: usize,
    pub action_accuracy: Option<f32>,
    pub explicitly_addressed_count: usize,
    pub addressed_response_count: usize,
    pub addressed_response_rate: Option<f32>,
    pub unaddressed_count: usize,
    pub silence_preserved_count: usize,
    pub silence_preservation_rate: Option<f32>,
    pub false_activation_count: usize,
    pub false_activation_rate: Option<f32>,
    pub human_resolution_count: usize,
    pub stop_compliant_count: usize,
    pub stop_on_resolution_rate: Option<f32>,
    pub answer_graded_count: usize,
    pub answers_correct_count: usize,
    pub answer_accuracy: Option<f32>,
    pub deadline_met_count: usize,
    pub deadline_met_rate: Option<f32>,
    pub median_decision_latency_ms: u32,
    pub p95_decision_latency_ms: u32,
}

pub fn verify_multiparty_scenario(
    scenario: &MultipartyScenarioV1,
) -> Result<(), MultipartyControlError> {
    if scenario.schema != MULTIPARTY_CONTROL_SCHEMA_V1 {
        return Err(MultipartyControlError::WrongSchema);
    }
    validate_identifier(&scenario.scenario_id)?;
    validate_identifier(&scenario.agent_participant_id)?;
    validate_identifier(&scenario.current_speaker_id)?;
    validate_digest(&scenario.source_event_digest)?;
    if !(3..=4).contains(&scenario.participant_ids.len()) {
        return Err(MultipartyControlError::InvalidParticipantCount);
    }
    if scenario.response_deadline_ms == 0 {
        return Err(MultipartyControlError::InvalidDeadline);
    }

    let mut participants = HashSet::with_capacity(scenario.participant_ids.len());
    for participant in &scenario.participant_ids {
        validate_identifier(participant)?;
        if !participants.insert(participant.as_str()) {
            return Err(MultipartyControlError::DuplicateParticipant);
        }
    }
    if !participants.contains(scenario.agent_participant_id.as_str()) {
        return Err(MultipartyControlError::UnknownAgent);
    }
    if !participants.contains(scenario.current_speaker_id.as_str()) {
        return Err(MultipartyControlError::UnknownSpeaker);
    }

    match scenario.addressing {
        MultipartyAddressing::DirectToAgent => {
            if scenario.addressee_id.as_deref()
                != Some(scenario.agent_participant_id.as_str())
            {
                return Err(MultipartyControlError::InvalidAddressee);
            }
        }
        MultipartyAddressing::DirectToOther | MultipartyAddressing::ImplicitToOther => {
            let addressee = scenario.addressee_id.as_deref()
                .ok_or(MultipartyControlError::MissingAddressee)?;
            if !participants.contains(addressee)
                || addressee == scenario.agent_participant_id
                || addressee == scenario.current_speaker_id
            {
                return Err(MultipartyControlError::InvalidAddressee);
            }
        }
        MultipartyAddressing::GroupBroadcast | MultipartyAddressing::Ambiguous => {
            if scenario.addressee_id.is_some() {
                return Err(MultipartyControlError::UnexpectedAddressee);
            }
        }
    }
    Ok(())
}

pub fn verify_multiparty_prediction(
    scenario: &MultipartyScenarioV1,
    prediction: &MultipartyPredictionV1,
) -> Result<(), MultipartyControlError> {
    verify_multiparty_scenario(scenario)?;
    validate_identifier(&prediction.scenario_id)?;
    if prediction.scenario_id != scenario.scenario_id {
        return Err(MultipartyControlError::UnexpectedPrediction);
    }

    let answers = matches!(
        prediction.decision,
        MultipartyDecision::Respond | MultipartyDecision::ContinueSpeaking
    );
    if answers != prediction.answer_correct.is_some() {
        return Err(if answers {
            MultipartyControlError::MissingAnswerGrade
        } else {
            MultipartyControlError::UnexpectedAnswerGrade
        });
    }
    if answers {
        let target = prediction.response_target_id.as_deref()
            .ok_or(MultipartyControlError::MissingResponseTarget)?;
        if target != scenario.current_speaker_id {
            return Err(MultipartyControlError::InvalidResponseTarget);
        }
    } else if prediction.response_target_id.is_some() {
        return Err(MultipartyControlError::UnexpectedResponseTarget);
    }
    Ok(())
}

/// Aggregate one prediction per scenario. Action, silence, human-stop, answer
/// and deadline metrics remain separate; no single weighted score hides them.
pub fn score_multiparty_control(
    scenarios: &[MultipartyScenarioV1],
    predictions: &[MultipartyPredictionV1],
) -> Result<MultipartyControlMetrics, MultipartyControlError> {
    if scenarios.is_empty() {
        return Err(MultipartyControlError::EmptyEvaluationSet);
    }
    let mut scenario_by_id = HashMap::with_capacity(scenarios.len());
    for scenario in scenarios {
        verify_multiparty_scenario(scenario)?;
        if scenario_by_id.insert(scenario.scenario_id.as_str(), scenario).is_some() {
            return Err(MultipartyControlError::DuplicateScenario);
        }
    }

    let mut prediction_by_id = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if prediction_by_id.insert(prediction.scenario_id.as_str(), prediction).is_some() {
            return Err(MultipartyControlError::DuplicatePrediction);
        }
        if !scenario_by_id.contains_key(prediction.scenario_id.as_str()) {
            return Err(MultipartyControlError::UnexpectedPrediction);
        }
    }
    if predictions.len() != scenarios.len() {
        return Err(MultipartyControlError::MissingPrediction);
    }

    let mut action_correct_count = 0;
    let mut explicitly_addressed_count = 0;
    let mut addressed_response_count = 0;
    let mut unaddressed_count = 0;
    let mut silence_preserved_count = 0;
    let mut false_activation_count = 0;
    let mut human_resolution_count = 0;
    let mut stop_compliant_count = 0;
    let mut answer_graded_count = 0;
    let mut answers_correct_count = 0;
    let mut deadline_met_count = 0;
    let mut latencies = Vec::with_capacity(predictions.len());

    for scenario in scenarios {
        let prediction = prediction_by_id[scenario.scenario_id.as_str()];
        verify_multiparty_prediction(scenario, prediction)?;
        let expected = scenario.expected_decision();
        let correct = prediction.decision == expected;
        action_correct_count += usize::from(correct);
        latencies.push(prediction.decision_latency_ms);
        deadline_met_count += usize::from(
            prediction.decision_latency_ms <= scenario.response_deadline_ms,
        );

        if matches!(
            scenario.addressing,
            MultipartyAddressing::DirectToAgent | MultipartyAddressing::GroupBroadcast
        ) && !scenario.human_resolved_request {
            explicitly_addressed_count += 1;
            addressed_response_count += usize::from(
                matches!(
                    prediction.decision,
                    MultipartyDecision::Respond | MultipartyDecision::ContinueSpeaking
                ),
            );
        }

        if matches!(
            scenario.addressing,
            MultipartyAddressing::DirectToOther | MultipartyAddressing::ImplicitToOther
        ) {
            unaddressed_count += 1;
            let silent = matches!(
                prediction.decision,
                MultipartyDecision::StaySilent | MultipartyDecision::StopSpeaking
            );
            silence_preserved_count += usize::from(silent);
            false_activation_count += usize::from(!silent);
        }

        if scenario.human_resolved_request {
            human_resolution_count += 1;
            let stopped = if scenario.agent_was_speaking {
                prediction.decision == MultipartyDecision::StopSpeaking
            } else {
                prediction.decision == MultipartyDecision::StaySilent
            };
            stop_compliant_count += usize::from(stopped);
        }

        if let Some(answer_correct) = prediction.answer_correct {
            answer_graded_count += 1;
            answers_correct_count += usize::from(answer_correct);
        }
    }

    latencies.sort_unstable();
    let median = percentile_nearest_rank(&latencies, 0.50);
    let p95 = percentile_nearest_rank(&latencies, 0.95);
    Ok(MultipartyControlMetrics {
        item_count: scenarios.len(),
        action_correct_count,
        action_accuracy: rate(action_correct_count, scenarios.len()),
        explicitly_addressed_count,
        addressed_response_count,
        addressed_response_rate: rate(addressed_response_count, explicitly_addressed_count),
        unaddressed_count,
        silence_preserved_count,
        silence_preservation_rate: rate(silence_preserved_count, unaddressed_count),
        false_activation_count,
        false_activation_rate: rate(false_activation_count, unaddressed_count),
        human_resolution_count,
        stop_compliant_count,
        stop_on_resolution_rate: rate(stop_compliant_count, human_resolution_count),
        answer_graded_count,
        answers_correct_count,
        answer_accuracy: rate(answers_correct_count, answer_graded_count),
        deadline_met_count,
        deadline_met_rate: rate(deadline_met_count, scenarios.len()),
        median_decision_latency_ms: median,
        p95_decision_latency_ms: p95,
    })
}

fn rate(numerator: usize, denominator: usize) -> Option<f32> {
    (denominator > 0).then_some(numerator as f32 / denominator as f32)
}

fn percentile_nearest_rank(sorted: &[u32], quantile: f32) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f32 * quantile).ceil() as usize).max(1);
    sorted[rank.min(sorted.len()) - 1]
}

fn validate_identifier(value: &str) -> Result<(), MultipartyControlError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(MultipartyControlError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), MultipartyControlError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MultipartyControlError::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: u8) -> String { std::iter::repeat_n(ch as char, 64).collect() }

    fn scenario(id: &str, addressing: MultipartyAddressing, resolved: bool, agent_speaking: bool) -> MultipartyScenarioV1 {
        let addressee_id = match addressing {
            MultipartyAddressing::DirectToAgent => Some("agent".to_owned()),
            MultipartyAddressing::DirectToOther | MultipartyAddressing::ImplicitToOther => Some("human-b".to_owned()),
            MultipartyAddressing::GroupBroadcast | MultipartyAddressing::Ambiguous => None,
        };
        MultipartyScenarioV1 {
            schema: MULTIPARTY_CONTROL_SCHEMA_V1.to_owned(),
            scenario_id: id.to_owned(),
            participant_ids: vec!["agent".to_owned(), "human-a".to_owned(), "human-b".to_owned()],
            agent_participant_id: "agent".to_owned(),
            current_speaker_id: "human-a".to_owned(),
            addressee_id,
            addressing,
            human_resolved_request: resolved,
            agent_was_speaking: agent_speaking,
            source_event_digest: digest(b'a'),
            response_deadline_ms: 800,
        }
    }

    fn prediction(scenario: &MultipartyScenarioV1, decision: MultipartyDecision) -> MultipartyPredictionV1 {
        let answers = matches!(decision, MultipartyDecision::Respond | MultipartyDecision::ContinueSpeaking);
        MultipartyPredictionV1 {
            scenario_id: scenario.scenario_id.clone(),
            decision,
            response_target_id: answers.then(|| scenario.current_speaker_id.clone()),
            answer_correct: answers.then_some(true),
            decision_latency_ms: 150,
        }
    }

    #[test]
    fn implicit_request_to_another_person_requires_silence() {
        let value = scenario("s1", MultipartyAddressing::ImplicitToOther, false, false);
        let guess = prediction(&value, MultipartyDecision::StaySilent);
        verify_multiparty_prediction(&value, &guess).unwrap();
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.silence_preservation_rate, Some(1.0));
        assert_eq!(metrics.false_activation_count, 0);
    }

    #[test]
    fn response_to_an_unaddressed_person_is_a_false_activation() {
        let value = scenario("s1", MultipartyAddressing::DirectToOther, false, false);
        let guess = prediction(&value, MultipartyDecision::Respond);
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.false_activation_count, 1);
        assert_eq!(metrics.silence_preservation_rate, Some(0.0));
    }

    #[test]
    fn assistant_stops_when_a_human_resolves_its_active_request() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, true, true);
        let guess = prediction(&value, MultipartyDecision::StopSpeaking);
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.stop_on_resolution_rate, Some(1.0));
    }

    #[test]
    fn assistant_does_not_invent_a_stop_when_already_silent() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, true, false);
        let guess = prediction(&value, MultipartyDecision::StaySilent);
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.stop_on_resolution_rate, Some(1.0));
    }

    #[test]
    fn ambiguous_addressee_requires_clarification() {
        let value = scenario("s1", MultipartyAddressing::Ambiguous, false, false);
        let guess = prediction(&value, MultipartyDecision::AskClarifying);
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.action_accuracy, Some(1.0));
    }

    #[test]
    fn explicit_agent_request_counts_as_addressed_response() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, false, false);
        let guess = prediction(&value, MultipartyDecision::Respond);
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.addressed_response_rate, Some(1.0));
        assert_eq!(metrics.answer_accuracy, Some(1.0));
    }

    #[test]
    fn direct_to_other_cannot_name_agent_as_addressee() {
        let mut value = scenario("s1", MultipartyAddressing::DirectToOther, false, false);
        value.addressee_id = Some("agent".to_owned());
        assert_eq!(
            verify_multiparty_scenario(&value),
            Err(MultipartyControlError::InvalidAddressee)
        );
    }

    #[test]
    fn missing_prediction_is_not_a_silent_success() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, false, false);
        assert_eq!(
            score_multiparty_control(&[value], &[]),
            Err(MultipartyControlError::MissingPrediction)
        );
    }

    #[test]
    fn duplicate_predictions_are_rejected() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, false, false);
        let guess = prediction(&value, MultipartyDecision::Respond);
        assert_eq!(
            score_multiparty_control(&[value], &[guess.clone(), guess]),
            Err(MultipartyControlError::DuplicatePrediction)
        );
    }


    #[test]
    fn undefined_rates_remain_none_when_a_category_has_no_items() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, true, false);
        let guess = prediction(&value, MultipartyDecision::StaySilent);
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.silence_preservation_rate, None);
        assert_eq!(metrics.false_activation_rate, None);
        assert_eq!(metrics.stop_on_resolution_rate, Some(1.0));
        assert_eq!(metrics.answer_accuracy, None);
    }

    #[test]
    fn empty_evaluation_set_is_not_reported_as_zero_percent() {
        assert_eq!(
            score_multiparty_control(&[], &[]),
            Err(MultipartyControlError::EmptyEvaluationSet)
        );
    }

    #[test]
    fn deadline_metrics_use_decision_latency() {
        let value = scenario("s1", MultipartyAddressing::DirectToAgent, false, false);
        let mut guess = prediction(&value, MultipartyDecision::Respond);
        guess.decision_latency_ms = 900;
        let metrics = score_multiparty_control(&[value], &[guess]).unwrap();
        assert_eq!(metrics.deadline_met_rate, Some(0.0));
    }
}
