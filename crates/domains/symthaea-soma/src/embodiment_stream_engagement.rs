// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Continuous-stream participation and response-window benchmark.
//!
//! Inspired by DIVE's separation of response quality, participation decision
//! and reaction timing. This evaluator consumes annotated opportunity windows
//! and independently graded predictions; it does not infer events from media.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;

pub const STREAM_ENGAGEMENT_SCHEMA_V1: &str = "symthaea.stream-engagement.v1";
pub const MAX_STREAM_OPPORTUNITY_WINDOW_NS: u64 = 30_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamOpportunityKind {
    ProactiveVisualCue,
    IncomingNewInformation,
    ListenerBackchannel,
    OtherDirectedSpeech,
    AmbientSpeech,
    HumanResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamAction {
    Intervene,
    AdaptOutput,
    ContinueOutput,
    StaySilent,
    StopOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOpportunityV1 {
    pub schema: String,
    pub opportunity_id: String,
    pub stream_id: String,
    pub clock_domain_id: String,
    pub event_start_monotonic_ns: u64,
    pub response_deadline_monotonic_ns: u64,
    pub kind: StreamOpportunityKind,
    pub agent_was_speaking: bool,
    pub source_event_digest: String,
}

impl StreamOpportunityV1 {
    pub fn expected_action(&self) -> StreamAction {
        match self.kind {
            StreamOpportunityKind::ProactiveVisualCue => StreamAction::Intervene,
            StreamOpportunityKind::IncomingNewInformation => {
                if self.agent_was_speaking {
                    StreamAction::AdaptOutput
                } else {
                    StreamAction::Intervene
                }
            }
            StreamOpportunityKind::ListenerBackchannel
            | StreamOpportunityKind::OtherDirectedSpeech
            | StreamOpportunityKind::AmbientSpeech => {
                if self.agent_was_speaking {
                    StreamAction::ContinueOutput
                } else {
                    StreamAction::StaySilent
                }
            }
            StreamOpportunityKind::HumanResolution => {
                if self.agent_was_speaking {
                    StreamAction::StopOutput
                } else {
                    StreamAction::StaySilent
                }
            }
        }
    }

    pub fn is_actionable(&self) -> bool {
        matches!(
            self.expected_action(),
            StreamAction::Intervene | StreamAction::AdaptOutput
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamPredictionV1 {
    pub opportunity_id: String,
    pub clock_domain_id: String,
    pub decision_at_monotonic_ns: u64,
    pub action: StreamAction,
    /// Independently graded [0,1] response quality; required only for
    /// Intervene/AdaptOutput actions.
    pub response_quality: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEngagementError {
    WrongSchema,
    InvalidIdentifier,
    InvalidDigest,
    InvalidWindow,
    WrongClockDomain,
    DecisionBeforeOpportunity,
    InvalidQualityGrade,
    MissingQualityGrade,
    UnexpectedQualityGrade,
    EmptyEvaluationSet,
    DuplicateOpportunity,
    DuplicatePrediction,
    MissingPrediction,
    UnexpectedPrediction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamEngagementMetrics {
    pub opportunity_count: usize,
    pub actionable_opportunity_count: usize,
    pub timely_action_count: usize,
    pub response_window_rate: Option<f32>,
    pub late_action_count: usize,
    pub missed_action_count: usize,
    pub correct_action_count: usize,
    pub correct_action_rate: Option<f32>,
    pub silence_opportunity_count: usize,
    pub correct_silence_count: usize,
    pub silence_preservation_rate: Option<f32>,
    pub false_intervention_count: usize,
    pub false_intervention_rate: Option<f32>,
    pub continuation_opportunity_count: usize,
    pub correct_continuation_count: usize,
    pub continuation_rate: Option<f32>,
    pub stop_opportunity_count: usize,
    pub correct_stop_count: usize,
    pub stop_compliance_rate: Option<f32>,
    pub response_quality_count: usize,
    pub mean_response_quality: Option<f32>,
    pub median_response_latency_ms: Option<u64>,
    pub p95_response_latency_ms: Option<u64>,
}

pub fn verify_stream_opportunity(
    opportunity: &StreamOpportunityV1,
) -> Result<(), StreamEngagementError> {
    if opportunity.schema != STREAM_ENGAGEMENT_SCHEMA_V1 {
        return Err(StreamEngagementError::WrongSchema);
    }
    validate_identifier(&opportunity.opportunity_id)?;
    validate_identifier(&opportunity.stream_id)?;
    validate_identifier(&opportunity.clock_domain_id)?;
    validate_digest(&opportunity.source_event_digest)?;
    if opportunity.response_deadline_monotonic_ns <= opportunity.event_start_monotonic_ns
        || opportunity.response_deadline_monotonic_ns
            - opportunity.event_start_monotonic_ns
            > MAX_STREAM_OPPORTUNITY_WINDOW_NS
    {
        return Err(StreamEngagementError::InvalidWindow);
    }
    Ok(())
}

pub fn verify_stream_prediction(
    opportunity: &StreamOpportunityV1,
    prediction: &StreamPredictionV1,
) -> Result<(), StreamEngagementError> {
    verify_stream_opportunity(opportunity)?;
    validate_identifier(&prediction.opportunity_id)?;
    validate_identifier(&prediction.clock_domain_id)?;
    if prediction.opportunity_id != opportunity.opportunity_id {
        return Err(StreamEngagementError::UnexpectedPrediction);
    }
    if prediction.clock_domain_id != opportunity.clock_domain_id {
        return Err(StreamEngagementError::WrongClockDomain);
    }
    if prediction.decision_at_monotonic_ns < opportunity.event_start_monotonic_ns {
        return Err(StreamEngagementError::DecisionBeforeOpportunity);
    }
    let produces_response = matches!(
        prediction.action,
        StreamAction::Intervene | StreamAction::AdaptOutput
    );
    match (produces_response, prediction.response_quality) {
        (true, None) => return Err(StreamEngagementError::MissingQualityGrade),
        (false, Some(_)) => return Err(StreamEngagementError::UnexpectedQualityGrade),
        (true, Some(value)) if !value.is_finite() || !(0.0..=1.0).contains(&value) => {
            return Err(StreamEngagementError::InvalidQualityGrade);
        }
        _ => {}
    }
    Ok(())
}

/// Score one prediction per annotated event window. The response-window rate
/// only credits the exact expected action when it occurs on or before the
/// deadline. Late responses are reported separately and never count as timely.
pub fn score_stream_engagement(
    opportunities: &[StreamOpportunityV1],
    predictions: &[StreamPredictionV1],
) -> Result<StreamEngagementMetrics, StreamEngagementError> {
    if opportunities.is_empty() {
        return Err(StreamEngagementError::EmptyEvaluationSet);
    }
    let mut opportunity_by_id = HashMap::with_capacity(opportunities.len());
    for opportunity in opportunities {
        verify_stream_opportunity(opportunity)?;
        if opportunity_by_id
            .insert(opportunity.opportunity_id.as_str(), opportunity)
            .is_some()
        {
            return Err(StreamEngagementError::DuplicateOpportunity);
        }
    }

    let mut prediction_by_id = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if prediction_by_id
            .insert(prediction.opportunity_id.as_str(), prediction)
            .is_some()
        {
            return Err(StreamEngagementError::DuplicatePrediction);
        }
        if !opportunity_by_id.contains_key(prediction.opportunity_id.as_str()) {
            return Err(StreamEngagementError::UnexpectedPrediction);
        }
    }
    if predictions.len() != opportunities.len() {
        return Err(StreamEngagementError::MissingPrediction);
    }

    let mut actionable = 0;
    let mut timely = 0;
    let mut late = 0;
    let mut missed = 0;
    let mut correct = 0;
    let mut silence_opportunities = 0;
    let mut correct_silence = 0;
    let mut false_interventions = 0;
    let mut continuation_opportunities = 0;
    let mut correct_continuations = 0;
    let mut stop_opportunities = 0;
    let mut correct_stops = 0;
    let mut response_quality_sum = 0.0_f64;
    let mut response_quality_count = 0;
    let mut response_latencies_ms = Vec::new();

    for opportunity in opportunities {
        let prediction = prediction_by_id[opportunity.opportunity_id.as_str()];
        verify_stream_prediction(opportunity, prediction)?;
        let expected = opportunity.expected_action();
        let is_correct = prediction.action == expected;
        correct += usize::from(is_correct);

        if opportunity.is_actionable() {
            actionable += 1;
            if is_correct {
                if prediction.decision_at_monotonic_ns
                    <= opportunity.response_deadline_monotonic_ns
                {
                    timely += 1;
                    response_latencies_ms.push(
                        (prediction.decision_at_monotonic_ns
                            - opportunity.event_start_monotonic_ns)
                            / 1_000_000,
                    );
                } else {
                    late += 1;
                }
            } else {
                missed += 1;
            }
        }

        if matches!(
            expected,
            StreamAction::StaySilent | StreamAction::ContinueOutput | StreamAction::StopOutput
        ) {
            if expected == StreamAction::StaySilent {
                silence_opportunities += 1;
                correct_silence += usize::from(prediction.action == StreamAction::StaySilent);
            } else if expected == StreamAction::ContinueOutput {
                continuation_opportunities += 1;
                correct_continuations += usize::from(
                    prediction.action == StreamAction::ContinueOutput,
                );
            } else {
                stop_opportunities += 1;
                correct_stops += usize::from(prediction.action == StreamAction::StopOutput);
            }
            false_interventions += usize::from(matches!(
                prediction.action,
                StreamAction::Intervene | StreamAction::AdaptOutput
            ));
        }

        if let Some(quality) = prediction.response_quality {
            response_quality_sum += quality as f64;
            response_quality_count += 1;
        }
    }

    response_latencies_ms.sort_unstable();
    Ok(StreamEngagementMetrics {
        opportunity_count: opportunities.len(),
        actionable_opportunity_count: actionable,
        timely_action_count: timely,
        response_window_rate: rate(timely, actionable),
        late_action_count: late,
        missed_action_count: missed,
        correct_action_count: correct,
        correct_action_rate: rate(correct, opportunities.len()),
        silence_opportunity_count: silence_opportunities,
        correct_silence_count: correct_silence,
        silence_preservation_rate: rate(correct_silence, silence_opportunities),
        false_intervention_count: false_interventions,
        false_intervention_rate: rate(false_interventions, silence_opportunities
            + continuation_opportunities + stop_opportunities),
        continuation_opportunity_count: continuation_opportunities,
        correct_continuation_count: correct_continuations,
        continuation_rate: rate(correct_continuations, continuation_opportunities),
        stop_opportunity_count: stop_opportunities,
        correct_stop_count: correct_stops,
        stop_compliance_rate: rate(correct_stops, stop_opportunities),
        response_quality_count: response_quality_count,
        mean_response_quality: (response_quality_count > 0)
            .then_some((response_quality_sum / response_quality_count as f64) as f32),
        median_response_latency_ms: percentile_nearest_rank(&response_latencies_ms, 0.50),
        p95_response_latency_ms: percentile_nearest_rank(&response_latencies_ms, 0.95),
    })
}

fn rate(numerator: usize, denominator: usize) -> Option<f32> {
    (denominator > 0).then_some(numerator as f32 / denominator as f32)
}

fn percentile_nearest_rank(values: &[u64], quantile: f32) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let rank = ((values.len() as f32 * quantile).ceil() as usize).max(1);
    Some(values[rank.min(values.len()) - 1])
}

fn validate_identifier(value: &str) -> Result<(), StreamEngagementError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(StreamEngagementError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), StreamEngagementError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(StreamEngagementError::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: u8) -> String { std::iter::repeat_n(ch as char, 64).collect() }

    fn opportunity(
        id: &str,
        kind: StreamOpportunityKind,
        speaking: bool,
    ) -> StreamOpportunityV1 {
        StreamOpportunityV1 {
            schema: STREAM_ENGAGEMENT_SCHEMA_V1.to_owned(),
            opportunity_id: id.to_owned(),
            stream_id: "stream-1".to_owned(),
            clock_domain_id: "clock:stream-1".to_owned(),
            event_start_monotonic_ns: 1_000_000,
            response_deadline_monotonic_ns: 501_000_000,
            kind,
            agent_was_speaking: speaking,
            source_event_digest: digest(b'a'),
        }
    }

    fn prediction(
        opportunity: &StreamOpportunityV1,
        action: StreamAction,
        latency_ms: u64,
    ) -> StreamPredictionV1 {
        let response = matches!(action, StreamAction::Intervene | StreamAction::AdaptOutput);
        StreamPredictionV1 {
            opportunity_id: opportunity.opportunity_id.clone(),
            clock_domain_id: opportunity.clock_domain_id.clone(),
            decision_at_monotonic_ns: opportunity.event_start_monotonic_ns
                + latency_ms * 1_000_000,
            action,
            response_quality: response.then_some(0.9),
        }
    }

    #[test]
    fn timely_visual_intervention_counts_inside_the_response_window() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::Intervene, 250)],
        ).unwrap();
        assert_eq!(result.response_window_rate, Some(1.0));
        assert_eq!(result.p95_response_latency_ms, Some(250));
        assert_eq!(result.mean_response_quality, Some(0.9));
    }

    #[test]
    fn late_intervention_does_not_count_as_a_window_success() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::Intervene, 700)],
        ).unwrap();
        assert_eq!(result.response_window_rate, Some(0.0));
        assert_eq!(result.late_action_count, 1);
    }

    #[test]
    fn missing_a_visual_opportunity_counts_as_missed() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::StaySilent, 100)],
        ).unwrap();
        assert_eq!(result.missed_action_count, 1);
        assert_eq!(result.response_window_rate, Some(0.0));
    }

    #[test]
    fn other_directed_speech_requires_continuation_or_silence_not_intervention() {
        let item = opportunity("op-1", StreamOpportunityKind::OtherDirectedSpeech, true);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::ContinueOutput, 100)],
        ).unwrap();
        assert_eq!(result.continuation_rate, Some(1.0));
        assert_eq!(result.false_intervention_count, 0);
    }

    #[test]
    fn ambient_speech_does_not_activate_a_silent_agent() {
        let item = opportunity("op-1", StreamOpportunityKind::AmbientSpeech, false);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::StaySilent, 100)],
        ).unwrap();
        assert_eq!(result.silence_preservation_rate, Some(1.0));
        assert_eq!(result.false_intervention_rate, Some(0.0));
    }

    #[test]
    fn human_resolution_requires_immediate_stop_when_agent_is_speaking() {
        let item = opportunity("op-1", StreamOpportunityKind::HumanResolution, true);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::StopOutput, 100)],
        ).unwrap();
        assert_eq!(result.stop_compliance_rate, Some(1.0));
    }

    #[test]
    fn new_information_adapts_an_active_response() {
        let item = opportunity("op-1", StreamOpportunityKind::IncomingNewInformation, true);
        let result = score_stream_engagement(
            &[item.clone()],
            &[prediction(&item, StreamAction::AdaptOutput, 100)],
        ).unwrap();
        assert_eq!(result.response_window_rate, Some(1.0));
        assert_eq!(result.missed_action_count, 0);
    }

    #[test]
    fn empty_evaluation_set_is_rejected() {
        assert_eq!(
            score_stream_engagement(&[], &[]),
            Err(StreamEngagementError::EmptyEvaluationSet)
        );
    }

    #[test]
    fn missing_prediction_is_not_scored_as_silence() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        assert_eq!(
            score_stream_engagement(&[item], &[]),
            Err(StreamEngagementError::MissingPrediction)
        );
    }

    #[test]
    fn duplicate_prediction_is_rejected() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        let guess = prediction(&item, StreamAction::Intervene, 100);
        assert_eq!(
            score_stream_engagement(&[item], &[guess.clone(), guess]),
            Err(StreamEngagementError::DuplicatePrediction)
        );
    }

    #[test]
    fn quality_grade_is_mandatory_for_intervention() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        let mut guess = prediction(&item, StreamAction::Intervene, 100);
        guess.response_quality = None;
        assert_eq!(
            verify_stream_prediction(&item, &guess),
            Err(StreamEngagementError::MissingQualityGrade)
        );
    }

    #[test]
    fn wrong_clock_domains_are_rejected() {
        let item = opportunity("op-1", StreamOpportunityKind::ProactiveVisualCue, false);
        let mut guess = prediction(&item, StreamAction::Intervene, 100);
        guess.clock_domain_id = "clock:other".to_owned();
        assert_eq!(
            verify_stream_prediction(&item, &guess),
            Err(StreamEngagementError::WrongClockDomain)
        );
    }
}
