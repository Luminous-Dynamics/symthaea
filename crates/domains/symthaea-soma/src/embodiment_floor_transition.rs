// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! False-positive-constrained end-of-turn and selective-interruption evaluator.
//!
//! Inspired by MultiPartyTurnBench. The two tasks are evaluated independently:
//! detecting that a turn ended is not treated as proof that an interruption or
//! floor takeover was appropriate.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const FLOOR_TRANSITION_SCHEMA_V1: &str = "symthaea.floor-transition.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloorEventKind {
    EndOfTurn,
    SuccessfulTakeover,
    TemporaryHold,
    Backchannel,
    OtherDirectedSpeech,
    AmbientSpeech,
    NoFloorChange,
}

impl FloorEventKind {
    pub fn is_end_of_turn_positive(self) -> bool {
        matches!(self, Self::EndOfTurn | Self::SuccessfulTakeover)
    }

    pub fn is_interruption_positive(self) -> bool {
        self == Self::SuccessfulTakeover
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloorTransitionEventV1 {
    pub schema: String,
    pub event_id: String,
    pub session_id: String,
    pub clock_domain_id: String,
    pub current_speaker_id: String,
    pub event_start_monotonic_ns: u64,
    pub ground_truth_at_monotonic_ns: u64,
    pub kind: FloorEventKind,
    pub source_event_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorTransitionPredictionV1 {
    pub event_id: String,
    pub clock_domain_id: String,
    pub end_of_turn_probability: f32,
    pub interruption_probability: f32,
    /// Set only when the model emits an EOT decision at the configured threshold.
    pub end_of_turn_detected_at_monotonic_ns: Option<u64>,
    /// Set only when the model emits an interruption decision at the configured threshold.
    pub interruption_detected_at_monotonic_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FloorTransitionConfigV1 {
    pub end_of_turn_threshold: f32,
    pub interruption_threshold: f32,
    pub maximum_false_positive_rate: f32,
}

impl Default for FloorTransitionConfigV1 {
    fn default() -> Self {
        Self {
            end_of_turn_threshold: 0.90,
            interruption_threshold: 0.95,
            maximum_false_positive_rate: 0.05,
        }
    }
}

impl FloorTransitionConfigV1 {
    pub fn validate(self) -> Result<(), FloorTransitionError> {
        for threshold in [
            self.end_of_turn_threshold,
            self.interruption_threshold,
            self.maximum_false_positive_rate,
        ] {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err(FloorTransitionError::InvalidThreshold);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloorTransitionError {
    WrongSchema,
    InvalidIdentifier,
    InvalidDigest,
    InvalidEventTime,
    WrongClockDomain,
    InvalidProbability,
    MissingDetectionTimestamp,
    UnexpectedDetectionTimestamp,
    DetectionBeforeEvent,
    LatencyOutOfRange,
    InvalidThreshold,
    EmptyEvaluationSet,
    DuplicateEvent,
    DuplicatePrediction,
    MissingPrediction,
    UnexpectedPrediction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloorTaskMetricsV1 {
    pub positive_count: usize,
    pub negative_count: usize,
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub true_negative_count: usize,
    pub false_negative_count: usize,
    pub recall: Option<f32>,
    pub false_positive_rate: Option<f32>,
    pub precision: Option<f32>,
    pub balanced_accuracy: Option<f32>,
    pub area_under_roc: Option<f32>,
    pub false_positive_budget_met: Option<bool>,
    /// Negative = early detection; positive = late detection.
    pub median_signed_detection_latency_ms: Option<i64>,
    pub p95_signed_detection_latency_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloorTransitionMetricsV1 {
    pub event_count: usize,
    pub independent_session_count: usize,
    pub end_of_turn: FloorTaskMetricsV1,
    pub selective_interruption: FloorTaskMetricsV1,
}

#[derive(Debug, Clone, Copy, Default)]
struct Confusion {
    tp: usize,
    fp: usize,
    tn: usize,
    fn_count: usize,
}

pub fn verify_floor_transition_event(
    event: &FloorTransitionEventV1,
) -> Result<(), FloorTransitionError> {
    if event.schema != FLOOR_TRANSITION_SCHEMA_V1 {
        return Err(FloorTransitionError::WrongSchema);
    }
    for value in [
        event.event_id.as_str(),
        event.session_id.as_str(),
        event.clock_domain_id.as_str(),
        event.current_speaker_id.as_str(),
    ] {
        validate_identifier(value)?;
    }
    validate_digest(&event.source_event_digest)?;
    if event.ground_truth_at_monotonic_ns < event.event_start_monotonic_ns {
        return Err(FloorTransitionError::InvalidEventTime);
    }
    Ok(())
}

pub fn verify_floor_transition_prediction(
    event: &FloorTransitionEventV1,
    prediction: &FloorTransitionPredictionV1,
    config: FloorTransitionConfigV1,
) -> Result<(), FloorTransitionError> {
    verify_floor_transition_event(event)?;
    config.validate()?;
    validate_identifier(&prediction.event_id)?;
    validate_identifier(&prediction.clock_domain_id)?;
    if prediction.event_id != event.event_id {
        return Err(FloorTransitionError::UnexpectedPrediction);
    }
    if prediction.clock_domain_id != event.clock_domain_id {
        return Err(FloorTransitionError::WrongClockDomain);
    }
    for probability in [
        prediction.end_of_turn_probability,
        prediction.interruption_probability,
    ] {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(FloorTransitionError::InvalidProbability);
        }
    }
    validate_detection_timestamp(
        event,
        prediction.end_of_turn_probability >= config.end_of_turn_threshold,
        prediction.end_of_turn_detected_at_monotonic_ns,
    )?;
    validate_detection_timestamp(
        event,
        prediction.interruption_probability >= config.interruption_threshold,
        prediction.interruption_detected_at_monotonic_ns,
    )?;
    Ok(())
}

fn validate_detection_timestamp(
    event: &FloorTransitionEventV1,
    emitted: bool,
    timestamp: Option<u64>,
) -> Result<(), FloorTransitionError> {
    match (emitted, timestamp) {
        (true, None) => Err(FloorTransitionError::MissingDetectionTimestamp),
        (false, Some(_)) => Err(FloorTransitionError::UnexpectedDetectionTimestamp),
        (true, Some(value)) if value < event.event_start_monotonic_ns => {
            Err(FloorTransitionError::DetectionBeforeEvent)
        }
        _ => Ok(()),
    }
}

pub fn score_floor_transitions(
    events: &[FloorTransitionEventV1],
    predictions: &[FloorTransitionPredictionV1],
    config: FloorTransitionConfigV1,
) -> Result<FloorTransitionMetricsV1, FloorTransitionError> {
    config.validate()?;
    if events.is_empty() {
        return Err(FloorTransitionError::EmptyEvaluationSet);
    }

    let mut events_by_id = HashMap::with_capacity(events.len());
    let mut sessions = HashSet::with_capacity(events.len());
    for event in events {
        verify_floor_transition_event(event)?;
        if events_by_id.insert(event.event_id.as_str(), event).is_some() {
            return Err(FloorTransitionError::DuplicateEvent);
        }
        sessions.insert(event.session_id.as_str());
    }

    let mut predictions_by_id = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if predictions_by_id.insert(prediction.event_id.as_str(), prediction).is_some() {
            return Err(FloorTransitionError::DuplicatePrediction);
        }
        if !events_by_id.contains_key(prediction.event_id.as_str()) {
            return Err(FloorTransitionError::UnexpectedPrediction);
        }
    }
    if predictions.len() != events.len() {
        return Err(FloorTransitionError::MissingPrediction);
    }

    let mut eot_confusion = Confusion::default();
    let mut interruption_confusion = Confusion::default();
    let mut eot_scores = Vec::with_capacity(events.len());
    let mut interruption_scores = Vec::with_capacity(events.len());
    let mut eot_latency = Vec::new();
    let mut interruption_latency = Vec::new();

    for event in events {
        let prediction = predictions_by_id[event.event_id.as_str()];
        verify_floor_transition_prediction(event, prediction, config)?;

        let eot_positive = event.kind.is_end_of_turn_positive();
        let eot_predicted =
            prediction.end_of_turn_probability >= config.end_of_turn_threshold;
        update_confusion(&mut eot_confusion, eot_positive, eot_predicted);
        eot_scores.push((prediction.end_of_turn_probability, eot_positive));
        if eot_positive && eot_predicted {
            eot_latency.push(signed_latency_ms(
                prediction.end_of_turn_detected_at_monotonic_ns.unwrap(),
                event.ground_truth_at_monotonic_ns,
            )?);
        }

        let interruption_positive = event.kind.is_interruption_positive();
        let interruption_predicted =
            prediction.interruption_probability >= config.interruption_threshold;
        update_confusion(
            &mut interruption_confusion,
            interruption_positive,
            interruption_predicted,
        );
        interruption_scores.push((prediction.interruption_probability, interruption_positive));
        if interruption_positive && interruption_predicted {
            interruption_latency.push(signed_latency_ms(
                prediction.interruption_detected_at_monotonic_ns.unwrap(),
                event.ground_truth_at_monotonic_ns,
            )?);
        }
    }

    Ok(FloorTransitionMetricsV1 {
        event_count: events.len(),
        independent_session_count: sessions.len(),
        end_of_turn: finish_task(
            eot_confusion,
            &eot_scores,
            config.maximum_false_positive_rate,
            &mut eot_latency,
        ),
        selective_interruption: finish_task(
            interruption_confusion,
            &interruption_scores,
            config.maximum_false_positive_rate,
            &mut interruption_latency,
        ),
    })
}

fn update_confusion(confusion: &mut Confusion, positive: bool, predicted: bool) {
    match (positive, predicted) {
        (true, true) => confusion.tp += 1,
        (true, false) => confusion.fn_count += 1,
        (false, true) => confusion.fp += 1,
        (false, false) => confusion.tn += 1,
    }
}

fn finish_task(
    confusion: Confusion,
    scores: &[(f32, bool)],
    maximum_false_positive_rate: f32,
    latencies: &mut Vec<i64>,
) -> FloorTaskMetricsV1 {
    latencies.sort_unstable();
    let recall = rate(confusion.tp, confusion.tp + confusion.fn_count);
    let false_positive_rate = rate(confusion.fp, confusion.fp + confusion.tn);
    let precision = rate(confusion.tp, confusion.tp + confusion.fp);
    let specificity = rate(confusion.tn, confusion.tn + confusion.fp);
    let balanced_accuracy = match (recall, specificity) {
        (Some(r), Some(s)) => Some((r + s) / 2.0),
        _ => None,
    };
    let auc = auc_mann_whitney(scores);
    FloorTaskMetricsV1 {
        positive_count: confusion.tp + confusion.fn_count,
        negative_count: confusion.tn + confusion.fp,
        true_positive_count: confusion.tp,
        false_positive_count: confusion.fp,
        true_negative_count: confusion.tn,
        false_negative_count: confusion.fn_count,
        recall,
        false_positive_rate,
        precision,
        balanced_accuracy,
        area_under_roc: auc,
        false_positive_budget_met: false_positive_rate
            .map(|value| value <= maximum_false_positive_rate),
        median_signed_detection_latency_ms: percentile_nearest_rank(latencies, 0.50),
        p95_signed_detection_latency_ms: percentile_nearest_rank(latencies, 0.95),
    }
}

fn auc_mann_whitney(scores: &[(f32, bool)]) -> Option<f32> {
    let positives: Vec<f32> = scores.iter()
        .filter_map(|(score, positive)| (*positive).then_some(*score))
        .collect();
    let negatives: Vec<f32> = scores.iter()
        .filter_map(|(score, positive)| (!*positive).then_some(*score))
        .collect();
    if positives.is_empty() || negatives.is_empty() {
        return None;
    }
    let mut wins = 0.0_f64;
    for positive in &positives {
        for negative in &negatives {
            if positive > negative {
                wins += 1.0;
            } else if positive == negative {
                wins += 0.5;
            }
        }
    }
    Some((wins / (positives.len() * negatives.len()) as f64) as f32)
}

fn signed_latency_ms(detected_at: u64, truth_at: u64) -> Result<i64, FloorTransitionError> {
    let delta = detected_at as i128 - truth_at as i128;
    let milliseconds = delta / 1_000_000;
    i64::try_from(milliseconds).map_err(|_| FloorTransitionError::LatencyOutOfRange)
}

fn rate(numerator: usize, denominator: usize) -> Option<f32> {
    (denominator > 0).then_some(numerator as f32 / denominator as f32)
}

fn percentile_nearest_rank(values: &[i64], quantile: f32) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    let rank = ((values.len() as f32 * quantile).ceil() as usize).max(1);
    Some(values[rank.min(values.len()) - 1])
}

fn validate_identifier(value: &str) -> Result<(), FloorTransitionError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(FloorTransitionError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), FloorTransitionError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FloorTransitionError::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: u8) -> String { std::iter::repeat_n(ch as char, 64).collect() }

    fn event(id: &str, session: &str, kind: FloorEventKind) -> FloorTransitionEventV1 {
        FloorTransitionEventV1 {
            schema: FLOOR_TRANSITION_SCHEMA_V1.to_owned(),
            event_id: id.to_owned(),
            session_id: session.to_owned(),
            clock_domain_id: "clock:floor-1".to_owned(),
            current_speaker_id: "speaker-a".to_owned(),
            event_start_monotonic_ns: 1_000_000,
            ground_truth_at_monotonic_ns: 2_000_000,
            kind,
            source_event_digest: digest(b'a'),
        }
    }

    fn prediction(
        event: &FloorTransitionEventV1,
        eot_probability: f32,
        interruption_probability: f32,
        eot_time: Option<u64>,
        interruption_time: Option<u64>,
    ) -> FloorTransitionPredictionV1 {
        FloorTransitionPredictionV1 {
            event_id: event.event_id.clone(),
            clock_domain_id: event.clock_domain_id.clone(),
            end_of_turn_probability: eot_probability,
            interruption_probability,
            end_of_turn_detected_at_monotonic_ns: eot_time,
            interruption_detected_at_monotonic_ns: interruption_time,
        }
    }

    fn config() -> FloorTransitionConfigV1 {
        FloorTransitionConfigV1 {
            end_of_turn_threshold: 0.90,
            interruption_threshold: 0.95,
            maximum_false_positive_rate: 0.05,
        }
    }

    #[test]
    fn eot_and_interruption_are_scored_as_separate_tasks() {
        let events = vec![
            event("eot-1", "session-1", FloorEventKind::EndOfTurn),
            event("interrupt-1", "session-2", FloorEventKind::SuccessfulTakeover),
            event("hold-1", "session-3", FloorEventKind::TemporaryHold),
        ];
        let predictions = vec![
            prediction(&events[0], 0.99, 0.10, Some(1_000_000), None),
            prediction(&events[1], 0.99, 0.99, Some(1_950_000), Some(3_000_000)),
            prediction(&events[2], 0.10, 0.10, None, None),
        ];
        let result = score_floor_transitions(&events, &predictions, config()).unwrap();
        assert_eq!(result.end_of_turn.recall, Some(1.0));
        assert_eq!(result.selective_interruption.recall, Some(1.0));
        assert_eq!(result.end_of_turn.median_signed_detection_latency_ms, Some(-1));
        assert_eq!(result.selective_interruption.median_signed_detection_latency_ms, Some(1));
    }

    #[test]
    fn false_interruption_on_temporary_hold_counts_against_fpr() {
        let events = vec![
            event("takeover-1", "session-1", FloorEventKind::SuccessfulTakeover),
            event("hold-1", "session-2", FloorEventKind::TemporaryHold),
            event("other-1", "session-3", FloorEventKind::OtherDirectedSpeech),
            event("amb-1", "session-4", FloorEventKind::Backchannel),
        ];
        let predictions = vec![
            prediction(&events[0], 0.99, 0.99, Some(2_000_000), Some(2_000_000)),
            prediction(&events[1], 0.20, 0.98, None, Some(2_100_000)),
            prediction(&events[2], 0.10, 0.20, None, None),
            prediction(&events[3], 0.10, 0.10, None, None),
        ];
        let result = score_floor_transitions(&events, &predictions, config()).unwrap();
        assert_eq!(result.selective_interruption.false_positive_count, 1);
        assert_eq!(result.selective_interruption.false_positive_rate, Some(1.0 / 3.0));
        assert_eq!(result.selective_interruption.false_positive_budget_met, Some(false));
    }

    #[test]
    fn early_and_late_detections_use_signed_latency() {
        let events = vec![
            event("early", "session-1", FloorEventKind::EndOfTurn),
            event("late", "session-2", FloorEventKind::EndOfTurn),
        ];
        let predictions = vec![
            prediction(&events[0], 0.99, 0.10, Some(1_000_000), None),
            prediction(&events[1], 0.99, 0.10, Some(3_000_000), None),
        ];
        let result = score_floor_transitions(&events, &predictions, config()).unwrap();
        assert_eq!(result.end_of_turn.median_signed_detection_latency_ms, Some(-1));
        assert_eq!(result.end_of_turn.p95_signed_detection_latency_ms, Some(1));
    }

    #[test]
    fn threshold_free_auc_uses_all_scores() {
        let events = vec![
            event("pos-1", "session-1", FloorEventKind::EndOfTurn),
            event("pos-2", "session-2", FloorEventKind::EndOfTurn),
            event("neg-1", "session-3", FloorEventKind::TemporaryHold),
            event("neg-2", "session-4", FloorEventKind::AmbientSpeech),
        ];
        let predictions = vec![
            prediction(&events[0], 0.95, 0.10, Some(2_000_000), None),
            prediction(&events[1], 0.85, 0.10, None, None),
            prediction(&events[2], 0.40, 0.10, None, None),
            prediction(&events[3], 0.20, 0.10, None, None),
        ];
        let result = score_floor_transitions(&events, &predictions, config()).unwrap();
        assert_eq!(result.end_of_turn.area_under_roc, Some(1.0));
        assert_eq!(result.end_of_turn.recall, Some(0.5));
    }

    #[test]
    fn high_probability_requires_a_detection_timestamp() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        let guess = prediction(&item, 0.99, 0.10, None, None);
        assert_eq!(
            verify_floor_transition_prediction(&item, &guess, config()),
            Err(FloorTransitionError::MissingDetectionTimestamp)
        );
    }

    #[test]
    fn low_probability_cannot_claim_an_emitted_detection() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        let guess = prediction(&item, 0.10, 0.10, Some(1_900_000), None);
        assert_eq!(
            verify_floor_transition_prediction(&item, &guess, config()),
            Err(FloorTransitionError::UnexpectedDetectionTimestamp)
        );
    }

    #[test]
    fn detection_timestamp_before_observation_is_rejected() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        let guess = prediction(&item, 0.99, 0.10, Some(900_000), None);
        assert_eq!(
            verify_floor_transition_prediction(&item, &guess, config()),
            Err(FloorTransitionError::DetectionBeforeEvent)
        );
    }

    #[test]
    fn missing_predictions_are_not_counted_as_negative_decisions() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        assert_eq!(
            score_floor_transitions(&[item], &[], config()),
            Err(FloorTransitionError::MissingPrediction)
        );
    }

    #[test]
    fn duplicate_prediction_ids_are_rejected() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        let guess = prediction(&item, 0.10, 0.10, None, None);
        assert_eq!(
            score_floor_transitions(&[item], &[guess.clone(), guess], config()),
            Err(FloorTransitionError::DuplicatePrediction)
        );
    }

    #[test]
    fn absent_negative_class_keeps_fpr_and_auc_undefined() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        let guess = prediction(&item, 0.99, 0.10, Some(2_000_000), None);
        let result = score_floor_transitions(&[item], &[guess], config()).unwrap();
        assert_eq!(result.end_of_turn.false_positive_rate, None);
        assert_eq!(result.end_of_turn.area_under_roc, None);
        assert_eq!(result.end_of_turn.false_positive_budget_met, None);
    }

    #[test]
    fn wrong_clock_domain_is_rejected() {
        let item = event("eot-1", "session-1", FloorEventKind::EndOfTurn);
        let mut guess = prediction(&item, 0.99, 0.10, Some(2_000_000), None);
        guess.clock_domain_id = "clock:other".to_owned();
        assert_eq!(
            verify_floor_transition_prediction(&item, &guess, config()),
            Err(FloorTransitionError::WrongClockDomain)
        );
    }

    #[test]
    fn empty_evaluation_is_rejected() {
        assert_eq!(
            score_floor_transitions(&[], &[], config()),
            Err(FloorTransitionError::EmptyEvaluationSet)
        );
    }
}
