#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact-clock, chunk-complete streaming musical turn-taking scorer.
//!
//! The scorer evaluates supplied event annotations and declared per-chunk
//! predictions. It does not infer invitations from PCM, authenticate the
//! external runner's actual model input, or claim human coordination quality.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const STREAM_SESSION_SCHEMA_V1: &str = "mel-rti-stream-session-v1";
pub const STREAM_CHUNK_SCHEMA_V1: &str = "mel-rti-stream-chunk-v1";
pub const STREAM_TRUTH_SCHEMA_V1: &str = "mel-rti-stream-truth-v1";
pub const STREAM_PREDICTION_SCHEMA_V1: &str = "mel-rti-stream-prediction-v1";
pub const STREAM_REPORT_SCHEMA_V1: &str = "mel-rti-stream-report-v1";
pub const STREAM_MAX_CHUNKS_V1: u32 = 100_000;
const STREAM_REPORT_DOMAIN_V1: &[u8] = b"luminous.mel.rti.streaming-report.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnActionV1 {
    Wait,
    Respond,
    Continue,
    Yield,
}

impl TurnActionV1 {
    pub const ALL: [Self; 4] = [Self::Wait, Self::Respond, Self::Continue, Self::Yield];

    pub fn confusion_index(self) -> usize {
        match self {
            Self::Wait => 0,
            Self::Respond => 1,
            Self::Continue => 2,
            Self::Yield => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictedTurnDecisionV1 {
    Wait,
    Respond,
    Continue,
    Yield,
    ParseFailure,
}

impl PredictedTurnDecisionV1 {
    pub fn action(self) -> Option<TurnActionV1> {
        match self {
            Self::Wait => Some(TurnActionV1::Wait),
            Self::Respond => Some(TurnActionV1::Respond),
            Self::Continue => Some(TurnActionV1::Continue),
            Self::Yield => Some(TurnActionV1::Yield),
            Self::ParseFailure => None,
        }
    }

    pub fn confusion_index(self) -> usize {
        match self {
            Self::Wait => 0,
            Self::Respond => 1,
            Self::Continue => 2,
            Self::Yield => 3,
            Self::ParseFailure => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameWindowV1 {
    pub start_frame: u64,
    pub end_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSessionRequestV1 {
    pub schema_version: String,
    pub session_id: String,
    pub source_audio_sha256: String,
    pub task_manifest_sha256: String,
    pub annotation_policy_sha256: String,
    pub immutable_policy_identity_sha256: String,
    pub runner_build_sha256: String,
    pub initial_policy_state_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub total_frames: u64,
    pub expected_chunk_count: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamChunkV1 {
    pub schema_version: String,
    pub chunk_index: u32,
    pub chunk_id: String,
    pub start_frame: u64,
    pub end_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamChunkTruthV1 {
    pub schema_version: String,
    pub chunk_index: u32,
    pub chunk_id: String,
    /// Sorted unique valid actions. More than one action preserves ambiguity.
    pub valid_actions: Vec<TurnActionV1>,
    /// Present only for an unambiguous RESPOND or YIELD truth label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_target_frame: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_window: Option<FrameWindowV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamChunkPredictionV1 {
    pub schema_version: String,
    pub chunk_index: u32,
    pub chunk_id: String,
    pub decision: PredictedTurnDecisionV1,
    /// Declared extent of the input prefix delivered to this policy call.
    pub observed_through_frame: u64,
    /// Monotonic runner clock, not the audio sample-frame clock.
    pub chunk_received_at_ns: u64,
    pub decision_at_ns: u64,
    pub effect_at_ns: Option<u64>,
    /// Sample-frame position of the actual/scheduled output effect.
    pub effect_frame: Option<u64>,
    pub input_prefix_sha256: String,
    pub pre_state_sha256: String,
    pub post_state_sha256: String,
    pub immutable_policy_identity_sha256: String,
    pub runner_build_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parse_failure_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionClassMetricsV1 {
    pub action: TurnActionV1,
    pub support: u32,
    pub true_positive: u32,
    pub false_positive: u32,
    pub false_negative: u32,
    pub precision_bp: Option<u32>,
    pub recall_bp: Option<u32>,
    pub f1_bp: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencyPercentilesV1 {
    pub sample_count: u32,
    pub p50_ns: Option<u64>,
    pub p95_ns: Option<u64>,
    pub p99_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameErrorPercentilesV1 {
    pub sample_count: u32,
    pub p50_absolute_frames: Option<u64>,
    pub p95_absolute_frames: Option<u64>,
    pub p99_absolute_frames: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamChunkScoreV1 {
    pub chunk_index: u32,
    pub chunk_id: String,
    pub valid_truth_actions: Vec<TurnActionV1>,
    pub predicted_decision: PredictedTurnDecisionV1,
    pub unambiguous_action_match: Option<bool>,
    pub ambiguous_valid_set_accept: Option<bool>,
    pub decision_latency_ns: u64,
    pub effect_latency_ns: Option<u64>,
    pub effect_frame_error: Option<i64>,
    pub effect_window_success: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamScoreSummaryV1 {
    /// Rows are truth WAIT/RESPOND/CONTINUE/YIELD; columns add ParseFailure as
    /// a fifth prediction category. Ambiguous truth sets are excluded here.
    pub confusion_matrix: [[u32; 5]; 4],
    pub unambiguous_chunk_count: u32,
    pub unambiguous_correct_count: u32,
    pub top1_accuracy_bp: Option<u32>,
    pub action_metrics: Vec<ActionClassMetricsV1>,
    pub macro_f1_bp: Option<u32>,
    pub ambiguous_chunk_count: u32,
    pub ambiguous_valid_set_accept_count: u32,
    pub ambiguous_valid_set_acceptance_bp: Option<u32>,
    pub parse_failure_count: u32,
    pub wait_false_activation_count: u32,
    pub continue_premature_yield_count: u32,
    pub post_yield_non_wait_leakage_count: u32,
    pub effect_eligible_count: u32,
    pub effect_window_attempt_count: u32,
    pub effect_window_success_count: u32,
    pub effect_window_success_bp: Option<u32>,
    pub decision_latency: LatencyPercentilesV1,
    pub effect_latency: LatencyPercentilesV1,
    pub effect_frame_error: FrameErrorPercentilesV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamScoreReportV1 {
    pub schema_version: String,
    pub request: StreamSessionRequestV1,
    pub chunks: Vec<StreamChunkV1>,
    pub truth: Vec<StreamChunkTruthV1>,
    pub predictions: Vec<StreamChunkPredictionV1>,
    pub scored_chunks: Vec<StreamChunkScoreV1>,
    pub summary: StreamScoreSummaryV1,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct StreamScorePayloadV1 {
    schema_version: String,
    request: StreamSessionRequestV1,
    chunks: Vec<StreamChunkV1>,
    truth: Vec<StreamChunkTruthV1>,
    predictions: Vec<StreamChunkPredictionV1>,
    scored_chunks: Vec<StreamChunkScoreV1>,
    summary: StreamScoreSummaryV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamScorerErrorV1 {
    InvalidRequest,
    InvalidDigest,
    InvalidAudioFormat,
    ChunkCountMismatch,
    ChunkSequenceMismatch,
    InvalidChunkInterval,
    ChunkCoverageGapOrOverlap,
    DuplicateIdentity,
    TruthCountMismatch,
    TruthIdentityMismatch,
    InvalidTruthActionSet,
    InvalidTruthEffectWindow,
    PredictionCountMismatch,
    PredictionIdentityMismatch,
    LookaheadDetected,
    ClockRegression,
    DecisionBeforeReceipt,
    InvalidActionEffect,
    BackdatedEffect,
    DuplicateOrRegressingEffect,
    InvalidParseFailure,
    PolicyIdentityMismatch,
    StateChainMismatch,
    ReportReplayMismatch,
    SerializationFailure,
}

impl fmt::Display for StreamScorerErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRequest => "stream session request schema, identity or bounds are invalid",
            Self::InvalidDigest => "a required SHA-256 commitment is malformed",
            Self::InvalidAudioFormat => "sample rate or channel count is unsupported",
            Self::ChunkCountMismatch => "chunk count differs from the frozen session request",
            Self::ChunkSequenceMismatch => "chunk indices must be contiguous and zero-based",
            Self::InvalidChunkInterval => "chunk interval is empty or outside the source artifact",
            Self::ChunkCoverageGapOrOverlap => "chunks must form one exact contiguous partition",
            Self::DuplicateIdentity => "chunk IDs must be nonempty and unique",
            Self::TruthCountMismatch => "there must be one truth record per chunk",
            Self::TruthIdentityMismatch => "truth chunk identity differs from its chunk",
            Self::InvalidTruthActionSet => "truth actions must be nonempty, sorted and unique",
            Self::InvalidTruthEffectWindow => "truth effect target/window is missing or invalid",
            Self::PredictionCountMismatch => "there must be exactly one prediction per chunk",
            Self::PredictionIdentityMismatch => "prediction identity differs from its chunk",
            Self::LookaheadDetected => "prediction observed a frame beyond the current chunk end",
            Self::ClockRegression => "monotonic receipt, decision or effect clocks regress",
            Self::DecisionBeforeReceipt => "a decision timestamp precedes chunk receipt",
            Self::InvalidActionEffect => "action and effect timestamp fields are inconsistent",
            Self::BackdatedEffect => "an action effect precedes the observed prefix or decision",
            Self::DuplicateOrRegressingEffect => "action effects are duplicated or regress in sample/monotonic time",
            Self::InvalidParseFailure => "parse failure must have an error digest and no action effect",
            Self::PolicyIdentityMismatch => "immutable policy or runner build differs from the session request",
            Self::StateChainMismatch => "policy state snapshots do not form the declared session chain",
            Self::ReportReplayMismatch => "supplied report differs from deterministic replay",
            Self::SerializationFailure => "canonical report serialization failed",
        })
    }
}
impl std::error::Error for StreamScorerErrorV1 {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn bp(numerator: u32, denominator: u32) -> Option<u32> {
    (denominator != 0).then(|| {
        ((u64::from(numerator) * 10_000) / u64::from(denominator)) as u32
    })
}

fn quantile_u64(values: &[u64], percentile: usize) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * percentile).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
}

fn latency_percentiles(values: &[u64]) -> LatencyPercentilesV1 {
    LatencyPercentilesV1 {
        sample_count: values.len() as u32,
        p50_ns: quantile_u64(values, 50),
        p95_ns: quantile_u64(values, 95),
        p99_ns: quantile_u64(values, 99),
    }
}

fn frame_error_percentiles(values: &[u64]) -> FrameErrorPercentilesV1 {
    FrameErrorPercentilesV1 {
        sample_count: values.len() as u32,
        p50_absolute_frames: quantile_u64(values, 50),
        p95_absolute_frames: quantile_u64(values, 95),
        p99_absolute_frames: quantile_u64(values, 99),
    }
}

fn validate_request(request: &StreamSessionRequestV1) -> Result<(), StreamScorerErrorV1> {
    if request.schema_version != STREAM_SESSION_SCHEMA_V1
        || request.session_id.trim().is_empty()
        || request.sample_rate_hz < 8_000
        || request.sample_rate_hz > 192_000
        || !(1..=2).contains(&request.channels)
        || request.total_frames == 0
        || request.total_frames > i64::MAX as u64
        || request.expected_chunk_count == 0
        || request.expected_chunk_count > STREAM_MAX_CHUNKS_V1
    {
        return Err(StreamScorerErrorV1::InvalidRequest);
    }
    for digest in [
        request.source_audio_sha256.as_str(),
        request.task_manifest_sha256.as_str(),
        request.annotation_policy_sha256.as_str(),
        request.immutable_policy_identity_sha256.as_str(),
        request.runner_build_sha256.as_str(),
        request.initial_policy_state_sha256.as_str(),
    ] {
        if !valid_sha256(digest) {
            return Err(StreamScorerErrorV1::InvalidDigest);
        }
    }
    Ok(())
}

fn validate_chunks(
    request: &StreamSessionRequestV1,
    chunks: &[StreamChunkV1],
) -> Result<(), StreamScorerErrorV1> {
    if chunks.len() != request.expected_chunk_count as usize {
        return Err(StreamScorerErrorV1::ChunkCountMismatch);
    }
    let mut seen = BTreeSet::new();
    let mut expected_start = 0_u64;
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.schema_version != STREAM_CHUNK_SCHEMA_V1
            || chunk.chunk_index != index as u32
        {
            return Err(StreamScorerErrorV1::ChunkSequenceMismatch);
        }
        if chunk.chunk_id.trim().is_empty() || !seen.insert(chunk.chunk_id.as_str()) {
            return Err(StreamScorerErrorV1::DuplicateIdentity);
        }
        if chunk.start_frame >= chunk.end_frame || chunk.end_frame > request.total_frames {
            return Err(StreamScorerErrorV1::InvalidChunkInterval);
        }
        if chunk.start_frame != expected_start {
            return Err(StreamScorerErrorV1::ChunkCoverageGapOrOverlap);
        }
        expected_start = chunk.end_frame;
    }
    if expected_start != request.total_frames {
        return Err(StreamScorerErrorV1::ChunkCoverageGapOrOverlap);
    }
    Ok(())
}

fn validate_truth(
    request: &StreamSessionRequestV1,
    chunks: &[StreamChunkV1],
    truth: &[StreamChunkTruthV1],
) -> Result<(), StreamScorerErrorV1> {
    if truth.len() != chunks.len() {
        return Err(StreamScorerErrorV1::TruthCountMismatch);
    }
    for (chunk, label) in chunks.iter().zip(truth) {
        if label.schema_version != STREAM_TRUTH_SCHEMA_V1
            || label.chunk_index != chunk.chunk_index
            || label.chunk_id != chunk.chunk_id
        {
            return Err(StreamScorerErrorV1::TruthIdentityMismatch);
        }
        if label.valid_actions.is_empty()
            || !label.valid_actions.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(StreamScorerErrorV1::InvalidTruthActionSet);
        }

        let single_action = (label.valid_actions.len() == 1)
            .then(|| label.valid_actions[0]);
        let needs_effect = matches!(
            single_action,
            Some(TurnActionV1::Respond | TurnActionV1::Yield)
        );
        match (needs_effect, label.effect_target_frame, label.effect_window) {
            (true, Some(target), Some(window))
                if window.start_frame >= chunk.end_frame
                    && window.start_frame < window.end_frame
                    && window.end_frame <= request.total_frames
                    && target >= window.start_frame
                    && target < window.end_frame =>
            {
            }
            (false, None, None) => {}
            _ => return Err(StreamScorerErrorV1::InvalidTruthEffectWindow),
        }
    }
    Ok(())
}

fn validate_predictions(
    request: &StreamSessionRequestV1,
    chunks: &[StreamChunkV1],
    predictions: &[StreamChunkPredictionV1],
) -> Result<(), StreamScorerErrorV1> {
    if predictions.len() != chunks.len() {
        return Err(StreamScorerErrorV1::PredictionCountMismatch);
    }
    let mut previous_receipt = None;
    let mut previous_decision = None;
    let mut previous_effect_time = None;
    let mut previous_effect_frame = None;
    let mut expected_state = request.initial_policy_state_sha256.as_str();

    for (chunk, prediction) in chunks.iter().zip(predictions) {
        if prediction.schema_version != STREAM_PREDICTION_SCHEMA_V1
            || prediction.chunk_index != chunk.chunk_index
            || prediction.chunk_id != chunk.chunk_id
        {
            return Err(StreamScorerErrorV1::PredictionIdentityMismatch);
        }
        if prediction.observed_through_frame != chunk.end_frame {
            return Err(StreamScorerErrorV1::LookaheadDetected);
        }
        if !valid_sha256(&prediction.input_prefix_sha256)
            || !valid_sha256(&prediction.pre_state_sha256)
            || !valid_sha256(&prediction.post_state_sha256)
        {
            return Err(StreamScorerErrorV1::InvalidDigest);
        }
        if prediction.immutable_policy_identity_sha256
            != request.immutable_policy_identity_sha256
            || prediction.runner_build_sha256 != request.runner_build_sha256
        {
            return Err(StreamScorerErrorV1::PolicyIdentityMismatch);
        }
        if prediction.pre_state_sha256 != expected_state {
            return Err(StreamScorerErrorV1::StateChainMismatch);
        }
        expected_state = prediction.post_state_sha256.as_str();

        if prediction.decision_at_ns < prediction.chunk_received_at_ns {
            return Err(StreamScorerErrorV1::DecisionBeforeReceipt);
        }
        if previous_receipt.is_some_and(|last| prediction.chunk_received_at_ns < last)
            || previous_decision.is_some_and(|last| prediction.decision_at_ns < last)
        {
            return Err(StreamScorerErrorV1::ClockRegression);
        }
        previous_receipt = Some(prediction.chunk_received_at_ns);
        previous_decision = Some(prediction.decision_at_ns);

        match prediction.decision {
            PredictedTurnDecisionV1::Wait | PredictedTurnDecisionV1::Continue => {
                if prediction.effect_frame.is_some()
                    || prediction.effect_at_ns.is_some()
                    || prediction.parse_failure_sha256.is_some()
                {
                    return Err(StreamScorerErrorV1::InvalidActionEffect);
                }
            }
            PredictedTurnDecisionV1::Respond | PredictedTurnDecisionV1::Yield => {
                if prediction.effect_frame.is_none()
                    || prediction.effect_at_ns.is_none()
                    || prediction.parse_failure_sha256.is_some()
                {
                    return Err(StreamScorerErrorV1::InvalidActionEffect);
                }
                let effect_frame = prediction.effect_frame.expect("presence checked");
                let effect_at_ns = prediction.effect_at_ns.expect("presence checked");
                if effect_frame < prediction.observed_through_frame
                    || effect_frame > request.total_frames
                    || effect_at_ns < prediction.decision_at_ns
                {
                    return Err(StreamScorerErrorV1::BackdatedEffect);
                }
                if previous_effect_frame.is_some_and(|last| effect_frame <= last)
                    || previous_effect_time.is_some_and(|last| effect_at_ns <= last)
                {
                    return Err(StreamScorerErrorV1::DuplicateOrRegressingEffect);
                }
                previous_effect_frame = Some(effect_frame);
                previous_effect_time = Some(effect_at_ns);
            }
            PredictedTurnDecisionV1::ParseFailure => {
                if prediction.effect_frame.is_some()
                    || prediction.effect_at_ns.is_some()
                    || !prediction
                        .parse_failure_sha256
                        .as_deref()
                        .is_some_and(valid_sha256)
                {
                    return Err(StreamScorerErrorV1::InvalidParseFailure);
                }
            }
        }
    }
    Ok(())
}

fn per_action_metrics(matrix: &[[u32; 5]; 4]) -> (Vec<ActionClassMetricsV1>, Option<u32>) {
    let mut metrics = Vec::with_capacity(4);
    let mut f1_values = Vec::new();
    for action in TurnActionV1::ALL {
        let index = action.confusion_index();
        let support = matrix[index].iter().sum::<u32>();
        let predicted = matrix.iter().map(|row| row[index]).sum::<u32>();
        let true_positive = matrix[index][index];
        let false_negative = support.saturating_sub(true_positive);
        let false_positive = predicted.saturating_sub(true_positive);
        let precision_bp = bp(true_positive, predicted);
        let recall_bp = bp(true_positive, support);
        let f1_denominator = true_positive
            .saturating_mul(2)
            .saturating_add(false_positive)
            .saturating_add(false_negative);
        let f1_bp = bp(true_positive.saturating_mul(2), f1_denominator);
        if support > 0 {
            if let Some(f1) = f1_bp {
                f1_values.push(f1);
            }
        }
        metrics.push(ActionClassMetricsV1 {
            action,
            support,
            true_positive,
            false_positive,
            false_negative,
            precision_bp,
            recall_bp,
            f1_bp,
        });
    }
    let macro_f1 = (!f1_values.is_empty()).then(|| {
        (f1_values.iter().map(|value| u64::from(*value)).sum::<u64>()
            / f1_values.len() as u64) as u32
    });
    (metrics, macro_f1)
}

pub fn score_streaming_session_v1(
    request: &StreamSessionRequestV1,
    chunks: &[StreamChunkV1],
    truth: &[StreamChunkTruthV1],
    predictions: &[StreamChunkPredictionV1],
) -> Result<StreamScoreReportV1, StreamScorerErrorV1> {
    validate_request(request)?;
    validate_chunks(request, chunks)?;
    validate_truth(request, chunks, truth)?;
    validate_predictions(request, chunks, predictions)?;

    let mut confusion_matrix = [[0_u32; 5]; 4];
    let mut unambiguous_count = 0_u32;
    let mut unambiguous_correct = 0_u32;
    let mut ambiguous_count = 0_u32;
    let mut ambiguous_valid_set_accepts = 0_u32;
    let mut parse_failures = 0_u32;
    let mut wait_false_activations = 0_u32;
    let mut continue_premature_yields = 0_u32;
    let mut effect_eligible = 0_u32;
    let mut effect_attempts = 0_u32;
    let mut effect_successes = 0_u32;
    let mut decision_latencies = Vec::with_capacity(predictions.len());
    let mut effect_latencies = Vec::new();
    let mut effect_frame_errors = Vec::new();
    let mut scored_chunks = Vec::with_capacity(chunks.len());

    for ((chunk, label), prediction) in chunks.iter().zip(truth).zip(predictions) {
        let latency = prediction
            .decision_at_ns
            .saturating_sub(prediction.chunk_received_at_ns);
        decision_latencies.push(latency);

        let predicted_action = prediction.decision.action();
        let single_truth = (label.valid_actions.len() == 1).then(|| label.valid_actions[0]);
        let action_match = single_truth.map(|expected| predicted_action == Some(expected));
        let ambiguous_match = (label.valid_actions.len() > 1).then(|| {
            predicted_action.is_some_and(|action| label.valid_actions.contains(&action))
        });

        if let Some(expected) = single_truth {
            unambiguous_count += 1;
            let predicted_index = prediction.decision.confusion_index();
            confusion_matrix[expected.confusion_index()][predicted_index] += 1;
            if action_match == Some(true) {
                unambiguous_correct += 1;
            }
            if expected == TurnActionV1::Wait && prediction.decision != PredictedTurnDecisionV1::Wait {
                wait_false_activations += 1;
            }
            if expected == TurnActionV1::Continue
                && prediction.decision == PredictedTurnDecisionV1::Yield
            {
                continue_premature_yields += 1;
            }
            if matches!(expected, TurnActionV1::Respond | TurnActionV1::Yield) {
                effect_eligible += 1;
            }
        } else {
            ambiguous_count += 1;
            if ambiguous_match == Some(true) {
                ambiguous_valid_set_accepts += 1;
            }
        }

        if prediction.decision == PredictedTurnDecisionV1::ParseFailure {
            parse_failures += 1;
        }

        // Latency includes every emitted effect, including false activations
        // and wrong-action effects. Do not condition the latency sample on a
        // reference match.
        let effect_latency = prediction.effect_at_ns.map(|effect_at_ns| {
            effect_at_ns.saturating_sub(prediction.decision_at_ns)
        });
        if let Some(latency) = effect_latency {
            effect_latencies.push(latency);
        }

        let mut effect_error = None;
        let mut effect_window_success = None;
        if let (Some(expected), Some(target), Some(window)) = (
            single_truth,
            label.effect_target_frame,
            label.effect_window,
        ) {
            if let (Some(effect_frame), Some(_effect_at_ns)) =
                (prediction.effect_frame, prediction.effect_at_ns)
            {
                effect_attempts += 1;
                let error = effect_frame as i64 - target as i64;
                effect_error = Some(error);
                effect_frame_errors.push(error.unsigned_abs());
                let success = prediction.decision.action() == Some(expected)
                    && window.start_frame <= effect_frame
                    && effect_frame < window.end_frame;
                effect_window_success = Some(success);
                if success {
                    effect_successes += 1;
                }
            }
        }

        scored_chunks.push(StreamChunkScoreV1 {
            chunk_index: chunk.chunk_index,
            chunk_id: chunk.chunk_id.clone(),
            valid_truth_actions: label.valid_actions.clone(),
            predicted_decision: prediction.decision,
            unambiguous_action_match: action_match,
            ambiguous_valid_set_accept: ambiguous_match,
            decision_latency_ns: latency,
            effect_latency_ns: effect_latency,
            effect_frame_error: effect_error,
            effect_window_success,
        });
    }

    let mut leakage_chunks = BTreeSet::new();
    for (index, prediction) in predictions.iter().enumerate() {
        if prediction.decision != PredictedTurnDecisionV1::Yield {
            continue;
        }
        for later_index in (index + 1)..predictions.len() {
            if truth[later_index].valid_actions.contains(&TurnActionV1::Respond) {
                break;
            }
            if predictions[later_index].decision != PredictedTurnDecisionV1::Wait {
                leakage_chunks.insert(later_index as u32);
            }
        }
    }

    let (action_metrics, macro_f1_bp) = per_action_metrics(&confusion_matrix);
    let summary = StreamScoreSummaryV1 {
        confusion_matrix,
        unambiguous_chunk_count: unambiguous_count,
        unambiguous_correct_count: unambiguous_correct,
        top1_accuracy_bp: bp(unambiguous_correct, unambiguous_count),
        action_metrics,
        macro_f1_bp,
        ambiguous_chunk_count: ambiguous_count,
        ambiguous_valid_set_accept_count: ambiguous_valid_set_accepts,
        ambiguous_valid_set_acceptance_bp: bp(ambiguous_valid_set_accepts, ambiguous_count),
        parse_failure_count: parse_failures,
        wait_false_activation_count: wait_false_activations,
        continue_premature_yield_count: continue_premature_yields,
        post_yield_non_wait_leakage_count: leakage_chunks.len() as u32,
        effect_eligible_count: effect_eligible,
        effect_window_attempt_count: effect_attempts,
        effect_window_success_count: effect_successes,
        effect_window_success_bp: bp(effect_successes, effect_eligible),
        decision_latency: latency_percentiles(&decision_latencies),
        effect_latency: latency_percentiles(&effect_latencies),
        effect_frame_error: frame_error_percentiles(&effect_frame_errors),
    };

    let payload = StreamScorePayloadV1 {
        schema_version: STREAM_REPORT_SCHEMA_V1.to_owned(),
        request: request.clone(),
        chunks: chunks.to_vec(),
        truth: truth.to_vec(),
        predictions: predictions.to_vec(),
        scored_chunks: scored_chunks.clone(),
        summary: summary.clone(),
    };
    let payload_bytes = serde_json::to_vec(&payload)
        .map_err(|_| StreamScorerErrorV1::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(STREAM_REPORT_DOMAIN_V1);
    hasher.update([0]);
    hasher.update(payload_bytes);
    let report_sha256 = format!("{:x}", hasher.finalize());

    Ok(StreamScoreReportV1 {
        schema_version: STREAM_REPORT_SCHEMA_V1.to_owned(),
        request: request.clone(),
        chunks: chunks.to_vec(),
        truth: truth.to_vec(),
        predictions: predictions.to_vec(),
        scored_chunks,
        summary,
        report_sha256,
    })
}

/// Replay the complete report from frozen annotations and the raw declared
/// prediction trace. A self-reported report digest alone cannot qualify a run.
pub fn verify_stream_score_report_v1(
    request: &StreamSessionRequestV1,
    chunks: &[StreamChunkV1],
    truth: &[StreamChunkTruthV1],
    predictions: &[StreamChunkPredictionV1],
    supplied: &StreamScoreReportV1,
) -> Result<(), StreamScorerErrorV1> {
    let replayed = score_streaming_session_v1(request, chunks, truth, predictions)?;
    if &replayed != supplied {
        return Err(StreamScorerErrorV1::ReportReplayMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(value: u64) -> String {
        format!("{value:064x}")
    }

    fn request() -> StreamSessionRequestV1 {
        StreamSessionRequestV1 {
            schema_version: STREAM_SESSION_SCHEMA_V1.to_owned(),
            session_id: "synthetic-five-chunk-session".to_owned(),
            source_audio_sha256: h(1),
            task_manifest_sha256: h(2),
            annotation_policy_sha256: h(3),
            immutable_policy_identity_sha256: h(4),
            runner_build_sha256: h(5),
            initial_policy_state_sha256: h(100),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 100,
            expected_chunk_count: 5,
            seed: 7,
        }
    }

    fn chunks() -> Vec<StreamChunkV1> {
        [(0, 0, 20), (1, 20, 40), (2, 40, 60), (3, 60, 80), (4, 80, 100)]
            .into_iter()
            .map(|(index, start, end)| StreamChunkV1 {
                schema_version: STREAM_CHUNK_SCHEMA_V1.to_owned(),
                chunk_index: index,
                chunk_id: format!("chunk-{index}"),
                start_frame: start,
                end_frame: end,
            })
            .collect()
    }

    fn truth() -> Vec<StreamChunkTruthV1> {
        vec![
            StreamChunkTruthV1 {
                schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
                chunk_index: 0, chunk_id: "chunk-0".to_owned(),
                valid_actions: vec![TurnActionV1::Wait],
                effect_target_frame: None, effect_window: None,
            },
            StreamChunkTruthV1 {
                schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
                chunk_index: 1, chunk_id: "chunk-1".to_owned(),
                valid_actions: vec![TurnActionV1::Respond],
                effect_target_frame: Some(50),
                effect_window: Some(FrameWindowV1 { start_frame: 40, end_frame: 60 }),
            },
            StreamChunkTruthV1 {
                schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
                chunk_index: 2, chunk_id: "chunk-2".to_owned(),
                valid_actions: vec![TurnActionV1::Continue],
                effect_target_frame: None, effect_window: None,
            },
            StreamChunkTruthV1 {
                schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
                chunk_index: 3, chunk_id: "chunk-3".to_owned(),
                valid_actions: vec![TurnActionV1::Yield],
                effect_target_frame: Some(90),
                effect_window: Some(FrameWindowV1 { start_frame: 80, end_frame: 100 }),
            },
            StreamChunkTruthV1 {
                schema_version: STREAM_TRUTH_SCHEMA_V1.to_owned(),
                chunk_index: 4, chunk_id: "chunk-4".to_owned(),
                valid_actions: vec![TurnActionV1::Wait],
                effect_target_frame: None, effect_window: None,
            },
        ]
    }

    fn predictions(req: &StreamSessionRequestV1) -> Vec<StreamChunkPredictionV1> {
        [
            (PredictedTurnDecisionV1::Wait, None, None),
            (PredictedTurnDecisionV1::Respond, Some(50), Some(220)),
            (PredictedTurnDecisionV1::Continue, None, None),
            (PredictedTurnDecisionV1::Yield, Some(90), Some(420)),
            (PredictedTurnDecisionV1::Wait, None, None),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (decision, effect_frame, effect_at_ns))| StreamChunkPredictionV1 {
            schema_version: STREAM_PREDICTION_SCHEMA_V1.to_owned(),
            chunk_index: index as u32,
            chunk_id: format!("chunk-{index}"),
            decision,
            observed_through_frame: (index as u64 + 1) * 20,
            chunk_received_at_ns: (index as u64 + 1) * 100,
            decision_at_ns: (index as u64 + 1) * 100 + 10,
            effect_at_ns,
            effect_frame,
            input_prefix_sha256: h(200 + index as u64),
            pre_state_sha256: if index == 0 { req.initial_policy_state_sha256.clone() } else { h(300 + index as u64) },
            post_state_sha256: h(301 + index as u64),
            immutable_policy_identity_sha256: req.immutable_policy_identity_sha256.clone(),
            runner_build_sha256: req.runner_build_sha256.clone(),
            parse_failure_sha256: None,
        })
        .collect()
    }

    fn run(
        req: &StreamSessionRequestV1,
        chunks: &[StreamChunkV1],
        truth: &[StreamChunkTruthV1],
        predictions: &[StreamChunkPredictionV1],
    ) -> Result<StreamScoreReportV1, StreamScorerErrorV1> {
        score_streaming_session_v1(req, chunks, truth, predictions)
    }

    #[test]
    fn complete_four_action_trace_scores_per_action_and_effect_timing() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let predictions = predictions(&req);
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        assert_eq!(report.summary.top1_accuracy_bp, Some(10_000));
        assert_eq!(report.summary.effect_eligible_count, 2);
        assert_eq!(report.summary.effect_window_success_count, 2);
        assert_eq!(report.summary.wait_false_activation_count, 0);
        assert_eq!(report.summary.continue_premature_yield_count, 0);
        assert_eq!(report.summary.decision_latency.p95_ns, Some(10));
        assert_eq!(report.summary.effect_latency.p95_ns, Some(10));
    }

    #[test]
    fn always_wait_baseline_is_not_mistaken_for_a_contingent_policy() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        for prediction in &mut predictions {
            prediction.decision = PredictedTurnDecisionV1::Wait;
            prediction.effect_frame = None;
            prediction.effect_at_ns = None;
            prediction.parse_failure_sha256 = None;
        }
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        assert_eq!(report.summary.top1_accuracy_bp, Some(4_000));
        assert_eq!(report.summary.wait_false_activation_count, 0);
        assert_eq!(report.summary.effect_window_success_count, 0);
        assert_eq!(report.summary.effect_window_success_bp, Some(0));
    }

    #[test]
    fn always_respond_baseline_false_activates_during_wait() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        for (index, prediction) in predictions.iter_mut().enumerate() {
            prediction.decision = PredictedTurnDecisionV1::Respond;
            prediction.effect_frame = Some((index as u64 + 1) * 20 + 1);
            prediction.effect_at_ns = Some(prediction.decision_at_ns + 20);
            prediction.parse_failure_sha256 = None;
        }
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        assert_eq!(report.summary.wait_false_activation_count, 2);
        assert_eq!(report.summary.effect_window_eligible_count, 2);
        assert_eq!(report.summary.effect_window_success_count, 1);
    }

    #[test]
    fn parse_failure_is_not_counted_as_wait() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[0].decision = PredictedTurnDecisionV1::ParseFailure;
        predictions[0].parse_failure_sha256 = Some(h(999));
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        assert_eq!(report.summary.parse_failure_count, 1);
        assert_eq!(report.summary.wait_false_activation_count, 1);
        assert_eq!(report.summary.confusion_matrix[0][4], 1);
    }

    #[test]
    fn ambiguous_truth_is_preserved_as_a_valid_action_set() {
        let req = request();
        let chunks = chunks();
        let mut truth = truth();
        truth[0].valid_actions = vec![TurnActionV1::Wait, TurnActionV1::Respond];
        let predictions = predictions(&req);
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        assert_eq!(report.summary.ambiguous_chunk_count, 1);
        assert_eq!(report.summary.ambiguous_valid_set_accept_count, 1);
        assert_eq!(report.summary.top1_accuracy_bp, Some(10_000));
    }

    #[test]
    fn missing_chunk_prediction_fails_closed() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions.pop();
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::PredictionCountMismatch)
        );
    }

    #[test]
    fn chunk_gap_is_rejected() {
        let req = request();
        let mut chunks = chunks();
        chunks[2].start_frame += 1;
        let truth = truth();
        let predictions = predictions(&req);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::ChunkCoverageGapOrOverlap)
        );
    }

    #[test]
    fn lookahead_is_rejected_even_by_one_frame() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[1].observed_through_frame += 1;
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::LookaheadDetected)
        );
    }

    #[test]
    fn decision_before_chunk_receipt_is_rejected() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[2].decision_at_ns = predictions[2].chunk_received_at_ns - 1;
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::DecisionBeforeReceipt)
        );
    }

    #[test]
    fn monotonic_runner_clock_regression_is_rejected_independently_of_audio_time() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[2].chunk_received_at_ns = predictions[1].chunk_received_at_ns - 1;
        predictions[2].decision_at_ns = predictions[2].chunk_received_at_ns + 1;
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::ClockRegression)
        );
    }

    #[test]
    fn backdated_sample_effect_is_rejected() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[1].effect_frame = Some(39);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::BackdatedEffect)
        );
    }

    #[test]
    fn backdated_monotonic_effect_is_rejected() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[1].effect_at_ns = Some(predictions[1].decision_at_ns - 1);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::BackdatedEffect)
        );
    }

    #[test]
    fn untimed_wait_or_continue_cannot_carry_an_effect() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[0].effect_frame = Some(10);
        predictions[0].effect_at_ns = Some(111);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::InvalidActionEffect)
        );
    }

    #[test]
    fn response_without_both_effect_clocks_is_rejected() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[1].effect_at_ns = None;
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::InvalidActionEffect)
        );
    }

    #[test]
    fn action_effect_frames_cannot_be_duplicated() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[3].effect_frame = predictions[1].effect_frame;
        predictions[3].effect_at_ns = Some(420);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::DuplicateOrRegressingEffect)
        );
    }

    #[test]
    fn yield_followed_by_non_wait_before_next_invitation_is_leakage() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[4].decision = PredictedTurnDecisionV1::Continue;
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        assert_eq!(report.summary.post_yield_non_wait_leakage_count, 1);
    }

    #[test]
    fn policy_state_chain_and_model_identity_are_bound() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let mut predictions = predictions(&req);
        predictions[1].pre_state_sha256 = h(998);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::StateChainMismatch)
        );
        let mut predictions = predictions(&req);
        predictions[0].immutable_policy_identity_sha256 = h(888);
        assert_eq!(
            run(&req, &chunks, &truth, &predictions),
            Err(StreamScorerErrorV1::PolicyIdentityMismatch)
        );
    }

    #[test]
    fn report_replay_rejects_tampered_summary() {
        let req = request();
        let chunks = chunks();
        let truth = truth();
        let predictions = predictions(&req);
        let report = run(&req, &chunks, &truth, &predictions).unwrap();
        verify_stream_score_report_v1(&req, &chunks, &truth, &predictions, &report).unwrap();
        let mut tampered = report;
        tampered.summary.top1_accuracy_bp = Some(0);
        assert_eq!(
            verify_stream_score_report_v1(&req, &chunks, &truth, &predictions, &tampered),
            Err(StreamScorerErrorV1::ReportReplayMismatch)
        );
    }
}
