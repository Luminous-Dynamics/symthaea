// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Artifact-bound evaluation of streaming musical turn-taking decisions.
//!
//! V1 evaluates supplied annotations and per-chunk agent traces. It does not
//! detect musical cues from waveforms, render audio, or prove that a runner
//! supplied only the prefix it declares. It does make the declared input
//! boundary, decision time and action-effect time explicit and rejects
//! backdated or lookahead-contaminated traces.
//!
//! The action vocabulary is intentionally small:
//! - Wait: preserve listening space.
//! - Respond: begin a new contribution after an invitation.
//! - Continue: sustain the current contribution rather than yield.
//! - Yield: relinquish the musical floor.
//!
//! These are interaction actions, not music-quality labels.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MUSICAL_TURNTAKING_SCHEMA_V1: &str = "mel-musical-turntaking-v1";
pub const MUSICAL_TURNTAKING_SCORER_V1: &str = "mel-musical-turntaking-scorer-v1";

/// Exact audio clock supplied by the dataset/runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalStreamArtifactV1 {
    pub audio_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub total_frames: u64,
}

/// A half-open sample-frame interval: [start_frame, end_frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MusicalFrameIntervalV1 {
    pub start_frame: u64,
    pub end_frame: u64,
}

impl MusicalFrameIntervalV1 {
    pub fn is_valid(self) -> bool {
        self.start_frame < self.end_frame
    }

    pub fn contains(self, frame: u64) -> bool {
        self.start_frame <= frame && frame < self.end_frame
    }
}

/// Action selected for the current musical interaction opportunity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicalTurnActionV1 {
    Wait,
    Respond,
    Continue,
    Yield,
}

impl MusicalTurnActionV1 {
    fn index(self) -> usize {
        match self {
            Self::Wait => 0,
            Self::Respond => 1,
            Self::Continue => 2,
            Self::Yield => 3,
        }
    }

    fn requires_effect_time(self) -> bool {
        matches!(self, Self::Respond | Self::Yield)
    }
}

/// Ground truth for one received audio chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalTurnTruthChunkV1 {
    pub chunk_index: u32,
    pub input_interval: MusicalFrameIntervalV1,
    pub expected_action: MusicalTurnActionV1,
    /// Expected response onset (Respond) or output-stop boundary (Yield).
    pub effect_target_frame: Option<u64>,
    /// Permitted half-open window for the expected onset/stop boundary.
    pub effect_window: Option<MusicalFrameIntervalV1>,
}

/// Ground truth for one complete streaming session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalTurnTruthV1 {
    pub schema_version: String,
    pub session_id: String,
    pub artifact: MusicalStreamArtifactV1,
    pub annotation_policy_version: String,
    pub annotation_set_sha256: String,
    pub chunks: Vec<MusicalTurnTruthChunkV1>,
}

/// Prediction for one chunk. A parsed action may still fail to produce the
/// expected output effect; that is measured separately from action accuracy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicalTurnDispositionV1 {
    Action {
        action: MusicalTurnActionV1,
        /// For Respond: actual/scheduled response onset. For Yield: actual/
        /// scheduled output-stop boundary. None means the action had no
        /// observable musical event timestamp in the session sample-frame clock.
        effect_frame: Option<u64>,
        /// Monotonic runner-clock time when the effect was actually emitted.
        /// Must be paired with effect_frame.
        effect_at_ns: Option<u64>,
    },
    Unparseable {
        raw_output_sha256: String,
        reason: String,
    },
}

/// One decision made after consuming exactly one audio chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalTurnPredictionChunkV1 {
    pub chunk_index: u32,
    pub input_interval: MusicalFrameIntervalV1,
    /// Last input frame available to the agent. Must equal chunk end.
    pub observed_through_frame: u64,
    /// Monotonic runner-clock time when the complete input chunk was received.
    pub chunk_received_at_ns: u64,
    /// Monotonic runner-clock time when the action decision was committed.
    pub decision_at_ns: u64,
    pub disposition: MusicalTurnDispositionV1,
}

/// Prediction log for one complete streaming session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalTurnPredictionV1 {
    pub schema_version: String,
    pub session_id: String,
    pub artifact: MusicalStreamArtifactV1,
    pub chunks: Vec<MusicalTurnPredictionChunkV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicalTurnTakingIssue {
    WrongTruthSchema,
    WrongPredictionSchema,
    EmptySessionId,
    EmptyAnnotationPolicyVersion,
    InvalidAudioDigest,
    InvalidAnnotationDigest,
    InvalidSampleRate,
    InvalidChannelCount,
    InvalidTotalFrames,
    EmptyChunkSequence,
    ChunkIndexMismatch { position: usize },
    InvalidChunkInterval { position: usize },
    ChunkPastArtifact { position: usize },
    ChunkGapOrOverlap { position: usize },
    IncompleteStreamCoverage,
    TimedActionMissingTarget { position: usize },
    UntimedActionHasTarget { position: usize },
    InvalidEffectWindow { position: usize },
    EffectTargetOutsideWindow { position: usize },
    EffectWindowBeforeDecisionOpportunity { position: usize },
    PredictionChunkCountMismatch,
    SessionMismatch,
    ArtifactMismatch { field: String },
    ObservedThroughMismatch { position: usize },
    DecisionBeforeChunkReceipt { position: usize },
    NonMonotonicChunkReceiptTime { position: usize },
    NonMonotonicDecisionTime { position: usize },
    NonMonotonicEffectTime { position: usize },
    EffectFrameBeforeObservedInput { position: usize },
    EffectBeforeDecision { position: usize },
    EffectTimestampPairMismatch { position: usize },
    UntimedActionHasEffect { position: usize },
    TimedActionParseDigestInvalid { position: usize },
    MissingParseFailureReason { position: usize },
    ParseFailureHasEffect { position: usize },
}

/// Per-action metrics. Missing precision is represented as None when the
/// action was never predicted; recall remains defined when the class exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalTurnActionMetricsV1 {
    pub action: MusicalTurnActionV1,
    pub support: usize,
    pub predicted: usize,
    pub true_positive: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

/// Replayable score for one session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalTurnTakingScoreV1 {
    pub scorer_version: String,
    pub session_id: String,
    pub audio_sha256: String,
    pub total_chunks: usize,
    pub parsed_chunks: usize,
    pub parse_failures: usize,
    pub correct_actions: usize,
    /// Action accuracy includes parse failures in the denominator.
    pub action_accuracy: Option<f64>,
    pub macro_f1: Option<f64>,
    pub balanced_accuracy: Option<f64>,
    /// Rows are ground truth [Wait, Respond, Continue, Yield], columns are
    /// predictions in the same order. Parse failures are counted separately.
    pub confusion_matrix: [[usize; 4]; 4],
    pub action_metrics: Vec<MusicalTurnActionMetricsV1>,
    pub timed_action_count: usize,
    pub timed_action_success_count: usize,
    pub timed_action_window_accuracy: Option<f64>,
    pub mean_signed_effect_error_ms: Option<f64>,
    pub p95_absolute_effect_error_ms: Option<f64>,
    pub wait_chunks: usize,
    pub wait_false_activation_count: usize,
    pub wait_false_activation_rate: Option<f64>,
    pub continue_chunks: usize,
    pub premature_yield_count: usize,
    /// Non-WAIT decisions while the reference stream remains in a post-yield
    /// listening interval before the next expected Respond action.
    pub post_yield_leakage_count: usize,
    pub decision_latency_p50_ms: Option<f64>,
    pub decision_latency_p95_ms: Option<f64>,
    pub decision_latency_p99_ms: Option<f64>,
    pub decision_latencies_ms: Vec<f64>,
    pub effect_latency_p50_ms: Option<f64>,
    pub effect_latency_p95_ms: Option<f64>,
    pub effect_latency_p99_ms: Option<f64>,
    pub effect_latencies_ms: Vec<f64>,
    pub signed_effect_errors_ms: Vec<f64>,
}

/// Validate a complete annotation/session contract.
pub fn validate_musical_turn_truth(
    truth: &MusicalTurnTruthV1,
) -> Vec<MusicalTurnTakingIssue> {
    let mut issues = Vec::new();
    if truth.schema_version != MUSICAL_TURNTAKING_SCHEMA_V1 {
        issues.push(MusicalTurnTakingIssue::WrongTruthSchema);
    }
    if truth.session_id.trim().is_empty() {
        issues.push(MusicalTurnTakingIssue::EmptySessionId);
    }
    if truth.annotation_policy_version.trim().is_empty() {
        issues.push(MusicalTurnTakingIssue::EmptyAnnotationPolicyVersion);
    }
    validate_artifact(&truth.artifact, &mut issues);
    if !is_lower_hex_digest(&truth.annotation_set_sha256) {
        issues.push(MusicalTurnTakingIssue::InvalidAnnotationDigest);
    }
    validate_chunk_coverage(
        truth.chunks.iter().map(|chunk| (chunk.chunk_index, chunk.input_interval)),
        truth.artifact.total_frames,
        &mut issues,
    );

    for (position, chunk) in truth.chunks.iter().enumerate() {
        let timed = chunk.expected_action.requires_effect_time();
        match (timed, chunk.effect_target_frame, chunk.effect_window) {
            (true, Some(target), Some(window)) => {
                if !window.is_valid() || window.end_frame > truth.artifact.total_frames {
                    issues.push(MusicalTurnTakingIssue::InvalidEffectWindow { position });
                } else {
                    if !window.contains(target) {
                        issues.push(MusicalTurnTakingIssue::EffectTargetOutsideWindow { position });
                    }
                    if window.start_frame < chunk.input_interval.end_frame {
                        issues.push(
                            MusicalTurnTakingIssue::EffectWindowBeforeDecisionOpportunity {
                                position,
                            },
                        );
                    }
                }
            }
            (true, _, _) => {
                issues.push(MusicalTurnTakingIssue::TimedActionMissingTarget { position });
            }
            (false, None, None) => {}
            (false, _, _) => {
                issues.push(MusicalTurnTakingIssue::UntimedActionHasTarget { position });
            }
        }
    }
    issues
}

/// Validate a complete prediction log. Missing chunks are rejected; WAIT
/// decisions must be explicit rather than inferred from an absent record.
pub fn validate_musical_turn_prediction(
    prediction: &MusicalTurnPredictionV1,
) -> Vec<MusicalTurnTakingIssue> {
    let mut issues = Vec::new();
    if prediction.schema_version != MUSICAL_TURNTAKING_SCHEMA_V1 {
        issues.push(MusicalTurnTakingIssue::WrongPredictionSchema);
    }
    if prediction.session_id.trim().is_empty() {
        issues.push(MusicalTurnTakingIssue::EmptySessionId);
    }
    validate_artifact(&prediction.artifact, &mut issues);
    validate_chunk_coverage(
        prediction
            .chunks
            .iter()
            .map(|chunk| (chunk.chunk_index, chunk.input_interval)),
        prediction.artifact.total_frames,
        &mut issues,
    );

    let mut previous_received_at = None;
    let mut previous_decision_at = None;
    let mut previous_effect_at = None;
    for (position, chunk) in prediction.chunks.iter().enumerate() {
        if chunk.observed_through_frame != chunk.input_interval.end_frame {
            issues.push(MusicalTurnTakingIssue::ObservedThroughMismatch { position });
        }
        if chunk.decision_at_ns < chunk.chunk_received_at_ns {
            issues.push(MusicalTurnTakingIssue::DecisionBeforeChunkReceipt { position });
        }
        if let Some(previous) = previous_received_at {
            if chunk.chunk_received_at_ns < previous {
                issues.push(MusicalTurnTakingIssue::NonMonotonicChunkReceiptTime { position });
            }
        }
        if let Some(previous) = previous_decision_at {
            if chunk.decision_at_ns < previous {
                issues.push(MusicalTurnTakingIssue::NonMonotonicDecisionTime { position });
            }
        }
        previous_received_at = Some(chunk.chunk_received_at_ns);
        previous_decision_at = Some(chunk.decision_at_ns);

        match &chunk.disposition {
            MusicalTurnDispositionV1::Action {
                action,
                effect_frame,
                effect_at_ns,
            } => {
                if action.requires_effect_time() {
                    if effect_frame.is_some() != effect_at_ns.is_some() {
                        issues.push(MusicalTurnTakingIssue::EffectTimestampPairMismatch { position });
                    }
                    if let Some(frame) = effect_frame {
                        if *frame < chunk.observed_through_frame {
                            issues.push(
                                MusicalTurnTakingIssue::EffectFrameBeforeObservedInput { position },
                            );
                        }
                        if *frame >= prediction.artifact.total_frames {
                            issues.push(MusicalTurnTakingIssue::EffectPastArtifact { position });
                        }
                    }
                    if let Some(effect_time) = effect_at_ns {
                        if *effect_time < chunk.decision_at_ns {
                            issues.push(MusicalTurnTakingIssue::EffectBeforeDecision { position });
                        }
                        if previous_effect_at.is_some_and(|previous| *effect_time < previous) {
                            issues.push(MusicalTurnTakingIssue::NonMonotonicEffectTime { position });
                        }
                        previous_effect_at = Some(*effect_time);
                    }
                } else if effect_frame.is_some() || effect_at_ns.is_some() {
                    issues.push(MusicalTurnTakingIssue::UntimedActionHasEffect { position });
                }
            }
            MusicalTurnDispositionV1::Unparseable {
                raw_output_sha256,
                reason,
            } => {
                if !is_lower_hex_digest(raw_output_sha256) {
                    issues.push(MusicalTurnTakingIssue::TimedActionParseDigestInvalid { position });
                }
                if reason.trim().is_empty() {
                    issues.push(MusicalTurnTakingIssue::MissingParseFailureReason { position });
                }
            }
        }
    }
    issues
}

/// Evaluate one complete session. This scores declared actions and timing
/// receipts; it does not inspect the audio waveform or verify runner isolation.
pub fn score_musical_turn_taking(
    truth: &MusicalTurnTruthV1,
    prediction: &MusicalTurnPredictionV1,
) -> Result<MusicalTurnTakingScoreV1, Vec<MusicalTurnTakingIssue>> {
    let mut issues = validate_musical_turn_truth(truth);
    issues.extend(validate_musical_turn_prediction(prediction));

    if truth.session_id != prediction.session_id {
        issues.push(MusicalTurnTakingIssue::SessionMismatch);
    }
    compare_artifact(&truth.artifact, &prediction.artifact, &mut issues);
    if truth.chunks.len() != prediction.chunks.len() {
        issues.push(MusicalTurnTakingIssue::PredictionChunkCountMismatch);
    } else {
        for (position, (expected, actual)) in
            truth.chunks.iter().zip(prediction.chunks.iter()).enumerate()
        {
            if expected.chunk_index != actual.chunk_index
                || expected.input_interval != actual.input_interval
            {
                issues.push(MusicalTurnTakingIssue::ChunkIndexMismatch { position });
            }
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut confusion_matrix = [[0usize; 4]; 4];
    let mut parse_failures = 0usize;
    let mut correct_actions = 0usize;
    let mut decision_latencies_ms = Vec::with_capacity(truth.chunks.len());
    let mut effect_latencies_ms = Vec::new();
    let mut signed_effect_errors_ms = Vec::new();
    let mut timed_action_count = 0usize;
    let mut timed_action_success_count = 0usize;
    let mut wait_chunks = 0usize;
    let mut wait_false_activation_count = 0usize;
    let mut continue_chunks = 0usize;
    let mut premature_yield_count = 0usize;
    let mut post_yield_leakage_count = 0usize;
    let mut awaiting_partner_after_yield = false;

    for (expected, actual) in truth.chunks.iter().zip(prediction.chunks.iter()) {
        let predicted_action = match &actual.disposition {
            MusicalTurnDispositionV1::Action { action, .. } => Some(*action),
            MusicalTurnDispositionV1::Unparseable { .. } => {
                parse_failures += 1;
                None
            }
        };

        if let Some(action) = predicted_action {
            confusion_matrix[expected.expected_action.index()][action.index()] += 1;
            if action == expected.expected_action {
                correct_actions += 1;
            }
        }

        let decision_latency_ns = actual
            .decision_at_ns
            .saturating_sub(actual.chunk_received_at_ns);
        decision_latencies_ms.push(decision_latency_ns as f64 / 1_000_000.0);
        if let MusicalTurnDispositionV1::Action {
            effect_at_ns: Some(effect_time),
            ..
        } = &actual.disposition
        {
            let effect_latency_ns = effect_time.saturating_sub(actual.decision_at_ns);
            effect_latencies_ms.push(effect_latency_ns as f64 / 1_000_000.0);
        }

        if expected.expected_action.requires_effect_time() {
            timed_action_count += 1;
            if let (
                Some(expected_target),
                Some(expected_window),
                Some(predicted),
                Some(predicted_effect),
            ) = (
                expected.effect_target_frame,
                expected.effect_window,
                predicted_action,
                action_effect_frame(&actual.disposition),
            ) {
                if predicted == expected.expected_action {
                    signed_effect_errors_ms.push(
                        signed_frame_delta(predicted_effect, expected_target) as f64 * 1000.0
                            / truth.artifact.sample_rate_hz as f64,
                    );
                    if expected_window.contains(predicted_effect) {
                        timed_action_success_count += 1;
                    }
                }
            }
        }

        if expected.expected_action == MusicalTurnActionV1::Wait {
            wait_chunks += 1;
            if predicted_action.is_some_and(|action| action != MusicalTurnActionV1::Wait) {
                wait_false_activation_count += 1;
            }
            if awaiting_partner_after_yield
                && predicted_action.is_some_and(|action| action != MusicalTurnActionV1::Wait)
            {
                post_yield_leakage_count += 1;
            }
        }

        if expected.expected_action == MusicalTurnActionV1::Continue {
            continue_chunks += 1;
            if predicted_action == Some(MusicalTurnActionV1::Yield) {
                premature_yield_count += 1;
            }
        }

        // A new expected response invitation ends the post-yield listening
        // interval. A Yield starts it. This measures action leakage in the
        // trace, not audible PCM leakage (which requires a separate verifier).
        if expected.expected_action == MusicalTurnActionV1::Respond {
            awaiting_partner_after_yield = false;
        } else if expected.expected_action == MusicalTurnActionV1::Yield {
            awaiting_partner_after_yield = true;
        }
    }

    let total_chunks = truth.chunks.len();
    let parsed_chunks = total_chunks - parse_failures;
    let action_accuracy = ratio(correct_actions, total_chunks);
    let mut action_metrics = Vec::with_capacity(4);
    let mut class_f1 = Vec::new();
    let mut class_recall = Vec::new();

    for action in [
        MusicalTurnActionV1::Wait,
        MusicalTurnActionV1::Respond,
        MusicalTurnActionV1::Continue,
        MusicalTurnActionV1::Yield,
    ] {
        let index = action.index();
        let true_positive = confusion_matrix[index][index];
        let support = confusion_matrix[index].iter().sum::<usize>()
            + parse_failures_for_action(truth, prediction, action);
        let predicted = (0..4)
            .map(|row| confusion_matrix[row][index])
            .sum::<usize>();
        let precision = if predicted == 0 {
            None
        } else {
            Some(true_positive as f64 / predicted as f64)
        };
        let recall = if support == 0 {
            None
        } else {
            Some(true_positive as f64 / support as f64)
        };
        let f1 = match (precision, recall) {
            (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
            (Some(_), Some(_)) => Some(0.0),
            (None, Some(_)) => Some(0.0),
            _ => None,
        };
        if support > 0 {
            class_f1.push(f1.unwrap_or(0.0));
            class_recall.push(recall.unwrap_or(0.0));
        }
        action_metrics.push(MusicalTurnActionMetricsV1 {
            action,
            support,
            predicted,
            true_positive,
            precision,
            recall,
            f1,
        });
    }

    Ok(MusicalTurnTakingScoreV1 {
        scorer_version: MUSICAL_TURNTAKING_SCORER_V1.into(),
        session_id: truth.session_id.clone(),
        audio_sha256: truth.artifact.audio_sha256.clone(),
        total_chunks,
        parsed_chunks,
        parse_failures,
        correct_actions,
        action_accuracy,
        macro_f1: mean(&class_f1),
        balanced_accuracy: mean(&class_recall),
        confusion_matrix,
        action_metrics,
        timed_action_count,
        timed_action_success_count,
        timed_action_window_accuracy: ratio(timed_action_success_count, timed_action_count),
        mean_signed_effect_error_ms: mean(&signed_effect_errors_ms),
        p95_absolute_effect_error_ms: percentile(
            &signed_effect_errors_ms.iter().map(|v| v.abs()).collect::<Vec<_>>(),
            0.95,
        ),
        wait_chunks,
        wait_false_activation_count,
        wait_false_activation_rate: ratio(wait_false_activation_count, wait_chunks),
        continue_chunks,
        premature_yield_count,
        post_yield_leakage_count,
        decision_latency_p50_ms: percentile(&decision_latencies_ms, 0.50),
        decision_latency_p95_ms: percentile(&decision_latencies_ms, 0.95),
        decision_latency_p99_ms: percentile(&decision_latencies_ms, 0.99),
        decision_latencies_ms,
        effect_latency_p50_ms: percentile(&effect_latencies_ms, 0.50),
        effect_latency_p95_ms: percentile(&effect_latencies_ms, 0.95),
        effect_latency_p99_ms: percentile(&effect_latencies_ms, 0.99),
        effect_latencies_ms,
        signed_effect_errors_ms,
    })
}

/// Commit the exact declared truth and prediction inputs. The hash binds the
/// serialized evaluator inputs; it does not authenticate the source WAV bytes.
pub fn musical_turntaking_input_commitment_sha256(
    truth: &MusicalTurnTruthV1,
    prediction: &MusicalTurnPredictionV1,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&(truth, prediction))?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn validate_artifact(
    artifact: &MusicalStreamArtifactV1,
    issues: &mut Vec<MusicalTurnTakingIssue>,
) {
    if !is_lower_hex_digest(&artifact.audio_sha256) {
        issues.push(MusicalTurnTakingIssue::InvalidAudioDigest);
    }
    if artifact.sample_rate_hz == 0 {
        issues.push(MusicalTurnTakingIssue::InvalidSampleRate);
    }
    if artifact.channels == 0 {
        issues.push(MusicalTurnTakingIssue::InvalidChannelCount);
    }
    if artifact.total_frames == 0 {
        issues.push(MusicalTurnTakingIssue::InvalidTotalFrames);
    }
}

fn validate_chunk_coverage(
    chunks: impl Iterator<Item = (u32, MusicalFrameIntervalV1)>,
    total_frames: u64,
    issues: &mut Vec<MusicalTurnTakingIssue>,
) {
    let chunks: Vec<_> = chunks.collect();
    if chunks.is_empty() {
        issues.push(MusicalTurnTakingIssue::EmptyChunkSequence);
        return;
    }
    let mut previous_end = 0u64;
    for (position, (chunk_index, interval)) in chunks.iter().copied().enumerate() {
        if chunk_index as usize != position {
            issues.push(MusicalTurnTakingIssue::ChunkIndexMismatch { position });
        }
        if !interval.is_valid() {
            issues.push(MusicalTurnTakingIssue::InvalidChunkInterval { position });
            continue;
        }
        if interval.end_frame > total_frames {
            issues.push(MusicalTurnTakingIssue::ChunkPastArtifact { position });
        }
        if interval.start_frame != previous_end {
            issues.push(MusicalTurnTakingIssue::ChunkGapOrOverlap { position });
        }
        previous_end = interval.end_frame;
    }
    if previous_end != total_frames {
        issues.push(MusicalTurnTakingIssue::IncompleteStreamCoverage);
    }
}

fn compare_artifact(
    expected: &MusicalStreamArtifactV1,
    actual: &MusicalStreamArtifactV1,
    issues: &mut Vec<MusicalTurnTakingIssue>,
) {
    if expected.audio_sha256 != actual.audio_sha256 {
        issues.push(MusicalTurnTakingIssue::ArtifactMismatch {
            field: "audio_sha256".into(),
        });
    }
    if expected.sample_rate_hz != actual.sample_rate_hz {
        issues.push(MusicalTurnTakingIssue::ArtifactMismatch {
            field: "sample_rate_hz".into(),
        });
    }
    if expected.channels != actual.channels {
        issues.push(MusicalTurnTakingIssue::ArtifactMismatch {
            field: "channels".into(),
        });
    }
    if expected.total_frames != actual.total_frames {
        issues.push(MusicalTurnTakingIssue::ArtifactMismatch {
            field: "total_frames".into(),
        });
    }
}

fn action_effect_frame(disposition: &MusicalTurnDispositionV1) -> Option<u64> {
    match disposition {
        MusicalTurnDispositionV1::Action { effect_frame, .. } => *effect_frame,
        MusicalTurnDispositionV1::Unparseable { .. } => None,
    }
}

fn parse_failures_for_action(
    truth: &MusicalTurnTruthV1,
    prediction: &MusicalTurnPredictionV1,
    action: MusicalTurnActionV1,
) -> usize {
    truth
        .chunks
        .iter()
        .zip(prediction.chunks.iter())
        .filter(|(expected, actual)| {
            expected.expected_action == action
                && matches!(
                    &actual.disposition,
                    MusicalTurnDispositionV1::Unparseable { .. }
                )
        })
        .count()
}

fn signed_frame_delta(predicted: u64, expected: u64) -> i64 {
    if predicted >= expected {
        i64::try_from(predicted - expected).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(expected - predicted).unwrap_or(i64::MAX)
    }
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (quantile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> MusicalStreamArtifactV1 {
        MusicalStreamArtifactV1 {
            audio_sha256: format!("{:x}", Sha256::digest(b"test stream bytes")),
            sample_rate_hz: 10_000,
            channels: 2,
            total_frames: 600,
        }
    }

    fn truth() -> MusicalTurnTruthV1 {
        MusicalTurnTruthV1 {
            schema_version: MUSICAL_TURNTAKING_SCHEMA_V1.into(),
            session_id: "session-001".into(),
            artifact: artifact(),
            annotation_policy_version: "music-turn-policy-v1".into(),
            annotation_set_sha256: format!("{:x}", Sha256::digest(b"annotations")),
            chunks: vec![
                MusicalTurnTruthChunkV1 {
                    chunk_index: 0,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 0, end_frame: 100 },
                    expected_action: MusicalTurnActionV1::Wait,
                    effect_target_frame: None,
                    effect_window: None,
                },
                MusicalTurnTruthChunkV1 {
                    chunk_index: 1,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 100, end_frame: 200 },
                    expected_action: MusicalTurnActionV1::Respond,
                    effect_target_frame: Some(220),
                    effect_window: Some(MusicalFrameIntervalV1 { start_frame: 200, end_frame: 240 }),
                },
                MusicalTurnTruthChunkV1 {
                    chunk_index: 2,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 200, end_frame: 300 },
                    expected_action: MusicalTurnActionV1::Continue,
                    effect_target_frame: None,
                    effect_window: None,
                },
                MusicalTurnTruthChunkV1 {
                    chunk_index: 3,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 300, end_frame: 400 },
                    expected_action: MusicalTurnActionV1::Yield,
                    effect_target_frame: Some(420),
                    effect_window: Some(MusicalFrameIntervalV1 { start_frame: 410, end_frame: 450 }),
                },
                MusicalTurnTruthChunkV1 {
                    chunk_index: 4,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 400, end_frame: 500 },
                    expected_action: MusicalTurnActionV1::Wait,
                    effect_target_frame: None,
                    effect_window: None,
                },
                MusicalTurnTruthChunkV1 {
                    chunk_index: 5,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 500, end_frame: 600 },
                    expected_action: MusicalTurnActionV1::Wait,
                    effect_target_frame: None,
                    effect_window: None,
                },
            ],
        }
    }

    fn prediction() -> MusicalTurnPredictionV1 {
        MusicalTurnPredictionV1 {
            schema_version: MUSICAL_TURNTAKING_SCHEMA_V1.into(),
            session_id: "session-001".into(),
            artifact: artifact(),
            chunks: vec![
                MusicalTurnPredictionChunkV1 {
                    chunk_index: 0,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 0, end_frame: 100 },
                    observed_through_frame: 100,
                    chunk_received_at_ns: 10_000_000,
                    decision_at_ns: 10_000_000,
                    disposition: MusicalTurnDispositionV1::Action {
                        action: MusicalTurnActionV1::Wait,
                        effect_frame: None,
                        effect_at_ns: None,
                    },
                },
                MusicalTurnPredictionChunkV1 {
                    chunk_index: 1,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 100, end_frame: 200 },
                    observed_through_frame: 200,
                    chunk_received_at_ns: 20_000_000,
                    decision_at_ns: 20_000_000,
                    disposition: MusicalTurnDispositionV1::Action {
                        action: MusicalTurnActionV1::Respond,
                        effect_frame: Some(220),
                        effect_at_ns: Some(22_000_000),
                    },
                },
                MusicalTurnPredictionChunkV1 {
                    chunk_index: 2,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 200, end_frame: 300 },
                    observed_through_frame: 300,
                    chunk_received_at_ns: 30_000_000,
                    decision_at_ns: 30_000_000,
                    disposition: MusicalTurnDispositionV1::Action {
                        action: MusicalTurnActionV1::Continue,
                        effect_frame: None,
                        effect_at_ns: None,
                    },
                },
                MusicalTurnPredictionChunkV1 {
                    chunk_index: 3,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 300, end_frame: 400 },
                    observed_through_frame: 400,
                    chunk_received_at_ns: 40_000_000,
                    decision_at_ns: 40_000_000,
                    disposition: MusicalTurnDispositionV1::Action {
                        action: MusicalTurnActionV1::Yield,
                        effect_frame: Some(420),
                        effect_at_ns: Some(42_000_000),
                    },
                },
                MusicalTurnPredictionChunkV1 {
                    chunk_index: 4,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 400, end_frame: 500 },
                    observed_through_frame: 500,
                    chunk_received_at_ns: 50_000_000,
                    decision_at_ns: 50_000_000,
                    disposition: MusicalTurnDispositionV1::Action {
                        action: MusicalTurnActionV1::Wait,
                        effect_frame: None,
                        effect_at_ns: None,
                    },
                },
                MusicalTurnPredictionChunkV1 {
                    chunk_index: 5,
                    input_interval: MusicalFrameIntervalV1 { start_frame: 500, end_frame: 600 },
                    observed_through_frame: 600,
                    chunk_received_at_ns: 60_000_000,
                    decision_at_ns: 60_000_000,
                    disposition: MusicalTurnDispositionV1::Action {
                        action: MusicalTurnActionV1::Wait,
                        effect_frame: None,
                        effect_at_ns: None,
                    },
                },
            ],
        }
    }

    #[test]
    fn exact_four_action_trace_scores_perfectly() {
        let score = score_musical_turn_taking(&truth(), &prediction()).unwrap();
        assert_eq!(score.total_chunks, 6);
        assert_eq!(score.parsed_chunks, 6);
        assert_eq!(score.correct_actions, 6);
        assert_eq!(score.action_accuracy, Some(1.0));
        assert_eq!(score.timed_action_count, 2);
        assert_eq!(score.timed_action_success_count, 2);
        assert_eq!(score.timed_action_window_accuracy, Some(1.0));
        assert_eq!(score.wait_false_activation_count, 0);
        assert_eq!(score.premature_yield_count, 0);
    }

    #[test]
    fn action_confusion_and_premature_yield_are_separate() {
        let mut predicted = prediction();
        predicted.chunks[2].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Yield,
            effect_frame: Some(320),
            effect_at_ns: Some(32_000_000),
        };
        let score = score_musical_turn_taking(&truth(), &predicted).unwrap();
        assert_eq!(score.correct_actions, 5);
        assert_eq!(score.premature_yield_count, 1);
        assert_eq!(score.confusion_matrix[2][3], 1);
    }

    #[test]
    fn unparseable_is_not_a_wait_decision() {
        let mut predicted = prediction();
        predicted.chunks[0].disposition = MusicalTurnDispositionV1::Unparseable {
            raw_output_sha256: format!("{:x}", Sha256::digest(b"raw output")),
            reason: "invalid action JSON".into(),
        };
        let score = score_musical_turn_taking(&truth(), &predicted).unwrap();
        assert_eq!(score.parse_failures, 1);
        assert_eq!(score.parsed_chunks, 5);
        assert_eq!(score.correct_actions, 5);
        assert_eq!(score.wait_false_activation_count, 0);
    }

    #[test]
    fn lookahead_is_rejected() {
        let mut predicted = prediction();
        predicted.chunks[1].observed_through_frame = 240;
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::ObservedThroughMismatch { position: 1 }));
    }

    #[test]
    fn backdated_decision_is_rejected() {
        let mut predicted = prediction();
        predicted.chunks[1].decision_at_ns = 19_000_000;
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::DecisionBeforeChunkReceipt { position: 1 }));
    }

    #[test]
    fn response_effect_cannot_precede_decision() {
        let mut predicted = prediction();
        predicted.chunks[1].decision_at_ns = 21_000_000;
        predicted.chunks[1].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Respond,
            effect_frame: Some(220),
            effect_at_ns: Some(20_900_000),
        };
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::EffectBeforeDecision { position: 1 }));
    }

    #[test]
    fn reports_runner_latency_separately_from_musical_frame_error() {
        let mut predicted = prediction();
        predicted.chunks[1].decision_at_ns = 20_500_000;
        predicted.chunks[1].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Respond,
            effect_frame: Some(220),
            effect_at_ns: Some(22_500_000),
        };
        let score = score_musical_turn_taking(&truth(), &predicted).unwrap();

        assert_eq!(score.decision_latency_p95_ms, Some(0.5));
        assert_eq!(score.effect_latency_p50_ms, Some(2.0));
        assert_eq!(score.signed_effect_errors_ms[0], 0.0);
    }

    #[test]
    fn effect_frame_cannot_be_backdated_before_observed_audio() {
        let mut predicted = prediction();
        predicted.chunks[1].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Respond,
            effect_frame: Some(199),
            effect_at_ns: Some(22_000_000),
        };
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::EffectFrameBeforeObservedInput { position: 1 }));
    }

    #[test]
    fn effect_times_must_be_monotonic_across_the_stream() {
        let mut predicted = prediction();
        predicted.chunks[1].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Respond,
            effect_frame: Some(220),
            effect_at_ns: Some(45_000_000),
        };
        predicted.chunks[3].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Yield,
            effect_frame: Some(420),
            effect_at_ns: Some(42_000_000),
        };
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::NonMonotonicEffectTime { position: 3 }));
    }

    #[test]
    fn non_monotonic_chunk_receipt_time_is_rejected() {
        let mut predicted = prediction();
        predicted.chunks[2].chunk_received_at_ns = 19_000_000;
        predicted.chunks[2].decision_at_ns = 30_000_000;
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::NonMonotonicChunkReceiptTime {
                position: 2,
            }));
    }

    #[test]
    fn event_frame_and_runner_effect_time_must_be_paired() {
        let mut predicted = prediction();
        predicted.chunks[1].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Respond,
            effect_frame: Some(220),
            effect_at_ns: None,
        };
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::EffectTimestampPairMismatch {
                position: 1,
            }));
    }

    #[test]
    fn missing_chunk_is_rejected() {
        let mut predicted = prediction();
        predicted.chunks.remove(2);
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::PredictionChunkCountMismatch));
    }

    #[test]
    fn chunk_gap_and_wrong_audio_identity_are_rejected() {
        let mut annotations = truth();
        annotations.chunks[1].input_interval.start_frame = 101;
        let issues = validate_musical_turn_truth(&annotations);
        assert!(issues.contains(&MusicalTurnTakingIssue::ChunkGapOrOverlap { position: 1 }));

        let mut predicted = prediction();
        predicted.artifact.audio_sha256 = format!("{:x}", Sha256::digest(b"other stream"));
        assert!(score_musical_turn_taking(&truth(), &predicted)
            .unwrap_err()
            .contains(&MusicalTurnTakingIssue::ArtifactMismatch {
                field: "audio_sha256".into()
            }));
    }

    #[test]
    fn wait_and_continue_cannot_claim_action_effect_frames() {
        let mut predicted = prediction();
        predicted.chunks[0].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Wait,
            effect_frame: Some(101),
            effect_at_ns: Some(10_100_000),
        };
        assert!(validate_musical_turn_prediction(&predicted)
            .contains(&MusicalTurnTakingIssue::UntimedActionHasEffect { position: 0 }));

        predicted = prediction();
        predicted.chunks[2].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Continue,
            effect_frame: Some(301),
            effect_at_ns: Some(30_100_000),
        };
        assert!(validate_musical_turn_prediction(&predicted)
            .contains(&MusicalTurnTakingIssue::UntimedActionHasEffect { position: 2 }));
    }

    #[test]
    fn commitment_binds_exact_truth_and_prediction() {
        let original =
            musical_turntaking_input_commitment_sha256(&truth(), &prediction()).unwrap();
        let mut changed = prediction();
        changed.chunks[1].decision_at_ns = 20_500_000;
        let changed_hash =
            musical_turntaking_input_commitment_sha256(&truth(), &changed).unwrap();
        assert_ne!(original, changed_hash);
    }

    #[test]
    fn post_yield_non_wait_decisions_are_reported_separately() {
        let annotations = truth();
        let mut predicted = prediction();
        predicted.chunks[4].disposition = MusicalTurnDispositionV1::Action {
            action: MusicalTurnActionV1::Respond,
            effect_frame: Some(510),
            effect_at_ns: Some(51_000_000),
        };
        let score = score_musical_turn_taking(&annotations, &predicted).unwrap();
        assert_eq!(score.post_yield_leakage_count, 1);
    }
}
