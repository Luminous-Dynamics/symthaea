// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![forbid(unsafe_code)]
//! Hardware-independent signal contracts for early Symthaea BCI research.
//!
//! This crate validates signal shape, quality, freshness, and stream continuity.
//! It does not decode intent, establish safety/clinical benefit, or authorize an
//! action. A separately reviewed action policy must remain between any decoder
//! output and an actuator. The API uses caller-supplied monotonic nanoseconds and
//! has no device I/O, wall-clock reads, network access, or third-party dependencies.

use std::fmt;

pub mod robustness;
pub use robustness::{
    evaluate_robustness, PerturbationFamily, RobustnessConfig, RobustnessError,
    RobustnessEvaluationReport, RobustnessModelReport, RobustnessScenario,
    RobustnessScenarioDelta, RobustnessScore, ROBUSTNESS_SCORE_SCALE_PPM,
};

pub mod selective;
pub use selective::{evaluate_risk_coverage, evaluate_selective, SelectiveMetricError, SelectiveMetrics, SelectiveObservation};

pub mod calibration;
pub use calibration::{
    evaluate_calibration_reduction, CalibrationBudgetScore, CalibrationEvaluationError,
    CalibrationReductionReport, NonInferiorityDecision,
};

pub mod statistics;
pub use statistics::{
    analyze_leaderboard, holm_adjust_p_values, paired_cell_mean_participant_bootstrap,
    paired_cell_mean_unit_bootstrap, paired_participant_bootstrap, BootstrapConfig,
    CellMeanPairedBootstrapReport, LeaderboardInferenceConfig, LeaderboardInferenceError,
    LeaderboardInferenceReport, LeaderboardUnitScore, ModelRankStability, PairwiseModelInference,
    PairedBootstrapError, PairedBootstrapReport, PairedCellBootstrapError,
    PairedLabeledPrediction, PairedParticipantScore, SCORE_SCALE_PPM,
};

pub mod benchmark;
pub use benchmark::{
    audit_split_assignments, evaluate_cell_mean_balanced_accuracy, evaluate_track02_reference_compatible_cell_mean, AssignedWindow,
    BenchmarkArtifactError, BenchmarkMetric, BenchmarkRunManifest, BenchmarkScorecard,
    CellBalancedAccuracy, CellBalancedAccuracyError, CellMeanBalancedAccuracyReport,
    ClassRecall, EvaluationCell, EvaluationSplit, LabeledPrediction, MetricDirection,
    SplitAuditConfig, SplitAuditError, SplitAuditReport,
};

pub const SIGNAL_SCHEMA_V1: u16 = 1;
const NS_PER_SECOND: u128 = 1_000_000_000;
const MAX_CHANNEL_ID_BYTES: usize = 64;

/// A sample-major interleaved signal frame. Element index is
/// `sample_index * channel_ids.len() + channel_index`.
///
/// The preprocessing fingerprint must identify a versioned manifest containing
/// filter/reference/resampling details. A channel-layout or preprocessing change
/// requires an explicit stream reset, never silent remapping.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalFrame {
    pub schema_version: u16,
    pub sequence: u64,
    pub clock_epoch: u64,
    pub first_sample_index: u64,
    /// Monotonic capture time of the first sample in this frame, not arrival time.
    pub captured_at_ns: u64,
    pub sampling_rate_hz: u32,
    pub channel_ids: Vec<String>,
    pub sample_count: u32,
    pub samples: Vec<f32>,
    pub preprocessing_fingerprint: [u8; 32],
    /// None is rejected; absent quality must not be interpreted as a clean signal.
    pub quality: Option<SignalQuality>,
}

/// Producer-computed quality summary. Score scale: 0 (unusable) to 1000 (best).
/// The computation and calibration must be declared by the experiment manifest;
/// this crate does not claim one universal quality score across recording devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignalQuality {
    pub quality_score_permille: u16,
    pub clipped_sample_count: u32,
    /// Indices into `SignalFrame::channel_ids`.
    pub flatline_channels: Vec<u16>,
}

/// Explicit policy. There is deliberately no universal Default threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatorConfig {
    pub max_frame_age_ns: u64,
    pub max_future_skew_ns: u64,
    /// Permitted deviation from sample-count/rate-derived inter-frame capture time.
    pub max_interframe_timing_error_ns: u64,
    pub minimum_quality_score_permille: u16,
    pub max_channels: usize,
    pub max_sample_values_per_frame: usize,
    pub maximum_clipped_samples: u32,
    pub reject_flatline_channels: bool,
}

impl ValidatorConfig {
    pub fn validate(&self) -> Result<(), FrameError> {
        if self.minimum_quality_score_permille > 1000 {
            return Err(FrameError::InvalidConfiguration("quality threshold must be <= 1000"));
        }
        if self.max_channels == 0 || self.max_sample_values_per_frame == 0 {
            return Err(FrameError::InvalidConfiguration("channel and sample limits must be non-zero"));
        }
        Ok(())
    }
}

/// Typed rejection reasons support telemetry without logging raw neural samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    InvalidConfiguration(&'static str),
    UnsupportedSchema(u16),
    ZeroSamplingRate,
    EmptyChannelList,
    TooManyChannels { actual: usize, maximum: usize },
    InvalidChannelId,
    DuplicateChannelId(String),
    ZeroSampleCount,
    SampleBufferLengthOverflow,
    SampleBufferLengthMismatch { expected: usize, actual: usize },
    FrameTooLarge { actual: usize, maximum: usize },
    NonFiniteSample { index: usize },
    MissingQualityMetadata,
    InvalidQualityScore(u16),
    ClippedSampleCountExceedsBuffer,
    TooManyClippedSamples { actual: u32, maximum: u32 },
    InvalidFlatlineChannelIndex(u16),
    DuplicateFlatlineChannelIndex(u16),
    FlatlineDetected,
    QualityBelowThreshold { actual: u16, minimum: u16 },
    FutureFrame,
    StaleFrame { age_ns: u64, maximum_ns: u64 },
    SequenceDiscontinuity { expected: u64, actual: u64 },
    SampleDiscontinuity { expected: u64, actual: u64 },
    ClockEpochChanged { expected: u64, actual: u64 },
    ClockEpochMustAdvanceAfterReset { previous: u64, actual: u64 },
    TimestampRegression { previous_ns: u64, actual_ns: u64 },
    FrameTimingDiscontinuity { expected_ns: u64, actual_ns: u64, tolerance_ns: u64 },
    SamplingRateChanged { expected: u32, actual: u32 },
    ChannelLayoutChanged,
    PreprocessingChanged,
    CounterOverflow,
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FrameError {}

/// Unforgeable outside this crate: only FrameValidator can create a receipt.
/// It binds one accepted frame's sequence, capture time, quality, and stream epoch.
#[derive(Debug, PartialEq, Eq)]
pub struct ValidatedFrameReceipt {
    sequence: u64,
    clock_epoch: u64,
    captured_at_ns: u64,
    quality_score_permille: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StreamCursor {
    sequence: u64,
    first_sample_index: u64,
    sample_count: u32,
    captured_at_ns: u64,
    sampling_rate_hz: u32,
    clock_epoch: u64,
    channel_ids: Vec<String>,
    preprocessing_fingerprint: [u8; 32],
}

/// Stateful fail-closed validator. Rejected frames never advance the stream cursor.
#[derive(Debug, Clone)]
pub struct FrameValidator {
    config: ValidatorConfig,
    cursor: Option<StreamCursor>,
    /// A reset after accepted data requires the next frame to explicitly advance epoch.
    minimum_next_clock_epoch: Option<u64>,
}

impl FrameValidator {
    /// Start a new stream when no prior accepted epoch needs to be enforced.
    /// After process/device restart, prefer new_with_epoch_floor using the last
    /// durably recorded epoch; this constructor alone cannot detect persisted replay.
    pub fn new(config: ValidatorConfig) -> Result<Self, FrameError> {
        config.validate()?;
        Ok(Self { config, cursor: None, minimum_next_clock_epoch: None })
    }

    /// Start after a known prior epoch. The first accepted frame must have a
    /// strictly greater epoch, allowing a caller to persist anti-replay state
    /// across process restarts without this crate owning a storage subsystem.
    pub fn new_with_epoch_floor(
        config: ValidatorConfig,
        last_accepted_clock_epoch: u64,
    ) -> Result<Self, FrameError> {
        config.validate()?;
        Ok(Self {
            config,
            cursor: None,
            minimum_next_clock_epoch: Some(last_accepted_clock_epoch),
        })
    }

    /// Validate using caller-supplied monotonic `now_ns`; don't pass wall-clock time.
    pub fn validate(&mut self, frame: &SignalFrame, now_ns: u64) -> Result<(), FrameError> {
        self.validate_for_control(frame, now_ns).map(|_| ())
    }

    /// Validate and issue an opaque receipt for an independently gated sandbox intent.
    /// A receipt is created only after every frame and continuity invariant passes.
    pub fn validate_for_control(
        &mut self,
        frame: &SignalFrame,
        now_ns: u64,
    ) -> Result<ValidatedFrameReceipt, FrameError> {
        validate_intrinsic(frame, self.config)?;
        if frame.captured_at_ns > now_ns.saturating_add(self.config.max_future_skew_ns) {
            return Err(FrameError::FutureFrame);
        }
        let age_ns = now_ns.saturating_sub(frame.captured_at_ns);
        if age_ns > self.config.max_frame_age_ns {
            return Err(FrameError::StaleFrame { age_ns, maximum_ns: self.config.max_frame_age_ns });
        }

        if self.cursor.is_none() {
            if let Some(previous_epoch) = self.minimum_next_clock_epoch {
                if frame.clock_epoch <= previous_epoch {
                    return Err(FrameError::ClockEpochMustAdvanceAfterReset {
                        previous: previous_epoch,
                        actual: frame.clock_epoch,
                    });
                }
            }
        }

        if let Some(prev) = &self.cursor {
            let expected_sequence = prev.sequence.checked_add(1).ok_or(FrameError::CounterOverflow)?;
            if frame.sequence != expected_sequence {
                return Err(FrameError::SequenceDiscontinuity { expected: expected_sequence, actual: frame.sequence });
            }
            let expected_sample = prev.first_sample_index.checked_add(u64::from(prev.sample_count))
                .ok_or(FrameError::CounterOverflow)?;
            if frame.first_sample_index != expected_sample {
                return Err(FrameError::SampleDiscontinuity { expected: expected_sample, actual: frame.first_sample_index });
            }
            if frame.clock_epoch != prev.clock_epoch {
                return Err(FrameError::ClockEpochChanged { expected: prev.clock_epoch, actual: frame.clock_epoch });
            }
            if frame.captured_at_ns <= prev.captured_at_ns {
                return Err(FrameError::TimestampRegression { previous_ns: prev.captured_at_ns, actual_ns: frame.captured_at_ns });
            }
            if frame.sampling_rate_hz != prev.sampling_rate_hz {
                return Err(FrameError::SamplingRateChanged { expected: prev.sampling_rate_hz, actual: frame.sampling_rate_hz });
            }
            let duration_ns = u64::try_from(
                (u128::from(prev.sample_count) * NS_PER_SECOND) / u128::from(prev.sampling_rate_hz),
            ).map_err(|_| FrameError::CounterOverflow)?;
            let expected_capture_ns = prev.captured_at_ns.checked_add(duration_ns).ok_or(FrameError::CounterOverflow)?;
            if frame.captured_at_ns.abs_diff(expected_capture_ns) > self.config.max_interframe_timing_error_ns {
                return Err(FrameError::FrameTimingDiscontinuity {
                    expected_ns: expected_capture_ns,
                    actual_ns: frame.captured_at_ns,
                    tolerance_ns: self.config.max_interframe_timing_error_ns,
                });
            }
            if frame.channel_ids != prev.channel_ids {
                return Err(FrameError::ChannelLayoutChanged);
            }
            if frame.preprocessing_fingerprint != prev.preprocessing_fingerprint {
                return Err(FrameError::PreprocessingChanged);
            }
        }

        // Re-check rather than panic if validation invariants change in a future schema.
        let quality_score_permille = frame.quality.as_ref()
            .ok_or(FrameError::MissingQualityMetadata)?
            .quality_score_permille;

        // Commit accepted stream state only after every invariant succeeds.
        self.cursor = Some(StreamCursor {
            sequence: frame.sequence,
            first_sample_index: frame.first_sample_index,
            sample_count: frame.sample_count,
            captured_at_ns: frame.captured_at_ns,
            sampling_rate_hz: frame.sampling_rate_hz,
            clock_epoch: frame.clock_epoch,
            channel_ids: frame.channel_ids.clone(),
            preprocessing_fingerprint: frame.preprocessing_fingerprint,
        });
        self.minimum_next_clock_epoch = None;
        Ok(ValidatedFrameReceipt {
            sequence: frame.sequence,
            clock_epoch: frame.clock_epoch,
            captured_at_ns: frame.captured_at_ns,
            quality_score_permille,
        })
    }

    /// Explicit reset after recording a device, clock, montage, or preprocessing change.
    /// If a stream was accepted, the next valid frame must use a strictly greater clock epoch.
    pub fn reset_stream(&mut self) {
        if let Some(cursor) = &self.cursor {
            self.minimum_next_clock_epoch = Some(cursor.clock_epoch);
        }
        self.cursor = None;
    }
}


/// Abstract actions supported by the first deterministic sandbox. These are
/// never sent directly to an OS pointer, assistive device, or actuator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposedAction {
    /// Fixed-point displacement; 1000 units represent one normalized cursor unit.
    MoveCursor { dx_milli: i16, dy_milli: i16 },
    /// Select an already-present sandbox target by stable target index.
    Select { target_id: u16 },
}

/// Decoder output paired with a validator-issued receipt. The receipt is opaque
/// and single-use because a proposal consumes it.
#[derive(Debug, PartialEq, Eq)]
pub struct IntentProposal {
    evidence: ValidatedFrameReceipt,
    pub confidence_permille: u16,
    pub runner_up_permille: u16,
    pub action: ProposedAction,
}

impl IntentProposal {
    pub fn new(
        evidence: ValidatedFrameReceipt,
        confidence_permille: u16,
        runner_up_permille: u16,
        action: ProposedAction,
    ) -> Self {
        Self { evidence, confidence_permille, runner_up_permille, action }
    }
}

/// Bounded output for a sandbox only. It is not a real-device authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxAction {
    MoveCursor { dx_milli: i16, dy_milli: i16 },
    Select { target_id: u16 },
}

/// All action limits are explicit; there is no permissive default configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionGateConfig {
    pub max_evidence_age_ns: u64,
    pub minimum_quality_score_permille: u16,
    pub minimum_confidence_permille: u16,
    pub minimum_margin_permille: u16,
    pub maximum_cursor_delta_milli: u16,
    pub allow_cursor_movement: bool,
    pub allow_selection: bool,
    pub require_confirmation_for_selection: bool,
}

impl ActionGateConfig {
    pub fn validate(&self) -> Result<(), ActionGateError> {
        if self.minimum_quality_score_permille > 1000
            || self.minimum_confidence_permille > 1000
            || self.minimum_margin_permille > 1000
        {
            return Err(ActionGateError::InvalidConfiguration("scores must be <= 1000"));
        }
        if self.maximum_cursor_delta_milli > 1000 {
            return Err(ActionGateError::InvalidConfiguration("cursor delta must be <= 1000 milli-units"));
        }
        if !self.allow_cursor_movement && !self.allow_selection {
            return Err(ActionGateError::InvalidConfiguration("at least one sandbox action must be allowed"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionGateError {
    InvalidConfiguration(&'static str),
    FutureEvidence,
    StaleEvidence { age_ns: u64, maximum_ns: u64 },
    LowSignalQuality { actual: u16, minimum: u16 },
    InvalidConfidenceScore(u16),
    InvalidRunnerUpScore(u16),
    RunnerUpExceedsConfidence { confidence: u16, runner_up: u16 },
    LowConfidence { actual: u16, minimum: u16 },
    AmbiguousIntent { margin: u16, minimum: u16 },
    NonMonotonicEvidenceSequence { last: u64, actual: u64 },
    NonMonotonicClockEpoch { last: u64, actual: u64 },
    CursorMovementDisabled,
    CursorMovementOutOfBounds { dx_milli: i16, dy_milli: i16, maximum: u16 },
    SelectionDisabled,
    UserConfirmationRequired,
}

impl fmt::Display for ActionGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ActionGateError {}

/// Independent policy gate for sandbox control. Instantiate per stream/session.
/// It returns bounded intent values only; the caller must not treat them as a
/// production actuator authorization. Rejections do not advance the accepted sequence.
#[derive(Debug, Clone)]
pub struct ActionGate {
    config: ActionGateConfig,
    last_authorized_sequence: Option<u64>,
    stream_epoch: Option<u64>,
}

impl ActionGate {
    pub fn new(config: ActionGateConfig) -> Result<Self, ActionGateError> {
        config.validate()?;
        Ok(Self { config, last_authorized_sequence: None, stream_epoch: None })
    }

    pub fn authorize(
        &mut self,
        proposal: IntentProposal,
        now_ns: u64,
        user_confirmed: bool,
    ) -> Result<SandboxAction, ActionGateError> {
        let evidence = proposal.evidence;
        if evidence.captured_at_ns > now_ns {
            return Err(ActionGateError::FutureEvidence);
        }
        let age_ns = now_ns - evidence.captured_at_ns;
        if age_ns > self.config.max_evidence_age_ns {
            return Err(ActionGateError::StaleEvidence {
                age_ns,
                maximum_ns: self.config.max_evidence_age_ns,
            });
        }
        if evidence.quality_score_permille < self.config.minimum_quality_score_permille {
            return Err(ActionGateError::LowSignalQuality {
                actual: evidence.quality_score_permille,
                minimum: self.config.minimum_quality_score_permille,
            });
        }
        if proposal.confidence_permille > 1000 {
            return Err(ActionGateError::InvalidConfidenceScore(proposal.confidence_permille));
        }
        if proposal.runner_up_permille > 1000 {
            return Err(ActionGateError::InvalidRunnerUpScore(proposal.runner_up_permille));
        }
        if proposal.runner_up_permille > proposal.confidence_permille {
            return Err(ActionGateError::RunnerUpExceedsConfidence {
                confidence: proposal.confidence_permille,
                runner_up: proposal.runner_up_permille,
            });
        }
        if proposal.confidence_permille < self.config.minimum_confidence_permille {
            return Err(ActionGateError::LowConfidence {
                actual: proposal.confidence_permille,
                minimum: self.config.minimum_confidence_permille,
            });
        }
        let margin = proposal.confidence_permille - proposal.runner_up_permille;
        if margin < self.config.minimum_margin_permille {
            return Err(ActionGateError::AmbiguousIntent {
                margin,
                minimum: self.config.minimum_margin_permille,
            });
        }

        if let Some(epoch) = self.stream_epoch {
            if evidence.clock_epoch < epoch {
                return Err(ActionGateError::NonMonotonicClockEpoch {
                    last: epoch,
                    actual: evidence.clock_epoch,
                });
            }
            if evidence.clock_epoch == epoch {
                if let Some(last) = self.last_authorized_sequence {
                    if evidence.sequence <= last {
                        return Err(ActionGateError::NonMonotonicEvidenceSequence {
                            last,
                            actual: evidence.sequence,
                        });
                    }
                }
            }
        }

        let action = match proposal.action {
            ProposedAction::MoveCursor { dx_milli, dy_milli } => {
                if !self.config.allow_cursor_movement {
                    return Err(ActionGateError::CursorMovementDisabled);
                }
                let max_delta = i32::from(self.config.maximum_cursor_delta_milli);
                if i32::from(dx_milli).abs() > max_delta || i32::from(dy_milli).abs() > max_delta {
                    return Err(ActionGateError::CursorMovementOutOfBounds {
                        dx_milli,
                        dy_milli,
                        maximum: self.config.maximum_cursor_delta_milli,
                    });
                }
                SandboxAction::MoveCursor { dx_milli, dy_milli }
            }
            ProposedAction::Select { target_id } => {
                if !self.config.allow_selection {
                    return Err(ActionGateError::SelectionDisabled);
                }
                if self.config.require_confirmation_for_selection && !user_confirmed {
                    return Err(ActionGateError::UserConfirmationRequired);
                }
                SandboxAction::Select { target_id }
            }
        };

        self.stream_epoch = Some(evidence.clock_epoch);
        self.last_authorized_sequence = Some(evidence.sequence);
        Ok(action)
    }
}

fn validate_intrinsic(frame: &SignalFrame, config: ValidatorConfig) -> Result<(), FrameError> {
    if frame.schema_version != SIGNAL_SCHEMA_V1 {
        return Err(FrameError::UnsupportedSchema(frame.schema_version));
    }
    if frame.sampling_rate_hz == 0 { return Err(FrameError::ZeroSamplingRate); }
    if frame.channel_ids.is_empty() { return Err(FrameError::EmptyChannelList); }
    if frame.channel_ids.len() > config.max_channels {
        return Err(FrameError::TooManyChannels { actual: frame.channel_ids.len(), maximum: config.max_channels });
    }
    for (i, id) in frame.channel_ids.iter().enumerate() {
        if id.trim().is_empty() || id.len() > MAX_CHANNEL_ID_BYTES || id.chars().any(char::is_control) {
            return Err(FrameError::InvalidChannelId);
        }
        if frame.channel_ids[..i].iter().any(|prior| prior == id) {
            return Err(FrameError::DuplicateChannelId(id.clone()));
        }
    }
    if frame.sample_count == 0 { return Err(FrameError::ZeroSampleCount); }
    let expected = usize::try_from(frame.sample_count).ok()
        .and_then(|n| n.checked_mul(frame.channel_ids.len()))
        .ok_or(FrameError::SampleBufferLengthOverflow)?;
    if expected > config.max_sample_values_per_frame {
        return Err(FrameError::FrameTooLarge { actual: expected, maximum: config.max_sample_values_per_frame });
    }
    if frame.samples.len() != expected {
        return Err(FrameError::SampleBufferLengthMismatch { expected, actual: frame.samples.len() });
    }
    if let Some((index, _)) = frame.samples.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(FrameError::NonFiniteSample { index });
    }

    let quality = frame.quality.as_ref().ok_or(FrameError::MissingQualityMetadata)?;
    if quality.quality_score_permille > 1000 {
        return Err(FrameError::InvalidQualityScore(quality.quality_score_permille));
    }
    if u64::from(quality.clipped_sample_count) > frame.samples.len() as u64 {
        return Err(FrameError::ClippedSampleCountExceedsBuffer);
    }
    if quality.clipped_sample_count > config.maximum_clipped_samples {
        return Err(FrameError::TooManyClippedSamples { actual: quality.clipped_sample_count, maximum: config.maximum_clipped_samples });
    }
    for (i, &channel) in quality.flatline_channels.iter().enumerate() {
        if usize::from(channel) >= frame.channel_ids.len() {
            return Err(FrameError::InvalidFlatlineChannelIndex(channel));
        }
        if quality.flatline_channels[..i].contains(&channel) {
            return Err(FrameError::DuplicateFlatlineChannelIndex(channel));
        }
    }
    if config.reject_flatline_channels && !quality.flatline_channels.is_empty() {
        return Err(FrameError::FlatlineDetected);
    }
    if quality.quality_score_permille < config.minimum_quality_score_permille {
        return Err(FrameError::QualityBelowThreshold {
            actual: quality.quality_score_permille,
            minimum: config.minimum_quality_score_permille,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: u64 = 2_000_000_000;

    fn config() -> ValidatorConfig {
        ValidatorConfig {
            max_frame_age_ns: 500_000_000,
            max_future_skew_ns: 2_000_000,
            max_interframe_timing_error_ns: 5_000_000,
            minimum_quality_score_permille: 600,
            max_channels: 32,
            max_sample_values_per_frame: 4096,
            maximum_clipped_samples: 0,
            reject_flatline_channels: true,
        }
    }
    fn gate_config() -> ActionGateConfig {
        ActionGateConfig {
            max_evidence_age_ns: 100_000_000,
            minimum_quality_score_permille: 600,
            minimum_confidence_permille: 800,
            minimum_margin_permille: 200,
            maximum_cursor_delta_milli: 100,
            allow_cursor_movement: true,
            allow_selection: true,
            require_confirmation_for_selection: true,
        }
    }
    fn receipt(sequence: u64, captured_at_ns: u64) -> ValidatedFrameReceipt {
        let mut validator = FrameValidator::new(config()).unwrap();
        validator.validate_for_control(&frame(sequence, sequence * 4, captured_at_ns), NOW).unwrap()
    }
    fn proposal(sequence: u64, action: ProposedAction) -> IntentProposal {
        IntentProposal::new(receipt(sequence, NOW - 10_000_000), 950, 100, action)
    }
    fn frame(sequence: u64, first_sample_index: u64, captured_at_ns: u64) -> SignalFrame {
        SignalFrame {
            schema_version: SIGNAL_SCHEMA_V1, sequence, clock_epoch: 1,
            first_sample_index, captured_at_ns, sampling_rate_hz: 250,
            channel_ids: vec!["C3".into(), "C4".into()], sample_count: 4,
            samples: vec![0.1; 8], preprocessing_fingerprint: [7; 32],
            quality: Some(SignalQuality { quality_score_permille: 900, clipped_sample_count: 0, flatline_channels: vec![] }),
        }
    }

    #[test]
    fn action_gate_authorizes_bounded_cursor_movement() {
        let mut gate = ActionGate::new(gate_config()).unwrap();
        assert_eq!(
            gate.authorize(proposal(1, ProposedAction::MoveCursor { dx_milli: 60, dy_milli: -40 }), NOW, false),
            Ok(SandboxAction::MoveCursor { dx_milli: 60, dy_milli: -40 })
        );
    }
    #[test]
    fn action_gate_rejects_low_confidence_and_ambiguous_intents() {
        let mut gate = ActionGate::new(gate_config()).unwrap();
        let low = IntentProposal::new(receipt(1, NOW - 10_000_000), 700, 100, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 1 });
        assert_eq!(gate.authorize(low, NOW, false), Err(ActionGateError::LowConfidence { actual: 700, minimum: 800 }));
        let ambiguous = IntentProposal::new(receipt(2, NOW - 10_000_000), 900, 750, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 1 });
        assert_eq!(gate.authorize(ambiguous, NOW, false), Err(ActionGateError::AmbiguousIntent { margin: 150, minimum: 200 }));
    }
    #[test]
    fn action_gate_rejects_stale_evidence_and_oversized_movement() {
        let mut gate = ActionGate::new(gate_config()).unwrap();
        let stale = IntentProposal::new(receipt(1, NOW - 150_000_000), 950, 100, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 1 });
        assert!(matches!(gate.authorize(stale, NOW, false), Err(ActionGateError::StaleEvidence { .. })));
        let large = proposal(2, ProposedAction::MoveCursor { dx_milli: 101, dy_milli: 0 });
        assert_eq!(gate.authorize(large, NOW, false), Err(ActionGateError::CursorMovementOutOfBounds { dx_milli: 101, dy_milli: 0, maximum: 100 }));
    }
    #[test]
    fn action_gate_requires_external_confirmation_for_selection() {
        let mut gate = ActionGate::new(gate_config()).unwrap();
        assert_eq!(
            gate.authorize(proposal(1, ProposedAction::Select { target_id: 3 }), NOW, false),
            Err(ActionGateError::UserConfirmationRequired)
        );
        assert_eq!(
            gate.authorize(proposal(2, ProposedAction::Select { target_id: 3 }), NOW, true),
            Ok(SandboxAction::Select { target_id: 3 })
        );
    }
    #[test]
    fn action_gate_rejects_replayed_or_older_sequences() {
        let mut gate = ActionGate::new(gate_config()).unwrap();
        gate.authorize(proposal(10, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 0 }), NOW, false).unwrap();
        assert_eq!(
            gate.authorize(proposal(10, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 0 }), NOW, false),
            Err(ActionGateError::NonMonotonicEvidenceSequence { last: 10, actual: 10 })
        );
        assert_eq!(
            gate.authorize(proposal(9, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 0 }), NOW, false),
            Err(ActionGateError::NonMonotonicEvidenceSequence { last: 10, actual: 9 })
        );
    }
    #[test]
    fn action_gate_rejects_older_clock_epoch_even_with_higher_sequence() {
        let mut gate = ActionGate::new(gate_config()).unwrap();
        gate.authorize(proposal(10, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 0 }), NOW, false).unwrap();
        let mut validator = FrameValidator::new(config()).unwrap();
        let mut old_epoch_frame = frame(99, 396, NOW - 10_000_000);
        old_epoch_frame.clock_epoch = 0;
        let evidence = validator.validate_for_control(&old_epoch_frame, NOW).unwrap();
        let old = IntentProposal::new(evidence, 950, 100, ProposedAction::MoveCursor { dx_milli: 1, dy_milli: 0 });
        assert_eq!(
            gate.authorize(old, NOW, false),
            Err(ActionGateError::NonMonotonicClockEpoch { last: 1, actual: 0 })
        );
    }

    #[test]
    fn action_gate_rejects_invalid_policy() {
        let mut config = gate_config();
        config.minimum_confidence_permille = 1001;
        assert_eq!(ActionGate::new(config).unwrap_err(), ActionGateError::InvalidConfiguration("scores must be <= 1000"));
    }
    #[test]
    fn restart_constructor_rejects_same_or_older_clock_epoch() {
        let mut validator = FrameValidator::new_with_epoch_floor(config(), 8).unwrap();
        let mut replay = frame(0, 0, NOW - 10_000_000);
        replay.clock_epoch = 8;
        assert_eq!(
            validator.validate(&replay, NOW),
            Err(FrameError::ClockEpochMustAdvanceAfterReset { previous: 8, actual: 8 })
        );
        replay.clock_epoch = 7;
        assert_eq!(
            validator.validate(&replay, NOW),
            Err(FrameError::ClockEpochMustAdvanceAfterReset { previous: 8, actual: 7 })
        );
        replay.clock_epoch = 9;
        assert_eq!(validator.validate(&replay, NOW), Ok(()));
    }

    #[test]
    fn one_hundred_thousand_deterministic_adversarial_frames_fail_closed() {
        let mut validator = FrameValidator::new(config()).unwrap();
        validator.validate(&frame(10, 100, NOW - 20_000_000), NOW).unwrap();
        let mut state = 0xC0FF_EE12_3456_789A_u64;

        for _ in 0..100_000 {
            // Fixed-seed LCG: reproducible mutations and no random crate/runtime dependency.
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let mut candidate = frame(11, 104, NOW - 4_000_000);
            match (state >> 32) % 10 {
                0 => candidate.schema_version = u16::MAX,
                1 => candidate.samples[0] = f32::NAN,
                2 => candidate.quality = None,
                3 => candidate.sample_count = 0,
                4 => candidate.sequence = 12,
                5 => candidate.captured_at_ns = NOW - 600_000_000,
                6 => candidate.channel_ids[1] = candidate.channel_ids[0].clone(),
                7 => candidate.preprocessing_fingerprint = [0; 32],
                8 => candidate.clock_epoch = 0,
                _ => candidate.quality.as_mut().unwrap().quality_score_permille = 1001,
            }
            assert!(
                validator.validate(&candidate, NOW).is_err(),
                "adversarial mutation unexpectedly accepted at state {state}"
            );
        }

        // None of the rejected candidates may have advanced the stream cursor.
        assert_eq!(validator.validate(&frame(11, 104, NOW - 4_000_000), NOW), Ok(()));
    }

    #[test]
    fn accepts_valid_frame_and_contiguous_successor() {
        let mut v = FrameValidator::new(config()).unwrap();
        assert_eq!(v.validate(&frame(10, 100, NOW - 20_000_000), NOW), Ok(()));
        assert_eq!(v.validate(&frame(11, 104, NOW - 4_000_000), NOW), Ok(()));
    }
    #[test]
    fn requires_quality_and_finite_samples() {
        let mut f = frame(0, 0, NOW); f.quality = None;
        assert_eq!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::MissingQualityMetadata));
        let mut f = frame(0, 0, NOW); f.samples[3] = f32::NAN;
        assert_eq!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::NonFiniteSample { index: 3 }));
    }
    #[test]
    fn rejects_malformed_shape_duplicate_channels_and_limits() {
        let mut f = frame(0, 0, NOW); f.samples.pop();
        assert!(matches!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::SampleBufferLengthMismatch { .. })));
        let mut f = frame(0, 0, NOW); f.sample_count = u32::MAX;
        assert!(matches!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::FrameTooLarge { .. })));
        let mut f = frame(0, 0, NOW); f.channel_ids[1] = "C3".into();
        assert_eq!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::DuplicateChannelId("C3".into())));
    }
    #[test]
    fn rejects_low_quality_clipping_and_flatlines() {
        let mut f = frame(0, 0, NOW); f.quality.as_mut().unwrap().quality_score_permille = 599;
        assert!(matches!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::QualityBelowThreshold { .. })));
        let mut f = frame(0, 0, NOW); f.quality.as_mut().unwrap().clipped_sample_count = 1;
        assert!(matches!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::TooManyClippedSamples { .. })));
        let mut f = frame(0, 0, NOW); f.quality.as_mut().unwrap().flatline_channels = vec![1];
        assert_eq!(FrameValidator::new(config()).unwrap().validate(&f, NOW), Err(FrameError::FlatlineDetected));
    }
    #[test]
    fn rejects_stale_and_future_frames() {
        let mut v = FrameValidator::new(config()).unwrap();
        assert!(matches!(v.validate(&frame(0, 0, NOW - 500_000_001), NOW), Err(FrameError::StaleFrame { .. })));
        assert_eq!(v.validate(&frame(0, 0, NOW + 2_000_001), NOW), Err(FrameError::FutureFrame));
    }
    #[test]
    fn rejects_interframe_timestamp_gap_even_if_sequence_is_contiguous() {
        let mut v = FrameValidator::new(config()).unwrap();
        v.validate(&frame(10, 100, NOW - 20_000_000), NOW).unwrap();
        let next = frame(11, 104, NOW - 10_000_000); // 10ms elapsed, but four samples at 250Hz imply 16ms.
        assert!(matches!(v.validate(&next, NOW), Err(FrameError::FrameTimingDiscontinuity { .. })));
        assert_eq!(v.validate(&frame(11, 104, NOW - 4_000_000), NOW), Ok(()));
    }
    #[test]
    fn rejected_frame_does_not_advance_cursor() {
        let mut v = FrameValidator::new(config()).unwrap();
        v.validate(&frame(10, 100, NOW - 20_000_000), NOW).unwrap();
        assert_eq!(v.validate(&frame(12, 104, NOW - 4_000_000), NOW), Err(FrameError::SequenceDiscontinuity { expected: 11, actual: 12 }));
        assert_eq!(v.validate(&frame(11, 104, NOW - 4_000_000), NOW), Ok(()));
    }
    #[test]
    fn rejects_stream_identity_and_continuity_changes() {
        let mut v = FrameValidator::new(config()).unwrap();
        v.validate(&frame(10, 100, NOW - 20_000_000), NOW).unwrap();
        let mut f = frame(11, 104, NOW - 10_000_000); f.clock_epoch = 2;
        assert!(matches!(v.validate(&f, NOW), Err(FrameError::ClockEpochChanged { .. })));
        let mut f = frame(11, 103, NOW - 10_000_000);
        assert!(matches!(v.validate(&f, NOW), Err(FrameError::SampleDiscontinuity { .. })));
        let mut f = frame(11, 104, NOW - 30_000_000);
        assert!(matches!(v.validate(&f, NOW), Err(FrameError::TimestampRegression { .. })));
        let mut f = frame(11, 104, NOW - 10_000_000); f.sampling_rate_hz = 200;
        assert!(matches!(v.validate(&f, NOW), Err(FrameError::SamplingRateChanged { .. })));
        let mut f = frame(11, 104, NOW - 4_000_000); f.channel_ids.swap(0, 1);
        assert_eq!(v.validate(&f, NOW), Err(FrameError::ChannelLayoutChanged));
        let mut f = frame(11, 104, NOW - 4_000_000); f.preprocessing_fingerprint = [8; 32];
        assert_eq!(v.validate(&f, NOW), Err(FrameError::PreprocessingChanged));
    }
    #[test]
    fn explicit_reset_requires_epoch_advance_and_allows_declared_reconfiguration() {
        let mut v = FrameValidator::new(config()).unwrap();
        v.validate(&frame(10, 100, NOW - 20_000_000), NOW).unwrap();
        v.reset_stream();
        let mut same_epoch = frame(0, 0, NOW - 10_000_000);
        same_epoch.channel_ids = vec!["Cz".into()];
        same_epoch.samples = vec![0.2; 4];
        assert_eq!(
            v.validate(&same_epoch, NOW),
            Err(FrameError::ClockEpochMustAdvanceAfterReset { previous: 1, actual: 1 })
        );
        let mut next_epoch = same_epoch;
        next_epoch.clock_epoch = 2;
        assert_eq!(v.validate(&next_epoch, NOW), Ok(()));
    }
    #[test]
    fn rejects_invalid_quality_policy() {
        let mut c = config(); c.minimum_quality_score_permille = 1001;
        assert_eq!(FrameValidator::new(c).unwrap_err(), FrameError::InvalidConfiguration("quality threshold must be <= 1000"));
    }
}
