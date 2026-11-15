#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline sample-accurate continuity and preservation contract.
//!
//! This harness proves only deterministic properties of supplied PCM samples:
//! artifact identity, immutable format/timebase, exact off-support preservation,
//! sample-frame ancestry, and declared boundary measurements. It does not
//! establish musical naturalness or replace EVAL-040 terminal authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const AUDIO_SEQUENCE_SCHEMA_V1: &str = "mel-music-audio-sequence-v1";
pub const AUDIO_STEP_SCHEMA_V1: &str = "mel-music-audio-step-v1";
pub const AUDIO_ARTIFACT_SCHEMA_V1: &str = "mel-pcm-audio-artifact-v1";
pub const AUDIO_MAX_STEPS_V1: u8 = 8;
const AUDIO_DOMAIN_V1: &[u8] = b"luminous.mel.pcm-audio-artifact.v1";
const AUDIO_REPORT_DOMAIN_V1: &[u8] = b"luminous.mel.audio-continuity-report.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PcmEncodingV1 {
    SignedInt16LittleEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PcmFormatV1 {
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub encoding: PcmEncodingV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PcmAudioArtifactV1 {
    pub format: PcmFormatV1,
    /// Interleaved signed 16-bit samples, one value per channel per frame.
    pub samples: Vec<i16>,
}

impl PcmAudioArtifactV1 {
    pub fn frame_count(&self) -> Result<u64, AudioContinuityErrorV1> {
        validate_format(self.format)?;
        let channels = usize::from(self.format.channels);
        if self.samples.is_empty() || !self.samples.len().is_multiple_of(channels) {
            return Err(AudioContinuityErrorV1::InvalidSampleLayout);
        }
        u64::try_from(self.samples.len() / channels)
            .map_err(|_| AudioContinuityErrorV1::InvalidSampleLayout)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameWindowV1 {
    /// Inclusive start frame, zero-based.
    pub start_frame: u64,
    /// Exclusive end frame, zero-based.
    pub end_frame: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioBoundaryKindV1 {
    HardCut,
    EqualPowerCrossfade,
    TransientPreservingSplice,
    OperationSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryPolicyV1 {
    pub kind: AudioBoundaryKindV1,
    /// Number of frames measured on each side of each authorized support edge.
    pub measurement_radius_frames: u16,
    /// None means report-only; both thresholds are required for a Verified gate.
    pub maximum_jump_delta_samples: Option<u32>,
    pub maximum_side_rms_delta_samples: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioGateVerdictV1 {
    Verified,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioUpstreamGatesV1 {
    pub rights_admission: AudioGateVerdictV1,
    pub source_lineage: AudioGateVerdictV1,
    pub realization_receipt: AudioGateVerdictV1,
    pub transaction_admission: AudioGateVerdictV1,
}

impl AudioUpstreamGatesV1 {
    fn all_verified(self) -> bool {
        self.rights_admission == AudioGateVerdictV1::Verified
            && self.source_lineage == AudioGateVerdictV1::Verified
            && self.realization_receipt == AudioGateVerdictV1::Verified
            && self.transaction_admission == AudioGateVerdictV1::Verified
    }

    fn any_rejected(self) -> bool {
        [
            self.rights_admission,
            self.source_lineage,
            self.realization_receipt,
            self.transaction_admission,
        ]
        .contains(&AudioGateVerdictV1::Rejected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSequenceRequestV1 {
    pub schema_version: String,
    pub session_id: String,
    pub source_lineage_sha256: String,
    pub source_audio_sha256: String,
    pub operation_plan_sha256: String,
    pub expected_step_count: u8,
    pub format: PcmFormatV1,
    pub frame_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSequenceStepV1 {
    pub schema_version: String,
    pub sequence: u8,
    pub step_id: String,
    pub operation_id: String,
    pub instruction_sha256: String,
    pub source_lineage_sha256: String,
    pub parent_audio_sha256: String,
    pub output_audio_sha256: String,
    /// The semantic edit core. Empty is invalid.
    pub edit_regions: Vec<FrameWindowV1>,
    /// Explicitly authorized transition/crossfade regions. These are not
    /// silently added by the verifier; they must be frozen before execution.
    #[serde(default)]
    pub transition_support: Vec<FrameWindowV1>,
    pub boundary_policy: AudioBoundaryPolicyV1,
    pub upstream_gates: AudioUpstreamGatesV1,
    pub candidate_attempt_count: u16,
    pub failed_candidate_attempt_count: u16,
    pub render_call_count: u16,
    pub retry_count: u16,
    pub manual_intervention_count: u16,
    pub output: PcmAudioArtifactV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryMeasurementV1 {
    pub frame: u64,
    pub channel: u8,
    pub parent_jump_samples: u32,
    pub output_jump_samples: u32,
    pub jump_delta_samples: u32,
    pub parent_side_rms_delta_samples: u32,
    pub output_side_rms_delta_samples: u32,
    pub side_rms_delta_drift_samples: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioStepEvaluationV1 {
    pub sequence: u8,
    pub step_id: String,
    pub operation_id: String,
    pub instruction_sha256: String,
    pub source_lineage_sha256: String,
    pub edit_regions: Vec<FrameWindowV1>,
    pub transition_support: Vec<FrameWindowV1>,
    pub boundary_policy: AudioBoundaryPolicyV1,
    pub upstream_gates: AudioUpstreamGatesV1,
    pub parent_audio_sha256: String,
    pub output_audio_sha256: String,
    pub authorized_frame_count: u64,
    pub changed_frame_count: u64,
    pub changed_sample_value_count: u64,
    pub off_support_changed_sample_value_count: u64,
    pub cumulative_authorized_frame_count: u64,
    pub source_changed_frame_count: u64,
    pub source_changed_sample_value_count: u64,
    pub source_max_absolute_delta_samples: u32,
    pub source_off_support_changed_sample_value_count: u64,
    pub boundary_measurements: Vec<AudioBoundaryMeasurementV1>,
    pub boundary_gate: AudioGateVerdictV1,
    pub upstream_gates_verified: bool,
    pub hard_gate_verified: bool,
    pub cumulative_attempt_count: u32,
    pub cumulative_failed_attempt_count: u32,
    pub cumulative_render_call_count: u32,
    pub cumulative_retry_count: u32,
    pub cumulative_manual_intervention_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioSequenceDispositionV1 {
    Accepted,
    RejectedAtStep,
    AbstainedAtStep,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSequenceEvaluationV1 {
    pub schema_version: String,
    pub request: AudioSequenceRequestV1,
    pub session_id: String,
    pub disposition: AudioSequenceDispositionV1,
    pub expected_step_count: u8,
    pub observed_step_count: u8,
    pub accepted_step_count: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_at_step: Option<u8>,
    pub source_audio_sha256: String,
    pub final_audio_sha256: String,
    pub cumulative_changed_frame_count: u64,
    pub cumulative_changed_sample_value_count: u64,
    pub final_source_changed_frame_count: u64,
    pub final_source_changed_sample_value_count: u64,
    pub final_source_max_absolute_delta_samples: u32,
    pub cumulative_authorized_frame_count: u64,
    pub cumulative_attempt_count: u32,
    pub cumulative_failed_attempt_count: u32,
    pub cumulative_render_call_count: u32,
    pub cumulative_retry_count: u32,
    pub cumulative_manual_intervention_count: u32,
    pub steps: Vec<AudioStepEvaluationV1>,
    pub report_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioContinuityErrorV1 {
    InvalidRequest,
    InvalidDigest,
    OperationPlanMismatch,
    InvalidFormat,
    InvalidSampleLayout,
    FrameCountMismatch,
    SourceDigestMismatch,
    ParentDigestMismatch,
    OutputDigestMismatch,
    SourceLineageMismatch,
    StepSequenceMismatch,
    DuplicateIdentity,
    InvalidWindow,
    NonCanonicalWindows,
    EmptyEditSupport,
    InvalidAttemptAccounting,
    NoAudioMutation,
    OffSupportMutation,
    ContinuationAfterBlockedStep,
    ReportReplayMismatch,
    SerializationFailure,
}

impl fmt::Display for AudioContinuityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRequest => "audio sequence request schema, identity or length is invalid",
            Self::InvalidDigest => "a required SHA-256 digest is malformed",
            Self::OperationPlanMismatch => "frozen operation-plan digest does not match the submitted steps",
            Self::InvalidFormat => "PCM format is invalid or changed during the session",
            Self::InvalidSampleLayout => "interleaved PCM sample layout is invalid",
            Self::FrameCountMismatch => "audio frame count changed during the session",
            Self::SourceDigestMismatch => "frozen source audio digest does not match actual PCM",
            Self::ParentDigestMismatch => "step did not consume the exact parent audio artifact",
            Self::OutputDigestMismatch => "declared output digest does not match actual PCM",
            Self::SourceLineageMismatch => "step source lineage differs from frozen session lineage",
            Self::StepSequenceMismatch => "audio steps are not contiguous and one-based",
            Self::DuplicateIdentity => "step or operation identity is empty or duplicated",
            Self::InvalidWindow => "a sample-frame window is empty or out of range",
            Self::NonCanonicalWindows => "windows must be sorted and non-overlapping",
            Self::EmptyEditSupport => "each edit step needs at least one semantic edit region",
            Self::InvalidAttemptAccounting => "candidate attempt counters are inconsistent",
            Self::NoAudioMutation => "an edit step did not change any PCM sample",
            Self::OffSupportMutation => "one or more PCM samples changed outside declared edit and transition support",
            Self::ContinuationAfterBlockedStep => "a later edit cannot launder a failed or unknown audio gate",
            Self::ReportReplayMismatch => "supplied audio continuity report differs from deterministic replay",
            Self::SerializationFailure => "canonical evidence serialization failed",
        })
    }
}
impl std::error::Error for AudioContinuityErrorV1 {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_format(format: PcmFormatV1) -> Result<(), AudioContinuityErrorV1> {
    if !(8_000..=192_000).contains(&format.sample_rate_hz)
        || !(1..=2).contains(&format.channels)
        || format.encoding != PcmEncodingV1::SignedInt16LittleEndian
    {
        return Err(AudioContinuityErrorV1::InvalidFormat);
    }
    Ok(())
}

pub fn pcm_audio_sha256_v1(
    artifact: &PcmAudioArtifactV1,
) -> Result<String, AudioContinuityErrorV1> {
    let frames = artifact.frame_count()?;
    let mut hasher = Sha256::new();
    hasher.update(AUDIO_DOMAIN_V1);
    hasher.update([0]);
    hasher.update(artifact.format.sample_rate_hz.to_le_bytes());
    hasher.update([artifact.format.channels]);
    hasher.update([16]); // signed PCM bit depth
    hasher.update(frames.to_le_bytes());
    for sample in &artifact.samples {
        hasher.update(sample.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_windows(
    windows: &[FrameWindowV1],
    frame_count: u64,
) -> Result<(), AudioContinuityErrorV1> {
    let mut previous_end = None;
    for window in windows {
        if window.start_frame >= window.end_frame || window.end_frame > frame_count {
            return Err(AudioContinuityErrorV1::InvalidWindow);
        }
        if previous_end.is_some_and(|end| end > window.start_frame) {
            return Err(AudioContinuityErrorV1::NonCanonicalWindows);
        }
        previous_end = Some(window.end_frame);
    }
    Ok(())
}

fn merge_support(
    edits: &[FrameWindowV1],
    transitions: &[FrameWindowV1],
) -> Vec<FrameWindowV1> {
    let mut all = Vec::with_capacity(edits.len() + transitions.len());
    all.extend_from_slice(edits);
    all.extend_from_slice(transitions);
    all.sort_by_key(|window| (window.start_frame, window.end_frame));
    let mut merged: Vec<FrameWindowV1> = Vec::with_capacity(all.len());
    for window in all {
        if let Some(last) = merged.last_mut()
            && window.start_frame <= last.end_frame
        {
            last.end_frame = last.end_frame.max(window.end_frame);
        } else {
            merged.push(window);
        }
    }
    merged
}

fn frame_is_authorized(frame: u64, support: &[FrameWindowV1]) -> bool {
    support
        .iter()
        .any(|window| window.start_frame <= frame && frame < window.end_frame)
}

fn isqrt(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut x = value;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + value / x) / 2;
    }
    x
}

fn side_rms(
    samples: &[i16],
    channels: usize,
    channel: usize,
    start: u64,
    end: u64,
) -> u32 {
    if start >= end {
        return 0;
    }
    let mut sum_squares = 0_u128;
    let mut count = 0_u128;
    for frame in start..end {
        let index = frame as usize * channels + channel;
        let value = i128::from(samples[index]);
        sum_squares += (value * value) as u128;
        count += 1;
    }
    isqrt(sum_squares / count).min(u128::from(u32::MAX)) as u32
}

fn abs_diff_u32(left: u32, right: u32) -> u32 {
    left.max(right) - left.min(right)
}

fn boundary_measurements(
    parent: &PcmAudioArtifactV1,
    output: &PcmAudioArtifactV1,
    support: &[FrameWindowV1],
    radius: u16,
) -> Vec<AudioBoundaryMeasurementV1> {
    let channels = usize::from(parent.format.channels);
    let frame_count = (parent.samples.len() / channels) as u64;
    let radius = u64::from(radius.max(1));
    let mut boundary_frames = BTreeSet::new();
    for window in support {
        if window.start_frame > 0 && window.start_frame < frame_count {
            boundary_frames.insert(window.start_frame);
        }
        if window.end_frame > 0 && window.end_frame < frame_count {
            boundary_frames.insert(window.end_frame);
        }
    }

    let mut measurements = Vec::new();
    for frame in boundary_frames {
        let left_start = frame.saturating_sub(radius);
        let left_end = frame;
        let right_start = frame;
        let right_end = (frame + radius).min(frame_count);
        for channel in 0..channels {
            let left_index = (frame as usize - 1) * channels + channel;
            let right_index = frame as usize * channels + channel;
            let parent_jump =
                (i32::from(parent.samples[right_index]) - i32::from(parent.samples[left_index]))
                    .unsigned_abs();
            let output_jump =
                (i32::from(output.samples[right_index]) - i32::from(output.samples[left_index]))
                    .unsigned_abs();
            let parent_left_rms =
                side_rms(&parent.samples, channels, channel, left_start, left_end);
            let parent_right_rms =
                side_rms(&parent.samples, channels, channel, right_start, right_end);
            let output_left_rms =
                side_rms(&output.samples, channels, channel, left_start, left_end);
            let output_right_rms =
                side_rms(&output.samples, channels, channel, right_start, right_end);
            let parent_side_delta = abs_diff_u32(parent_left_rms, parent_right_rms);
            let output_side_delta = abs_diff_u32(output_left_rms, output_right_rms);
            measurements.push(AudioBoundaryMeasurementV1 {
                frame,
                channel: channel as u8,
                parent_jump_samples: parent_jump,
                output_jump_samples: output_jump,
                jump_delta_samples: abs_diff_u32(parent_jump, output_jump),
                parent_side_rms_delta_samples: parent_side_delta,
                output_side_rms_delta_samples: output_side_delta,
                side_rms_delta_drift_samples: abs_diff_u32(parent_side_delta, output_side_delta),
            });
        }
    }
    measurements
}

fn evaluate_boundary_gate(
    measurements: &[AudioBoundaryMeasurementV1],
    policy: AudioBoundaryPolicyV1,
) -> AudioGateVerdictV1 {
    if measurements.is_empty() {
        return AudioGateVerdictV1::Verified;
    }
    let (Some(max_jump), Some(max_rms)) = (
        policy.maximum_jump_delta_samples,
        policy.maximum_side_rms_delta_samples,
    ) else {
        return AudioGateVerdictV1::Unknown;
    };
    if measurements.iter().any(|measurement| {
        measurement.jump_delta_samples > max_jump
            || measurement.side_rms_delta_drift_samples > max_rms
    }) {
        AudioGateVerdictV1::Rejected
    } else {
        AudioGateVerdictV1::Verified
    }
}

fn count_authorized_frames(support: &[FrameWindowV1]) -> u64 {
    support.iter().map(|window| window.end_frame - window.start_frame).sum()
}

#[derive(Serialize)]
struct AudioOperationPlanStepV1<'a> {
    schema_version: &'a str,
    sequence: u8,
    step_id: &'a str,
    operation_id: &'a str,
    instruction_sha256: &'a str,
    source_lineage_sha256: &'a str,
    parent_audio_sha256: &'a str,
    output_audio_sha256: &'a str,
    edit_regions: &'a [FrameWindowV1],
    transition_support: &'a [FrameWindowV1],
    boundary_policy: AudioBoundaryPolicyV1,
    upstream_gates: AudioUpstreamGatesV1,
    candidate_attempt_count: u16,
    failed_candidate_attempt_count: u16,
    render_call_count: u16,
    retry_count: u16,
    manual_intervention_count: u16,
}

/// Hash the complete immutable edit plan without duplicating raw audio bytes;
/// parent/output artifact hashes commit to the actual PCM payloads.
pub fn audio_operation_plan_sha256_v1(
    steps: &[AudioSequenceStepV1],
) -> Result<String, AudioContinuityErrorV1> {
    let projection: Vec<AudioOperationPlanStepV1<'_>> = steps.iter().map(|step| {
        AudioOperationPlanStepV1 {
            schema_version: &step.schema_version,
            sequence: step.sequence,
            step_id: &step.step_id,
            operation_id: &step.operation_id,
            instruction_sha256: &step.instruction_sha256,
            source_lineage_sha256: &step.source_lineage_sha256,
            parent_audio_sha256: &step.parent_audio_sha256,
            output_audio_sha256: &step.output_audio_sha256,
            edit_regions: &step.edit_regions,
            transition_support: &step.transition_support,
            boundary_policy: step.boundary_policy,
            upstream_gates: step.upstream_gates,
            candidate_attempt_count: step.candidate_attempt_count,
            failed_candidate_attempt_count: step.failed_candidate_attempt_count,
            render_call_count: step.render_call_count,
            retry_count: step.retry_count,
            manual_intervention_count: step.manual_intervention_count,
        }
    }).collect();
    let bytes = serde_json::to_vec(&projection)
        .map_err(|_| AudioContinuityErrorV1::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous.mel.audio-operation-plan.v1");
    hasher.update([0]);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AudioReportPayloadV1 {
    schema_version: String,
    request: AudioSequenceRequestV1,
    session_id: String,
    disposition: AudioSequenceDispositionV1,
    expected_step_count: u8,
    observed_step_count: u8,
    accepted_step_count: u8,
    blocked_at_step: Option<u8>,
    source_audio_sha256: String,
    final_audio_sha256: String,
    cumulative_changed_frame_count: u64,
    cumulative_changed_sample_value_count: u64,
    final_source_changed_frame_count: u64,
    final_source_changed_sample_value_count: u64,
    final_source_max_absolute_delta_samples: u32,
    cumulative_authorized_frame_count: u64,
    cumulative_attempt_count: u32,
    cumulative_failed_attempt_count: u32,
    cumulative_render_call_count: u32,
    cumulative_retry_count: u32,
    cumulative_manual_intervention_count: u32,
    steps: Vec<AudioStepEvaluationV1>,
}

pub fn evaluate_audio_sequence_v1(
    request: &AudioSequenceRequestV1,
    source: &PcmAudioArtifactV1,
    steps: &[AudioSequenceStepV1],
) -> Result<AudioSequenceEvaluationV1, AudioContinuityErrorV1> {
    if request.schema_version != AUDIO_SEQUENCE_SCHEMA_V1
        || request.session_id.trim().is_empty()
        || ![1, 2, 4, 8].contains(&request.expected_step_count)
        || steps.len() > usize::from(request.expected_step_count)
    {
        return Err(AudioContinuityErrorV1::InvalidRequest);
    }
    for digest in [
        request.source_lineage_sha256.as_str(),
        request.source_audio_sha256.as_str(),
        request.operation_plan_sha256.as_str(),
    ] {
        if !valid_sha256(digest) {
            return Err(AudioContinuityErrorV1::InvalidDigest);
        }
    }
    let actual_plan_sha256 = audio_operation_plan_sha256_v1(steps)?;
    if actual_plan_sha256 != request.operation_plan_sha256 {
        return Err(AudioContinuityErrorV1::OperationPlanMismatch);
    }
    validate_format(request.format)?;
    let source_frames = source.frame_count()?;
    if source.format != request.format {
        return Err(AudioContinuityErrorV1::InvalidFormat);
    }
    if source_frames != request.frame_count {
        return Err(AudioContinuityErrorV1::FrameCountMismatch);
    }
    let actual_source_sha = pcm_audio_sha256_v1(source)?;
    if actual_source_sha != request.source_audio_sha256 {
        return Err(AudioContinuityErrorV1::SourceDigestMismatch);
    }

    let mut current = source.clone();
    let mut current_sha = actual_source_sha.clone();
    let mut step_ids = BTreeSet::new();
    let mut operation_ids = BTreeSet::new();
    let mut reports = Vec::with_capacity(steps.len());
    let mut total_changed_frames = 0_u64;
    let mut total_changed_samples = 0_u64;
    let mut cumulative_support = Vec::<FrameWindowV1>::new();
    let mut final_source_changed_frames = 0_u64;
    let mut final_source_changed_samples = 0_u64;
    let mut final_source_max_delta = 0_u32;
    let mut final_cumulative_authorized_frames = 0_u64;
    let mut total_attempts = 0_u32;
    let mut total_failed_attempts = 0_u32;
    let mut total_render_calls = 0_u32;
    let mut total_retries = 0_u32;
    let mut total_manual_interventions = 0_u32;
    let mut accepted_steps = 0_u8;
    let mut blocked_at_step = None;
    let mut disposition = AudioSequenceDispositionV1::Incomplete;

    for (index, step) in steps.iter().enumerate() {
        if blocked_at_step.is_some() {
            return Err(AudioContinuityErrorV1::ContinuationAfterBlockedStep);
        }
        if step.schema_version != AUDIO_STEP_SCHEMA_V1
            || step.sequence != index as u8 + 1
        {
            return Err(AudioContinuityErrorV1::StepSequenceMismatch);
        }
        if step.step_id.trim().is_empty()
            || step.operation_id.trim().is_empty()
            || !step_ids.insert(step.step_id.as_str())
            || !operation_ids.insert(step.operation_id.as_str())
        {
            return Err(AudioContinuityErrorV1::DuplicateIdentity);
        }
        for digest in [
            step.instruction_sha256.as_str(),
            step.source_lineage_sha256.as_str(),
            step.parent_audio_sha256.as_str(),
            step.output_audio_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(AudioContinuityErrorV1::InvalidDigest);
            }
        }
        if step.source_lineage_sha256 != request.source_lineage_sha256 {
            return Err(AudioContinuityErrorV1::SourceLineageMismatch);
        }
        if step.parent_audio_sha256 != current_sha {
            return Err(AudioContinuityErrorV1::ParentDigestMismatch);
        }
        if step.candidate_attempt_count == 0
            || step.failed_candidate_attempt_count > step.candidate_attempt_count
            || step.retry_count > step.candidate_attempt_count
        {
            return Err(AudioContinuityErrorV1::InvalidAttemptAccounting);
        }
        if step.edit_regions.is_empty() {
            return Err(AudioContinuityErrorV1::EmptyEditSupport);
        }
        validate_windows(&step.edit_regions, request.frame_count)?;
        validate_windows(&step.transition_support, request.frame_count)?;

        if step.output.format != request.format {
            return Err(AudioContinuityErrorV1::InvalidFormat);
        }
        if step.output.frame_count()? != request.frame_count {
            return Err(AudioContinuityErrorV1::FrameCountMismatch);
        }
        let actual_output_sha = pcm_audio_sha256_v1(&step.output)?;
        if actual_output_sha != step.output_audio_sha256 {
            return Err(AudioContinuityErrorV1::OutputDigestMismatch);
        }

        let support = merge_support(&step.edit_regions, &step.transition_support);
        cumulative_support.extend_from_slice(&step.edit_regions);
        cumulative_support.extend_from_slice(&step.transition_support);
        cumulative_support.sort_by_key(|window| (window.start_frame, window.end_frame));
        cumulative_support = merge_support(&cumulative_support, &[]);
        let cumulative_authorized_frame_count = count_authorized_frames(&cumulative_support);
        let channels = usize::from(request.format.channels);
        let mut changed_frames = BTreeSet::new();
        let mut changed_sample_values = 0_u64;
        let mut off_support_changes = 0_u64;
        for (sample_index, (before, after)) in current
            .samples
            .iter()
            .zip(step.output.samples.iter())
            .enumerate()
        {
            if before != after {
                changed_sample_values += 1;
                let frame = (sample_index / channels) as u64;
                changed_frames.insert(frame);
                if !frame_is_authorized(frame, &support) {
                    off_support_changes += 1;
                }
            }
        }
        if changed_sample_values == 0 {
            return Err(AudioContinuityErrorV1::NoAudioMutation);
        }
        if off_support_changes != 0 {
            return Err(AudioContinuityErrorV1::OffSupportMutation);
        }

        // Re-measure drift against the frozen session source after every step.
        let mut source_changed_frames = BTreeSet::new();
        let mut source_changed_samples = 0_u64;
        let mut source_max_delta = 0_u32;
        let mut source_off_support_changes = 0_u64;
        for (sample_index, (source_sample, output_sample)) in source
            .samples
            .iter()
            .zip(step.output.samples.iter())
            .enumerate()
        {
            if source_sample != output_sample {
                source_changed_samples = source_changed_samples.saturating_add(1);
                let frame = (sample_index / channels) as u64;
                source_changed_frames.insert(frame);
                let delta =
                    (i32::from(*source_sample) - i32::from(*output_sample)).unsigned_abs();
                source_max_delta = source_max_delta.max(delta);
                if !frame_is_authorized(frame, &cumulative_support) {
                    source_off_support_changes = source_off_support_changes.saturating_add(1);
                }
            }
        }
        if source_off_support_changes != 0 {
            return Err(AudioContinuityErrorV1::OffSupportMutation);
        }
        final_source_changed_frames = source_changed_frames.len() as u64;
        final_source_changed_samples = source_changed_samples;
        final_source_max_delta = source_max_delta;
        final_cumulative_authorized_frames = cumulative_authorized_frame_count;

        let measurements = boundary_measurements(
            &current,
            &step.output,
            &support,
            step.boundary_policy.measurement_radius_frames,
        );
        let boundary_gate = evaluate_boundary_gate(&measurements, step.boundary_policy);
        let upstream_verified = step.upstream_gates.all_verified();
        let upstream_rejected = step.upstream_gates.any_rejected();
        let hard_gate_verified =
            upstream_verified && boundary_gate == AudioGateVerdictV1::Verified;
        let hard_gate_rejected =
            upstream_rejected || boundary_gate == AudioGateVerdictV1::Rejected;

        total_attempts = total_attempts.saturating_add(u32::from(step.candidate_attempt_count));
        total_failed_attempts =
            total_failed_attempts.saturating_add(u32::from(step.failed_candidate_attempt_count));
        total_render_calls = total_render_calls.saturating_add(u32::from(step.render_call_count));
        total_retries = total_retries.saturating_add(u32::from(step.retry_count));
        total_manual_interventions =
            total_manual_interventions.saturating_add(u32::from(step.manual_intervention_count));
        total_changed_frames = total_changed_frames.saturating_add(changed_frames.len() as u64);
        total_changed_samples = total_changed_samples.saturating_add(changed_sample_values);

        if hard_gate_verified {
            accepted_steps = accepted_steps.saturating_add(1);
        } else if hard_gate_rejected {
            blocked_at_step = Some(step.sequence);
            disposition = AudioSequenceDispositionV1::RejectedAtStep;
        } else {
            blocked_at_step = Some(step.sequence);
            disposition = AudioSequenceDispositionV1::AbstainedAtStep;
        }

        reports.push(AudioStepEvaluationV1 {
            sequence: step.sequence,
            step_id: step.step_id.clone(),
            operation_id: step.operation_id.clone(),
            instruction_sha256: step.instruction_sha256.clone(),
            source_lineage_sha256: step.source_lineage_sha256.clone(),
            edit_regions: step.edit_regions.clone(),
            transition_support: step.transition_support.clone(),
            boundary_policy: step.boundary_policy,
            upstream_gates: step.upstream_gates,
            parent_audio_sha256: current_sha.clone(),
            output_audio_sha256: actual_output_sha.clone(),
            authorized_frame_count: count_authorized_frames(&support),
            changed_frame_count: changed_frames.len() as u64,
            changed_sample_value_count: changed_sample_values,
            off_support_changed_sample_value_count: off_support_changes,
            cumulative_authorized_frame_count,
            source_changed_frame_count: source_changed_frames.len() as u64,
            source_changed_sample_value_count: source_changed_samples,
            source_max_absolute_delta_samples: source_max_delta,
            source_off_support_changed_sample_value_count: source_off_support_changes,
            boundary_measurements: measurements,
            boundary_gate,
            upstream_gates_verified: upstream_verified,
            hard_gate_verified,
            cumulative_attempt_count: total_attempts,
            cumulative_failed_attempt_count: total_failed_attempts,
            cumulative_render_call_count: total_render_calls,
            cumulative_retry_count: total_retries,
            cumulative_manual_intervention_count: total_manual_interventions,
        });

        current = step.output.clone();
        current_sha = actual_output_sha;
    }

    if blocked_at_step.is_none() && steps.len() == usize::from(request.expected_step_count) {
        disposition = AudioSequenceDispositionV1::Accepted;
    }

    let payload = AudioReportPayloadV1 {
        schema_version: "mel-music-audio-sequence-report-v1".to_owned(),
        request: request.clone(),
        session_id: request.session_id.clone(),
        disposition,
        expected_step_count: request.expected_step_count,
        observed_step_count: steps.len() as u8,
        accepted_step_count: accepted_steps,
        blocked_at_step,
        source_audio_sha256: actual_source_sha,
        final_audio_sha256: current_sha,
        cumulative_changed_frame_count: total_changed_frames,
        cumulative_changed_sample_value_count: total_changed_samples,
        final_source_changed_frame_count: final_source_changed_frames,
        final_source_changed_sample_value_count: final_source_changed_samples,
        final_source_max_absolute_delta_samples: final_source_max_delta,
        cumulative_authorized_frame_count: final_cumulative_authorized_frames,
        cumulative_attempt_count: total_attempts,
        cumulative_failed_attempt_count: total_failed_attempts,
        cumulative_render_call_count: total_render_calls,
        cumulative_retry_count: total_retries,
        cumulative_manual_intervention_count: total_manual_interventions,
        steps: reports.clone(),
    };
    let bytes =
        serde_json::to_vec(&payload).map_err(|_| AudioContinuityErrorV1::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(AUDIO_REPORT_DOMAIN_V1);
    hasher.update([0]);
    hasher.update(bytes);
    let report_sha256 = format!("{:x}", hasher.finalize());

    Ok(AudioSequenceEvaluationV1 {
        schema_version: "mel-music-audio-sequence-report-v1".to_owned(),
        request: payload.request,
        session_id: payload.session_id,
        disposition,
        expected_step_count: request.expected_step_count,
        observed_step_count: steps.len() as u8,
        accepted_step_count: accepted_steps,
        blocked_at_step,
        source_audio_sha256: payload.source_audio_sha256,
        final_audio_sha256: payload.final_audio_sha256,
        cumulative_changed_frame_count: total_changed_frames,
        cumulative_changed_sample_value_count: total_changed_samples,
        final_source_changed_frame_count: payload.final_source_changed_frame_count,
        final_source_changed_sample_value_count: payload.final_source_changed_sample_value_count,
        final_source_max_absolute_delta_samples: payload.final_source_max_absolute_delta_samples,
        cumulative_authorized_frame_count: payload.cumulative_authorized_frame_count,
        cumulative_attempt_count: total_attempts,
        cumulative_failed_attempt_count: total_failed_attempts,
        cumulative_render_call_count: total_render_calls,
        cumulative_retry_count: total_retries,
        cumulative_manual_intervention_count: total_manual_interventions,
        steps: reports,
        report_sha256,
    })
}

/// Recompute the entire audio-session report from the frozen request,
/// original PCM and submitted steps. A matching self-hash alone is not enough.
pub fn verify_audio_sequence_report_v1(
    request: &AudioSequenceRequestV1,
    source: &PcmAudioArtifactV1,
    steps: &[AudioSequenceStepV1],
    supplied: &AudioSequenceEvaluationV1,
) -> Result<(), AudioContinuityErrorV1> {
    let replayed = evaluate_audio_sequence_v1(request, source, steps)?;
    if &replayed != supplied {
        return Err(AudioContinuityErrorV1::ReportReplayMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(channels: u8) -> PcmFormatV1 {
        PcmFormatV1 {
            sample_rate_hz: 48_000,
            channels,
            encoding: PcmEncodingV1::SignedInt16LittleEndian,
        }
    }

    fn source(frames: usize, channels: u8) -> PcmAudioArtifactV1 {
        let samples = (0..frames * usize::from(channels))
            .map(|index| ((index as i32 * 97 % 20_000) - 10_000) as i16)
            .collect();
        PcmAudioArtifactV1 {
            format: format(channels),
            samples,
        }
    }

    fn h(value: u64) -> String {
        format!("{value:064x}")
    }

    fn request(
        source: &PcmAudioArtifactV1,
        count: u8,
        steps: &[AudioSequenceStepV1],
    ) -> AudioSequenceRequestV1 {
        AudioSequenceRequestV1 {
            schema_version: AUDIO_SEQUENCE_SCHEMA_V1.to_owned(),
            session_id: format!("audio-session-{count}"),
            source_lineage_sha256: h(10),
            source_audio_sha256: pcm_audio_sha256_v1(source).unwrap(),
            operation_plan_sha256: audio_operation_plan_sha256_v1(steps).unwrap(),
            expected_step_count: count,
            format: source.format,
            frame_count: source.frame_count().unwrap(),
        }
    }

    fn run(
        source: &PcmAudioArtifactV1,
        count: u8,
        steps: &[AudioSequenceStepV1],
    ) -> Result<AudioSequenceEvaluationV1, AudioContinuityErrorV1> {
        let frozen_request = request(source, count, steps);
        evaluate_audio_sequence_v1(&frozen_request, source, steps)
    }

    fn gates(verdict: AudioGateVerdictV1) -> AudioUpstreamGatesV1 {
        AudioUpstreamGatesV1 {
            rights_admission: verdict,
            source_lineage: verdict,
            realization_receipt: verdict,
            transaction_admission: verdict,
        }
    }

    fn edited(
        parent: &PcmAudioArtifactV1,
        window: FrameWindowV1,
        delta: i16,
    ) -> PcmAudioArtifactV1 {
        let mut output = parent.clone();
        let channels = usize::from(parent.format.channels);
        for frame in window.start_frame..window.end_frame {
            for channel in 0..channels {
                let index = frame as usize * channels + channel;
                output.samples[index] = output.samples[index].saturating_add(delta);
            }
        }
        output
    }

    fn step(
        sequence: u8,
        parent: &PcmAudioArtifactV1,
        output: PcmAudioArtifactV1,
        edit: FrameWindowV1,
    ) -> AudioSequenceStepV1 {
        AudioSequenceStepV1 {
            schema_version: AUDIO_STEP_SCHEMA_V1.to_owned(),
            sequence,
            step_id: format!("audio-step-{sequence}"),
            operation_id: format!("audio-op-{sequence}"),
            instruction_sha256: h(20 + u64::from(sequence)),
            source_lineage_sha256: h(10),
            parent_audio_sha256: pcm_audio_sha256_v1(parent).unwrap(),
            output_audio_sha256: pcm_audio_sha256_v1(&output).unwrap(),
            edit_regions: vec![edit],
            transition_support: Vec::new(),
            boundary_policy: AudioBoundaryPolicyV1 {
                kind: AudioBoundaryKindV1::OperationSpecific,
                measurement_radius_frames: 4,
                maximum_jump_delta_samples: Some(32_767),
                maximum_side_rms_delta_samples: Some(32_767),
            },
            upstream_gates: gates(AudioGateVerdictV1::Verified),
            candidate_attempt_count: 2,
            failed_candidate_attempt_count: 1,
            render_call_count: 2,
            retry_count: 1,
            manual_intervention_count: 0,
            output,
        }
    }

    #[test]
    fn one_step_preserves_every_sample_outside_authorized_region() {
        let original = source(128, 2);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let output = edited(&original, window, 400);
        let result = run(&original, 1, &[step(1, &original, output, window)]).unwrap();
        assert_eq!(result.disposition, AudioSequenceDispositionV1::Accepted);
        assert_eq!(result.steps[0].off_support_changed_sample_value_count, 0);
        assert_eq!(result.steps[0].changed_frame_count, 24);
    }

    #[test]
    fn one_sample_leakage_outside_support_is_rejected() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let mut output = edited(&original, window, 400);
        output.samples[39] = output.samples[39].saturating_add(1);
        assert_eq!(
            run(&original, 1, &[step(1, &original, output, window)]),
            Err(AudioContinuityErrorV1::OffSupportMutation)
        );
    }

    #[test]
    fn unchanged_output_cannot_be_counted_as_a_completed_edit() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let unchanged = original.clone();
        assert_eq!(
            run(&original, 1, &[step(1, &original, unchanged, window)]),
            Err(AudioContinuityErrorV1::NoAudioMutation)
        );
    }

    #[test]
    fn explicitly_declared_transition_support_allows_only_that_extra_region() {
        let original = source(128, 1);
        let edit = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let transition = FrameWindowV1 { start_frame: 36, end_frame: 68 };
        let output = edited(&original, transition, 200);
        let mut edit_step = step(1, &original, output, edit);
        edit_step.transition_support = vec![transition];
        let result = run(&original, 1, &[edit_step]).unwrap();
        assert_eq!(result.disposition, AudioSequenceDispositionV1::Accepted);
        assert_eq!(result.steps[0].authorized_frame_count, 32);
    }

    #[test]
    fn undeclared_transition_dilation_is_rejected() {
        let original = source(128, 1);
        let edit = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let enlarged = FrameWindowV1 { start_frame: 36, end_frame: 68 };
        let output = edited(&original, enlarged, 200);
        assert_eq!(
            run(&original, 1, &[step(1, &original, output, edit)]),
            Err(AudioContinuityErrorV1::OffSupportMutation)
        );
    }

    #[test]
    fn stale_parent_audio_digest_is_rejected() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let output = edited(&original, window, 200);
        let mut edit_step = step(1, &original, output, window);
        edit_step.parent_audio_sha256 = h(999);
        assert_eq!(
            run(&original, 1, &[edit_step]),
            Err(AudioContinuityErrorV1::ParentDigestMismatch)
        );
    }

    #[test]
    fn implicit_resampling_or_channel_change_is_rejected() {
        let original = source(128, 2);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let mut output = edited(&original, window, 200);
        output.format.sample_rate_hz = 44_100;
        assert_eq!(
            run(&original, 1, &[step(1, &original, output, window)]),
            Err(AudioContinuityErrorV1::InvalidFormat)
        );
    }

    #[test]
    fn frame_count_change_is_rejected() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let mut output = edited(&original, window, 200);
        output.samples.pop();
        assert_eq!(
            run(&original, 1, &[step(1, &original, output, window)]),
            Err(AudioContinuityErrorV1::FrameCountMismatch)
        );
    }

    #[test]
    fn operation_plan_digest_binds_regions_thresholds_and_upstream_gates() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let output = edited(&original, window, 200);
        let operation_step = step(1, &original, output, window);
        let frozen = request(&original, 1, std::slice::from_ref(&operation_step));
        let mut changed = operation_step;
        changed.boundary_policy.maximum_jump_delta_samples = Some(0);
        assert_eq!(
            evaluate_audio_sequence_v1(&frozen, &original, &[changed]),
            Err(AudioContinuityErrorV1::OperationPlanMismatch)
        );
    }

    #[test]
    fn report_verification_replays_request_plan_measurements_and_decision() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let steps = vec![step(1, &original, edited(&original, window, 200), window)];
        let frozen_request = request(&original, 1, &steps);
        let report = evaluate_audio_sequence_v1(&frozen_request, &original, &steps).unwrap();
        verify_audio_sequence_report_v1(&frozen_request, &original, &steps, &report).unwrap();

        let mut altered = report.clone();
        altered.steps[0].boundary_policy.maximum_jump_delta_samples = Some(0);
        assert_eq!(
            verify_audio_sequence_report_v1(&frozen_request, &original, &steps, &altered),
            Err(AudioContinuityErrorV1::ReportReplayMismatch)
        );
    }

    #[test]
    fn digest_is_bound_to_pcm_format_and_samples() {
        let original = source(128, 1);
        let first = pcm_audio_sha256_v1(&original).unwrap();
        let mut changed = original.clone();
        changed.format.sample_rate_hz = 44_100;
        assert_ne!(first, pcm_audio_sha256_v1(&changed).unwrap());
        changed = original.clone();
        changed.samples[0] = changed.samples[0].saturating_add(1);
        assert_ne!(first, pcm_audio_sha256_v1(&changed).unwrap());
    }

    #[test]
    fn missing_boundary_thresholds_produce_unknown_not_success() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let output = edited(&original, window, 200);
        let mut edit_step = step(1, &original, output, window);
        edit_step.boundary_policy.maximum_jump_delta_samples = None;
        let result = run(&original, 1, &[edit_step]).unwrap();
        assert_eq!(result.disposition, AudioSequenceDispositionV1::AbstainedAtStep);
        assert_eq!(result.steps[0].boundary_gate, AudioGateVerdictV1::Unknown);
    }

    #[test]
    fn boundary_threshold_failure_is_retained_as_rejected_step() {
        let original = source(128, 1);
        let window = FrameWindowV1 { start_frame: 40, end_frame: 64 };
        let output = edited(&original, window, 20_000);
        let mut edit_step = step(1, &original, output, window);
        edit_step.boundary_policy.maximum_jump_delta_samples = Some(0);
        edit_step.boundary_policy.maximum_side_rms_delta_samples = Some(0);
        let result = run(&original, 1, &[edit_step]).unwrap();
        assert_eq!(result.disposition, AudioSequenceDispositionV1::RejectedAtStep);
        assert_eq!(result.blocked_at_step, Some(1));
    }

    #[test]
    fn no_later_step_can_launder_unknown_boundary_evidence() {
        let original = source(128, 1);
        let first_window = FrameWindowV1 { start_frame: 20, end_frame: 40 };
        let second_window = FrameWindowV1 { start_frame: 72, end_frame: 88 };
        let first_output = edited(&original, first_window, 100);
        let second_output = edited(&first_output, second_window, 100);
        let mut first = step(1, &original, first_output.clone(), first_window);
        first.boundary_policy.maximum_side_rms_delta_samples = None;
        let second = step(2, &first_output, second_output, second_window);
        assert_eq!(
            run(&original, 2, &[first, second]),
            Err(AudioContinuityErrorV1::ContinuationAfterBlockedStep)
        );
    }

    #[test]
    fn cumulative_sequence_chains_exact_audio_parent_and_counts_all_work() {
        let original = source(256, 2);
        let first_window = FrameWindowV1 { start_frame: 32, end_frame: 48 };
        let first_output = edited(&original, first_window, 100);
        let second_window = FrameWindowV1 { start_frame: 160, end_frame: 184 };
        let second_output = edited(&first_output, second_window, -75);
        let result = run(&original, 2, &[
                step(1, &original, first_output.clone(), first_window),
                step(2, &first_output, second_output, second_window),
            ]).unwrap();
        assert_eq!(result.disposition, AudioSequenceDispositionV1::Accepted);
        assert_eq!(result.observed_step_count, 2);
        assert_eq!(result.accepted_step_count, 2);
        assert_eq!(result.cumulative_attempt_count, 4);
        assert_eq!(result.cumulative_failed_attempt_count, 2);
        assert_eq!(result.cumulative_render_call_count, 4);
        assert_eq!(result.cumulative_changed_frame_count, 40);
    }

    #[test]
    fn cumulative_source_drift_is_measured_against_the_frozen_session_root() {
        let original = source(256, 2);
        let first_window = FrameWindowV1 { start_frame: 32, end_frame: 48 };
        let first_output = edited(&original, first_window, 100);
        let second_window = FrameWindowV1 { start_frame: 160, end_frame: 184 };
        let second_output = edited(&first_output, second_window, -75);
        let steps = [
            step(1, &original, first_output.clone(), first_window),
            step(2, &first_output, second_output, second_window),
        ];
        let result = run(&original, 2, &steps).unwrap();
        assert_eq!(result.steps[0].source_changed_frame_count, 16);
        assert_eq!(result.steps[1].source_changed_frame_count, 40);
        assert_eq!(result.final_source_changed_frame_count, 40);
        assert_eq!(result.final_source_changed_sample_value_count, 80);
        assert_eq!(result.final_source_max_absolute_delta_samples, 100);
        assert_eq!(result.cumulative_authorized_frame_count, 40);
        assert_eq!(result.steps[1].source_off_support_changed_sample_value_count, 0);
    }

    #[test]
    fn window_order_overlap_and_bounds_are_rejected() {
        let original = source(128, 1);
        let first = FrameWindowV1 { start_frame: 48, end_frame: 64 };
        let second = FrameWindowV1 { start_frame: 40, end_frame: 56 };
        let output = edited(&original, first, 200);
        let mut edit_step = step(1, &original, output, first);
        edit_step.transition_support = vec![second, first];
        assert_eq!(
            run(&original, 1, &[edit_step]),
            Err(AudioContinuityErrorV1::NonCanonicalWindows)
        );
        let mut out_of_range = step(
            1, &original, edited(&original, first, 200),
            FrameWindowV1 { start_frame: 100, end_frame: 129 },
        );
        out_of_range.edit_regions = vec![FrameWindowV1 { start_frame: 100, end_frame: 129 }];
        assert_eq!(
            run(&original, 1, &[out_of_range]),
            Err(AudioContinuityErrorV1::InvalidWindow)
        );
    }
}
