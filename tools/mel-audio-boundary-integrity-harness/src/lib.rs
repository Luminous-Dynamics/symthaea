#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic PCM boundary-integrity contract.
//!
//! This crate checks exact PCM identity, protected-region preservation,
//! support locality and predeclared boundary signal thresholds. Signal
//! statistics are evidence features, not substitutes for human judgments of
//! musicality, timbral fit or whether a transition sounds natural.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const AUDIO_BOUNDARY_REQUEST_SCHEMA_V1: &str = "mel-audio-boundary-request-v1";
pub const AUDIO_BOUNDARY_RECEIPT_SCHEMA_V1: &str = "mel-audio-boundary-receipt-v1";
const PCM_DOMAIN: &[u8] = b"luminous.mel.audio-pcm-interleaved-s32le.v1";
const REGION_DOMAIN: &[u8] = b"luminous.mel.audio-protected-region.v1";
const REQUEST_DOMAIN: &[u8] = b"luminous.mel.audio-boundary-request.v1";
const RECEIPT_DOMAIN: &[u8] = b"luminous.mel.audio-boundary-receipt.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioPcmBufferV1 {
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    /// Interleaved signed 32-bit PCM samples. Frame count is len / channels.
    pub samples_interleaved_s32: Vec<i32>,
}

impl AudioPcmBufferV1 {
    pub fn frame_count(&self) -> Result<u64, AudioBoundaryErrorV1> {
        if self.channel_count == 0
            || !self.samples_interleaved_s32.len().is_multiple_of(usize::from(self.channel_count))
        {
            return Err(AudioBoundaryErrorV1::InvalidAudioBuffer);
        }
        u64::try_from(self.samples_interleaved_s32.len() / usize::from(self.channel_count))
            .map_err(|_| AudioBoundaryErrorV1::InvalidAudioBuffer)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioEditWindowV1 {
    /// Inclusive start frame.
    pub start_frame: u64,
    /// Exclusive end frame.
    pub end_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioProtectedRegionV1 {
    pub region_id: String,
    pub start_frame: u64,
    pub end_frame: u64,
    /// Digest over the exact canonical PCM in this region and its channel/rate metadata.
    pub expected_region_sha256: String,
}

/// A predeclared boundary policy. Thresholds are task/profile-specific; this
/// crate does not claim that one universal waveform threshold guarantees
/// audibility or musical quality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryThresholdV1 {
    pub boundary_id: String,
    /// First frame after the seam.
    pub boundary_frame: u64,
    pub left_window_frames: u16,
    pub right_window_frames: u16,
    pub maximum_seam_jump_abs: u64,
    pub maximum_slope_delta_abs: u64,
    pub maximum_local_step_abs: u64,
    pub maximum_rms_ratio_deviation_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryRequestV1 {
    pub schema_version: String,
    pub operation_id: String,
    pub source_lineage_sha256: String,
    pub operation_plan_sha256: String,
    pub source_pcm_sha256: String,
    pub expected_output_pcm_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub frame_count: u64,
    /// Only these regions may differ from the source in this fixed-timeline v1 contract.
    pub edit_windows: Vec<AudioEditWindowV1>,
    pub protected_regions: Vec<AudioProtectedRegionV1>,
    /// Canonically sorted by boundary_id and frozen before evaluation.
    pub boundaries: Vec<AudioBoundaryThresholdV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedRegionResultV1 {
    pub region_id: String,
    pub source_region_sha256: String,
    pub output_region_sha256: String,
    pub exact_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelBoundaryMetricsV1 {
    pub channel_index: u16,
    pub seam_jump_abs: u64,
    pub slope_delta_abs: u64,
    pub maximum_local_step_abs: u64,
    pub rms_ratio_deviation_ppm: u32,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryResultV1 {
    pub boundary_id: String,
    pub boundary_frame: u64,
    pub channels: Vec<ChannelBoundaryMetricsV1>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryReceiptPayloadV1 {
    pub schema_version: String,
    pub request_sha256: String,
    pub source_pcm_sha256: String,
    pub output_pcm_sha256: String,
    pub frame_count: u64,
    pub changed_frame_count: u64,
    pub protected_regions: Vec<ProtectedRegionResultV1>,
    pub boundaries: Vec<AudioBoundaryResultV1>,
    /// False means the audio fails at least one predeclared boundary threshold.
    /// It is an evidence result, not a transaction decision.
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBoundaryReceiptV1 {
    pub payload: AudioBoundaryReceiptPayloadV1,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioBoundaryErrorV1 {
    InvalidRequest,
    InvalidAudioBuffer,
    InvalidDigest,
    NonCanonicalEditWindows,
    NonCanonicalProtectedRegions,
    NonCanonicalBoundaries,
    InvalidRegion,
    ProtectedRegionOverlapsEditSupport,
    SourceDigestMismatch,
    OutputDigestMismatch,
    AudioMetadataMismatch,
    FrameCountMismatch,
    UnauthorizedFrameMutation,
    ProtectedRegionMismatch,
    ReceiptDigestMismatch,
    RequestBindingMismatch,
}

impl fmt::Display for AudioBoundaryErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidRequest => "audio-boundary request schema, identity or metadata is invalid",
            Self::InvalidAudioBuffer => "PCM buffer has invalid channel or interleaving metadata",
            Self::InvalidDigest => "a required SHA-256 digest is malformed",
            Self::NonCanonicalEditWindows => "edit windows must be sorted, nonempty and nonoverlapping",
            Self::NonCanonicalProtectedRegions => "protected regions must be sorted and unique",
            Self::NonCanonicalBoundaries => "boundary policies must be sorted by unique boundary ID",
            Self::InvalidRegion => "audio region or boundary window is outside the frozen frame range",
            Self::ProtectedRegionOverlapsEditSupport => "protected audio region overlaps an authorized edit window",
            Self::SourceDigestMismatch => "source PCM bytes differ from the frozen source digest",
            Self::OutputDigestMismatch => "output PCM bytes differ from the frozen expected digest",
            Self::AudioMetadataMismatch => "source and output sample rate or channel layout differs",
            Self::FrameCountMismatch => "source or output frame count differs from the frozen request",
            Self::UnauthorizedFrameMutation => "PCM samples changed outside authorized edit support",
            Self::ProtectedRegionMismatch => "a protected PCM region changed",
            Self::ReceiptDigestMismatch => "audio-boundary receipt digest is invalid",
            Self::RequestBindingMismatch => "receipt does not replay against the exact request and PCM buffers",
        };
        f.write_str(message)
    }
}

impl std::error::Error for AudioBoundaryErrorV1 {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn audio_pcm_sha256_v1(audio: &AudioPcmBufferV1) -> Result<String, AudioBoundaryErrorV1> {
    let _ = audio.frame_count()?;
    let mut bytes = Vec::with_capacity(audio.samples_interleaved_s32.len() * 4 + 8);
    bytes.extend_from_slice(&audio.sample_rate_hz.to_le_bytes());
    bytes.extend_from_slice(&audio.channel_count.to_le_bytes());
    for sample in &audio.samples_interleaved_s32 {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(digest(PCM_DOMAIN, &bytes))
}

fn samples_for_region(
    audio: &AudioPcmBufferV1,
    start_frame: u64,
    end_frame: u64,
) -> Result<Vec<i32>, AudioBoundaryErrorV1> {
    let frames = audio.frame_count()?;
    if start_frame >= end_frame || end_frame > frames {
        return Err(AudioBoundaryErrorV1::InvalidRegion);
    }
    let channels = usize::from(audio.channel_count);
    let start = usize::try_from(start_frame)
        .ok()
        .and_then(|frame| frame.checked_mul(channels))
        .ok_or(AudioBoundaryErrorV1::InvalidRegion)?;
    let end = usize::try_from(end_frame)
        .ok()
        .and_then(|frame| frame.checked_mul(channels))
        .ok_or(AudioBoundaryErrorV1::InvalidRegion)?;
    Ok(audio.samples_interleaved_s32[start..end].to_vec())
}

fn region_sha256_v1(
    audio: &AudioPcmBufferV1,
    region_id: &str,
    start_frame: u64,
    end_frame: u64,
) -> Result<String, AudioBoundaryErrorV1> {
    let samples = samples_for_region(audio, start_frame, end_frame)?;
    let mut bytes = Vec::with_capacity(region_id.len() + samples.len() * 4 + 32);
    bytes.extend_from_slice(region_id.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&audio.sample_rate_hz.to_le_bytes());
    bytes.extend_from_slice(&audio.channel_count.to_le_bytes());
    bytes.extend_from_slice(&start_frame.to_le_bytes());
    bytes.extend_from_slice(&end_frame.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(digest(REGION_DOMAIN, &bytes))
}

fn validate_request_v1(request: &AudioBoundaryRequestV1) -> Result<(), AudioBoundaryErrorV1> {
    if request.schema_version != AUDIO_BOUNDARY_REQUEST_SCHEMA_V1
        || request.operation_id.trim().is_empty()
        || request.sample_rate_hz < 8_000
        || request.sample_rate_hz > 384_000
        || request.channel_count == 0
        || request.channel_count > 8
        || request.frame_count < 4
        || !valid_sha256(&request.source_lineage_sha256)
        || !valid_sha256(&request.operation_plan_sha256)
        || !valid_sha256(&request.source_pcm_sha256)
        || !valid_sha256(&request.expected_output_pcm_sha256)
    {
        return Err(AudioBoundaryErrorV1::InvalidRequest);
    }

    let mut prior_end = 0_u64;
    for (index, window) in request.edit_windows.iter().enumerate() {
        if window.start_frame >= window.end_frame
            || window.end_frame > request.frame_count
            || (index > 0 && window.start_frame < prior_end)
        {
            return Err(AudioBoundaryErrorV1::NonCanonicalEditWindows);
        }
        prior_end = window.end_frame;
    }

    let mut protected_ids = BTreeSet::new();
    let mut previous_region_end = 0_u64;
    for (index, region) in request.protected_regions.iter().enumerate() {
        if region.region_id.trim().is_empty()
            || region.start_frame >= region.end_frame
            || region.end_frame > request.frame_count
            || !valid_sha256(&region.expected_region_sha256)
        {
            return Err(AudioBoundaryErrorV1::InvalidRegion);
        }
        if !protected_ids.insert(region.region_id.as_str())
            || (index > 0 && region.start_frame < previous_region_end)
        {
            return Err(AudioBoundaryErrorV1::NonCanonicalProtectedRegions);
        }
        if request.edit_windows.iter().any(|window| {
            region.start_frame < window.end_frame && window.start_frame < region.end_frame
        }) {
            return Err(AudioBoundaryErrorV1::ProtectedRegionOverlapsEditSupport);
        }
        previous_region_end = region.end_frame;
    }

    if !request
        .boundaries
        .windows(2)
        .all(|pair| pair[0].boundary_id < pair[1].boundary_id)
    {
        return Err(AudioBoundaryErrorV1::NonCanonicalBoundaries);
    }
    for boundary in &request.boundaries {
        let left = u64::from(boundary.left_window_frames);
        let right = u64::from(boundary.right_window_frames);
        let is_edit_edge = request.edit_windows.iter().any(|window| {
            boundary.boundary_frame == window.start_frame
                || boundary.boundary_frame == window.end_frame
        });
        if boundary.boundary_id.trim().is_empty()
            || left < 2
            || right < 2
            || boundary.boundary_frame < left
            || boundary.boundary_frame.checked_add(right).is_none_or(|end| end > request.frame_count)
            || !is_edit_edge
        {
            return Err(AudioBoundaryErrorV1::InvalidRegion);
        }
    }
    Ok(())
}

fn changed_frame_count_and_validate_support(
    request: &AudioBoundaryRequestV1,
    source: &AudioPcmBufferV1,
    output: &AudioPcmBufferV1,
) -> Result<u64, AudioBoundaryErrorV1> {
    let channels = usize::from(source.channel_count);
    let mut changed_frames = 0_u64;
    for frame in 0..request.frame_count {
        let start = usize::try_from(frame)
            .ok()
            .and_then(|value| value.checked_mul(channels))
            .ok_or(AudioBoundaryErrorV1::InvalidAudioBuffer)?;
        let end = start + channels;
        if source.samples_interleaved_s32[start..end] != output.samples_interleaved_s32[start..end] {
            changed_frames += 1;
            let allowed = request
                .edit_windows
                .iter()
                .any(|window| frame >= window.start_frame && frame < window.end_frame);
            if !allowed {
                return Err(AudioBoundaryErrorV1::UnauthorizedFrameMutation);
            }
        }
    }
    Ok(changed_frames)
}

fn integer_sqrt(value: u128) -> u128 {
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

fn rms_abs(samples: &[i32]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let sum_squares = samples.iter().fold(0_u128, |sum, sample| {
        let value = i128::from(*sample);
        sum.saturating_add((value * value) as u128)
    });
    integer_sqrt(sum_squares / samples.len() as u128).min(u128::from(u64::MAX)) as u64
}

fn absolute_delta(left: i32, right: i32) -> u64 {
    (i64::from(left) - i64::from(right)).unsigned_abs()
}

fn boundary_metrics_for_channel(
    audio: &AudioPcmBufferV1,
    policy: &AudioBoundaryThresholdV1,
    channel_index: u16,
) -> Result<ChannelBoundaryMetricsV1, AudioBoundaryErrorV1> {
    let channels = usize::from(audio.channel_count);
    let channel = usize::from(channel_index);
    let boundary = usize::try_from(policy.boundary_frame)
        .map_err(|_| AudioBoundaryErrorV1::InvalidRegion)?;
    let left_start = usize::try_from(policy.boundary_frame - u64::from(policy.left_window_frames))
        .map_err(|_| AudioBoundaryErrorV1::InvalidRegion)?;
    let right_end = usize::try_from(policy.boundary_frame + u64::from(policy.right_window_frames))
        .map_err(|_| AudioBoundaryErrorV1::InvalidRegion)?;

    let sample = |frame: usize| -> i32 {
        audio.samples_interleaved_s32[frame * channels + channel]
    };

    let seam_jump_abs = absolute_delta(sample(boundary - 1), sample(boundary));
    let left_slope = i64::from(sample(boundary - 1)) - i64::from(sample(boundary - 2));
    let right_slope = i64::from(sample(boundary + 1)) - i64::from(sample(boundary));
    let slope_delta_abs = (right_slope - left_slope).unsigned_abs();

    let mut maximum_local_step_abs = 0_u64;
    for frame in (left_start + 1)..right_end {
        maximum_local_step_abs = maximum_local_step_abs.max(absolute_delta(sample(frame - 1), sample(frame)));
    }

    let left_samples: Vec<i32> = (left_start..boundary).map(sample).collect();
    let right_samples: Vec<i32> = (boundary..right_end).map(sample).collect();
    let left_rms = rms_abs(&left_samples);
    let right_rms = rms_abs(&right_samples);
    let maximum_rms = left_rms.max(right_rms);
    let rms_ratio_deviation_ppm = if maximum_rms == 0 {
        0
    } else {
        ((left_rms.abs_diff(right_rms) as u128 * 1_000_000_u128)
            / u128::from(maximum_rms))
        .min(u128::from(u32::MAX)) as u32
    };

    let passed = seam_jump_abs <= policy.maximum_seam_jump_abs
        && slope_delta_abs <= policy.maximum_slope_delta_abs
        && maximum_local_step_abs <= policy.maximum_local_step_abs
        && rms_ratio_deviation_ppm <= policy.maximum_rms_ratio_deviation_ppm;

    Ok(ChannelBoundaryMetricsV1 {
        channel_index,
        seam_jump_abs,
        slope_delta_abs,
        maximum_local_step_abs,
        rms_ratio_deviation_ppm,
        passed,
    })
}

fn evaluate_boundary_v1(
    audio: &AudioPcmBufferV1,
    policy: &AudioBoundaryThresholdV1,
) -> Result<AudioBoundaryResultV1, AudioBoundaryErrorV1> {
    let mut channels = Vec::with_capacity(usize::from(audio.channel_count));
    for channel_index in 0..audio.channel_count {
        channels.push(boundary_metrics_for_channel(audio, policy, channel_index)?);
    }
    let passed = channels.iter().all(|metrics| metrics.passed);
    Ok(AudioBoundaryResultV1 {
        boundary_id: policy.boundary_id.clone(),
        boundary_frame: policy.boundary_frame,
        channels,
        passed,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AudioBoundaryRequestPayloadV1<'a> {
    schema_version: &'a str,
    operation_id: &'a str,
    source_lineage_sha256: &'a str,
    operation_plan_sha256: &'a str,
    source_pcm_sha256: &'a str,
    expected_output_pcm_sha256: &'a str,
    sample_rate_hz: u32,
    channel_count: u16,
    frame_count: u64,
    edit_windows: &'a [AudioEditWindowV1],
    protected_regions: &'a [AudioProtectedRegionV1],
    boundaries: &'a [AudioBoundaryThresholdV1],
}

fn request_sha256_v1(request: &AudioBoundaryRequestV1) -> Result<String, AudioBoundaryErrorV1> {
    let payload = AudioBoundaryRequestPayloadV1 {
        schema_version: &request.schema_version,
        operation_id: &request.operation_id,
        source_lineage_sha256: &request.source_lineage_sha256,
        operation_plan_sha256: &request.operation_plan_sha256,
        source_pcm_sha256: &request.source_pcm_sha256,
        expected_output_pcm_sha256: &request.expected_output_pcm_sha256,
        sample_rate_hz: request.sample_rate_hz,
        channel_count: request.channel_count,
        frame_count: request.frame_count,
        edit_windows: &request.edit_windows,
        protected_regions: &request.protected_regions,
        boundaries: &request.boundaries,
    };
    let encoded = serde_json::to_vec(&payload).map_err(|_| AudioBoundaryErrorV1::InvalidRequest)?;
    Ok(digest(REQUEST_DOMAIN, &encoded))
}

fn receipt_sha256_v1(payload: &AudioBoundaryReceiptPayloadV1) -> Result<String, AudioBoundaryErrorV1> {
    let encoded = serde_json::to_vec(payload).map_err(|_| AudioBoundaryErrorV1::ReceiptDigestMismatch)?;
    Ok(digest(RECEIPT_DOMAIN, &encoded))
}

/// Verify exact bytes, protected regions and predeclared boundary features.
/// Boundary threshold failures are retained in a negative receipt. Structural
/// corruption (wrong bytes, unauthorized changes or protected-region drift)
/// returns an error rather than a green receipt.
pub fn evaluate_audio_edit_boundary_v1(
    request: &AudioBoundaryRequestV1,
    source: &AudioPcmBufferV1,
    output: &AudioPcmBufferV1,
) -> Result<AudioBoundaryReceiptV1, AudioBoundaryErrorV1> {
    validate_request_v1(request)?;
    let source_frames = source.frame_count()?;
    let output_frames = output.frame_count()?;
    if source.sample_rate_hz != request.sample_rate_hz
        || output.sample_rate_hz != request.sample_rate_hz
        || source.channel_count != request.channel_count
        || output.channel_count != request.channel_count
    {
        return Err(AudioBoundaryErrorV1::AudioMetadataMismatch);
    }
    if source_frames != request.frame_count || output_frames != request.frame_count {
        return Err(AudioBoundaryErrorV1::FrameCountMismatch);
    }

    let source_digest = audio_pcm_sha256_v1(source)?;
    if source_digest != request.source_pcm_sha256 {
        return Err(AudioBoundaryErrorV1::SourceDigestMismatch);
    }
    let output_digest = audio_pcm_sha256_v1(output)?;
    if output_digest != request.expected_output_pcm_sha256 {
        return Err(AudioBoundaryErrorV1::OutputDigestMismatch);
    }

    let mut protected_results = Vec::with_capacity(request.protected_regions.len());
    for region in &request.protected_regions {
        let source_region = region_sha256_v1(
            source,
            &region.region_id,
            region.start_frame,
            region.end_frame,
        )?;
        let output_region = region_sha256_v1(
            output,
            &region.region_id,
            region.start_frame,
            region.end_frame,
        )?;
        let exact_match = source_region == output_region
            && source_region == region.expected_region_sha256;
        protected_results.push(ProtectedRegionResultV1 {
            region_id: region.region_id.clone(),
            source_region_sha256: source_region,
            output_region_sha256: output_region,
            exact_match,
        });
        if !exact_match {
            return Err(AudioBoundaryErrorV1::ProtectedRegionMismatch);
        }
    }
    let changed_frame_count = changed_frame_count_and_validate_support(request, source, output)?;

    let mut boundary_results = Vec::with_capacity(request.boundaries.len());
    for boundary in &request.boundaries {
        boundary_results.push(evaluate_boundary_v1(output, boundary)?);
    }
    let passed = protected_results.iter().all(|region| region.exact_match)
        && boundary_results.iter().all(|boundary| boundary.passed);
    let payload = AudioBoundaryReceiptPayloadV1 {
        schema_version: AUDIO_BOUNDARY_RECEIPT_SCHEMA_V1.to_owned(),
        request_sha256: request_sha256_v1(request)?,
        source_pcm_sha256: source_digest,
        output_pcm_sha256: output_digest,
        frame_count: request.frame_count,
        changed_frame_count,
        protected_regions: protected_results,
        boundaries: boundary_results,
        passed,
    };
    let receipt_sha256 = receipt_sha256_v1(&payload)?;
    Ok(AudioBoundaryReceiptV1 {
        payload,
        receipt_sha256,
    })
}

pub fn verify_audio_boundary_receipt_v1(
    request: &AudioBoundaryRequestV1,
    source: &AudioPcmBufferV1,
    output: &AudioPcmBufferV1,
    receipt: &AudioBoundaryReceiptV1,
) -> Result<(), AudioBoundaryErrorV1> {
    let expected = evaluate_audio_edit_boundary_v1(request, source, output)?;
    if expected != *receipt {
        return Err(AudioBoundaryErrorV1::RequestBindingMismatch);
    }
    if receipt.payload.schema_version != AUDIO_BOUNDARY_RECEIPT_SCHEMA_V1
        || receipt_sha256_v1(&receipt.payload)? != receipt.receipt_sha256
    {
        return Err(AudioBoundaryErrorV1::ReceiptDigestMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(value: u64) -> String {
        format!("{value:064x}")
    }

    fn audio(samples: Vec<i32>, channels: u16) -> AudioPcmBufferV1 {
        AudioPcmBufferV1 {
            sample_rate_hz: 48_000,
            channel_count: channels,
            samples_interleaved_s32: samples,
        }
    }

    fn base_mono() -> AudioPcmBufferV1 {
        audio(vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], 1)
    }

    fn request_for(source: &AudioPcmBufferV1, output: &AudioPcmBufferV1) -> AudioBoundaryRequestV1 {
        let protected_digest = region_sha256_v1(source, "protected-head", 0, 3).unwrap();
        AudioBoundaryRequestV1 {
            schema_version: AUDIO_BOUNDARY_REQUEST_SCHEMA_V1.to_owned(),
            operation_id: "replace-hit-001".to_owned(),
            source_lineage_sha256: h(1),
            operation_plan_sha256: h(2),
            source_pcm_sha256: audio_pcm_sha256_v1(source).unwrap(),
            expected_output_pcm_sha256: audio_pcm_sha256_v1(output).unwrap(),
            sample_rate_hz: 48_000,
            channel_count: 1,
            frame_count: 12,
            edit_windows: vec![AudioEditWindowV1 {
                start_frame: 3,
                end_frame: 9,
            }],
            protected_regions: vec![AudioProtectedRegionV1 {
                region_id: "protected-head".to_owned(),
                start_frame: 0,
                end_frame: 3,
                expected_region_sha256: protected_digest,
            }],
            boundaries: vec![
                AudioBoundaryThresholdV1 {
                    boundary_id: "boundary-left".to_owned(),
                    boundary_frame: 3,
                    left_window_frames: 2,
                    right_window_frames: 2,
                    maximum_seam_jump_abs: 1_000,
                    maximum_slope_delta_abs: 1_000,
                    maximum_local_step_abs: 1_000,
                    maximum_rms_ratio_deviation_ppm: 1_000_000,
                },
                AudioBoundaryThresholdV1 {
                    boundary_id: "boundary-right".to_owned(),
                    boundary_frame: 9,
                    left_window_frames: 2,
                    right_window_frames: 2,
                    maximum_seam_jump_abs: 1_000,
                    maximum_slope_delta_abs: 1_000,
                    maximum_local_step_abs: 1_000,
                    maximum_rms_ratio_deviation_ppm: 1_000_000,
                },
            ],
        }
    }

    #[test]
    fn untouched_regions_are_exact_and_authorized_changes_are_counted() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[4] = 45;
        output.samples_interleaved_s32[7] = 75;
        let request = request_for(&source, &output);
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        assert_eq!(receipt.payload.changed_frame_count, 2);
        assert!(receipt.payload.protected_regions.iter().all(|region| region.exact_match));
        assert!(receipt.payload.passed);
        verify_audio_boundary_receipt_v1(&request, &source, &output, &receipt).unwrap();
    }

    #[test]
    fn changed_pcm_outside_authorized_window_is_rejected() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[10] = 111;
        let request = request_for(&source, &output);
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &source, &output),
            Err(AudioBoundaryErrorV1::UnauthorizedFrameMutation)
        );
    }

    #[test]
    fn protected_region_digest_change_is_rejected() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[1] = 999;
        let mut request = request_for(&source, &source);
        request.expected_output_pcm_sha256 = audio_pcm_sha256_v1(&output).unwrap();
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &source, &output),
            Err(AudioBoundaryErrorV1::ProtectedRegionMismatch)
        );
    }

    #[test]
    fn impulse_at_edit_boundary_fails_frozen_jump_threshold() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[3] = 1_000_000;
        let mut request = request_for(&source, &output);
        for boundary in &mut request.boundaries {
            if boundary.boundary_id == "boundary-left" {
                boundary.maximum_seam_jump_abs = 100;
                boundary.maximum_local_step_abs = 100;
            }
        }
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        assert!(!receipt.payload.passed);
        let left = receipt.payload.boundaries.iter().find(|b| b.boundary_id == "boundary-left").unwrap();
        assert!(!left.passed);
        assert!(left.channels[0].seam_jump_abs > 100);
    }

    #[test]
    fn channel_specific_seam_failure_is_not_averaged_away() {
        let mut samples = Vec::new();
        for frame in 0..12_i32 {
            samples.push(frame * 10);
            samples.push(frame * 10);
        }
        let source = audio(samples, 2);
        let mut output = source.clone();
        output.samples_interleaved_s32[6] = 2_000_000;
        let mut request = request_for(&source, &output);
        request.channel_count = 2;
        request.protected_regions[0].expected_region_sha256 =
            region_sha256_v1(&source, "protected-head", 0, 3).unwrap();
        for boundary in &mut request.boundaries {
            if boundary.boundary_id == "boundary-left" {
                boundary.maximum_seam_jump_abs = 100;
            }
        }
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        let left = receipt.payload.boundaries.iter().find(|b| b.boundary_id == "boundary-left").unwrap();
        assert!(!left.passed);
        assert!(left.channels.iter().any(|channel| !channel.passed));
    }

    #[test]
    fn request_rejects_protected_region_inside_edit_support() {
        let source = base_mono();
        let request = request_for(&source, &source);
        let mut invalid = request.clone();
        invalid.protected_regions.push(AudioProtectedRegionV1 {
            region_id: "protected-middle".to_owned(),
            start_frame: 4,
            end_frame: 5,
            expected_region_sha256: h(77),
        });
        invalid.protected_regions.sort_by(|a, b| a.region_id.cmp(&b.region_id));
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&invalid, &source, &source),
            Err(AudioBoundaryErrorV1::ProtectedRegionOverlapsEditSupport)
        );
    }

    #[test]
    fn boundary_policy_must_name_an_edit_window_edge() {
        let source = base_mono();
        let request = request_for(&source, &source);
        let mut invalid = request.clone();
        invalid.boundaries[0].boundary_frame = 5;
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&invalid, &source, &source),
            Err(AudioBoundaryErrorV1::InvalidRegion)
        );
    }

    #[test]
    fn boundary_policy_is_frozen_and_receipt_tampering_is_detected() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[4] = 45;
        let request = request_for(&source, &output);
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        let mut altered = receipt.clone();
        altered.payload.changed_frame_count += 1;
        assert_eq!(
            verify_audio_boundary_receipt_v1(&request, &source, &output, &altered),
            Err(AudioBoundaryErrorV1::RequestBindingMismatch)
        );
        let mut changed_policy = request.clone();
        changed_policy.boundaries[0].maximum_seam_jump_abs += 1;
        assert_eq!(
            verify_audio_boundary_receipt_v1(&changed_policy, &source, &output, &receipt),
            Err(AudioBoundaryErrorV1::RequestBindingMismatch)
        );
    }

    #[test]
    fn source_and_output_pcm_digests_are_bound() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[4] = 45;
        let request = request_for(&source, &output);
        let mut wrong_source = source.clone();
        wrong_source.samples_interleaved_s32[0] = 1;
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &wrong_source, &output),
            Err(AudioBoundaryErrorV1::SourceDigestMismatch)
        );
        let mut wrong_output = output.clone();
        wrong_output.samples_interleaved_s32[5] = 99;
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &source, &wrong_output),
            Err(AudioBoundaryErrorV1::OutputDigestMismatch)
        );
    }

    #[test]
    fn stereo_interleaving_and_metadata_are_validated() {
        let source = audio(vec![0, 1, 2], 2);
        assert_eq!(source.frame_count(), Err(AudioBoundaryErrorV1::InvalidAudioBuffer));
        let source = base_mono();
        let output = audio(source.samples_interleaved_s32.clone(), 2);
        let request = request_for(&source, &source);
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &source, &output),
            Err(AudioBoundaryErrorV1::AudioMetadataMismatch)
        );
    }

    #[test]
    fn boundary_feature_report_is_deterministic() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[4] = 45;
        let request = request_for(&source, &output);
        let first = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        let second = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.receipt_sha256, second.receipt_sha256);
    }

    #[test]
    fn abrupt_zero_fill_at_both_edit_edges_fails_boundary_policy() {
        let source = base_mono();
        let mut output = source.clone();
        for frame in 3..9 {
            output.samples_interleaved_s32[frame] = 0;
        }
        let mut request = request_for(&source, &output);
        for boundary in &mut request.boundaries {
            boundary.maximum_seam_jump_abs = 15;
            boundary.maximum_local_step_abs = 15;
        }
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        assert!(!receipt.payload.passed);
        assert!(receipt.payload.boundaries.iter().any(|boundary| !boundary.passed));
    }

    #[test]
    fn polarity_inversion_inside_edit_support_fails_boundary_policy() {
        let source = base_mono();
        let mut output = source.clone();
        for frame in 3..9 {
            output.samples_interleaved_s32[frame] = -source.samples_interleaved_s32[frame];
        }
        let mut request = request_for(&source, &output);
        for boundary in &mut request.boundaries {
            boundary.maximum_seam_jump_abs = 15;
            boundary.maximum_slope_delta_abs = 15;
        }
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        assert!(!receipt.payload.passed);
        assert!(receipt.payload.boundaries.iter().any(|boundary| !boundary.passed));
    }

    #[test]
    fn one_frame_latency_shift_inside_edit_window_is_detected_at_seam() {
        let source = base_mono();
        let mut output = source.clone();
        for frame in 3..9 {
            output.samples_interleaved_s32[frame] = source.samples_interleaved_s32[frame - 1];
        }
        let mut request = request_for(&source, &output);
        request.boundaries[1].maximum_seam_jump_abs = 15;
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        let right = receipt.payload.boundaries.iter()
            .find(|boundary| boundary.boundary_id == "boundary-right")
            .unwrap();
        assert!(!right.passed);
        assert!(right.channels[0].seam_jump_abs > 15);
    }

    #[test]
    fn truncated_output_is_rejected_even_when_its_digest_is_frozen() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32.pop();
        let request = request_for(&source, &output);
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &source, &output),
            Err(AudioBoundaryErrorV1::FrameCountMismatch)
        );
    }

    #[test]
    fn channel_swap_fails_against_frozen_expected_output_bytes() {
        let mut samples = Vec::new();
        for frame in 0..12_i32 {
            samples.push(frame * 10);
            samples.push(10_000 + frame * 100);
        }
        let source = audio(samples, 2);
        let expected_output = source.clone();
        let mut request = request_for(&source, &expected_output);
        request.channel_count = 2;

        let mut swapped = source.clone();
        for frame in 0..12_usize {
            swapped.samples_interleaved_s32.swap(frame * 2, frame * 2 + 1);
        }
        assert_eq!(
            evaluate_audio_edit_boundary_v1(&request, &source, &swapped),
            Err(AudioBoundaryErrorV1::OutputDigestMismatch)
        );
    }

    #[test]
    fn isolated_transient_near_edit_seam_fails_local_step_limit() {
        let source = base_mono();
        let mut output = source.clone();
        output.samples_interleaved_s32[4] = 1_000_000;
        let mut request = request_for(&source, &output);
        request.boundaries[0].maximum_local_step_abs = 100;
        let receipt = evaluate_audio_edit_boundary_v1(&request, &source, &output).unwrap();
        let left = receipt.payload.boundaries.iter()
            .find(|boundary| boundary.boundary_id == "boundary-left")
            .unwrap();
        assert!(!left.passed);
        assert!(left.channels[0].maximum_local_step_abs > 100);
    }

}
