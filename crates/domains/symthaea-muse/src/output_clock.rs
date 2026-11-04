// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned output-device clock anchors and deterministic frame-clock fitting.
//!
//! This module is independent of CPAL so persisted evidence can be replayed
//! and validated without opening an audio device. CPAL integration records
//! native stream timestamps as anchors; this module fits the mapping from the
//! stream's playback clock to the cumulative output sample-frame index.
//!
//! A fit residual is not a measurement-accuracy guarantee. When the timestamp
//! source does not provide a qualified uncertainty bound, the result remains
//! point-estimate-only and must not be promoted to an absolute device-latency
//! or acoustic-output claim.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const OUTPUT_CLOCK_ANCHOR_SCHEMA_V1: &str = "mel-output-clock-anchor-v1";
pub const OUTPUT_CLOCK_FIT_SCHEMA_V1: &str = "mel-output-clock-fit-v1";
pub const OUTPUT_CLOCK_FIT_BUNDLE_SCHEMA_V1: &str = "mel-output-clock-fit-bundle-v1";



pub const OUTPUT_CLOCK_STREAM_DESCRIPTOR_SCHEMA_V1: &str =
    "mel-output-clock-stream-descriptor-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputClockSampleFormatV1 {
    F32,
}

/// Stream-level metadata shared by all callback anchors in one CPAL stream.
///
/// The raw CPAL device ID is deliberately not persisted. Its SHA-256
/// fingerprint supports device-stratified analysis without storing the raw
/// host/device string. It remains linkable and must not be treated as
/// anonymization. The configuration digest
/// binds that fingerprint to the negotiated output format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputClockStreamDescriptorV1 {
    pub schema_version: String,
    pub stream_instance_id: u64,
    pub device_identity_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub sample_format: OutputClockSampleFormatV1,
    pub configuration_digest_sha256: String,
}

impl OutputClockStreamDescriptorV1 {
    fn expected_configuration_digest_sha256(&self) -> String {
        let canonical = format!(
            "{OUTPUT_CLOCK_STREAM_DESCRIPTOR_SCHEMA_V1}\\n{}\\n{}\\n{}\\nf32",
            self.device_identity_sha256,
            self.sample_rate_hz,
            self.channel_count
        );
        format!("{:x}", Sha256::digest(canonical.as_bytes()))
    }

    /// Validate schema, identifier digest shapes, and the canonical
    /// configuration digest without needing access to the original device ID.
    pub fn is_valid(&self) -> bool {
        self.schema_version == OUTPUT_CLOCK_STREAM_DESCRIPTOR_SCHEMA_V1
            && self.sample_rate_hz > 0
            && self.channel_count > 0
            && is_lower_hex_digest(&self.device_identity_sha256)
            && is_lower_hex_digest(&self.configuration_digest_sha256)
            && self.configuration_digest_sha256
                == self.expected_configuration_digest_sha256()
    }

    /// Build a stable, pseudonymous stream descriptor from CPAL's stable
    /// device ID and the negotiated output configuration. The SHA-256
    /// fingerprint is linkable across runs and is not anonymization.
    pub fn from_cpal_device_id(
        stream_instance_id: u64,
        device_id: &str,
        sample_rate_hz: u32,
        channel_count: u16,
    ) -> Option<Self> {
        if device_id.trim().is_empty() || sample_rate_hz == 0 || channel_count == 0 {
            return None;
        }
        let device_identity_sha256 =
            format!("{:x}", Sha256::digest(device_id.as_bytes()));
        let canonical = format!(
            "{OUTPUT_CLOCK_STREAM_DESCRIPTOR_SCHEMA_V1}\n{device_identity_sha256}\n{sample_rate_hz}\n{channel_count}\nf32"
        );
        let configuration_digest_sha256 =
            format!("{:x}", Sha256::digest(canonical.as_bytes()));

        let descriptor = Self {
            schema_version: OUTPUT_CLOCK_STREAM_DESCRIPTOR_SCHEMA_V1.to_owned(),
            stream_instance_id,
            device_identity_sha256,
            sample_rate_hz,
            channel_count,
            sample_format: OutputClockSampleFormatV1::F32,
            configuration_digest_sha256,
        };
        Some(descriptor)
    }
}

/// Versioned schema identity for a callback anchor. An enum rather than a
/// String makes the record Copy and safe to enqueue from a real-time callback
/// without heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputClockAnchorSchemaV1 {
    #[serde(rename = "mel-output-clock-anchor-v1")]
    V1,
}

/// Provenance for one output callback's native clock observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputClockAnchorSourceV1 {
    CpalOutputStreamTimestampV1,
    SyntheticFixtureV1,
}

/// One callback's mapping between its predicted playback instant and the
/// first output sample frame contained in that callback's buffer.
///
/// All stream timestamps are nanoseconds relative to the first callback
/// timestamp in this stream instance. The CPAL playback timestamp is a
/// backend prediction, not an independently observed DAC/acoustic event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputClockAnchorV1 {
    pub schema_version: OutputClockAnchorSchemaV1,
    pub source: OutputClockAnchorSourceV1,
    pub stream_instance_id: u64,
    pub anchor_index: u64,
    pub first_output_frame_index: u64,
    pub frame_count: u32,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub callback_stream_time_ns: u64,
    pub playback_stream_time_ns: u64,
    pub callback_to_playback_ns: u64,
    /// Optional externally qualified bound on the timestamp observation.
    /// CPAL does not currently expose a universal backend accuracy bound;
    /// callers must leave this null unless independently established.
    pub timestamp_uncertainty_ns: Option<u64>,
}

/// Frozen acceptance policy for a clock-fit window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputClockFitPolicyV1 {
    pub minimum_anchor_count: usize,
    pub maximum_abs_drift_ppm: f64,
    pub maximum_abs_residual_frames: f64,
    pub maximum_extrapolation_ns: u64,
}

impl Default for OutputClockFitPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_anchor_count: 4,
            maximum_abs_drift_ppm: 500.0,
            maximum_abs_residual_frames: 1.0,
            maximum_extrapolation_ns: 100_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputClockAccuracyStatusV1 {
    /// Residuals describe fit consistency only. Timestamp measurement error
    /// and regression-parameter uncertainty have not been propagated into a
    /// certified absolute frame-position interval.
    PointEstimateOnly,
}

/// Qualified only as a mathematical fit under the supplied policy.
/// Physical output and absolute device timing require external observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputClockFitV1 {
    pub schema_version: String,
    pub stream_instance_id: u64,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    /// Output frames per nanosecond of the CPAL playback stream clock.
    pub frames_per_ns: f64,
    /// Frame position at reference_playback_stream_time_ns.
    pub reference_frame_position: f64,
    pub reference_playback_stream_time_ns: u64,
    pub valid_from_playback_stream_time_ns: u64,
    pub valid_until_playback_stream_time_ns: u64,
    pub maximum_extrapolation_ns: u64,
    pub anchor_count: usize,
    pub maximum_abs_drift_ppm: f64,
    pub estimated_drift_ppm: f64,
    pub median_abs_residual_frames: f64,
    pub p95_abs_residual_frames: f64,
    pub maximum_abs_residual_frames: f64,
    pub accuracy_status: OutputClockAccuracyStatusV1,
    /// Always None in v1. Raw per-anchor uncertainty is retained, but a
    /// conservative regression-parameter uncertainty propagation is not yet
    /// qualified.
    pub reported_uncertainty_frames: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputClockFitErrorV1 {
    TooFewAnchors { actual: usize, required: usize },
    InvalidSampleRate,
    InvalidChannelCount,
    InvalidAnchorSchema { index: usize },
    InvalidAnchorFrameCount { index: usize },
    CallbackPlaybackMismatch { index: usize },
    NonMonotonicCallbackTime { index: usize },
    InvalidStreamDescriptor,
    StreamDescriptorMismatch { index: usize },
    StreamIdentityMismatch { index: usize },
    SampleRateMismatch { index: usize },
    ChannelCountMismatch { index: usize },
    NonMonotonicAnchorIndex { index: usize },
    NonMonotonicPlaybackTime { index: usize },
    NonContiguousFrameIndex { index: usize },
    FrameIndexOverlap { index: usize },
    ZeroTimeVariance,
    NonFiniteFit,
    DriftOutsidePolicy { drift_ppm: f64, maximum_abs_ppm: f64 },
    ResidualOutsidePolicy { residual_frames: f64, maximum_abs_frames: f64 },
    QueryOutsideValidity { query_ns: u64 },
}

impl std::fmt::Display for OutputClockFitErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for OutputClockFitErrorV1 {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputFrameEstimateV1 {
    pub frame_position: f64,
    /// Fit residual envelope only; it is not a physical timing accuracy bound.
    pub fit_residual_bound_frames: f64,
    pub accuracy_status: OutputClockAccuracyStatusV1,
    pub reported_uncertainty_frames: Option<f64>,
}

impl OutputClockFitV1 {
    /// Map a stream playback-clock timestamp to fractional output frame.
    /// The mapping is rejected beyond its validity interval plus the frozen
    /// maximum extrapolation horizon.
    pub fn frame_at_playback_time(
        &self,
        playback_stream_time_ns: u64,
    ) -> Result<OutputFrameEstimateV1, OutputClockFitErrorV1> {
        let lower = self
            .valid_from_playback_stream_time_ns
            .saturating_sub(self.maximum_extrapolation_ns);
        let upper = self
            .valid_until_playback_stream_time_ns
            .saturating_add(self.maximum_extrapolation_ns);
        if playback_stream_time_ns < lower || playback_stream_time_ns > upper {
            return Err(OutputClockFitErrorV1::QueryOutsideValidity {
                query_ns: playback_stream_time_ns,
            });
        }

        let delta_ns = playback_stream_time_ns as i128
            - self.reference_playback_stream_time_ns as i128;
        let frame_position =
            self.reference_frame_position + self.frames_per_ns * delta_ns as f64;
        if !frame_position.is_finite() {
            return Err(OutputClockFitErrorV1::NonFiniteFit);
        }

        Ok(OutputFrameEstimateV1 {
            frame_position,
            fit_residual_bound_frames: self.maximum_abs_residual_frames,
            accuracy_status: self.accuracy_status,
            reported_uncertainty_frames: self.reported_uncertainty_frames,
        })
    }
}

/// Deterministic ordinary-least-squares fit over one continuous stream segment.
///
/// Anchors must be ordered, frame-contiguous and from one stream/sample rate.
/// The function never silently removes outliers: every residual is checked
/// against the frozen maximum. Split the stream into a new segment after a
/// restart, sample-rate change, frame discontinuity or timestamp reset.
pub fn fit_output_clock_v1(
    anchors: &[OutputClockAnchorV1],
    policy: OutputClockFitPolicyV1,
) -> Result<OutputClockFitV1, OutputClockFitErrorV1> {
    let required_anchors = policy.minimum_anchor_count.max(2);
    if anchors.len() < required_anchors {
        return Err(OutputClockFitErrorV1::TooFewAnchors {
            actual: anchors.len(),
            required: required_anchors,
        });
    }
    if !policy.maximum_abs_drift_ppm.is_finite()
        || policy.maximum_abs_drift_ppm < 0.0
        || !policy.maximum_abs_residual_frames.is_finite()
        || policy.maximum_abs_residual_frames < 0.0
    {
        return Err(OutputClockFitErrorV1::NonFiniteFit);
    }

    for (index, anchor) in anchors.iter().enumerate() {
        if anchor.schema_version != OutputClockAnchorSchemaV1::V1 {
            return Err(OutputClockFitErrorV1::InvalidAnchorSchema { index });
        }
        if anchor.sample_rate_hz == 0 {
            return Err(OutputClockFitErrorV1::InvalidSampleRate);
        }
        if anchor.frame_count == 0 {
            return Err(OutputClockFitErrorV1::InvalidAnchorFrameCount { index });
        }
        if anchor.channel_count == 0 {
            return Err(OutputClockFitErrorV1::InvalidChannelCount);
        }
        let delta = anchor
            .playback_stream_time_ns
            .checked_sub(anchor.callback_stream_time_ns);
        if delta != Some(anchor.callback_to_playback_ns) {
            return Err(OutputClockFitErrorV1::CallbackPlaybackMismatch { index });
        }
        if index > 0 {
            let previous = &anchors[index - 1];
            if anchor.stream_instance_id != previous.stream_instance_id {
                return Err(OutputClockFitErrorV1::StreamIdentityMismatch { index });
            }
            if anchor.sample_rate_hz != previous.sample_rate_hz {
                return Err(OutputClockFitErrorV1::SampleRateMismatch { index });
            }
            if anchor.channel_count != previous.channel_count {
                return Err(OutputClockFitErrorV1::ChannelCountMismatch { index });
            }
            if anchor.anchor_index <= previous.anchor_index {
                return Err(OutputClockFitErrorV1::NonMonotonicAnchorIndex { index });
            }
            if anchor.callback_stream_time_ns <= previous.callback_stream_time_ns {
                return Err(OutputClockFitErrorV1::NonMonotonicCallbackTime { index });
            }
            if anchor.playback_stream_time_ns <= previous.playback_stream_time_ns {
                return Err(OutputClockFitErrorV1::NonMonotonicPlaybackTime { index });
            }
            let expected_frame = previous
                .first_output_frame_index
                .checked_add(previous.frame_count as u64);
            let Some(expected_frame) = expected_frame else {
                return Err(OutputClockFitErrorV1::NonFiniteFit);
            };
            if anchor.first_output_frame_index < expected_frame {
                return Err(OutputClockFitErrorV1::FrameIndexOverlap { index });
            }
            let missing_anchor_records = anchor.anchor_index - previous.anchor_index - 1;
            if missing_anchor_records == 0
                && anchor.first_output_frame_index != expected_frame
            {
                // Consecutive anchor records must describe consecutive audio
                // blocks. A gap here is a clock/frame discontinuity.
                return Err(OutputClockFitErrorV1::NonContiguousFrameIndex { index });
            }
            if missing_anchor_records > 0
                && anchor.first_output_frame_index == expected_frame
            {
                // A skipped anchor index must correspond to at least one
                // omitted non-empty callback block.
                return Err(OutputClockFitErrorV1::NonContiguousFrameIndex { index });
            }
        }
    }

    let sample_rate_hz = anchors[0].sample_rate_hz;
    let t0 = anchors[0].playback_stream_time_ns;
    let f0 = anchors[0].first_output_frame_index as f64;

    let xs = anchors
        .iter()
        .map(|anchor| (anchor.playback_stream_time_ns - t0) as f64)
        .collect::<Vec<_>>();
    let ys = anchors
        .iter()
        .map(|anchor| anchor.first_output_frame_index as f64 - f0)
        .collect::<Vec<_>>();

    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let covariance = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    if variance <= 0.0 || !variance.is_finite() {
        return Err(OutputClockFitErrorV1::ZeroTimeVariance);
    }

    let frames_per_ns = covariance / variance;
    let reference_delta_ns = mean_x;
    let reference_playback_stream_time_ns =
        t0.saturating_add(reference_delta_ns.max(0.0).round() as u64);
    let rounded_reference_delta_ns =
        reference_playback_stream_time_ns.saturating_sub(t0) as f64;
    let reference_frame_position =
        f0 + mean_y + frames_per_ns * (rounded_reference_delta_ns - reference_delta_ns);
    if !frames_per_ns.is_finite() || frames_per_ns <= 0.0 {
        return Err(OutputClockFitErrorV1::NonFiniteFit);
    }

    let nominal_frames_per_ns = sample_rate_hz as f64 / 1_000_000_000.0;
    let estimated_drift_ppm =
        (frames_per_ns / nominal_frames_per_ns - 1.0) * 1_000_000.0;
    let mut residuals = anchors
        .iter()
        .zip(&xs)
        .map(|(anchor, x)| {
            let predicted = f0 + mean_y + frames_per_ns * (x - mean_x);
            (predicted - anchor.first_output_frame_index as f64).abs()
        })
        .collect::<Vec<_>>();
    if residuals.iter().any(|residual| !residual.is_finite()) {
        return Err(OutputClockFitErrorV1::NonFiniteFit);
    }
    residuals.sort_by(f64::total_cmp);
    let maximum_abs_residual_frames = *residuals.last().unwrap_or(&0.0);
    if maximum_abs_residual_frames > policy.maximum_abs_residual_frames {
        return Err(OutputClockFitErrorV1::ResidualOutsidePolicy {
            residual_frames: maximum_abs_residual_frames,
            maximum_abs_frames: policy.maximum_abs_residual_frames,
        });
    }
    let middle = residuals.len() / 2;
    let median_abs_residual_frames = if residuals.len() % 2 == 0 {
        (residuals[middle - 1] + residuals[middle]) * 0.5
    } else {
        residuals[middle]
    };
    let p95_index = ((residuals.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(residuals.len() - 1);
    let p95_abs_residual_frames = residuals[p95_index];

    if !estimated_drift_ppm.is_finite() {
        return Err(OutputClockFitErrorV1::NonFiniteFit);
    }
    if estimated_drift_ppm.abs() > policy.maximum_abs_drift_ppm {
        return Err(OutputClockFitErrorV1::DriftOutsidePolicy {
            drift_ppm: estimated_drift_ppm,
            maximum_abs_ppm: policy.maximum_abs_drift_ppm,
        });
    }

    // Retain each raw timestamp uncertainty on its anchor, but do not
    // promote it to a frame-position bound until regression uncertainty is
    // propagated with a separately qualified method.
    let accuracy_status = OutputClockAccuracyStatusV1::PointEstimateOnly;
    let reported_uncertainty_frames = None;

    Ok(OutputClockFitV1 {
        schema_version: OUTPUT_CLOCK_FIT_SCHEMA_V1.to_owned(),
        stream_instance_id: anchors[0].stream_instance_id,
        sample_rate_hz,
        channel_count: anchors[0].channel_count,
        frames_per_ns,
        reference_frame_position,
        reference_playback_stream_time_ns,
        valid_from_playback_stream_time_ns: anchors[0].playback_stream_time_ns,
        valid_until_playback_stream_time_ns: anchors
            .last()
            .map(|anchor| anchor.playback_stream_time_ns)
            .unwrap_or(t0),
        maximum_extrapolation_ns: policy.maximum_extrapolation_ns,
        anchor_count: anchors.len(),
        maximum_abs_drift_ppm: policy.maximum_abs_drift_ppm,
        estimated_drift_ppm,
        median_abs_residual_frames,
        p95_abs_residual_frames,
        maximum_abs_residual_frames,
        accuracy_status,
        reported_uncertainty_frames,
    })
}


/// A fit bound to the stream/device/configuration descriptor that produced
/// its raw callback anchors. Persist this as the report artifact rather than
/// exporting a bare fit detached from its clock domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputClockFitBundleV1 {
    pub schema_version: String,
    pub stream_descriptor: OutputClockStreamDescriptorV1,
    pub fit: OutputClockFitV1,
}

/// Validate descriptor-to-anchor identity and fit the complete stream bundle.
pub fn fit_output_clock_bundle_v1(
    descriptor: &OutputClockStreamDescriptorV1,
    anchors: &[OutputClockAnchorV1],
    policy: OutputClockFitPolicyV1,
) -> Result<OutputClockFitBundleV1, OutputClockFitErrorV1> {
    if !descriptor.is_valid() {
        return Err(OutputClockFitErrorV1::InvalidStreamDescriptor);
    }
    for (index, anchor) in anchors.iter().enumerate() {
        if anchor.stream_instance_id != descriptor.stream_instance_id
            || anchor.sample_rate_hz != descriptor.sample_rate_hz
            || anchor.channel_count != descriptor.channel_count
        {
            return Err(OutputClockFitErrorV1::StreamDescriptorMismatch { index });
        }
    }
    let fit = fit_output_clock_v1(anchors, policy)?;
    Ok(OutputClockFitBundleV1 {
        schema_version: OUTPUT_CLOCK_FIT_BUNDLE_SCHEMA_V1.to_owned(),
        stream_descriptor: descriptor.clone(),
        fit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(index: u64, first_frame: u64, playback_ns: u64) -> OutputClockAnchorV1 {
        OutputClockAnchorV1 {
            schema_version: OutputClockAnchorSchemaV1::V1,
            source: OutputClockAnchorSourceV1::SyntheticFixtureV1,
            stream_instance_id: 7,
            anchor_index: index,
            first_output_frame_index: first_frame,
            frame_count: 480,
            sample_rate_hz: 48_000,
            channel_count: 2,
            callback_stream_time_ns: playback_ns.saturating_sub(5_000_000),
            playback_stream_time_ns: playback_ns,
            callback_to_playback_ns: 5_000_000,
            timestamp_uncertainty_ns: None,
        }
    }

    fn policy() -> OutputClockFitPolicyV1 {
        OutputClockFitPolicyV1 {
            minimum_anchor_count: 4,
            maximum_abs_drift_ppm: 100.0,
            maximum_abs_residual_frames: 0.25,
            maximum_extrapolation_ns: 20_000_000,
        }
    }

    #[test]
    fn nominal_clock_fit_maps_playback_time_to_frame_position() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        let fit = fit_output_clock_v1(&anchors, policy()).unwrap();
        let estimate = fit.frame_at_playback_time(20_000_000).unwrap();

        assert!((fit.estimated_drift_ppm).abs() < 1e-6);
        assert!((estimate.frame_position - 720.0).abs() < 1e-6);
        assert_eq!(fit.accuracy_status, OutputClockAccuracyStatusV1::PointEstimateOnly);
        assert!(estimate.reported_uncertainty_frames.is_none());
    }

    #[test]
    fn small_clock_drift_is_estimated_without_replacing_nominal_rate() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_250),
            anchor(2, 960, 25_000_500),
            anchor(3, 1440, 35_000_750),
        ];
        let fit = fit_output_clock_v1(&anchors, policy()).unwrap();

        assert!((fit.estimated_drift_ppm + 25.0).abs() < 0.1);
    }

    #[test]
    fn outlier_is_rejected_instead_of_silently_smoothed() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 27_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::ResidualOutsidePolicy { .. })
        ));
    }

    #[test]
    fn frame_discontinuity_requires_a_new_clock_segment() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 1000, 25_000_000),
            anchor(3, 1480, 35_000_000),
        ];
        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::NonContiguousFrameIndex { index: 2 })
        ));
    }


    #[test]
    fn skipped_anchor_records_do_not_masquerade_as_frame_discontinuity() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(3, 1440, 35_000_000),
            anchor(4, 1920, 45_000_000),
        ];
        let fit = fit_output_clock_v1(&anchors, policy()).unwrap();
        assert!((fit.estimated_drift_ppm).abs() < 1e-6);
        assert_eq!(fit.anchor_count, 4);
    }

    #[test]
    fn overlapping_output_frame_ranges_are_rejected_even_when_anchor_records_were_skipped() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(3, 400, 35_000_000),
            anchor(4, 880, 45_000_000),
        ];
        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::FrameIndexOverlap { index: 2 })
        ));
    }

    #[test]
    fn stale_calibration_cannot_be_extrapolated_without_limit() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        let fit = fit_output_clock_v1(&anchors, policy()).unwrap();

        assert!(matches!(
            fit.frame_at_playback_time(100_000_000),
            Err(OutputClockFitErrorV1::QueryOutsideValidity { .. })
        ));
    }

    #[test]
    fn declared_anchor_uncertainty_is_not_promoted_without_a_propagation_model() {
        let mut anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        for value in &mut anchors {
            value.timestamp_uncertainty_ns = Some(1_000);
        }
        let fit = fit_output_clock_v1(&anchors, policy()).unwrap();

        assert_eq!(
            fit.accuracy_status,
            OutputClockAccuracyStatusV1::PointEstimateOnly
        );
        assert!(fit.reported_uncertainty_frames.is_none());
    }


    #[test]
    fn positive_clock_drift_is_estimated_separately() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 14_999_750),
            anchor(2, 960, 24_999_500),
            anchor(3, 1440, 34_999_250),
        ];
        let fit = fit_output_clock_v1(&anchors, policy()).unwrap();

        assert!((fit.estimated_drift_ppm - 25.0).abs() < 0.1);
    }

    #[test]
    fn fewer_than_two_anchors_is_rejected_even_if_policy_is_misconfigured() {
        let anchors = vec![anchor(0, 0, 5_000_000)];
        let loose_policy = OutputClockFitPolicyV1 {
            minimum_anchor_count: 0,
            ..policy()
        };

        assert!(matches!(
            fit_output_clock_v1(&anchors, loose_policy),
            Err(OutputClockFitErrorV1::TooFewAnchors {
                actual: 1,
                required: 2
            })
        ));
    }

    #[test]
    fn callback_playback_delta_must_match_the_two_stream_timestamps() {
        let mut anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        anchors[2].callback_to_playback_ns += 1;

        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::CallbackPlaybackMismatch { index: 2 })
        ));
    }


    #[derive(Deserialize)]
    struct OutputClockFixtureV1 {
        schema_version: String,
        fixture_id: String,
        expected_disposition: String,
        stream_descriptor: OutputClockStreamDescriptorV1,
        anchors: Vec<OutputClockAnchorV1>,
    }

    #[test]
    fn checked_in_clock_fixtures_match_the_rust_evaluator() {
        let fixtures = [
            include_str!("../../../../benchmarks/mel/clock/clock-exact-v1.json"),
            include_str!("../../../../benchmarks/mel/clock/clock-drift-minus-25ppm-v1.json"),
            include_str!("../../../../benchmarks/mel/clock/clock-outlier-v1.json"),
            include_str!("../../../../benchmarks/mel/clock/clock-discontinuity-v1.json"),
        ];

        for fixture_json in fixtures {
            let fixture: OutputClockFixtureV1 =
                serde_json::from_str(fixture_json).expect("fixture must parse");
            assert_eq!(
                fixture.schema_version,
                "mel-output-clock-anchor-set-v1",
                "{}",
                fixture.fixture_id
            );
            assert_eq!(
                fixture.stream_descriptor.stream_instance_id,
                fixture.anchors[0].stream_instance_id,
                "{} stream descriptor identity",
                fixture.fixture_id
            );
            assert_eq!(
                fixture.stream_descriptor.sample_rate_hz,
                fixture.anchors[0].sample_rate_hz,
                "{} stream descriptor sample rate",
                fixture.fixture_id
            );
            assert_eq!(
                fixture.stream_descriptor.channel_count,
                fixture.anchors[0].channel_count,
                "{} stream descriptor channel count",
                fixture.fixture_id
            );
            assert_eq!(
                fixture.stream_descriptor.sample_format,
                OutputClockSampleFormatV1::F32,
                "{} stream sample format",
                fixture.fixture_id
            );
            assert!(
                fixture.stream_descriptor.is_valid(),
                "{} stream descriptor digest",
                fixture.fixture_id
            );

            let result = fit_output_clock_bundle_v1(
                &fixture.stream_descriptor,
                &fixture.anchors,
                policy(),
            );
            match fixture.expected_disposition.as_str() {
                "fit" => {
                    let bundle = result.unwrap_or_else(|error| {
                        panic!("{} unexpectedly rejected: {error}", fixture.fixture_id)
                    });
                    assert_eq!(
                        bundle.schema_version,
                        OUTPUT_CLOCK_FIT_BUNDLE_SCHEMA_V1
                    );
                    assert_eq!(
                        bundle.fit.channel_count,
                        fixture.stream_descriptor.channel_count
                    );
                }
                "reject_residual" => assert!(
                    matches!(result, Err(OutputClockFitErrorV1::ResidualOutsidePolicy { .. })),
                    "{} must fail the residual gate",
                    fixture.fixture_id
                ),
                "reject_discontinuity" => assert!(
                    matches!(result, Err(OutputClockFitErrorV1::NonContiguousFrameIndex { .. })),
                    "{} must fail the frame-contiguity gate",
                    fixture.fixture_id
                ),
                other => panic!("{} has unsupported expected disposition {other}", fixture.fixture_id),
            }
        }
    }



    #[test]
    fn callback_clock_regression_is_rejected_even_when_playback_time_increases() {
        let mut anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        anchors[2].callback_stream_time_ns = 9_000_000;
        anchors[2].callback_to_playback_ns = 16_000_000;

        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::NonMonotonicCallbackTime { index: 2 })
        ));
    }

    #[test]
    fn descriptor_mismatch_prevents_fit_bundle_creation() {
        let anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        let descriptor = OutputClockStreamDescriptorV1::from_cpal_device_id(
            99,
            "different-device",
            48_000,
            2,
        ).unwrap();

        assert!(matches!(
            fit_output_clock_bundle_v1(&descriptor, &anchors, policy()),
            Err(OutputClockFitErrorV1::StreamDescriptorMismatch { index: 0 })
        ));
    }

    #[test]
    fn stream_descriptor_hashes_device_id_and_binds_output_configuration() {
        let descriptor = OutputClockStreamDescriptorV1::from_cpal_device_id(
            12,
            "synthetic-device-private-id",
            48_000,
            2,
        )
        .unwrap();
        assert_eq!(descriptor.schema_version, OUTPUT_CLOCK_STREAM_DESCRIPTOR_SCHEMA_V1);
        assert_ne!(descriptor.device_identity_sha256, "synthetic-device-private-id");
        assert_eq!(descriptor.sample_rate_hz, 48_000);
        assert_eq!(descriptor.channel_count, 2);
        assert_eq!(descriptor.sample_format, OutputClockSampleFormatV1::F32);
        assert_eq!(descriptor.configuration_digest_sha256.len(), 64);
        assert!(descriptor.is_valid());

        let known = OutputClockStreamDescriptorV1::from_cpal_device_id(
            7,
            "synthetic-device-7",
            48_000,
            2,
        )
        .unwrap();
        assert_eq!(
            known.device_identity_sha256,
            "91ecf2e58f41c6da44a5a98518827711459b170bf6bf3c62761f3e8ae281e33f"
        );
        assert_eq!(
            known.configuration_digest_sha256,
            "a18d7bb1d0c087b9502a2d858e4354c724486bc73a8648458054ba03005653aa"
        );
        let serialized = serde_json::to_string(&known).unwrap();
        assert!(!serialized.contains("synthetic-device-7"));
        let mut tampered = descriptor.clone();
        tampered.channel_count = 1;
        assert!(!tampered.is_valid());
    }

    #[test]
    fn channel_count_change_requires_a_new_clock_segment() {
        let mut anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        anchors[2].channel_count = 1;

        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::ChannelCountMismatch { index: 2 })
        ));
    }

    #[test]
    fn sample_rate_change_is_not_mixed_into_one_segment() {
        let mut anchors = vec![
            anchor(0, 0, 5_000_000),
            anchor(1, 480, 15_000_000),
            anchor(2, 960, 25_000_000),
            anchor(3, 1440, 35_000_000),
        ];
        anchors[2].sample_rate_hz = 44_100;

        assert!(matches!(
            fit_output_clock_v1(&anchors, policy()),
            Err(OutputClockFitErrorV1::SampleRateMismatch { index: 2 })
        ));
    }
}
