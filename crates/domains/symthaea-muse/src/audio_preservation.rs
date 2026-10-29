// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Localized, fail-closed checks for rendered-audio preservation.
//!
//! This module is a deterministic *measurement primitive*, not a perceptual
//! quality model. It compares protected time windows channel by channel using
//! RMS-level change and a phase-insensitive spectral-magnitude distance.
//! Thresholds are meaningful only when selected from a versioned calibration
//! artifact for the relevant renderer/domain. A calibration ID is recorded,
//! not authenticated or validated by this module.
//!
//! The caller supplies edit masks in audio-frame coordinates (one frame means
//! one sample per channel). Frames that intersect an edit mask expanded by the
//! configured boundary guard are excluded. There is no automatic alignment:
//! mismatched rates, channels, lengths, or non-finite samples are errors, never
//! implicit permission to accept. An audio digest binds the receipt to exact
//! input samples and format metadata; it does not authenticate origin.

use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Receipt/version string. Bump when metric or verdict semantics change.
pub const AUDIO_PRESERVATION_VERIFIER_VERSION: &str = "mel-mr-eval-006-audio-preservation-v0.1";

/// Interleaved PCM-like samples. The evaluator does not decode, resample,
/// loudness-normalize, or align audio; those steps would change the evidence.
#[derive(Debug, Clone, Copy)]
pub struct AudioView<'a> {
    pub interleaved_samples: &'a [f32],
    pub sample_rate_hz: u32,
    pub channels: usize,
}

/// Half-open edit interval in audio frames (not scalar interleaved samples).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameInterval {
    pub start_frame: usize,
    pub end_frame: usize,
}

/// Thresholds plus the calibration identity that justifies them.
///
/// The identifier must resolve externally to a frozen calibration report
/// containing held-out composition/renderer results. Supplying an arbitrary
/// string does not itself make the thresholds valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationConfig {
    /// Analysis window length, in audio frames; must be >= 16.
    pub frame_size: usize,
    /// Distance between analysis-window starts; must be in 1..=frame_size.
    pub hop_size: usize,
    /// Expand each edit mask by this many frames on both sides before analysis.
    /// This is a declared boundary/spillover guard, not an inferred reverb model.
    pub boundary_guard_frames: usize,
    /// Maximum allowed absolute RMS level difference, in dB.
    pub rms_delta_db_tolerance: f32,
    /// Maximum normalized L1 spectral-magnitude distance in [0, 1].
    pub spectral_distance_tolerance: f32,
    /// Minimum number of distinct protected time windows required for acceptance.
    pub minimum_protected_windows: usize,
    /// ID of the external calibration artifact; None forbids a passing verdict.
    pub calibration_id: Option<String>,
}

impl Default for PreservationConfig {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 256,
            boundary_guard_frames: 0,
            // Conservative placeholders only. A real evaluation must calibrate
            // these on a development split and freeze them before test scoring.
            rms_delta_db_tolerance: 0.5,
            spectral_distance_tolerance: 0.08,
            minimum_protected_windows: 8,
            calibration_id: None,
        }
    }
}

/// Overall operational verdict. Pass is impossible without a calibration ID
/// and the minimum number of protected windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreservationVerdict {
    Pass,
    Fail,
    Abstain,
}

/// One protected time-window/channel measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectedWindowMeasurement {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    pub rms_delta_db: f32,
    pub spectral_distance: f32,
    pub exceeded_threshold: bool,
}

/// Content-bound evidence from this deterministic measurement primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPreservationReport {
    pub verifier_version: String,
    pub calibration_id: Option<String>,
    pub reference_audio_sha256: String,
    pub candidate_audio_sha256: String,
    /// Count of unique time windows, not channel-window measurements.
    pub protected_windows: usize,
    pub audited_channel_windows: usize,
    pub violating_channel_windows: usize,
    pub worst_rms_delta_db: f32,
    pub worst_spectral_distance: f32,
    pub verdict: PreservationVerdict,
    pub reasons: Vec<String>,
    pub measurements: Vec<ProtectedWindowMeasurement>,
}

/// Invalid inputs are errors rather than implicit passes or heuristic fixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioAuditInputError {
    ZeroSampleRate,
    ZeroChannels,
    EmptyAudio,
    InterleavedLengthNotDivisibleByChannels,
    NonFiniteSample,
    FormatMismatch,
    LengthMismatch,
    InvalidEditInterval,
    InvalidConfig(&'static str),
}

impl std::fmt::Display for AudioAuditInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "sample rate must be non-zero"),
            Self::ZeroChannels => write!(f, "channel count must be non-zero"),
            Self::EmptyAudio => write!(f, "audio must contain at least one frame"),
            Self::InterleavedLengthNotDivisibleByChannels => {
                write!(f, "interleaved sample count is not divisible by channel count")
            }
            Self::NonFiniteSample => write!(f, "audio contains NaN or infinite samples"),
            Self::FormatMismatch => write!(f, "reference and candidate formats differ"),
            Self::LengthMismatch => write!(f, "reference and candidate frame counts differ"),
            Self::InvalidEditInterval => write!(f, "edit interval is empty or outside the audio"),
            Self::InvalidConfig(reason) => write!(f, "invalid preservation config: {reason}"),
        }
    }
}

impl std::error::Error for AudioAuditInputError {}

/// Audit every complete analysis window that is outside all guarded edit masks.
///
/// A protected channel-window fails if *either* metric exceeds its threshold.
/// Metrics are not averaged across time or channels: one violating protected
/// window prevents acceptance. This intentionally prioritizes leakage detection
/// over a compensatory global score.
pub fn audit_protected_audio(
    reference: AudioView<'_>,
    candidate: AudioView<'_>,
    authorized_edits: &[FrameInterval],
    config: &PreservationConfig,
) -> Result<AudioPreservationReport, AudioAuditInputError> {
    validate_view(reference)?;
    validate_view(candidate)?;
    if reference.sample_rate_hz != candidate.sample_rate_hz
        || reference.channels != candidate.channels
    {
        return Err(AudioAuditInputError::FormatMismatch);
    }
    if reference.interleaved_samples.len() != candidate.interleaved_samples.len() {
        return Err(AudioAuditInputError::LengthMismatch);
    }
    validate_config(config)?;

    let total_frames = reference.interleaved_samples.len() / reference.channels;
    for interval in authorized_edits {
        if interval.start_frame >= interval.end_frame || interval.end_frame > total_frames {
            return Err(AudioAuditInputError::InvalidEditInterval);
        }
    }

    let guarded_masks = merged_guarded_masks(
        authorized_edits,
        config.boundary_guard_frames,
        total_frames,
    );
    let mut protected_starts = Vec::new();
    if config.frame_size <= total_frames {
        let mut start = 0usize;
        while start <= total_frames - config.frame_size {
            let end = start + config.frame_size;
            let intersects_edit = guarded_masks
                .iter()
                .any(|mask| start < mask.end_frame && end > mask.start_frame);
            if !intersects_edit {
                protected_starts.push(start);
            }
            start = match start.checked_add(config.hop_size) {
                Some(next) => next,
                None => break,
            };
        }
    }

    let reference_audio_sha256 = audio_digest(reference);
    let candidate_audio_sha256 = audio_digest(candidate);
    let mut measurements = Vec::with_capacity(protected_starts.len() * reference.channels);
    let mut worst_rms_delta_db = 0.0f32;
    let mut worst_spectral_distance = 0.0f32;
    let mut violating_channel_windows = 0usize;

    if !protected_starts.is_empty() {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.frame_size);
        let window = hann_window(config.frame_size);

        for start_frame in protected_starts.iter().copied() {
            let end_frame = start_frame + config.frame_size;
            for channel in 0..reference.channels {
                let x = channel_window(reference, start_frame, config.frame_size, channel);
                let y = channel_window(candidate, start_frame, config.frame_size, channel);
                let (rms_delta_db, spectral_distance) =
                    compare_channel_windows(&x, &y, &window, fft.as_ref());
                worst_rms_delta_db = worst_rms_delta_db.max(rms_delta_db);
                worst_spectral_distance = worst_spectral_distance.max(spectral_distance);
                let exceeded_threshold = rms_delta_db > config.rms_delta_db_tolerance
                    || spectral_distance > config.spectral_distance_tolerance;
                if exceeded_threshold {
                    violating_channel_windows += 1;
                }
                measurements.push(ProtectedWindowMeasurement {
                    start_frame,
                    end_frame,
                    channel,
                    rms_delta_db,
                    spectral_distance,
                    exceeded_threshold,
                });
            }
        }
    }

    let mut reasons = Vec::new();
    if violating_channel_windows > 0 {
        reasons.push(
            "one or more protected channel-windows exceeded a configured threshold".to_owned(),
        );
    }
    if config.calibration_id.is_none() {
        reasons.push(
            "no calibration ID supplied; a non-violating run cannot be accepted".to_owned(),
        );
    }
    if protected_starts.len() < config.minimum_protected_windows {
        reasons.push(format!(
            "only {} protected windows were available; {} required",
            protected_starts.len(),
            config.minimum_protected_windows
        ));
    }

    let verdict = if violating_channel_windows > 0 {
        PreservationVerdict::Fail
    } else if config.calibration_id.is_none()
        || protected_starts.len() < config.minimum_protected_windows
    {
        PreservationVerdict::Abstain
    } else {
        PreservationVerdict::Pass
    };

    Ok(AudioPreservationReport {
        verifier_version: AUDIO_PRESERVATION_VERIFIER_VERSION.to_owned(),
        calibration_id: config.calibration_id.clone(),
        reference_audio_sha256,
        candidate_audio_sha256,
        protected_windows: protected_starts.len(),
        audited_channel_windows: measurements.len(),
        violating_channel_windows,
        worst_rms_delta_db,
        worst_spectral_distance,
        verdict,
        reasons,
        measurements,
    })
}

fn validate_view(view: AudioView<'_>) -> Result<(), AudioAuditInputError> {
    if view.sample_rate_hz == 0 {
        return Err(AudioAuditInputError::ZeroSampleRate);
    }
    if view.channels == 0 {
        return Err(AudioAuditInputError::ZeroChannels);
    }
    if view.interleaved_samples.is_empty() {
        return Err(AudioAuditInputError::EmptyAudio);
    }
    if view.interleaved_samples.len() % view.channels != 0 {
        return Err(AudioAuditInputError::InterleavedLengthNotDivisibleByChannels);
    }
    if view.interleaved_samples.iter().any(|sample| !sample.is_finite()) {
        return Err(AudioAuditInputError::NonFiniteSample);
    }
    Ok(())
}

fn validate_config(config: &PreservationConfig) -> Result<(), AudioAuditInputError> {
    if config.frame_size < 16 {
        return Err(AudioAuditInputError::InvalidConfig(
            "frame_size must be at least 16",
        ));
    }
    if config.hop_size == 0 || config.hop_size > config.frame_size {
        return Err(AudioAuditInputError::InvalidConfig(
            "hop_size must be in 1..=frame_size",
        ));
    }
    if !config.rms_delta_db_tolerance.is_finite()
        || config.rms_delta_db_tolerance < 0.0
        || config.rms_delta_db_tolerance > 240.0
    {
        return Err(AudioAuditInputError::InvalidConfig(
            "RMS dB tolerance must be finite and in 0..=240",
        ));
    }
    if !config.spectral_distance_tolerance.is_finite()
        || !(0.0..=1.0).contains(&config.spectral_distance_tolerance)
    {
        return Err(AudioAuditInputError::InvalidConfig(
            "spectral distance tolerance must be finite and in 0..=1",
        ));
    }
    if config.minimum_protected_windows == 0 {
        return Err(AudioAuditInputError::InvalidConfig(
            "minimum_protected_windows must be non-zero",
        ));
    }
    if config
        .calibration_id
        .as_ref()
        .is_some_and(|id| id.trim().is_empty())
    {
        return Err(AudioAuditInputError::InvalidConfig(
            "calibration_id must be non-empty when present",
        ));
    }
    Ok(())
}

fn merged_guarded_masks(
    intervals: &[FrameInterval],
    guard_frames: usize,
    total_frames: usize,
) -> Vec<FrameInterval> {
    let mut masks: Vec<FrameInterval> = intervals
        .iter()
        .map(|interval| FrameInterval {
            start_frame: interval.start_frame.saturating_sub(guard_frames),
            end_frame: interval.end_frame.saturating_add(guard_frames).min(total_frames),
        })
        .collect();
    masks.sort_by_key(|interval| interval.start_frame);

    let mut merged: Vec<FrameInterval> = Vec::with_capacity(masks.len());
    for mask in masks {
        if let Some(last) = merged.last_mut() {
            if mask.start_frame <= last.end_frame {
                last.end_frame = last.end_frame.max(mask.end_frame);
                continue;
            }
        }
        merged.push(mask);
    }
    merged
}

fn channel_window(
    view: AudioView<'_>,
    start_frame: usize,
    frame_size: usize,
    channel: usize,
) -> Vec<f32> {
    (start_frame..start_frame + frame_size)
        .map(|frame| view.interleaved_samples[frame * view.channels + channel])
        .collect()
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            0.5 - 0.5
                * (std::f32::consts::TAU * index as f32 / (size - 1) as f32).cos()
        })
        .collect()
}

fn compare_channel_windows(
    reference: &[f32],
    candidate: &[f32],
    window: &[f32],
    fft: &dyn Fft<f32>,
) -> (f32, f32) {
    let rms_delta_db = (rms_db(reference) - rms_db(candidate)).abs();

    let mut reference_spectrum: Vec<Complex<f32>> = reference
        .iter()
        .zip(window)
        .map(|(sample, weight)| Complex::new(sample * weight, 0.0))
        .collect();
    let mut candidate_spectrum: Vec<Complex<f32>> = candidate
        .iter()
        .zip(window)
        .map(|(sample, weight)| Complex::new(sample * weight, 0.0))
        .collect();
    fft.process(&mut reference_spectrum);
    fft.process(&mut candidate_spectrum);

    let bins = reference.len() / 2 + 1;
    let mut absolute_difference = 0.0f64;
    let mut total_magnitude = 0.0f64;
    for index in 0..bins {
        let a = reference_spectrum[index].norm() as f64;
        let b = candidate_spectrum[index].norm() as f64;
        absolute_difference += (a - b).abs();
        total_magnitude += a + b;
    }
    let spectral_distance = if total_magnitude <= f64::EPSILON {
        0.0
    } else {
        (absolute_difference / total_magnitude).clamp(0.0, 1.0) as f32
    };
    (rms_delta_db, spectral_distance)
}

fn rms_db(samples: &[f32]) -> f32 {
    let mean_square = samples
        .iter()
        .map(|sample| {
            let value = *sample as f64;
            value * value
        })
        .sum::<f64>()
        / samples.len() as f64;
    20.0 * mean_square.sqrt().max(1.0e-12).log10() as f32
}

/// SHA-256 over domain-separated, canonical little-endian f32 sample bits and
/// sample-format metadata. This detects input substitution; it does not prove
/// authorship or trusted capture.
fn audio_digest(view: AudioView<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-audio-preservation-buffer-v1\0");
    hasher.update(view.sample_rate_hz.to_le_bytes());
    hasher.update((view.channels as u64).to_le_bytes());
    hasher.update(((view.interleaved_samples.len() / view.channels) as u64).to_le_bytes());
    for sample in view.interleaved_samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frames: usize, sample_rate_hz: u32, frequency_hz: f32) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                (std::f32::consts::TAU * frequency_hz * index as f32
                    / sample_rate_hz as f32)
                    .sin()
            })
            .collect()
    }

    fn calibrated_config() -> PreservationConfig {
        PreservationConfig {
            frame_size: 128,
            hop_size: 128,
            boundary_guard_frames: 0,
            rms_delta_db_tolerance: 0.5,
            spectral_distance_tolerance: 0.08,
            minimum_protected_windows: 4,
            calibration_id: Some("synthetic-fixture-calibration-v0".to_owned()),
        }
    }

    fn view(samples: &[f32]) -> AudioView<'_> {
        AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 16_000,
            channels: 1,
        }
    }

    #[test]
    fn identical_render_passes_only_with_calibration_and_enough_coverage() {
        let samples = sine(4096, 16_000, 440.0);
        let report = audit_protected_audio(
            view(&samples),
            view(&samples),
            &[FrameInterval {
                start_frame: 1024,
                end_frame: 2048,
            }],
            &calibrated_config(),
        )
        .expect("valid identical audio");

        assert_eq!(report.verdict, PreservationVerdict::Pass);
        assert_eq!(report.violating_channel_windows, 0);
        assert_eq!(report.reference_audio_sha256, report.candidate_audio_sha256);
        assert_eq!(report.worst_rms_delta_db, 0.0);
        assert_eq!(report.worst_spectral_distance, 0.0);
    }

    #[test]
    fn one_localized_protected_mutation_fails_without_global_averaging() {
        let reference = sine(8192, 16_000, 440.0);
        let mut candidate = reference.clone();
        // Damage one 128-frame protected window while leaving the rest intact.
        candidate[1024..1152].fill(0.0);
        let report = audit_protected_audio(
            view(&reference),
            view(&candidate),
            &[FrameInterval {
                start_frame: 3072,
                end_frame: 4096,
            }],
            &calibrated_config(),
        )
        .expect("valid matched audio");

        assert_eq!(report.verdict, PreservationVerdict::Fail);
        assert!(report.violating_channel_windows >= 1);
        assert!(report.worst_spectral_distance > 0.08);
        assert_ne!(report.reference_audio_sha256, report.candidate_audio_sha256);
        assert!(report.measurements.iter().any(|measurement| {
            measurement.start_frame == 1024 && measurement.exceeded_threshold
        }));
    }

    #[test]
    fn changes_inside_the_authorized_mask_are_not_mislabelled_as_collateral() {
        let reference = sine(4096, 16_000, 440.0);
        let mut candidate = reference.clone();
        candidate[1024..2048].fill(0.0);
        let report = audit_protected_audio(
            view(&reference),
            view(&candidate),
            &[FrameInterval {
                start_frame: 1024,
                end_frame: 2048,
            }],
            &calibrated_config(),
        )
        .expect("valid matched audio");

        assert_eq!(report.verdict, PreservationVerdict::Pass);
        assert_eq!(report.violating_channel_windows, 0);
        assert!(!report.measurements.iter().any(|measurement| {
            measurement.start_frame >= 1024 && measurement.start_frame < 2048
        }));
    }

    #[test]
    fn missing_calibration_cannot_produce_a_pass() {
        let samples = sine(4096, 16_000, 440.0);
        let mut config = calibrated_config();
        config.calibration_id = None;
        let report = audit_protected_audio(
            view(&samples),
            view(&samples),
            &[],
            &config,
        )
        .expect("valid audio and config");

        assert_eq!(report.verdict, PreservationVerdict::Abstain);
        assert!(report.reasons.iter().any(|reason| reason.contains("no calibration ID")));
    }

    #[test]
    fn a_mutated_stereo_channel_is_checked_independently() {
        let mono = sine(4096, 16_000, 440.0);
        let mut reference = Vec::with_capacity(mono.len() * 2);
        let mut candidate = Vec::with_capacity(mono.len() * 2);
        for (index, sample) in mono.iter().copied().enumerate() {
            reference.push(sample);
            reference.push(sample * 0.5);
            candidate.push(sample);
            candidate.push(if (1024..1152).contains(&index) {
                0.0
            } else {
                sample * 0.5
            });
        }
        let reference_view = AudioView {
            interleaved_samples: &reference,
            sample_rate_hz: 16_000,
            channels: 2,
        };
        let candidate_view = AudioView {
            interleaved_samples: &candidate,
            sample_rate_hz: 16_000,
            channels: 2,
        };
        let report = audit_protected_audio(
            reference_view,
            candidate_view,
            &[],
            &calibrated_config(),
        )
        .expect("valid stereo buffers");

        assert_eq!(report.verdict, PreservationVerdict::Fail);
        assert!(report.measurements.iter().any(|measurement| {
            measurement.channel == 1
                && measurement.start_frame == 1024
                && measurement.exceeded_threshold
        }));
    }

    #[test]
    fn mismatched_lengths_are_rejected_instead_of_implicitly_aligned() {
        let reference = sine(4096, 16_000, 440.0);
        let candidate = sine(4095, 16_000, 440.0);
        assert_eq!(
            audit_protected_audio(
                view(&reference),
                view(&candidate),
                &[],
                &calibrated_config(),
            )
            .expect_err("different duration must fail closed"),
            AudioAuditInputError::LengthMismatch
        );
    }

    #[test]
    fn non_finite_audio_is_rejected() {
        let reference = sine(4096, 16_000, 440.0);
        let mut candidate = reference.clone();
        candidate[100] = f32::NAN;
        assert_eq!(
            audit_protected_audio(
                view(&reference),
                view(&candidate),
                &[],
                &calibrated_config(),
            )
            .expect_err("NaN must fail closed"),
            AudioAuditInputError::NonFiniteSample
        );
    }

    #[test]
    fn boundary_guard_excludes_windows_that_might_capture_expected_spillover() {
        let samples = sine(4096, 16_000, 440.0);
        let mut config = calibrated_config();
        config.boundary_guard_frames = 64;
        let report = audit_protected_audio(
            view(&samples),
            view(&samples),
            &[FrameInterval {
                start_frame: 1024,
                end_frame: 2048,
            }],
            &config,
        )
        .expect("valid guarded audio");

        assert_eq!(report.verdict, PreservationVerdict::Pass);
        assert!(report.measurements.iter().all(|measurement| {
            measurement.end_frame <= 960 || measurement.start_frame >= 2112
        }));
    }
}
