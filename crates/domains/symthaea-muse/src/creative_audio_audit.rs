// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Waveform-integrity evidence for generated creative audio.
//!
//! These measurements answer narrow engineering questions (empty, silent,
//! non-finite, out-of-range, near-ceiling, and sustained activity). They do
//! not evaluate musicality, loudness compliance, genre fit, or listener value.
//! Near-ceiling sample peaks are not a true-peak measurement.

use crate::AudioData;
use serde::{Deserialize, Serialize};

const MAX_WINDOW_COUNT: usize = 4096;

/// Configuration for deterministic audio-signal measurements.
///
/// Thresholds are normalized sample amplitudes. They must be declared
/// alongside reports and are not universal perceptual or delivery standards.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CreativeAudioAuditConfigV1 {
    pub window_count: usize,
    pub sample_silence_floor: f32,
    pub active_window_rms_floor: f32,
    pub near_ceiling_threshold: f32,
}

impl Default for CreativeAudioAuditConfigV1 {
    fn default() -> Self {
        Self {
            window_count: 64,
            sample_silence_floor: 1.0e-5,
            active_window_rms_floor: 1.0e-4,
            near_ceiling_threshold: 0.999,
        }
    }
}

/// Why audio-signal measurements could not be computed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreativeAudioAuditError {
    EmptyAudio,
    InvalidSampleRate,
    InvalidWindowCount,
    InvalidSampleSilenceFloor,
    InvalidActiveWindowRmsFloor,
    InvalidNearCeilingThreshold,
}

/// Descriptive measurements over decoded audio frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreativeAudioAuditV1 {
    pub schema_version: u32,
    pub sample_rate_hz: u32,
    pub frame_count: usize,
    pub channel_count: usize,
    pub duration_secs: f64,
    pub non_finite_sample_count: usize,
    pub out_of_range_sample_count: usize,
    pub sample_peak: f32,
    pub sample_rms: f32,
    /// Per-channel arithmetic means after non-finite samples are replaced by zero.
    pub channel_dc_offsets: Vec<f32>,
    pub near_ceiling_sample_ratio: f32,
    pub near_zero_sample_ratio: f32,
    pub active_window_coverage_ratio: f32,
    pub maximum_silent_run_secs: f64,
    /// A declared-threshold diagnostic, not a claim about human audibility.
    pub effectively_silent: bool,
    pub evaluated_window_count: usize,
}

/// Audit already-rendered audio without applying normalization or other DSP.
///
/// Non-finite float samples are counted and treated as zero only for aggregate
/// arithmetic, so the report remains inspectable. Treat a nonzero
/// non-finite_sample_count as an integrity failure. Integer PCM is normalized
/// by 32768, preserving the asymmetric signed-i16 range. This function does
/// not resample or re-render.
pub fn audit_creative_audio(
    audio: &AudioData,
    sample_rate_hz: u32,
    config: CreativeAudioAuditConfigV1,
) -> Result<CreativeAudioAuditV1, CreativeAudioAuditError> {
    if sample_rate_hz == 0 {
        return Err(CreativeAudioAuditError::InvalidSampleRate);
    }
    if audio.is_empty() {
        return Err(CreativeAudioAuditError::EmptyAudio);
    }
    if config.window_count == 0 || config.window_count > MAX_WINDOW_COUNT {
        return Err(CreativeAudioAuditError::InvalidWindowCount);
    }
    if !config.sample_silence_floor.is_finite()
        || config.sample_silence_floor < 0.0
        || config.sample_silence_floor >= 1.0
    {
        return Err(CreativeAudioAuditError::InvalidSampleSilenceFloor);
    }
    if !config.active_window_rms_floor.is_finite()
        || config.active_window_rms_floor < 0.0
        || config.active_window_rms_floor >= 1.0
    {
        return Err(CreativeAudioAuditError::InvalidActiveWindowRmsFloor);
    }
    if !config.near_ceiling_threshold.is_finite()
        || config.near_ceiling_threshold <= 0.0
        || config.near_ceiling_threshold > 1.0
    {
        return Err(CreativeAudioAuditError::InvalidNearCeilingThreshold);
    }

    let frame_count = audio.len();
    let channel_count = match audio {
        AudioData::StereoF32(_) => 2,
        AudioData::I16(_) | AudioData::F32(_) => 1,
    };
    let sample_count = frame_count * channel_count;
    let mut non_finite_sample_count = 0usize;
    let mut out_of_range_sample_count = 0usize;
    let mut near_ceiling_sample_count = 0usize;
    let mut near_zero_sample_count = 0usize;
    let mut sample_peak = 0.0_f32;
    let mut square_sum = 0.0_f64;
    let mut channel_sums = vec![0.0_f64; channel_count];
    let mut maximum_silent_run_frames = 0usize;
    let mut current_silent_run_frames = 0usize;

    for frame_index in 0..frame_count {
        let frame = sample_frame(audio, frame_index);
        let mut frame_silent = true;
        for (channel, sample) in frame.iter().copied().enumerate().take(channel_count) {
            let value = if sample.is_finite() {
                sample
            } else {
                non_finite_sample_count += 1;
                0.0
            };
            let magnitude = value.abs();
            if magnitude > 1.0 {
                out_of_range_sample_count += 1;
            }
            if magnitude >= config.near_ceiling_threshold {
                near_ceiling_sample_count += 1;
            }
            if magnitude <= config.sample_silence_floor {
                near_zero_sample_count += 1;
            } else {
                frame_silent = false;
            }
            sample_peak = sample_peak.max(magnitude);
            square_sum += f64::from(value) * f64::from(value);
            channel_sums[channel] += f64::from(value);
        }

        if frame_silent {
            current_silent_run_frames += 1;
            maximum_silent_run_frames =
                maximum_silent_run_frames.max(current_silent_run_frames);
        } else {
            current_silent_run_frames = 0;
        }
    }

    let evaluated_window_count = config.window_count.min(frame_count);
    let mut active_window_count = 0usize;
    for window in 0..evaluated_window_count {
        let start = window * frame_count / evaluated_window_count;
        let end = (window + 1) * frame_count / evaluated_window_count;
        let mut window_square_sum = 0.0_f64;
        for frame_index in start..end {
            for sample in sample_frame(audio, frame_index)
                .iter()
                .copied()
                .take(channel_count)
            {
                let value = if sample.is_finite() { sample } else { 0.0 };
                window_square_sum += f64::from(value) * f64::from(value);
            }
        }
        let window_sample_count = (end - start) * channel_count;
        let window_rms = (window_square_sum / window_sample_count as f64).sqrt();
        if window_rms > 0.0 && window_rms >= f64::from(config.active_window_rms_floor) {
            active_window_count += 1;
        }
    }

    let total_sample_frames = frame_count as f64;
    let sample_rms = (square_sum / sample_count as f64).sqrt() as f32;
    let channel_dc_offsets = channel_sums
        .into_iter()
        .map(|sum| (sum / total_sample_frames) as f32)
        .collect();

    Ok(CreativeAudioAuditV1 {
        schema_version: 1,
        sample_rate_hz,
        frame_count,
        channel_count,
        duration_secs: total_sample_frames / f64::from(sample_rate_hz),
        non_finite_sample_count,
        out_of_range_sample_count,
        sample_peak,
        sample_rms,
        channel_dc_offsets,
        near_ceiling_sample_ratio: near_ceiling_sample_count as f32 / sample_count as f32,
        near_zero_sample_ratio: near_zero_sample_count as f32 / sample_count as f32,
        active_window_coverage_ratio: active_window_count as f32 / evaluated_window_count as f32,
        maximum_silent_run_secs: maximum_silent_run_frames as f64 / f64::from(sample_rate_hz),
        effectively_silent: sample_rms <= config.active_window_rms_floor,
        evaluated_window_count,
    })
}

/// Convert one source frame to normalized floating-point channels.
fn sample_frame(audio: &AudioData, frame_index: usize) -> [f32; 2] {
    match audio {
        AudioData::I16(samples) => {
            let sample = f32::from(samples[frame_index]) / 32768.0;
            [sample, sample]
        }
        AudioData::F32(samples) => {
            let sample = samples[frame_index];
            [sample, sample]
        }
        AudioData::StereoF32(samples) => samples[frame_index],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_empty_audio_and_invalid_measurement_configuration() {
        assert_eq!(
            audit_creative_audio(
                &AudioData::F32(Vec::new()),
                48_000,
                CreativeAudioAuditConfigV1::default()
            ),
            Err(CreativeAudioAuditError::EmptyAudio)
        );
        assert_eq!(
            audit_creative_audio(
                &AudioData::F32(vec![0.1]),
                0,
                CreativeAudioAuditConfigV1::default()
            ),
            Err(CreativeAudioAuditError::InvalidSampleRate)
        );
        assert_eq!(
            audit_creative_audio(
                &AudioData::F32(vec![0.1]),
                48_000,
                CreativeAudioAuditConfigV1 {
                    window_count: 0,
                    ..Default::default()
                }
            ),
            Err(CreativeAudioAuditError::InvalidWindowCount)
        );
    }

    #[test]
    fn reports_exact_silence_without_calling_it_a_quality_score() {
        let report = audit_creative_audio(
            &AudioData::StereoF32(vec![[0.0, 0.0]; 48_000]),
            48_000,
            CreativeAudioAuditConfigV1::default(),
        )
        .unwrap();

        assert_eq!(report.frame_count, 48_000);
        assert_eq!(report.channel_count, 2);
        assert_eq!(report.duration_secs, 1.0);
        assert_eq!(report.sample_peak, 0.0);
        assert_eq!(report.sample_rms, 0.0);
        assert_eq!(report.active_window_coverage_ratio, 0.0);
        assert_eq!(report.maximum_silent_run_secs, 1.0);
        assert!(report.effectively_silent);
    }

    #[test]
    fn measures_peak_rms_and_dc_offset_for_constant_signal() {
        let report = audit_creative_audio(
            &AudioData::F32(vec![0.25; 8]),
            8,
            CreativeAudioAuditConfigV1 {
                window_count: 4,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(report.sample_peak, 0.25);
        assert_eq!(report.sample_rms, 0.25);
        assert_eq!(report.channel_dc_offsets, vec![0.25]);
        assert_eq!(report.active_window_coverage_ratio, 1.0);
        assert_eq!(report.maximum_silent_run_secs, 0.0);
    }

    #[test]
    fn reports_non_finite_and_out_of_range_samples_without_hiding_them() {
        let report = audit_creative_audio(
            &AudioData::F32(vec![0.5, f32::NAN, f32::INFINITY, -1.25]),
            4,
            CreativeAudioAuditConfigV1::default(),
        )
        .unwrap();

        assert_eq!(report.non_finite_sample_count, 2);
        assert_eq!(report.out_of_range_sample_count, 1);
        assert_eq!(report.sample_peak, 1.25);
        assert!(
            (report.sample_rms - ((0.25_f64 + 1.5625) / 4.0).sqrt() as f32).abs() < 1e-6
        );
    }

    #[test]
    fn measures_near_ceiling_samples_without_claiming_true_peak() {
        let report = audit_creative_audio(
            &AudioData::I16(vec![0, 32_760, -32_768, 100]),
            4,
            CreativeAudioAuditConfigV1 {
                window_count: 2,
                near_ceiling_threshold: 0.999,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(report.near_ceiling_sample_ratio, 0.5);
        assert_eq!(report.out_of_range_sample_count, 0);
        assert!(!report.effectively_silent);
    }

    #[test]
    fn silence_and_active_windows_are_independent_of_channel_layout() {
        let mono = AudioData::F32(vec![0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2, 0.2]);
        let stereo = AudioData::StereoF32(vec![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.2, 0.2],
            [0.2, 0.2],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.2, 0.2],
            [0.2, 0.2],
        ]);
        let config = CreativeAudioAuditConfigV1 {
            window_count: 4,
            ..Default::default()
        };
        let m = audit_creative_audio(&mono, 8, config).unwrap();
        let s = audit_creative_audio(&stereo, 8, config).unwrap();

        assert_eq!(m.active_window_coverage_ratio, s.active_window_coverage_ratio);
        assert_eq!(m.maximum_silent_run_secs, s.maximum_silent_run_secs);
        assert_eq!(m.sample_peak, s.sample_peak);
        assert_eq!(m.sample_rms, s.sample_rms);
    }

    #[test]
    fn handles_window_counts_larger_than_frame_count_without_empty_bins() {
        let report = audit_creative_audio(
            &AudioData::F32(vec![0.0, 0.5]),
            2,
            CreativeAudioAuditConfigV1 {
                window_count: 64,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(report.evaluated_window_count, 2);
        assert_eq!(report.active_window_coverage_ratio, 0.5);
    }

    #[test]
    fn audits_audio_from_the_real_muse_renderer() {
        let notes = [crate::Note {
            frequency: 440.0,
            start_time: 0.05,
            duration: 0.7,
            velocity: 0.7,
        }];
        let rendered = crate::synth::render_notes(
            &notes,
            44_100,
            44_100,
            &crate::MusicalState::default(),
        );
        let report = audit_creative_audio(
            &rendered,
            44_100,
            CreativeAudioAuditConfigV1::default(),
        )
        .unwrap();

        assert_eq!(report.non_finite_sample_count, 0);
        assert_eq!(report.out_of_range_sample_count, 0);
        assert!(!report.effectively_silent);
        assert!(report.sample_rms > 1.0e-4);
        assert!(report.active_window_coverage_ratio > 0.0);
    }

    #[test]
    fn detects_i16_full_scale_and_normalizes_negative_full_scale() {
        let report = audit_creative_audio(
            &AudioData::I16(vec![i16::MIN, i16::MAX]),
            2,
            CreativeAudioAuditConfigV1::default(),
        )
        .unwrap();

        assert_eq!(report.sample_peak, 1.0);
        assert_eq!(report.near_ceiling_sample_ratio, 1.0);
        assert!(report.channel_dc_offsets[0] < 0.0);
    }
}
