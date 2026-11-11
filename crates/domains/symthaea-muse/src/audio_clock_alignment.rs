// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provisional global audio-minus-MIDI clock alignment.
//!
//! The estimator cross-correlates MIDI note-on times with a broadband spectral
//! flux envelope. It estimates one common offset for a rendered WAV. The
//! estimate may include leading silence, file pre-roll, and average renderer
//! attack/detector delay. It is not an instrument-specific onset detector and
//! does not prove note identity, local timing fidelity, or beat-map correctness.

use crate::audio_event_observation::{
    AudioEventObservationIssueV1, AudioEventObservationWavInfoV1, decode_wav,
    expected_frame, sha256_hex, validate_midi_artifact,
};
use crate::midi_performance::MidiPerformanceArtifactV1;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use sha2::{Digest, Sha256};

pub const AUDIO_CLOCK_ALIGNMENT_SCHEMA_V1: &str = "muse-audio-clock-alignment-v1";
pub const AUDIO_CLOCK_ALIGNMENT_ALGORITHM_V1: &str = "hann-positive-spectral-flux-cross-correlation-v1";
pub const AUDIO_CLOCK_ALIGNMENT_DEFAULT_FFT_FRAMES_V1: u32 = 2_048;
pub const AUDIO_CLOCK_ALIGNMENT_DEFAULT_HOP_FRAMES_V1: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioClockAlignmentStatusV1 {
    /// A reproducible candidate was found. This is not a calibrated pass.
    Candidate,
    /// A competing correlation peak is close enough to make the result ambiguous.
    AmbiguousCandidate,
    /// The MIDI artifact has too few distinct note-on times for this estimator.
    InsufficientOnsetDiversity,
    /// The WAV does not contain enough of the expected onset sequence to estimate one global offset.
    InsufficientAudioCoverage,
    /// No positive onset evidence was measured in the WAV.
    NoPositiveEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioClockAlignmentConfigV1 {
    pub fft_window_frames: u32,
    pub hop_frames: u32,
    /// Search lower bound in milliseconds; negative values permit trimmed audio.
    pub minimum_offset_ms: i32,
    /// Search upper bound in milliseconds; positive values permit lead-in silence.
    pub maximum_offset_ms: u32,
    pub minimum_unique_onsets: u16,
    /// Competing peaks inside this many hops of the best peak are one peak family.
    pub peak_exclusion_hops: u16,
    /// Local spectral-flux baseline is sampled between these two hop radii.
    pub baseline_inner_hops: u16,
    pub baseline_outer_hops: u16,
    /// Diagnostic support interval includes contiguous scores at or above this fraction.
    /// This is not a statistical confidence level.
    pub support_fraction_basis_points: u16,
    /// A competing peak at or above this fraction of the best marks the result ambiguous.
    /// This is a fixed diagnostic heuristic, not a calibrated probability threshold.
    pub ambiguity_ratio_basis_points: u16,
}

impl Default for AudioClockAlignmentConfigV1 {
    fn default() -> Self {
        Self {
            fft_window_frames: AUDIO_CLOCK_ALIGNMENT_DEFAULT_FFT_FRAMES_V1,
            hop_frames: AUDIO_CLOCK_ALIGNMENT_DEFAULT_HOP_FRAMES_V1,
            minimum_offset_ms: -250,
            maximum_offset_ms: 2_000,
            minimum_unique_onsets: 4,
            peak_exclusion_hops: 5,
            baseline_inner_hops: 3,
            baseline_outer_hops: 16,
            support_fraction_basis_points: 9_500,
            ambiguity_ratio_basis_points: 9_800,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioClockAlignmentProfileV1 {
    pub schema: String,
    pub algorithm: String,
    pub source_score_sha256: String,
    pub midi_sha256: String,
    pub midi_manifest_sha256: String,
    pub wav_sha256: String,
    pub wav_info: AudioEventObservationWavInfoV1,
    pub config: AudioClockAlignmentConfigV1,
    pub status: AudioClockAlignmentStatusV1,
    pub note_event_count: u32,
    pub unique_expected_onset_count: u32,
    pub matched_expected_onset_count: u32,
    /// Positive means the corresponding MIDI event is observed later in WAV time.
    /// This combines file origin/pre-roll and average renderer/detector attack delay.
    pub audio_minus_midi_offset_frames: Option<i64>,
    pub audio_minus_midi_offset_ms: Option<f64>,
    /// Maximum normalized local spectral-flux support at a candidate lag.
    pub best_peak_score: Option<f64>,
    /// Best competing peak outside the local exclusion radius.
    pub competing_peak_score: Option<f64>,
    pub peak_margin: Option<f64>,
    pub competing_peak_ratio: Option<f64>,
    /// Contiguous candidate-lag support region near the winning peak.
    /// This is an empirical correlation-peak width, not a confidence interval.
    pub support_start_frames: Option<i64>,
    pub support_end_frames: Option<i64>,
    pub profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioClockAlignmentIssueV1 {
    MidiArtifactInvalid {
        issues: Vec<AudioEventObservationIssueV1>,
    },
    WavInvalid {
        issue: AudioEventObservationIssueV1,
    },
    InvalidConfig {
        field: String,
    },
    TooFewAudioFrames,
    SerializationFailed,
}

#[derive(Serialize)]
struct AudioClockAlignmentCommitmentV1<'a> {
    schema: &'a str,
    algorithm: &'a str,
    source_score_sha256: &'a str,
    midi_sha256: &'a str,
    midi_manifest_sha256: &'a str,
    wav_sha256: &'a str,
    wav_info: &'a AudioEventObservationWavInfoV1,
    config: &'a AudioClockAlignmentConfigV1,
    status: AudioClockAlignmentStatusV1,
    note_event_count: u32,
    unique_expected_onset_count: u32,
    matched_expected_onset_count: u32,
    audio_minus_midi_offset_frames: Option<i64>,
    audio_minus_midi_offset_ms: Option<f64>,
    best_peak_score: Option<f64>,
    competing_peak_score: Option<f64>,
    peak_margin: Option<f64>,
    competing_peak_ratio: Option<f64>,
    support_start_frames: Option<i64>,
    support_end_frames: Option<i64>,
}

fn domain_separated_sha256(schema: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn alignment_commitment(
    profile: &AudioClockAlignmentProfileV1,
) -> Result<String, serde_json::Error> {
    let commitment = AudioClockAlignmentCommitmentV1 {
        schema: &profile.schema,
        algorithm: &profile.algorithm,
        source_score_sha256: &profile.source_score_sha256,
        midi_sha256: &profile.midi_sha256,
        midi_manifest_sha256: &profile.midi_manifest_sha256,
        wav_sha256: &profile.wav_sha256,
        wav_info: &profile.wav_info,
        config: &profile.config,
        status: profile.status,
        note_event_count: profile.note_event_count,
        unique_expected_onset_count: profile.unique_expected_onset_count,
        matched_expected_onset_count: profile.matched_expected_onset_count,
        audio_minus_midi_offset_frames: profile.audio_minus_midi_offset_frames,
        audio_minus_midi_offset_ms: profile.audio_minus_midi_offset_ms,
        best_peak_score: profile.best_peak_score,
        competing_peak_score: profile.competing_peak_score,
        peak_margin: profile.peak_margin,
        competing_peak_ratio: profile.competing_peak_ratio,
        support_start_frames: profile.support_start_frames,
        support_end_frames: profile.support_end_frames,
    };
    Ok(domain_separated_sha256(
        AUDIO_CLOCK_ALIGNMENT_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

fn validate_config(
    config: &AudioClockAlignmentConfigV1,
) -> Result<(), AudioClockAlignmentIssueV1> {
    if config.fft_window_frames < 256
        || !config.fft_window_frames.is_power_of_two()
    {
        return Err(AudioClockAlignmentIssueV1::InvalidConfig {
            field: "fft_window_frames".into(),
        });
    }
    if config.hop_frames == 0 || config.hop_frames > config.fft_window_frames {
        return Err(AudioClockAlignmentIssueV1::InvalidConfig {
            field: "hop_frames".into(),
        });
    }
    if config.minimum_offset_ms >= 0
        || config.maximum_offset_ms == 0
        || config.minimum_unique_onsets < 2
        || config.peak_exclusion_hops == 0
        || config.baseline_inner_hops == 0
        || config.baseline_outer_hops <= config.baseline_inner_hops
        || !(5_000..10_000).contains(&config.support_fraction_basis_points)
        || !(5_000..10_000).contains(&config.ambiguity_ratio_basis_points)
    {
        return Err(AudioClockAlignmentIssueV1::InvalidConfig {
            field: "alignment_bounds_or_thresholds".into(),
        });
    }
    Ok(())
}

fn mono_samples(
    wav: &crate::audio_event_observation::DecodedWavV1,
) -> Vec<f32> {
    let channels = usize::from(wav.info.channels);
    wav.interleaved_samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn spectral_flux(
    samples: &[f32],
    fft_frames: usize,
    hop_frames: usize,
) -> Result<Vec<f64>, AudioClockAlignmentIssueV1> {
    if samples.len() < fft_frames {
        return Err(AudioClockAlignmentIssueV1::TooFewAudioFrames);
    }

    // Prefix one complete window of silence. A flux frame's timestamp is its
    // center relative to the original WAV, so the first frames can observe an
    // onset at sample frame zero without inventing a positive lead-in.
    let mut padded = vec![0.0f32; fft_frames];
    padded.extend_from_slice(samples);
    padded.extend(std::iter::repeat_n(0.0f32, fft_frames));

    let frame_count = (padded.len() - fft_frames) / hop_frames + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_frames);
    let mut previous: Option<Vec<f64>> = None;
    let mut flux = Vec::with_capacity(frame_count);
    let window_denominator = (fft_frames - 1) as f64;

    for frame_index in 0..frame_count {
        let start = frame_index * hop_frames;
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); fft_frames];
        for (index, value) in buffer.iter_mut().enumerate() {
            let hann = 0.5
                - 0.5
                    * (std::f64::consts::TAU * index as f64 / window_denominator).cos();
            value.re = padded[start + index] * hann as f32;
        }
        fft.process(&mut buffer);

        let magnitude: Vec<f64> = buffer[..=fft_frames / 2]
            .iter()
            .skip(1)
            .map(|value| f64::from(value.norm()))
            .collect();

        let value = if let Some(previous) = &previous {
            let positive_difference = magnitude
                .iter()
                .zip(previous)
                .map(|(current, prior)| (current - prior).max(0.0))
                .sum::<f64>();
            let prior_energy = previous.iter().sum::<f64>();
            positive_difference / (prior_energy + 1e-9)
        } else {
            0.0
        };
        flux.push(value);
        previous = Some(magnitude);
    }
    Ok(flux)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        0.0
    } else if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

fn local_novelty(
    flux: &[f64],
    center: usize,
    inner_radius: usize,
    outer_radius: usize,
) -> f64 {
    if center >= flux.len() {
        return 0.0;
    }
    let mut baseline = Vec::with_capacity(2 * (outer_radius - inner_radius + 1));
    for distance in inner_radius..=outer_radius {
        if let Some(left) = center.checked_sub(distance) {
            if left < flux.len() {
                baseline.push(flux[left]);
            }
        }
        if let Some(right) = center.checked_add(distance) {
            if right < flux.len() {
                baseline.push(flux[right]);
            }
        }
    }
    let background = median(&mut baseline);
    (flux[center] - background).max(0.0)
}

fn candidate_score(
    flux: &[f64],
    expected_onsets: &[u64],
    lag_frames: i64,
    fft_frames: usize,
    hop_frames: usize,
    inner_radius: usize,
    outer_radius: usize,
) -> (f64, usize) {
    let mut total = 0.0;
    let mut matched = 0usize;
    for expected in expected_onsets {
        let Ok(expected) = i64::try_from(*expected) else {
            continue;
        };
        let Some(target) = expected.checked_add(lag_frames) else {
            continue;
        };
        let centered = target.saturating_add((fft_frames / 2) as i64);
        if centered < 0 {
            continue;
        }
        let index = ((centered + (hop_frames as i64 / 2)) / hop_frames as i64) as usize;
        if index >= flux.len() {
            continue;
        }
        total += local_novelty(flux, index, inner_radius, outer_radius);
        matched += 1;
    }
    let score = if expected_onsets.is_empty() {
        0.0
    } else {
        total / expected_onsets.len() as f64
    };
    (score, matched)
}

fn interpolation_delta(left: f64, center: f64, right: f64) -> f64 {
    let denominator = left - 2.0 * center + right;
    if !denominator.is_finite() || denominator >= -1e-12 {
        0.0
    } else {
        (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
    }
}

fn profile_with_no_estimate(
    midi: &MidiPerformanceArtifactV1,
    wav_bytes: &[u8],
    wav_info: AudioEventObservationWavInfoV1,
    config: AudioClockAlignmentConfigV1,
    status: AudioClockAlignmentStatusV1,
    unique_onsets: &[u64],
) -> Result<AudioClockAlignmentProfileV1, AudioClockAlignmentIssueV1> {
    let mut profile = AudioClockAlignmentProfileV1 {
        schema: AUDIO_CLOCK_ALIGNMENT_SCHEMA_V1.into(),
        algorithm: AUDIO_CLOCK_ALIGNMENT_ALGORITHM_V1.into(),
        source_score_sha256: midi.manifest.score_sha256.clone(),
        midi_sha256: midi.manifest.midi_sha256.clone(),
        midi_manifest_sha256: midi.manifest_sha256.clone(),
        wav_sha256: sha256_hex(wav_bytes),
        wav_info,
        config,
        status,
        note_event_count: u32::try_from(midi.manifest.notes.len()).unwrap_or(u32::MAX),
        unique_expected_onset_count: u32::try_from(unique_onsets.len()).unwrap_or(u32::MAX),
        matched_expected_onset_count: 0,
        audio_minus_midi_offset_frames: None,
        audio_minus_midi_offset_ms: None,
        best_peak_score: None,
        competing_peak_score: None,
        peak_margin: None,
        competing_peak_ratio: None,
        support_start_frames: None,
        support_end_frames: None,
        profile_sha256: String::new(),
    };
    profile.profile_sha256 = alignment_commitment(&profile)
        .map_err(|_| AudioClockAlignmentIssueV1::SerializationFailed)?;
    Ok(profile)
}

/// Estimate one global offset between exact MIDI note-on events and a WAV's
/// broadband spectral-flux envelope. This does not replace the symbolic clock
/// or estimate local per-note timing.
pub fn estimate_audio_minus_midi_clock_offset_v1(
    wav_bytes: &[u8],
    midi: &MidiPerformanceArtifactV1,
    config: AudioClockAlignmentConfigV1,
) -> Result<AudioClockAlignmentProfileV1, Vec<AudioClockAlignmentIssueV1>> {
    validate_config(&config).map_err(|issue| vec![issue])?;
    let midi_issues = validate_midi_artifact(midi);
    if !midi_issues.is_empty() {
        return Err(vec![AudioClockAlignmentIssueV1::MidiArtifactInvalid {
            issues: midi_issues,
        }]);
    }
    let wav = decode_wav(wav_bytes).map_err(|issue| {
        vec![AudioClockAlignmentIssueV1::WavInvalid { issue }]
    })?;
    let fft_frames = config.fft_window_frames as usize;
    let hop_frames = config.hop_frames as usize;
    let mut onset_set = BTreeSet::new();
    for note in &midi.manifest.notes {
        let Some(frame) = expected_frame(
            note.start_tick,
            midi.manifest.tempo_microseconds_per_quarter,
            midi.manifest.ticks_per_quarter,
            wav.info.sample_rate_hz,
        ) else {
            return Err(vec![AudioClockAlignmentIssueV1::InvalidConfig {
                field: "midi_event_time_out_of_range".into(),
            }]);
        };
        onset_set.insert(frame);
    }
    let expected_onsets: Vec<u64> = onset_set.into_iter().collect();

    if expected_onsets.len() < usize::from(config.minimum_unique_onsets) {
        return profile_with_no_estimate(
            midi,
            wav_bytes,
            wav.info,
            config,
            AudioClockAlignmentStatusV1::InsufficientOnsetDiversity,
            &expected_onsets,
        )
        .map_err(|issue| vec![issue]);
    }

    let samples = mono_samples(&wav);
    let flux = spectral_flux(&samples, fft_frames, hop_frames)
        .map_err(|issue| vec![issue])?;
    let sample_rate = i64::from(wav.info.sample_rate_hz);
    let minimum_lag = i64::from(config.minimum_offset_ms) * sample_rate / 1_000;
    let maximum_lag = i64::from(config.maximum_offset_ms) * sample_rate / 1_000;
    let exclusion = usize::from(config.peak_exclusion_hops);
    let inner_radius = usize::from(config.baseline_inner_hops);
    let outer_radius = usize::from(config.baseline_outer_hops);

    let first_lag = minimum_lag.div_euclid(hop_frames as i64) * hop_frames as i64;
    let mut lags = Vec::new();
    let mut scores = Vec::new();
    let mut matched_counts = Vec::new();
    let mut lag = first_lag;
    while lag <= maximum_lag {
        let (score, matched) = candidate_score(
            &flux,
            &expected_onsets,
            lag,
            fft_frames,
            hop_frames,
            inner_radius,
            outer_radius,
        );
        lags.push(lag);
        scores.push(score);
        matched_counts.push(matched);
        lag = lag.saturating_add(hop_frames as i64);
        if lags.len() > 1_000_000 {
            return Err(vec![AudioClockAlignmentIssueV1::InvalidConfig {
                field: "offset_search_too_large".into(),
            }]);
        }
    }

    let Some(best_index) = scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
    else {
        return Err(vec![AudioClockAlignmentIssueV1::TooFewAudioFrames]);
    };
    if matched_counts[best_index] < usize::from(config.minimum_unique_onsets) {
        return profile_with_no_estimate(
            midi,
            wav_bytes,
            wav.info,
            config,
            AudioClockAlignmentStatusV1::InsufficientAudioCoverage,
            &expected_onsets,
        )
        .map_err(|issue| vec![issue]);
    }
    let best_score = scores[best_index];
    let competing_score = scores
        .iter()
        .enumerate()
        .filter(|(index, _)| index.abs_diff(best_index) > exclusion)
        .map(|(_, score)| *score)
        .fold(0.0f64, f64::max);
    let margin = (best_score - competing_score).max(0.0);
    let ratio = (best_score > 0.0).then_some(competing_score / best_score);
    let delta = if best_index > 0 && best_index + 1 < scores.len() {
        interpolation_delta(
            scores[best_index - 1],
            scores[best_index],
            scores[best_index + 1],
        )
    } else {
        0.0
    };
    let refined_offset =
        lags[best_index].saturating_add((delta * hop_frames as f64).round() as i64);

    let support_threshold = best_score
        * f64::from(config.support_fraction_basis_points)
        / 10_000.0;
    let mut left = best_index;
    while left > 0 && scores[left - 1] >= support_threshold {
        left -= 1;
    }
    let mut right = best_index;
    while right + 1 < scores.len() && scores[right + 1] >= support_threshold {
        right += 1;
    }
    let support_start = lags[left];
    let support_end = lags[right];

    let ambiguity_threshold =
        f64::from(config.ambiguity_ratio_basis_points) / 10_000.0;
    let status = if best_score <= 1e-12 {
        AudioClockAlignmentStatusV1::NoPositiveEvidence
    } else if ratio.is_some_and(|value| value >= ambiguity_threshold) {
        AudioClockAlignmentStatusV1::AmbiguousCandidate
    } else {
        AudioClockAlignmentStatusV1::Candidate
    };

    let mut profile = AudioClockAlignmentProfileV1 {
        schema: AUDIO_CLOCK_ALIGNMENT_SCHEMA_V1.into(),
        algorithm: AUDIO_CLOCK_ALIGNMENT_ALGORITHM_V1.into(),
        source_score_sha256: midi.manifest.score_sha256.clone(),
        midi_sha256: midi.manifest.midi_sha256.clone(),
        midi_manifest_sha256: midi.manifest_sha256.clone(),
        wav_sha256: sha256_hex(wav_bytes),
        wav_info: wav.info,
        config,
        status,
        note_event_count: u32::try_from(midi.manifest.notes.len()).unwrap_or(u32::MAX),
        unique_expected_onset_count: u32::try_from(expected_onsets.len()).unwrap_or(u32::MAX),
        matched_expected_onset_count: u32::try_from(matched_counts[best_index]).unwrap_or(u32::MAX),
        audio_minus_midi_offset_frames: Some(refined_offset),
        audio_minus_midi_offset_ms: Some(
            refined_offset as f64 * 1_000.0 / f64::from(wav.info.sample_rate_hz),
        ),
        best_peak_score: Some(best_score),
        competing_peak_score: Some(competing_score),
        peak_margin: Some(margin),
        competing_peak_ratio: ratio,
        support_start_frames: Some(support_start),
        support_end_frames: Some(support_end),
        profile_sha256: String::new(),
    };
    profile.profile_sha256 = alignment_commitment(&profile)
        .map_err(|_| vec![AudioClockAlignmentIssueV1::SerializationFailed])?;
    Ok(profile)
}

pub fn validate_audio_clock_alignment_profile_v1(
    profile: &AudioClockAlignmentProfileV1,
    wav_bytes: &[u8],
    midi: &MidiPerformanceArtifactV1,
) -> Vec<AudioClockAlignmentIssueV1> {
    let mut issues = Vec::new();
    let midi_issues = validate_midi_artifact(midi);
    if !midi_issues.is_empty() {
        issues.push(AudioClockAlignmentIssueV1::MidiArtifactInvalid {
            issues: midi_issues,
        });
    }
    let wav = decode_wav(wav_bytes);
    match &wav {
        Ok(decoded) if decoded.info == profile.wav_info => {}
        Ok(_) => issues.push(AudioClockAlignmentIssueV1::WavInvalid {
            issue: AudioEventObservationIssueV1::ReceiptInputBindingMismatch,
        }),
        Err(issue) => issues.push(AudioClockAlignmentIssueV1::WavInvalid {
            issue: issue.clone(),
        }),
    }
    if profile.schema != AUDIO_CLOCK_ALIGNMENT_SCHEMA_V1
        || profile.algorithm != AUDIO_CLOCK_ALIGNMENT_ALGORITHM_V1
        || profile.source_score_sha256 != midi.manifest.score_sha256
        || profile.midi_sha256 != midi.manifest.midi_sha256
        || profile.midi_manifest_sha256 != midi.manifest_sha256
        || profile.wav_sha256 != sha256_hex(wav_bytes)
    {
        issues.push(AudioClockAlignmentIssueV1::SerializationFailed);
    }
    match alignment_commitment(profile) {
        Ok(actual) if actual == profile.profile_sha256 => {}
        _ => issues.push(AudioClockAlignmentIssueV1::SerializationFailed),
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_performance::{
        MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
        MIDI_PERFORMANCE_SCHEMA_V1,
        MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1,
        MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1,
        MidiPerformanceManifestV1,
        MidiPerformanceNoteReceiptV1,
        MidiPerformanceTrackReceiptV1,
    };
    use hound::{SampleFormat, WavWriter};
    use std::io::Cursor;

    fn write_vlq(output: &mut Vec<u8>, mut value: u64) {
        let mut bytes = [0u8; 10];
        let mut index = bytes.len() - 1;
        bytes[index] = (value & 0x7f) as u8;
        value >>= 7;
        while value > 0 {
            index -= 1;
            bytes[index] = ((value & 0x7f) as u8) | 0x80;
            value >>= 7;
        }
        output.extend_from_slice(&bytes[index..]);
    }

    fn multi_note_midi(notes: &[(u8, u64, u64)]) -> Vec<u8> {
        let mut events: Vec<(u64, u8, Vec<u8>)> = vec![(0, 1, vec![0xc0, 0])];
        for (pitch, start, end) in notes {
            events.push((*start, 2, vec![0x90, *pitch, 100]));
            events.push((*end, 0, vec![0x80, *pitch, 0]));
        }
        events.sort_by_key(|event| (event.0, event.1));
        let mut track = Vec::new();
        let mut prior = 0u64;
        for (tick, _, bytes) in events {
            write_vlq(&mut track, tick - prior);
            track.extend_from_slice(&bytes);
            prior = tick;
        }
        track.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);

        let tempo_track = vec![
            0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
            0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&2u16.to_be_bytes());
        bytes.extend_from_slice(&MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1.to_be_bytes());
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(tempo_track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&tempo_track);
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&track);
        bytes
    }

    fn artifact(notes: &[(u8, u64, u64)]) -> MidiPerformanceArtifactV1 {
        let midi_bytes = multi_note_midi(notes);
        let max_end = notes.iter().map(|note| note.2).max().unwrap_or(9_600);
        let manifest = MidiPerformanceManifestV1 {
            schema: MIDI_PERFORMANCE_SCHEMA_V1.into(),
            score_digest_schema: symthaea_music_theory::orchestration_timeline::ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: "a".repeat(64),
            timeline_digest_schema: MIDI_PERFORMANCE_TIMELINE_DIGEST_SCHEMA_V1.into(),
            timeline_sha256: "b".repeat(64),
            binding_digest_schema: MIDI_PERFORMANCE_BINDING_DIGEST_SCHEMA_V1.into(),
            binding_sha256: "c".repeat(64),
            capability_digest_schema: MIDI_PERFORMANCE_CAPABILITY_DIGEST_SCHEMA_V1.into(),
            capability_sha256: "d".repeat(64),
            midi_sha256: sha256_hex(&midi_bytes),
            ticks_per_quarter: MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1,
            tempo_microseconds_per_quarter: 500_000,
            meter_numerator: 4,
            meter_denominator: 4,
            tracks: vec![MidiPerformanceTrackReceiptV1 {
                track_index: 1,
                track_name: "Part 1 segment-a instrument-a".into(),
                part_id: 1,
                segment_id: "segment-a".into(),
                instrument_id: "instrument-a".into(),
                renderer_id: "test-renderer".into(),
                patch_id: "test-patch".into(),
                gm_program: 0,
                midi_channel: 0,
                segment_start_tick: 0,
                segment_end_tick: max_end + 1,
            }],
            notes: notes
                .iter()
                .enumerate()
                .map(|(ordinal, (pitch, start, end))| MidiPerformanceNoteReceiptV1 {
                    source_note_ordinal: ordinal,
                    part_id: 1,
                    segment_id: "segment-a".into(),
                    instrument_id: "instrument-a".into(),
                    track_index: 1,
                    start_tick: *start,
                    end_tick: *end,
                    pitch_midi: *pitch,
                    velocity_midi: 100,
                })
                .collect(),
        };
        let manifest_sha256 = domain_separated_sha256(
            MIDI_PERFORMANCE_MANIFEST_DIGEST_SCHEMA_V1,
            &serde_json::to_vec(&manifest).unwrap(),
        );
        MidiPerformanceArtifactV1 {
            midi_bytes,
            manifest,
            manifest_sha256,
        }
    }

    fn rendered_bursts(
        notes: &[(u8, u64, u64)],
        offset_frames: usize,
        sample_rate: u32,
    ) -> Vec<u8> {
        let tempo_us = 500_000u32;
        let ppq = MIDI_PERFORMANCE_TICKS_PER_QUARTER_V1;
        let final_tick = notes.iter().map(|note| note.1).max().unwrap_or(0);
        let final_frame = expected_frame(final_tick, tempo_us, ppq, sample_rate).unwrap() as usize
            + offset_frames
            + (sample_rate as usize / 2);
        let mut samples = vec![0.0f32; final_frame.max(sample_rate as usize)];
        for (index, (pitch, start_tick, end_tick)) in notes.iter().enumerate() {
            let start = expected_frame(*start_tick, tempo_us, ppq, sample_rate).unwrap() as usize
                + offset_frames;
            let end = expected_frame(*end_tick, tempo_us, ppq, sample_rate).unwrap() as usize
                + offset_frames;
            let frequency = 440.0f32
                * 2.0f32.powf((f32::from(*pitch) - 69.0) / 12.0);
            let duration = end.saturating_sub(start).max(1);
            for local in 0..duration {
                let attack = (local as f32 / (sample_rate as f32 * 0.01)).clamp(0.0, 1.0);
                let release_frames = (sample_rate as usize / 12).max(1);
                let release = if local + release_frames >= duration {
                    ((duration.saturating_sub(local)) as f32 / release_frames as f32)
                        .clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let envelope = attack.min(release) * 0.18;
                let sample = envelope
                    * (std::f32::consts::TAU * frequency * local as f32
                        / sample_rate as f32
                        + index as f32 * 0.173)
                        .sin();
                if start + local < samples.len() {
                    samples[start + local] += sample;
                }
            }
        }
        let mut cursor = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            };
            let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
            for sample in samples {
                writer.write_sample(sample.clamp(-0.98, 0.98)).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    fn irregular_notes() -> Vec<(u8, u64, u64)> {
        vec![
            (69, 0, 480),
            (72, 700, 1_180),
            (76, 1_900, 2_380),
            (67, 3_100, 3_580),
            (74, 4_400, 4_880),
        ]
    }

    #[test]
    fn estimates_known_global_audio_minus_midi_offset() {
        let notes = irregular_notes();
        let midi = artifact(&notes);
        let expected_offset = 9_600usize; // 200 ms at 48 kHz
        let wav = rendered_bursts(&notes, expected_offset, 48_000);
        let config = AudioClockAlignmentConfigV1 {
            minimum_offset_ms: -100,
            maximum_offset_ms: 500,
            ..Default::default()
        };
        let profile =
            estimate_audio_minus_midi_clock_offset_v1(&wav, &midi, config).unwrap();
        assert!(
            matches!(
                profile.status,
                AudioClockAlignmentStatusV1::Candidate
                    | AudioClockAlignmentStatusV1::AmbiguousCandidate
            ),
            "{profile:?}"
        );
        let estimated = profile.audio_minus_midi_offset_frames.unwrap();
        assert!(
            (estimated - expected_offset as i64).abs() <= 2_048,
            "estimated={estimated}, expected={expected_offset}, profile={profile:?}"
        );
        assert_eq!(profile.unique_expected_onset_count, 5);
        assert!(validate_audio_clock_alignment_profile_v1(&profile, &wav, &midi).is_empty());
    }

    #[test]
    fn silence_is_not_misreported_as_a_clock_estimate() {
        let notes = irregular_notes();
        let midi = artifact(&notes);
        let wav = rendered_bursts(&[], 0, 48_000);
        let profile = estimate_audio_minus_midi_clock_offset_v1(
            &wav,
            &midi,
            AudioClockAlignmentConfigV1 {
                minimum_offset_ms: -100,
                maximum_offset_ms: 500,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            profile.status,
            AudioClockAlignmentStatusV1::NoPositiveEvidence
        );
        assert!(profile.audio_minus_midi_offset_frames.is_none());
    }

    #[test]
    fn insufficient_unique_midi_onsets_abstain() {
        let notes = vec![(69, 0, 960), (72, 0, 960), (76, 0, 960)];
        let midi = artifact(&notes);
        let wav = rendered_bursts(&notes, 0, 48_000);
        let profile = estimate_audio_minus_midi_clock_offset_v1(
            &wav,
            &midi,
            AudioClockAlignmentConfigV1::default(),
        )
        .unwrap();
        assert_eq!(
            profile.status,
            AudioClockAlignmentStatusV1::InsufficientOnsetDiversity
        );
        assert!(profile.audio_minus_midi_offset_frames.is_none());
    }

    #[test]
    fn truncated_wav_cannot_estimate_from_only_a_subset_of_expected_onsets() {
        let notes = irregular_notes();
        let midi = artifact(&notes);
        let wav = rendered_bursts(&notes[..2], 0, 48_000);
        let profile = estimate_audio_minus_midi_clock_offset_v1(
            &wav,
            &midi,
            AudioClockAlignmentConfigV1 {
                minimum_offset_ms: -100,
                maximum_offset_ms: 500,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            profile.status,
            AudioClockAlignmentStatusV1::InsufficientAudioCoverage
        );
        assert!(profile.audio_minus_midi_offset_frames.is_none());
    }

    #[test]
    fn tampered_midi_artifact_is_rejected_before_alignment() {
        let notes = irregular_notes();
        let mut midi = artifact(&notes);
        midi.midi_bytes.push(0);
        let wav = rendered_bursts(&notes, 0, 48_000);
        let error = estimate_audio_minus_midi_clock_offset_v1(
            &wav,
            &midi,
            AudioClockAlignmentConfigV1::default(),
        )
        .unwrap_err();
        assert!(matches!(
            &error[0],
            AudioClockAlignmentIssueV1::MidiArtifactInvalid { .. }
        ));
    }

    #[test]
    fn changed_wav_invalidates_alignment_profile_binding() {
        let notes = irregular_notes();
        let midi = artifact(&notes);
        let wav = rendered_bursts(&notes, 9_600, 48_000);
        let profile = estimate_audio_minus_midi_clock_offset_v1(
            &wav,
            &midi,
            AudioClockAlignmentConfigV1 {
                minimum_offset_ms: -100,
                maximum_offset_ms: 500,
                ..Default::default()
            },
        )
        .unwrap();
        let replacement = rendered_bursts(&notes, 4_800, 48_000);
        assert!(!validate_audio_clock_alignment_profile_v1(&profile, &replacement, &midi).is_empty());
    }
}
