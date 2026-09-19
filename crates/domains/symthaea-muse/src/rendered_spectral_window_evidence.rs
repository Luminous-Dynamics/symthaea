// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive frequency-domain evidence around one predeclared performed attack.
//!
//! This module intentionally does not implement an onset/audibility/quality gate.
//! It records matched log-mel evidence using its own frozen frequency-domain
//! geometry so the result cannot be confused with the shorter C3/C5 time-domain
//! windows.

use crate::mel_extractor::{MelConfig, MelExtractor};
use serde::{Deserialize, Serialize};

pub const RENDERED_SPECTRAL_WINDOW_EVIDENCE_VERSION: &str =
    "rendered-spectral-window-evidence-v1";
pub const RENDERED_SPECTRAL_REPRESENTATION_ID: &str =
    "muse-logmel-hann-magnitude-ln-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralWindowConfigV1 {
    /// Exact number of source samples in each pre/post analysis window and FFT.
    pub fft_size: usize,
    pub mel_bands: usize,
    pub f_min_hz: f32,
    pub f_max_hz: f32,
}

impl Default for RenderedSpectralWindowConfigV1 {
    fn default() -> Self {
        Self {
            // ~23.22 ms per side at 44.1 kHz. This is deliberately distinct
            // from C3/C5's 5 ms pre / 50 ms post time-domain geometry.
            fft_size: 1_024,
            mel_bands: 64,
            f_min_hz: 20.0,
            f_max_hz: 16_000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StereoLogMelFrameV1 {
    /// Left/right channels remain separate so stereo phase cancellation cannot
    /// silently change the evidence through downmixing.
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedSpectralWindowEvidenceV1 {
    pub evidence_version: String,
    pub representation_id: String,
    pub sample_rate: u32,
    pub attack_time_seconds: f32,
    pub center_sample: usize,
    pub pre_start_sample: usize,
    pub post_end_sample: usize,
    pub config: RenderedSpectralWindowConfigV1,
    pub baseline_pre: StereoLogMelFrameV1,
    pub baseline_post: StereoLogMelFrameV1,
    pub candidate_pre: StereoLogMelFrameV1,
    pub candidate_post: StereoLogMelFrameV1,
    /// Mean positive log-mel change from pre to post, pooled over both channels.
    pub baseline_positive_logmel_flux: f64,
    pub candidate_positive_logmel_flux: f64,
    /// Candidate flux minus matched baseline flux. Descriptive; no sign is a verdict.
    pub candidate_flux_excess: f64,
    /// RMS log-mel distance between arms before the marked attack.
    pub pre_between_arm_rms_distance: f64,
    /// RMS log-mel distance between arms after the marked attack.
    pub post_between_arm_rms_distance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedSpectralWindowEvidenceErrorV1 {
    UnsupportedEvidenceVersion,
    UnsupportedRepresentation,
    InvalidSampleRate,
    InvalidAttackTime,
    InvalidConfig,
    EmptyAudio,
    LengthMismatch,
    WindowOutOfBounds,
    NonFiniteAudio,
    ExtractionFailed,
    InvalidStoredFrame,
    InconsistentStoredGeometry,
    InconsistentStoredSummary,
}

pub fn measure_rendered_spectral_window(
    baseline: &[[f32; 2]],
    candidate: &[[f32; 2]],
    sample_rate: u32,
    attack_time_seconds: f32,
    config: RenderedSpectralWindowConfigV1,
) -> Result<RenderedSpectralWindowEvidenceV1, RenderedSpectralWindowEvidenceErrorV1> {
    validate_inputs(
        baseline,
        candidate,
        sample_rate,
        attack_time_seconds,
        config,
    )?;

    let center_sample = (attack_time_seconds * sample_rate as f32).round() as usize;
    let pre_start_sample = center_sample
        .checked_sub(config.fft_size)
        .ok_or(RenderedSpectralWindowEvidenceErrorV1::WindowOutOfBounds)?;
    let post_end_sample = center_sample
        .checked_add(config.fft_size)
        .ok_or(RenderedSpectralWindowEvidenceErrorV1::WindowOutOfBounds)?;
    if post_end_sample > baseline.len() {
        return Err(RenderedSpectralWindowEvidenceErrorV1::WindowOutOfBounds);
    }

    let baseline_pre = extract_stereo_logmel(
        &baseline[pre_start_sample..center_sample],
        sample_rate,
        config,
    )?;
    let baseline_post = extract_stereo_logmel(
        &baseline[center_sample..post_end_sample],
        sample_rate,
        config,
    )?;
    let candidate_pre = extract_stereo_logmel(
        &candidate[pre_start_sample..center_sample],
        sample_rate,
        config,
    )?;
    let candidate_post = extract_stereo_logmel(
        &candidate[center_sample..post_end_sample],
        sample_rate,
        config,
    )?;

    let baseline_positive_logmel_flux = positive_flux(&baseline_pre, &baseline_post);
    let candidate_positive_logmel_flux = positive_flux(&candidate_pre, &candidate_post);
    let candidate_flux_excess =
        candidate_positive_logmel_flux - baseline_positive_logmel_flux;
    let pre_between_arm_rms_distance = rms_distance(&baseline_pre, &candidate_pre);
    let post_between_arm_rms_distance = rms_distance(&baseline_post, &candidate_post);

    Ok(RenderedSpectralWindowEvidenceV1 {
        evidence_version: RENDERED_SPECTRAL_WINDOW_EVIDENCE_VERSION.into(),
        representation_id: RENDERED_SPECTRAL_REPRESENTATION_ID.into(),
        sample_rate,
        attack_time_seconds,
        center_sample,
        pre_start_sample,
        post_end_sample,
        config,
        baseline_pre,
        baseline_post,
        candidate_pre,
        candidate_post,
        baseline_positive_logmel_flux,
        candidate_positive_logmel_flux,
        candidate_flux_excess,
        pre_between_arm_rms_distance,
        post_between_arm_rms_distance,
    })
}

/// Verify all deterministic fields available from serialized C6A evidence.
///
/// This does not recreate the FFT without the source waveform. It does ensure
/// that stored vectors, geometry, and summary statistics are internally
/// consistent and have not been hand-edited into a stronger result.
pub fn verify_rendered_spectral_window_evidence(
    evidence: &RenderedSpectralWindowEvidenceV1,
) -> Result<(), RenderedSpectralWindowEvidenceErrorV1> {
    if evidence.evidence_version != RENDERED_SPECTRAL_WINDOW_EVIDENCE_VERSION {
        return Err(RenderedSpectralWindowEvidenceErrorV1::UnsupportedEvidenceVersion);
    }
    if evidence.representation_id != RENDERED_SPECTRAL_REPRESENTATION_ID {
        return Err(RenderedSpectralWindowEvidenceErrorV1::UnsupportedRepresentation);
    }
    validate_config(evidence.sample_rate, evidence.config)?;
    if !evidence.attack_time_seconds.is_finite() || evidence.attack_time_seconds < 0.0 {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InvalidAttackTime);
    }

    let expected_center =
        (evidence.attack_time_seconds * evidence.sample_rate as f32).round() as usize;
    let Some(expected_start) = expected_center.checked_sub(evidence.config.fft_size) else {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InconsistentStoredGeometry);
    };
    let Some(expected_end) = expected_center.checked_add(evidence.config.fft_size) else {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InconsistentStoredGeometry);
    };
    if evidence.center_sample != expected_center
        || evidence.pre_start_sample != expected_start
        || evidence.post_end_sample != expected_end
    {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InconsistentStoredGeometry);
    }

    for frame in [
        &evidence.baseline_pre,
        &evidence.baseline_post,
        &evidence.candidate_pre,
        &evidence.candidate_post,
    ] {
        if !valid_frame(frame, evidence.config.mel_bands) {
            return Err(RenderedSpectralWindowEvidenceErrorV1::InvalidStoredFrame);
        }
    }

    let baseline_flux = positive_flux(&evidence.baseline_pre, &evidence.baseline_post);
    let candidate_flux = positive_flux(&evidence.candidate_pre, &evidence.candidate_post);
    let flux_excess = candidate_flux - baseline_flux;
    let pre_distance = rms_distance(&evidence.baseline_pre, &evidence.candidate_pre);
    let post_distance = rms_distance(&evidence.baseline_post, &evidence.candidate_post);

    if evidence.baseline_positive_logmel_flux != baseline_flux
        || evidence.candidate_positive_logmel_flux != candidate_flux
        || evidence.candidate_flux_excess != flux_excess
        || evidence.pre_between_arm_rms_distance != pre_distance
        || evidence.post_between_arm_rms_distance != post_distance
    {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InconsistentStoredSummary);
    }

    Ok(())
}

fn validate_inputs(
    baseline: &[[f32; 2]],
    candidate: &[[f32; 2]],
    sample_rate: u32,
    attack_time_seconds: f32,
    config: RenderedSpectralWindowConfigV1,
) -> Result<(), RenderedSpectralWindowEvidenceErrorV1> {
    validate_config(sample_rate, config)?;
    if !attack_time_seconds.is_finite() || attack_time_seconds < 0.0 {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InvalidAttackTime);
    }
    if baseline.is_empty() || candidate.is_empty() {
        return Err(RenderedSpectralWindowEvidenceErrorV1::EmptyAudio);
    }
    if baseline.len() != candidate.len() {
        return Err(RenderedSpectralWindowEvidenceErrorV1::LengthMismatch);
    }
    if !finite_audio(baseline) || !finite_audio(candidate) {
        return Err(RenderedSpectralWindowEvidenceErrorV1::NonFiniteAudio);
    }
    Ok(())
}

fn validate_config(
    sample_rate: u32,
    config: RenderedSpectralWindowConfigV1,
) -> Result<(), RenderedSpectralWindowEvidenceErrorV1> {
    if sample_rate == 0 {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InvalidSampleRate);
    }
    let nyquist = sample_rate as f32 / 2.0;
    if config.fft_size < 64
        || config.fft_size > 16_384
        || !config.fft_size.is_power_of_two()
        || config.mel_bands == 0
        || config.mel_bands > 512
        || config.mel_bands > config.fft_size / 2
        || !config.f_min_hz.is_finite()
        || !config.f_max_hz.is_finite()
        || config.f_min_hz < 0.0
        || config.f_max_hz <= config.f_min_hz
        || config.f_max_hz >= nyquist
    {
        return Err(RenderedSpectralWindowEvidenceErrorV1::InvalidConfig);
    }
    Ok(())
}

fn finite_audio(frames: &[[f32; 2]]) -> bool {
    frames
        .iter()
        .all(|frame| frame[0].is_finite() && frame[1].is_finite())
}

fn extract_stereo_logmel(
    frames: &[[f32; 2]],
    sample_rate: u32,
    config: RenderedSpectralWindowConfigV1,
) -> Result<StereoLogMelFrameV1, RenderedSpectralWindowEvidenceErrorV1> {
    if frames.len() != config.fft_size {
        return Err(RenderedSpectralWindowEvidenceErrorV1::ExtractionFailed);
    }

    let left: Vec<f32> = frames.iter().map(|frame| frame[0]).collect();
    let right: Vec<f32> = frames.iter().map(|frame| frame[1]).collect();
    let mel_config = MelConfig {
        sample_rate,
        n_fft: config.fft_size,
        hop_length: config.fft_size,
        n_mels: config.mel_bands,
        f_min: config.f_min_hz,
        f_max: config.f_max_hz,
    };

    // Fresh extractors keep every pre/post/channel frame independent of any
    // future state that MelExtractor might acquire.
    let mut left_extractor = MelExtractor::new(mel_config.clone());
    let mut right_extractor = MelExtractor::new(mel_config);
    let left_frames = left_extractor.extract(&left);
    let right_frames = right_extractor.extract(&right);
    if left_frames.len() != 1 || right_frames.len() != 1 {
        return Err(RenderedSpectralWindowEvidenceErrorV1::ExtractionFailed);
    }

    Ok(StereoLogMelFrameV1 {
        left: left_frames.into_iter().next().unwrap(),
        right: right_frames.into_iter().next().unwrap(),
    })
}

fn valid_frame(frame: &StereoLogMelFrameV1, mel_bands: usize) -> bool {
    frame.left.len() == mel_bands
        && frame.right.len() == mel_bands
        && frame.left.iter().all(|value| value.is_finite())
        && frame.right.iter().all(|value| value.is_finite())
}

fn positive_flux(pre: &StereoLogMelFrameV1, post: &StereoLogMelFrameV1) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for (pre_channel, post_channel) in [(&pre.left, &post.left), (&pre.right, &post.right)] {
        for (&before, &after) in pre_channel.iter().zip(post_channel) {
            sum += (after as f64 - before as f64).max(0.0);
            count += 1;
        }
    }
    sum / count as f64
}

fn rms_distance(left: &StereoLogMelFrameV1, right: &StereoLogMelFrameV1) -> f64 {
    let mut squared_sum = 0.0_f64;
    let mut count = 0usize;
    for (left_channel, right_channel) in [(&left.left, &right.left), (&left.right, &right.right)] {
        for (&a, &b) in left_channel.iter().zip(right_channel) {
            let delta = b as f64 - a as f64;
            squared_sum += delta * delta;
            count += 1;
        }
    }
    (squared_sum / count as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 8_192;
    const ATTACK_TIME: f32 = 0.5;

    fn test_config() -> RenderedSpectralWindowConfigV1 {
        RenderedSpectralWindowConfigV1 {
            fft_size: 256,
            mel_bands: 24,
            f_min_hz: 20.0,
            f_max_hz: 3_500.0,
        }
    }

    fn silence() -> Vec<[f32; 2]> {
        vec![[0.0, 0.0]; SAMPLE_RATE as usize]
    }

    fn write_tone(audio: &mut [[f32; 2]], start: usize, amplitude: f32) {
        // 512 Hz is exactly FFT-bin 16 for the 256/8192 test geometry.
        for (index, frame) in audio.iter_mut().enumerate().skip(start) {
            let phase = std::f32::consts::TAU * 512.0 * index as f32 / SAMPLE_RATE as f32;
            let sample = amplitude * phase.sin();
            *frame = [sample, sample * 0.75];
        }
    }

    #[test]
    fn identical_silence_has_zero_between_arm_distance_and_flux() {
        let baseline = silence();
        let evidence = measure_rendered_spectral_window(
            &baseline,
            &baseline,
            SAMPLE_RATE,
            ATTACK_TIME,
            test_config(),
        )
        .unwrap();
        assert_eq!(evidence.baseline_positive_logmel_flux, 0.0);
        assert_eq!(evidence.candidate_positive_logmel_flux, 0.0);
        assert_eq!(evidence.candidate_flux_excess, 0.0);
        assert_eq!(evidence.pre_between_arm_rms_distance, 0.0);
        assert_eq!(evidence.post_between_arm_rms_distance, 0.0);
        verify_rendered_spectral_window_evidence(&evidence).unwrap();
    }

    #[test]
    fn candidate_only_tone_onset_is_retained_without_assigning_a_gate() {
        let baseline = silence();
        let mut candidate = baseline.clone();
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;
        write_tone(&mut candidate, center, 0.2);
        let evidence = measure_rendered_spectral_window(
            &baseline,
            &candidate,
            SAMPLE_RATE,
            ATTACK_TIME,
            test_config(),
        )
        .unwrap();
        assert!(
            evidence.candidate_positive_logmel_flux
                > evidence.baseline_positive_logmel_flux
        );
        assert!(evidence.candidate_flux_excess > 0.0);
        assert_eq!(evidence.pre_between_arm_rms_distance, 0.0);
        assert!(evidence.post_between_arm_rms_distance > 0.0);
    }

    #[test]
    fn shared_onset_is_not_relabelled_candidate_specific() {
        let mut shared = silence();
        let center = (ATTACK_TIME * SAMPLE_RATE as f32).round() as usize;
        write_tone(&mut shared, center, 0.2);
        let evidence = measure_rendered_spectral_window(
            &shared,
            &shared,
            SAMPLE_RATE,
            ATTACK_TIME,
            test_config(),
        )
        .unwrap();
        assert!(evidence.baseline_positive_logmel_flux > 0.0);
        assert_eq!(
            evidence.baseline_positive_logmel_flux,
            evidence.candidate_positive_logmel_flux
        );
        assert_eq!(evidence.candidate_flux_excess, 0.0);
        assert_eq!(evidence.pre_between_arm_rms_distance, 0.0);
        assert_eq!(evidence.post_between_arm_rms_distance, 0.0);
    }

    #[test]
    fn persistent_candidate_spectrum_is_distinct_from_a_new_spectral_change() {
        let baseline = silence();
        let mut candidate = baseline.clone();
        write_tone(&mut candidate, 0, 0.2);
        let evidence = measure_rendered_spectral_window(
            &baseline,
            &candidate,
            SAMPLE_RATE,
            ATTACK_TIME,
            test_config(),
        )
        .unwrap();
        assert!(evidence.pre_between_arm_rms_distance > 0.0);
        assert!(evidence.post_between_arm_rms_distance > 0.0);
        assert!(evidence.candidate_positive_logmel_flux.abs() < 1.0e-4);
    }

    #[test]
    fn malformed_inputs_and_serialized_summary_fail_closed() {
        let baseline = silence();
        let mut bad_config = test_config();
        bad_config.f_max_hz = SAMPLE_RATE as f32;
        assert_eq!(
            measure_rendered_spectral_window(
                &baseline,
                &baseline,
                SAMPLE_RATE,
                ATTACK_TIME,
                bad_config,
            ),
            Err(RenderedSpectralWindowEvidenceErrorV1::InvalidConfig)
        );

        let mut evidence = measure_rendered_spectral_window(
            &baseline,
            &baseline,
            SAMPLE_RATE,
            ATTACK_TIME,
            test_config(),
        )
        .unwrap();
        evidence.candidate_flux_excess = 1.0;
        assert_eq!(
            verify_rendered_spectral_window_evidence(&evidence),
            Err(RenderedSpectralWindowEvidenceErrorV1::InconsistentStoredSummary)
        );
    }
}
