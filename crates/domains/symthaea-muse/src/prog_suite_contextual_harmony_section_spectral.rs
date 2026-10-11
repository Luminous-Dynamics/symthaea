// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive section-wide log-mel evidence for the frozen ProgSuite
//! contextual-harmony experiment.
//!
//! The section windows come exclusively from the already-verified performed-
//! event localization layer. This module does not select excerpts, search for
//! favorable spectral frames, or apply a threshold. Time-domain and log-mel
//! views share the same PCM and are not independent replications.

use super::prog_suite_contextual_harmony_audio_protocol::ProgSuiteContextualHarmonyAudioSurvivalProtocolV1;
use super::prog_suite_contextual_harmony_section_audio::{
    ProgSuiteSectionAudioErrorV1, ProgSuiteSectionAudioLocalizationV1,
    verify_prog_suite_section_audio_localization,
};
use crate::mel_extractor::{MelConfig, MelExtractor};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::prog_suite_contextual_harmony_comparison::ProgSuiteContextualHarmonyComparisonV1;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_SECTION_SPECTRAL_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-section-spectral-v1";
pub const PROG_SUITE_SECTION_SPECTRAL_REPRESENTATION_ID: &str =
    "muse-logmel-hann-magnitude-ln-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionSpectralConfigV1 {
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min_hz: f32,
    pub f_max_hz: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSectionSpectralNonClaimV1 {
    SpectralViewSharesPcmWithTimeDomainEvidence,
    RepresentationDiversityDoesNotEstablishIndependentReplication,
    LogMelDistanceDoesNotEstablishAudibility,
    LogMelDistanceDoesNotEstablishHarmonicCorrectness,
    LogMelDistanceDoesNotEstablishListenerPreference,
    LogMelDistanceDoesNotEstablishArtisticQuality,
    SectionFramesDoNotEstablishStatisticalIndependence,
    FixedRepresentationDoesNotEstablishRepresentationGeneralization,
    FixedRendererDoesNotEstablishRendererGeneralization,
    LockboxDoesNotEstablishUniversalMusicGeneralization,
    EvidenceDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionSpectralFrameV1 {
    pub frame_index: usize,
    pub absolute_start_sample: usize,
    pub rms_logmel_distance: f64,
    pub mean_absolute_logmel_distance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionSpectralSummaryV1 {
    pub frame_count: usize,
    pub covered_sample_count: usize,
    pub uncovered_tail_sample_count: usize,
    pub mean_rms_logmel_distance: f64,
    pub max_rms_logmel_distance: f64,
    pub mean_absolute_logmel_distance: f64,
    pub max_mean_absolute_logmel_distance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionSpectralEvidenceV1 {
    pub section_index: usize,
    pub analysis_start_sample: usize,
    pub analysis_end_sample: usize,
    pub waveform_difference_observed_in_analyzed_window: bool,
    pub complete_time_domain_window: bool,
    pub frames: Vec<ProgSuiteSectionSpectralFrameV1>,
    pub summary: ProgSuiteSectionSpectralSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionSpectralLocalizationV1 {
    pub version: String,
    pub representation_id: String,
    pub subject_index: usize,
    pub subject_id: String,
    pub sample_rate: u32,
    pub config: ProgSuiteSectionSpectralConfigV1,
    pub source_section_audio: ProgSuiteSectionAudioLocalizationV1,
    pub sections: Vec<ProgSuiteSectionSpectralEvidenceV1>,
    pub nonclaims: Vec<ProgSuiteSectionSpectralNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteSectionSpectralErrorV1 {
    SectionAudio(ProgSuiteSectionAudioErrorV1),
    NonCanonicalConfig,
    SectionWindowTooShort {
        section_index: usize,
        samples: usize,
        required: usize,
    },
    ExtractionShapeMismatch { section_index: usize },
    InvalidSpectralValue {
        section_index: usize,
        frame_index: usize,
    },
    FrameGeometryOverflow { section_index: usize },
    EvidenceMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn measure_prog_suite_section_spectral_localization(
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    section_audio: &ProgSuiteSectionAudioLocalizationV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<ProgSuiteSectionSpectralLocalizationV1, ProgSuiteSectionSpectralErrorV1> {
    verify_prog_suite_section_audio_localization(
        section_audio,
        protocol,
        comparison,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )
    .map_err(ProgSuiteSectionSpectralErrorV1::SectionAudio)?;

    let config = canonical_config();
    validate_config(config, section_audio.sample_rate)?;
    let mut sections = Vec::with_capacity(section_audio.sections.len());
    for section in &section_audio.sections {
        let start = section.difference.analysis_start_sample;
        let end = section.difference.aligned_end_sample;
        let window_len = end.saturating_sub(start);
        if window_len < config.n_fft {
            return Err(ProgSuiteSectionSpectralErrorV1::SectionWindowTooShort {
                section_index: section.section_index,
                samples: window_len,
                required: config.n_fft,
            });
        }
        let source = &source_render_a[start..end];
        let contextual = &contextual_render_a[start..end];
        let source_mel = extract_stereo(source, section_audio.sample_rate, config);
        let contextual_mel = extract_stereo(contextual, section_audio.sample_rate, config);
        if source_mel.left.len() != source_mel.right.len()
            || source_mel.left.len() != contextual_mel.left.len()
            || source_mel.left.len() != contextual_mel.right.len()
            || source_mel.left.is_empty()
        {
            return Err(ProgSuiteSectionSpectralErrorV1::ExtractionShapeMismatch {
                section_index: section.section_index,
            });
        }

        let frame_count = source_mel.left.len();
        let mut frames = Vec::with_capacity(frame_count);
        for frame_index in 0..frame_count {
            let absolute_start_sample = frame_index
                .checked_mul(config.hop_length)
                .and_then(|offset| start.checked_add(offset))
                .ok_or(ProgSuiteSectionSpectralErrorV1::FrameGeometryOverflow {
                    section_index: section.section_index,
                })?;
            let (rms, mean_abs) = stereo_frame_distance(
                &source_mel.left[frame_index],
                &source_mel.right[frame_index],
                &contextual_mel.left[frame_index],
                &contextual_mel.right[frame_index],
            )
            .ok_or(ProgSuiteSectionSpectralErrorV1::InvalidSpectralValue {
                section_index: section.section_index,
                frame_index,
            })?;
            frames.push(ProgSuiteSectionSpectralFrameV1 {
                frame_index,
                absolute_start_sample,
                rms_logmel_distance: rms,
                mean_absolute_logmel_distance: mean_abs,
            });
        }

        let covered_sample_count = (frame_count - 1)
            .checked_mul(config.hop_length)
            .and_then(|value| value.checked_add(config.n_fft))
            .ok_or(ProgSuiteSectionSpectralErrorV1::FrameGeometryOverflow {
                section_index: section.section_index,
            })?;
        let uncovered_tail_sample_count = window_len.saturating_sub(covered_sample_count);
        let summary = summarize(&frames, covered_sample_count, uncovered_tail_sample_count)
            .ok_or(ProgSuiteSectionSpectralErrorV1::InvalidSpectralValue {
                section_index: section.section_index,
                frame_index: 0,
            })?;
        sections.push(ProgSuiteSectionSpectralEvidenceV1 {
            section_index: section.section_index,
            analysis_start_sample: start,
            analysis_end_sample: end,
            waveform_difference_observed_in_analyzed_window:
                section.difference.changed_frame_count > 0,
            complete_time_domain_window: section.difference.complete_window_in_both_arms,
            frames,
            summary,
        });
    }

    Ok(ProgSuiteSectionSpectralLocalizationV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_SECTION_SPECTRAL_VERSION.into(),
        representation_id: PROG_SUITE_SECTION_SPECTRAL_REPRESENTATION_ID.into(),
        subject_index: section_audio.subject_index,
        subject_id: section_audio.subject_id.clone(),
        sample_rate: section_audio.sample_rate,
        config,
        source_section_audio: section_audio.clone(),
        sections,
        nonclaims: required_nonclaims(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_prog_suite_section_spectral_localization(
    evidence: &ProgSuiteSectionSpectralLocalizationV1,
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<(), ProgSuiteSectionSpectralErrorV1> {
    if evidence.config != canonical_config() {
        return Err(ProgSuiteSectionSpectralErrorV1::NonCanonicalConfig);
    }
    let canonical = measure_prog_suite_section_spectral_localization(
        protocol,
        comparison,
        &evidence.source_section_audio,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )?;
    if &canonical != evidence {
        return Err(ProgSuiteSectionSpectralErrorV1::EvidenceMismatch);
    }
    Ok(())
}

struct StereoMelFrames {
    left: Vec<Vec<f32>>,
    right: Vec<Vec<f32>>,
}

fn extract_stereo(
    frames: &[[f32; 2]],
    sample_rate: u32,
    config: ProgSuiteSectionSpectralConfigV1,
) -> StereoMelFrames {
    let left = frames.iter().map(|frame| frame[0]).collect::<Vec<_>>();
    let right = frames.iter().map(|frame| frame[1]).collect::<Vec<_>>();
    let mel = MelConfig {
        sample_rate,
        n_fft: config.n_fft,
        hop_length: config.hop_length,
        n_mels: config.n_mels,
        f_min: config.f_min_hz,
        f_max: config.f_max_hz,
    };
    let mut left_extractor = MelExtractor::new(mel.clone());
    let mut right_extractor = MelExtractor::new(mel);
    StereoMelFrames {
        left: left_extractor.extract(&left),
        right: right_extractor.extract(&right),
    }
}

fn stereo_frame_distance(
    source_left: &[f32],
    source_right: &[f32],
    contextual_left: &[f32],
    contextual_right: &[f32],
) -> Option<(f64, f64)> {
    if source_left.len() != source_right.len()
        || source_left.len() != contextual_left.len()
        || source_left.len() != contextual_right.len()
        || source_left.is_empty()
    {
        return None;
    }
    let mut squared_sum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut count = 0usize;
    for (source, contextual) in [(source_left, contextual_left), (source_right, contextual_right)] {
        for (&left, &right) in source.iter().zip(contextual) {
            if !left.is_finite() || !right.is_finite() {
                return None;
            }
            let delta = f64::from(right) - f64::from(left);
            squared_sum += delta * delta;
            absolute_sum += delta.abs();
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let denominator = count as f64;
    let rms = (squared_sum / denominator).sqrt();
    let mean_abs = absolute_sum / denominator;
    (rms.is_finite() && mean_abs.is_finite()).then_some((rms, mean_abs))
}

fn summarize(
    frames: &[ProgSuiteSectionSpectralFrameV1],
    covered_sample_count: usize,
    uncovered_tail_sample_count: usize,
) -> Option<ProgSuiteSectionSpectralSummaryV1> {
    if frames.is_empty() {
        return None;
    }
    let mut rms_sum = 0.0_f64;
    let mut rms_max = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut abs_max = 0.0_f64;
    for frame in frames {
        if !frame.rms_logmel_distance.is_finite()
            || !frame.mean_absolute_logmel_distance.is_finite()
            || frame.rms_logmel_distance < 0.0
            || frame.mean_absolute_logmel_distance < 0.0
        {
            return None;
        }
        rms_sum += frame.rms_logmel_distance;
        rms_max = rms_max.max(frame.rms_logmel_distance);
        abs_sum += frame.mean_absolute_logmel_distance;
        abs_max = abs_max.max(frame.mean_absolute_logmel_distance);
    }
    let denominator = frames.len() as f64;
    Some(ProgSuiteSectionSpectralSummaryV1 {
        frame_count: frames.len(),
        covered_sample_count,
        uncovered_tail_sample_count,
        mean_rms_logmel_distance: rms_sum / denominator,
        max_rms_logmel_distance: rms_max,
        mean_absolute_logmel_distance: abs_sum / denominator,
        max_mean_absolute_logmel_distance: abs_max,
    })
}

fn validate_config(
    config: ProgSuiteSectionSpectralConfigV1,
    sample_rate: u32,
) -> Result<(), ProgSuiteSectionSpectralErrorV1> {
    let canonical = canonical_config();
    if config != canonical
        || sample_rate == 0
        || config.n_fft == 0
        || config.hop_length == 0
        || config.n_mels == 0
        || !config.f_min_hz.is_finite()
        || !config.f_max_hz.is_finite()
        || config.f_min_hz < 0.0
        || config.f_max_hz <= config.f_min_hz
        || config.f_max_hz >= sample_rate as f32 / 2.0
    {
        return Err(ProgSuiteSectionSpectralErrorV1::NonCanonicalConfig);
    }
    Ok(())
}

fn canonical_config() -> ProgSuiteSectionSpectralConfigV1 {
    ProgSuiteSectionSpectralConfigV1 {
        n_fft: 2_048,
        hop_length: 512,
        n_mels: 128,
        f_min_hz: 20.0,
        f_max_hz: 16_000.0,
    }
}

fn required_nonclaims() -> Vec<ProgSuiteSectionSpectralNonClaimV1> {
    vec![
        ProgSuiteSectionSpectralNonClaimV1::SpectralViewSharesPcmWithTimeDomainEvidence,
        ProgSuiteSectionSpectralNonClaimV1::RepresentationDiversityDoesNotEstablishIndependentReplication,
        ProgSuiteSectionSpectralNonClaimV1::LogMelDistanceDoesNotEstablishAudibility,
        ProgSuiteSectionSpectralNonClaimV1::LogMelDistanceDoesNotEstablishHarmonicCorrectness,
        ProgSuiteSectionSpectralNonClaimV1::LogMelDistanceDoesNotEstablishListenerPreference,
        ProgSuiteSectionSpectralNonClaimV1::LogMelDistanceDoesNotEstablishArtisticQuality,
        ProgSuiteSectionSpectralNonClaimV1::SectionFramesDoNotEstablishStatisticalIndependence,
        ProgSuiteSectionSpectralNonClaimV1::FixedRepresentationDoesNotEstablishRepresentationGeneralization,
        ProgSuiteSectionSpectralNonClaimV1::FixedRendererDoesNotEstablishRendererGeneralization,
        ProgSuiteSectionSpectralNonClaimV1::LockboxDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteSectionSpectralNonClaimV1::EvidenceDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_config_is_frozen_before_lockbox_execution() {
        let config = canonical_config();
        assert_eq!(config.n_fft, 2_048);
        assert_eq!(config.hop_length, 512);
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.f_min_hz, 20.0);
        assert_eq!(config.f_max_hz, 16_000.0);
    }

    #[test]
    fn identical_logmel_frames_have_zero_distance() {
        let frame = vec![0.25_f32, -0.5, 1.0];
        let (rms, mean_abs) = stereo_frame_distance(&frame, &frame, &frame, &frame).unwrap();
        assert_eq!(rms, 0.0);
        assert_eq!(mean_abs, 0.0);
    }

    #[test]
    fn frame_summary_retains_tail_accounting() {
        let frames = vec![
            ProgSuiteSectionSpectralFrameV1 {
                frame_index: 0,
                absolute_start_sample: 100,
                rms_logmel_distance: 1.0,
                mean_absolute_logmel_distance: 0.5,
            },
            ProgSuiteSectionSpectralFrameV1 {
                frame_index: 1,
                absolute_start_sample: 612,
                rms_logmel_distance: 3.0,
                mean_absolute_logmel_distance: 1.5,
            },
        ];
        let summary = summarize(&frames, 2_560, 127).unwrap();
        assert_eq!(summary.frame_count, 2);
        assert_eq!(summary.covered_sample_count, 2_560);
        assert_eq!(summary.uncovered_tail_sample_count, 127);
        assert_eq!(summary.mean_rms_logmel_distance, 2.0);
        assert_eq!(summary.max_rms_logmel_distance, 3.0);
        assert_eq!(summary.mean_absolute_logmel_distance, 1.0);
    }
}
