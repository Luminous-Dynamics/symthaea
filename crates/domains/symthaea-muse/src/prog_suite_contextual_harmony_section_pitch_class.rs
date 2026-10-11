// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive 12-bin pitch-class magnitude projection for the frozen
//! ProgSuite contextual-harmony experiment.
//!
//! This is intentionally not called HPCP/chroma recognition. It is a simple,
//! transparent acoustic projection: Hann-windowed stereo FFT magnitudes are
//! summed without phase-cancelling downmix, continuously distributed between
//! neighboring equal-tempered semitone classes, and L1-normalized per frame.
//! Harmonics and timbre remain in the projection; no chord label is inferred.

use super::prog_suite_contextual_harmony_audio_protocol::ProgSuiteContextualHarmonyAudioSurvivalProtocolV1;
use super::prog_suite_contextual_harmony_section_spectral::{
    ProgSuiteSectionSpectralErrorV1, ProgSuiteSectionSpectralLocalizationV1,
    verify_prog_suite_section_spectral_localization,
};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::prog_suite_contextual_harmony_comparison::ProgSuiteContextualHarmonyComparisonV1;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_SECTION_PITCH_CLASS_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-section-pitch-class-v1";
pub const PROG_SUITE_SECTION_PITCH_CLASS_REPRESENTATION_ID: &str =
    "stereo-fft-equal-tempered-pitch-class-magnitude-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePitchClassProjectionConfigV1 {
    pub n_fft: usize,
    pub hop_length: usize,
    pub f_min_hz: f64,
    pub f_max_hz: f64,
    pub reference_a_hz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuitePitchClassNonClaimV1 {
    ProjectionIsNotHpcpOrChordRecognition,
    HarmonicsAndTimbreRemainInPitchClassMagnitude,
    PitchClassDistanceDoesNotEstablishHarmonicCorrectness,
    PitchClassDistanceDoesNotEstablishAudibility,
    PitchClassDistanceDoesNotEstablishListenerPreference,
    PitchClassDistanceDoesNotEstablishArtisticQuality,
    PitchClassViewSharesPcmWithOtherAcousticEvidence,
    RepresentationDiversityDoesNotEstablishIndependentReplication,
    FrameCountsDoNotEstablishStatisticalIndependence,
    FixedFrequencyRangeDoesNotEstablishRepresentationGeneralization,
    FixedRendererDoesNotEstablishRendererGeneralization,
    LockboxDoesNotEstablishUniversalMusicGeneralization,
    EvidenceDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePitchClassFrameV1 {
    pub frame_index: usize,
    pub absolute_start_sample: usize,
    /// Index 0 = C, 1 = C#/Db, ..., 9 = A.
    pub source_normalized_profile: [f64; 12],
    pub contextual_normalized_profile: [f64; 12],
    pub source_projected_magnitude: f64,
    pub contextual_projected_magnitude: f64,
    /// L1 distance between normalized profiles, range [0, 2].
    pub normalized_profile_l1_distance: f64,
    /// `None` if either frame has zero projected magnitude.
    pub cosine_distance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePitchClassSectionSummaryV1 {
    pub frame_count: usize,
    pub covered_sample_count: usize,
    pub uncovered_tail_sample_count: usize,
    pub cosine_defined_frame_count: usize,
    pub mean_source_normalized_profile: [f64; 12],
    pub mean_contextual_normalized_profile: [f64; 12],
    pub mean_profile_l1_distance: f64,
    pub mean_frame_l1_distance: f64,
    pub max_frame_l1_distance: f64,
    pub mean_cosine_distance: Option<f64>,
    pub max_cosine_distance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePitchClassSectionEvidenceV1 {
    pub section_index: usize,
    pub progression_changed: bool,
    pub symbolic_event_stream_changed: bool,
    pub analysis_start_sample: usize,
    pub analysis_end_sample: usize,
    pub frames: Vec<ProgSuitePitchClassFrameV1>,
    pub summary: ProgSuitePitchClassSectionSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePitchClassLocalizationV1 {
    pub version: String,
    pub representation_id: String,
    pub subject_index: usize,
    pub subject_id: String,
    pub sample_rate: u32,
    pub config: ProgSuitePitchClassProjectionConfigV1,
    pub source_spectral_evidence: ProgSuiteSectionSpectralLocalizationV1,
    pub sections: Vec<ProgSuitePitchClassSectionEvidenceV1>,
    pub nonclaims: Vec<ProgSuitePitchClassNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuitePitchClassErrorV1 {
    Spectral(ProgSuiteSectionSpectralErrorV1),
    NonCanonicalConfig,
    WindowTooShort {
        section_index: usize,
        samples: usize,
        required: usize,
    },
    InvalidFrame { section_index: usize, frame_index: usize },
    FrameGeometryOverflow { section_index: usize },
    SummaryFailure { section_index: usize },
    EvidenceMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn measure_prog_suite_pitch_class_localization(
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    spectral: &ProgSuiteSectionSpectralLocalizationV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<ProgSuitePitchClassLocalizationV1, ProgSuitePitchClassErrorV1> {
    verify_prog_suite_section_spectral_localization(
        spectral,
        protocol,
        comparison,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )
    .map_err(ProgSuitePitchClassErrorV1::Spectral)?;

    let config = canonical_config();
    validate_config(config, spectral.sample_rate)?;
    let projector = PitchClassProjector::new(config, spectral.sample_rate);
    let source_sections = &spectral.source_section_audio.sections;
    let mut sections = Vec::with_capacity(spectral.sections.len());

    for (position, spectral_section) in spectral.sections.iter().enumerate() {
        let source_section = source_sections
            .get(position)
            .ok_or(ProgSuitePitchClassErrorV1::SummaryFailure {
                section_index: spectral_section.section_index,
            })?;
        if source_section.section_index != spectral_section.section_index {
            return Err(ProgSuitePitchClassErrorV1::SummaryFailure {
                section_index: spectral_section.section_index,
            });
        }
        let start = spectral_section.analysis_start_sample;
        let end = spectral_section.analysis_end_sample;
        let window_len = end.saturating_sub(start);
        if window_len < config.n_fft {
            return Err(ProgSuitePitchClassErrorV1::WindowTooShort {
                section_index: spectral_section.section_index,
                samples: window_len,
                required: config.n_fft,
            });
        }
        let source = &source_render_a[start..end];
        let contextual = &contextual_render_a[start..end];
        let frame_count = 1 + (window_len - config.n_fft) / config.hop_length;
        let mut frames = Vec::with_capacity(frame_count);
        for frame_index in 0..frame_count {
            let offset = frame_index
                .checked_mul(config.hop_length)
                .ok_or(ProgSuitePitchClassErrorV1::FrameGeometryOverflow {
                    section_index: spectral_section.section_index,
                })?;
            let frame_end = offset
                .checked_add(config.n_fft)
                .ok_or(ProgSuitePitchClassErrorV1::FrameGeometryOverflow {
                    section_index: spectral_section.section_index,
                })?;
            let absolute_start_sample = start
                .checked_add(offset)
                .ok_or(ProgSuitePitchClassErrorV1::FrameGeometryOverflow {
                    section_index: spectral_section.section_index,
                })?;
            let source_projection = projector.project(&source[offset..frame_end]);
            let contextual_projection = projector.project(&contextual[offset..frame_end]);
            let frame = compare_projection(
                frame_index,
                absolute_start_sample,
                source_projection,
                contextual_projection,
            )
            .ok_or(ProgSuitePitchClassErrorV1::InvalidFrame {
                section_index: spectral_section.section_index,
                frame_index,
            })?;
            frames.push(frame);
        }
        let covered_sample_count = (frame_count - 1)
            .checked_mul(config.hop_length)
            .and_then(|value| value.checked_add(config.n_fft))
            .ok_or(ProgSuitePitchClassErrorV1::FrameGeometryOverflow {
                section_index: spectral_section.section_index,
            })?;
        let summary = summarize(
            &frames,
            covered_sample_count,
            window_len.saturating_sub(covered_sample_count),
        )
        .ok_or(ProgSuitePitchClassErrorV1::SummaryFailure {
            section_index: spectral_section.section_index,
        })?;
        sections.push(ProgSuitePitchClassSectionEvidenceV1 {
            section_index: spectral_section.section_index,
            progression_changed: source_section.progression_changed,
            symbolic_event_stream_changed: source_section.symbolic_event_stream_changed,
            analysis_start_sample: start,
            analysis_end_sample: end,
            frames,
            summary,
        });
    }

    Ok(ProgSuitePitchClassLocalizationV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_SECTION_PITCH_CLASS_VERSION.into(),
        representation_id: PROG_SUITE_SECTION_PITCH_CLASS_REPRESENTATION_ID.into(),
        subject_index: spectral.subject_index,
        subject_id: spectral.subject_id.clone(),
        sample_rate: spectral.sample_rate,
        config,
        source_spectral_evidence: spectral.clone(),
        sections,
        nonclaims: required_nonclaims(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_prog_suite_pitch_class_localization(
    evidence: &ProgSuitePitchClassLocalizationV1,
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<(), ProgSuitePitchClassErrorV1> {
    if evidence.config != canonical_config() {
        return Err(ProgSuitePitchClassErrorV1::NonCanonicalConfig);
    }
    let canonical = measure_prog_suite_pitch_class_localization(
        protocol,
        comparison,
        &evidence.source_spectral_evidence,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )?;
    if &canonical != evidence {
        return Err(ProgSuitePitchClassErrorV1::EvidenceMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PitchClassProjection {
    profile: [f64; 12],
    total_magnitude: f64,
}

struct PitchClassProjector {
    config: ProgSuitePitchClassProjectionConfigV1,
    sample_rate: u32,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    hann: Vec<f32>,
}

impl PitchClassProjector {
    fn new(config: ProgSuitePitchClassProjectionConfigV1, sample_rate: u32) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(config.n_fft);
        let hann = (0..config.n_fft)
            .map(|index| {
                0.5_f32
                    * (1.0
                        - (std::f32::consts::TAU * index as f32
                            / (config.n_fft - 1) as f32)
                            .cos())
            })
            .collect();
        Self {
            config,
            sample_rate,
            fft,
            hann,
        }
    }

    fn project(&self, frames: &[[f32; 2]]) -> PitchClassProjection {
        debug_assert_eq!(frames.len(), self.config.n_fft);
        let mut left = frames
            .iter()
            .zip(&self.hann)
            .map(|(frame, window)| Complex {
                re: frame[0] * window,
                im: 0.0,
            })
            .collect::<Vec<_>>();
        let mut right = frames
            .iter()
            .zip(&self.hann)
            .map(|(frame, window)| Complex {
                re: frame[1] * window,
                im: 0.0,
            })
            .collect::<Vec<_>>();
        self.fft.process(&mut left);
        self.fft.process(&mut right);

        let mut profile = [0.0_f64; 12];
        for bin in 1..=self.config.n_fft / 2 {
            let frequency = bin as f64 * f64::from(self.sample_rate) / self.config.n_fft as f64;
            if frequency < self.config.f_min_hz || frequency > self.config.f_max_hz {
                continue;
            }
            let left_magnitude = (f64::from(left[bin].re).powi(2)
                + f64::from(left[bin].im).powi(2))
            .sqrt();
            let right_magnitude = (f64::from(right[bin].re).powi(2)
                + f64::from(right[bin].im).powi(2))
            .sqrt();
            let magnitude = left_magnitude + right_magnitude;
            if !magnitude.is_finite() || magnitude <= 0.0 {
                continue;
            }
            accumulate_pitch_class(
                &mut profile,
                frequency,
                magnitude,
                self.config.reference_a_hz,
            );
        }
        let total_magnitude = profile.iter().sum::<f64>();
        if total_magnitude > 0.0 && total_magnitude.is_finite() {
            for value in &mut profile {
                *value /= total_magnitude;
            }
        } else {
            profile = [0.0; 12];
        }
        PitchClassProjection {
            profile,
            total_magnitude,
        }
    }
}

fn accumulate_pitch_class(
    profile: &mut [f64; 12],
    frequency_hz: f64,
    magnitude: f64,
    reference_a_hz: f64,
) {
    let midi = 69.0 + 12.0 * (frequency_hz / reference_a_hz).log2();
    if !midi.is_finite() {
        return;
    }
    let lower = midi.floor();
    let fraction = midi - lower;
    let lower_pc = (lower as i64).rem_euclid(12) as usize;
    let upper_pc = ((lower as i64) + 1).rem_euclid(12) as usize;
    profile[lower_pc] += magnitude * (1.0 - fraction);
    profile[upper_pc] += magnitude * fraction;
}

fn compare_projection(
    frame_index: usize,
    absolute_start_sample: usize,
    source: PitchClassProjection,
    contextual: PitchClassProjection,
) -> Option<ProgSuitePitchClassFrameV1> {
    if !source.total_magnitude.is_finite()
        || !contextual.total_magnitude.is_finite()
        || source.total_magnitude < 0.0
        || contextual.total_magnitude < 0.0
        || source.profile.iter().any(|value| !value.is_finite() || *value < 0.0)
        || contextual
            .profile
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return None;
    }
    let mut l1 = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut source_squared = 0.0_f64;
    let mut contextual_squared = 0.0_f64;
    for index in 0..12 {
        l1 += (source.profile[index] - contextual.profile[index]).abs();
        dot += source.profile[index] * contextual.profile[index];
        source_squared += source.profile[index] * source.profile[index];
        contextual_squared += contextual.profile[index] * contextual.profile[index];
    }
    let cosine_distance = if source_squared > 0.0 && contextual_squared > 0.0 {
        let similarity = (dot / (source_squared.sqrt() * contextual_squared.sqrt())).clamp(-1.0, 1.0);
        Some(1.0 - similarity)
    } else {
        None
    };
    if !l1.is_finite() || cosine_distance.is_some_and(|value| !value.is_finite()) {
        return None;
    }
    Some(ProgSuitePitchClassFrameV1 {
        frame_index,
        absolute_start_sample,
        source_normalized_profile: source.profile,
        contextual_normalized_profile: contextual.profile,
        source_projected_magnitude: source.total_magnitude,
        contextual_projected_magnitude: contextual.total_magnitude,
        normalized_profile_l1_distance: l1,
        cosine_distance,
    })
}

fn summarize(
    frames: &[ProgSuitePitchClassFrameV1],
    covered_sample_count: usize,
    uncovered_tail_sample_count: usize,
) -> Option<ProgSuitePitchClassSectionSummaryV1> {
    if frames.is_empty() {
        return None;
    }
    let mut source_mean = [0.0_f64; 12];
    let mut contextual_mean = [0.0_f64; 12];
    let mut l1_sum = 0.0_f64;
    let mut l1_max = 0.0_f64;
    let mut cosine_sum = 0.0_f64;
    let mut cosine_max = 0.0_f64;
    let mut cosine_count = 0usize;
    for frame in frames {
        if !frame.normalized_profile_l1_distance.is_finite()
            || frame.normalized_profile_l1_distance < 0.0
        {
            return None;
        }
        for index in 0..12 {
            source_mean[index] += frame.source_normalized_profile[index];
            contextual_mean[index] += frame.contextual_normalized_profile[index];
        }
        l1_sum += frame.normalized_profile_l1_distance;
        l1_max = l1_max.max(frame.normalized_profile_l1_distance);
        if let Some(cosine) = frame.cosine_distance {
            if !cosine.is_finite() || cosine < 0.0 {
                return None;
            }
            cosine_sum += cosine;
            cosine_max = cosine_max.max(cosine);
            cosine_count += 1;
        }
    }
    let denominator = frames.len() as f64;
    for index in 0..12 {
        source_mean[index] /= denominator;
        contextual_mean[index] /= denominator;
    }
    let mean_profile_l1_distance = (0..12)
        .map(|index| (source_mean[index] - contextual_mean[index]).abs())
        .sum::<f64>();
    Some(ProgSuitePitchClassSectionSummaryV1 {
        frame_count: frames.len(),
        covered_sample_count,
        uncovered_tail_sample_count,
        cosine_defined_frame_count: cosine_count,
        mean_source_normalized_profile: source_mean,
        mean_contextual_normalized_profile: contextual_mean,
        mean_profile_l1_distance,
        mean_frame_l1_distance: l1_sum / denominator,
        max_frame_l1_distance: l1_max,
        mean_cosine_distance: (cosine_count > 0).then_some(cosine_sum / cosine_count as f64),
        max_cosine_distance: (cosine_count > 0).then_some(cosine_max),
    })
}

fn validate_config(
    config: ProgSuitePitchClassProjectionConfigV1,
    sample_rate: u32,
) -> Result<(), ProgSuitePitchClassErrorV1> {
    if config != canonical_config()
        || sample_rate == 0
        || config.n_fft < 64
        || !config.n_fft.is_power_of_two()
        || config.hop_length == 0
        || config.hop_length > config.n_fft
        || !config.f_min_hz.is_finite()
        || !config.f_max_hz.is_finite()
        || !config.reference_a_hz.is_finite()
        || config.f_min_hz <= 0.0
        || config.f_max_hz <= config.f_min_hz
        || config.f_max_hz >= f64::from(sample_rate) / 2.0
        || config.reference_a_hz <= 0.0
    {
        return Err(ProgSuitePitchClassErrorV1::NonCanonicalConfig);
    }
    Ok(())
}

fn canonical_config() -> ProgSuitePitchClassProjectionConfigV1 {
    ProgSuitePitchClassProjectionConfigV1 {
        n_fft: 8_192,
        hop_length: 2_048,
        f_min_hz: 55.0,
        f_max_hz: 5_000.0,
        reference_a_hz: 440.0,
    }
}

fn required_nonclaims() -> Vec<ProgSuitePitchClassNonClaimV1> {
    vec![
        ProgSuitePitchClassNonClaimV1::ProjectionIsNotHpcpOrChordRecognition,
        ProgSuitePitchClassNonClaimV1::HarmonicsAndTimbreRemainInPitchClassMagnitude,
        ProgSuitePitchClassNonClaimV1::PitchClassDistanceDoesNotEstablishHarmonicCorrectness,
        ProgSuitePitchClassNonClaimV1::PitchClassDistanceDoesNotEstablishAudibility,
        ProgSuitePitchClassNonClaimV1::PitchClassDistanceDoesNotEstablishListenerPreference,
        ProgSuitePitchClassNonClaimV1::PitchClassDistanceDoesNotEstablishArtisticQuality,
        ProgSuitePitchClassNonClaimV1::PitchClassViewSharesPcmWithOtherAcousticEvidence,
        ProgSuitePitchClassNonClaimV1::RepresentationDiversityDoesNotEstablishIndependentReplication,
        ProgSuitePitchClassNonClaimV1::FrameCountsDoNotEstablishStatisticalIndependence,
        ProgSuitePitchClassNonClaimV1::FixedFrequencyRangeDoesNotEstablishRepresentationGeneralization,
        ProgSuitePitchClassNonClaimV1::FixedRendererDoesNotEstablishRendererGeneralization,
        ProgSuitePitchClassNonClaimV1::LockboxDoesNotEstablishUniversalMusicGeneralization,
        ProgSuitePitchClassNonClaimV1::EvidenceDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_config_is_frozen_before_pcm_execution() {
        let config = canonical_config();
        assert_eq!(config.n_fft, 8_192);
        assert_eq!(config.hop_length, 2_048);
        assert_eq!(config.f_min_hz, 55.0);
        assert_eq!(config.f_max_hz, 5_000.0);
        assert_eq!(config.reference_a_hz, 440.0);
    }

    #[test]
    fn equal_tempered_reference_maps_a440_to_pitch_class_a() {
        let mut profile = [0.0_f64; 12];
        accumulate_pitch_class(&mut profile, 440.0, 1.0, 440.0);
        assert!((profile[9] - 1.0).abs() < 1.0e-12);
        assert_eq!(profile.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn half_semitone_frequency_distributes_between_adjacent_classes() {
        let mut profile = [0.0_f64; 12];
        let frequency = 440.0 * 2.0_f64.powf(0.5 / 12.0);
        accumulate_pitch_class(&mut profile, frequency, 1.0, 440.0);
        assert!((profile[9] - 0.5).abs() < 1.0e-9);
        assert!((profile[10] - 0.5).abs() < 1.0e-9);
    }

    #[test]
    fn identical_profiles_have_zero_distances() {
        let projection = PitchClassProjection {
            profile: [1.0 / 12.0; 12],
            total_magnitude: 12.0,
        };
        let frame = compare_projection(0, 0, projection.clone(), projection).unwrap();
        assert_eq!(frame.normalized_profile_l1_distance, 0.0);
        assert_eq!(frame.cosine_distance, Some(0.0));
    }
}
