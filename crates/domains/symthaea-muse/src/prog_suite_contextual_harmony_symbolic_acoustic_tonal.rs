// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive symbolic↔acoustic tonal survival evidence for the frozen
//! ProgSuite contextual-harmony experiment.
//!
//! This layer asks two separate questions for each exact A/B/C/ReturnA section:
//! how each arm's symbolic pitch-class occupancy relates to the rendered acoustic
//! pitch-class shape, and whether the *direction* of the source→contextual symbolic
//! redistribution is also present acoustically.
//!
//! Acoustic mean profiles from the upstream frame projection may have L1 mass
//! below one when some frames contain zero projected pitch-class magnitude. V1
//! therefore retains that mass explicitly and renormalizes only the nonzero mean
//! profile for tonal-shape comparisons. Silence/activity is not allowed to hide
//! inside a pitch-class-shape distance.

use super::prog_suite_contextual_harmony_audio_protocol::
    ProgSuiteContextualHarmonyAudioSurvivalProtocolV1;
use super::prog_suite_contextual_harmony_section_pitch_class::{
    ProgSuitePitchClassErrorV1, ProgSuitePitchClassLocalizationV1,
    verify_prog_suite_pitch_class_localization,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use symthaea_music_theory::prog_suite_contextual_harmony_comparison::
    ProgSuiteContextualHarmonyComparisonV1;
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::score::Score;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_SYMBOLIC_ACOUSTIC_TONAL_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-symbolic-acoustic-tonal-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSymbolicPitchClassWeightingV1 {
    EventCount,
    /// Primary V1 reference, selected before lockbox observation because the
    /// acoustic section representation is also aggregated over time.
    DurationWeighted,
    /// Sensitivity view only. Score velocity is not a calibrated cross-
    /// instrument acoustic-amplitude model.
    DurationTimesVelocity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTonalSurvivalDispositionV1 {
    NoPrimarySymbolicDistributionChange,
    PrimarySymbolicChangeWithEqualAcousticMeanShape,
    PrimarySymbolicAndAcousticMeanShapesChanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSymbolicAcousticTonalNonClaimV1 {
    SymbolicAcousticDistanceDoesNotEstablishHarmonicCorrectness,
    SmallerDistanceDoesNotEstablishBetterHarmony,
    ChangeVectorAlignmentDoesNotEstablishAudibility,
    ChangeVectorAlignmentDoesNotEstablishListenerPreference,
    ChangeVectorAlignmentDoesNotEstablishArtisticQuality,
    DurationWeightingDoesNotModelInstrumentTimbreOrAmplitude,
    VelocitySensitivityDoesNotModelCalibratedAcousticEnergy,
    AcousticPitchClassProjectionStillContainsHarmonicsAndTimbre,
    AcousticMeanRenormalizationRemovesActivityMassFromShapeComparison,
    MeanProfilesCanHideWithinSectionTemporalDifferences,
    SharedPcmRepresentationsDoNotEstablishIndependentReplication,
    SectionEventsAndFramesDoNotEstablishStatisticalIndependence,
    FixedRendererAndRepresentationDoNotEstablishGeneralization,
    LockboxDoesNotEstablishUniversalMusicGeneralization,
    EvidenceDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSymbolicPitchClassProfileV1 {
    pub weighting: ProgSuiteSymbolicPitchClassWeightingV1,
    pub note_count: usize,
    pub total_weight: f64,
    /// L1-normalized when total_weight > 0. Index 0 = C ... 9 = A.
    pub normalized_profile: [f64; 12],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSymbolicPitchClassReferencesV1 {
    pub event_count: ProgSuiteSymbolicPitchClassProfileV1,
    pub duration_weighted: ProgSuiteSymbolicPitchClassProfileV1,
    pub duration_times_velocity: ProgSuiteSymbolicPitchClassProfileV1,
}

impl ProgSuiteSymbolicPitchClassReferencesV1 {
    pub fn primary(&self) -> &ProgSuiteSymbolicPitchClassProfileV1 {
        &self.duration_weighted
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteAcousticMeanPitchClassProfileV1 {
    /// Upstream mean of per-frame normalized profiles. Its L1 mass may be below
    /// one when zero-projection frames are present.
    pub raw_mean_profile: [f64; 12],
    /// L1 mass of raw_mean_profile. With upstream V1 semantics this is also the
    /// fraction-like contribution of nonzero projected frames to the mean.
    pub profile_mass: f64,
    /// raw_mean_profile renormalized to L1=1 when profile_mass > 0, else zeros.
    pub normalized_shape: [f64; 12],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePitchClassProfileDistanceV1 {
    pub l1_distance: f64,
    /// `None` when either profile has zero L2 norm.
    pub cosine_distance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonalChangeVectorAgreementV1 {
    pub symbolic_change_vector: [f64; 12],
    pub acoustic_change_vector: [f64; 12],
    pub symbolic_change_l1_norm: f64,
    pub acoustic_change_l1_norm: f64,
    pub change_vector_l1_mismatch: f64,
    /// Directional similarity in [-1,1], or `None` if either change vector has
    /// zero L2 norm. It is a survival descriptor, not a quality score.
    pub change_vector_cosine_similarity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSymbolicAcousticTonalSectionV1 {
    pub section_index: usize,
    pub progression_changed: bool,
    pub symbolic_event_stream_changed: bool,
    pub source_symbolic: ProgSuiteSymbolicPitchClassReferencesV1,
    pub contextual_symbolic: ProgSuiteSymbolicPitchClassReferencesV1,
    pub source_acoustic: ProgSuiteAcousticMeanPitchClassProfileV1,
    pub contextual_acoustic: ProgSuiteAcousticMeanPitchClassProfileV1,
    pub source_primary_to_acoustic_shape: ProgSuitePitchClassProfileDistanceV1,
    pub contextual_primary_to_acoustic_shape: ProgSuitePitchClassProfileDistanceV1,
    pub change_agreement: ProgSuiteTonalChangeVectorAgreementV1,
    pub disposition: ProgSuiteTonalSurvivalDispositionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSymbolicAcousticTonalEvidenceV1 {
    pub version: String,
    pub subject_index: usize,
    pub subject_id: String,
    pub primary_weighting: ProgSuiteSymbolicPitchClassWeightingV1,
    pub sensitivity_weightings: Vec<ProgSuiteSymbolicPitchClassWeightingV1>,
    pub source_pitch_class_evidence: ProgSuitePitchClassLocalizationV1,
    pub sections: Vec<ProgSuiteSymbolicAcousticTonalSectionV1>,
    pub nonclaims: Vec<ProgSuiteSymbolicAcousticTonalNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteSymbolicAcousticTonalErrorV1 {
    PitchClass(ProgSuitePitchClassErrorV1),
    SectionShapeMismatch,
    SectionIdentityMismatch { position: usize },
    EmptySymbolicSection { section_index: usize },
    InvalidSymbolicDuration { section_index: usize },
    InvalidSymbolicVelocity { section_index: usize },
    InvalidSymbolicProfile { section_index: usize },
    InvalidAcousticProfile { section_index: usize },
    EvidenceMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn measure_prog_suite_symbolic_acoustic_tonal_evidence(
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    pitch_class: &ProgSuitePitchClassLocalizationV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<ProgSuiteSymbolicAcousticTonalEvidenceV1, ProgSuiteSymbolicAcousticTonalErrorV1> {
    verify_prog_suite_pitch_class_localization(
        pitch_class,
        protocol,
        comparison,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )
    .map_err(ProgSuiteSymbolicAcousticTonalErrorV1::PitchClass)?;

    if comparison.sections.len() != pitch_class.sections.len() {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::SectionShapeMismatch);
    }

    let mut sections = Vec::with_capacity(comparison.sections.len());
    for (position, (symbolic_section, acoustic_section)) in comparison
        .sections
        .iter()
        .zip(&pitch_class.sections)
        .enumerate()
    {
        if symbolic_section.section_index != acoustic_section.section_index
            || symbolic_section.progression_changed != acoustic_section.progression_changed
            || (!symbolic_section.all_events.exact_event_stream_match)
                != acoustic_section.symbolic_event_stream_changed
        {
            return Err(ProgSuiteSymbolicAcousticTonalErrorV1::SectionIdentityMismatch {
                position,
            });
        }

        let section_index = symbolic_section.section_index;
        let source_symbolic = symbolic_reference_set(
            &comparison.source_realization.score,
            symbolic_section.start,
            symbolic_section.end,
            section_index,
        )?;
        let contextual_symbolic = symbolic_reference_set(
            &comparison.contextual_realization.score,
            symbolic_section.start,
            symbolic_section.end,
            section_index,
        )?;
        let source_acoustic = acoustic_mean_shape(
            acoustic_section.summary.mean_source_normalized_profile,
            section_index,
        )?;
        let contextual_acoustic = acoustic_mean_shape(
            acoustic_section.summary.mean_contextual_normalized_profile,
            section_index,
        )?;

        let source_primary_to_acoustic_shape = profile_distance(
            &source_symbolic.primary().normalized_profile,
            &source_acoustic.normalized_shape,
        )
        .ok_or(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidAcousticProfile {
            section_index,
        })?;
        let contextual_primary_to_acoustic_shape = profile_distance(
            &contextual_symbolic.primary().normalized_profile,
            &contextual_acoustic.normalized_shape,
        )
        .ok_or(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidAcousticProfile {
            section_index,
        })?;

        let symbolic_change_vector = subtract_profiles(
            &contextual_symbolic.primary().normalized_profile,
            &source_symbolic.primary().normalized_profile,
        );
        let acoustic_change_vector = subtract_profiles(
            &contextual_acoustic.normalized_shape,
            &source_acoustic.normalized_shape,
        );
        let symbolic_change_l1_norm = l1_norm(&symbolic_change_vector);
        let acoustic_change_l1_norm = l1_norm(&acoustic_change_vector);
        let mismatch = subtract_profiles(&acoustic_change_vector, &symbolic_change_vector);
        let change_vector_l1_mismatch = l1_norm(&mismatch);
        let change_vector_cosine_similarity =
            cosine_similarity(&symbolic_change_vector, &acoustic_change_vector);
        if !symbolic_change_l1_norm.is_finite()
            || !acoustic_change_l1_norm.is_finite()
            || !change_vector_l1_mismatch.is_finite()
            || change_vector_cosine_similarity.is_some_and(|value| !value.is_finite())
        {
            return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidAcousticProfile {
                section_index,
            });
        }

        let disposition = if source_symbolic.primary().normalized_profile
            == contextual_symbolic.primary().normalized_profile
        {
            ProgSuiteTonalSurvivalDispositionV1::NoPrimarySymbolicDistributionChange
        } else if source_acoustic.normalized_shape == contextual_acoustic.normalized_shape {
            ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicChangeWithEqualAcousticMeanShape
        } else {
            ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicAndAcousticMeanShapesChanged
        };

        sections.push(ProgSuiteSymbolicAcousticTonalSectionV1 {
            section_index,
            progression_changed: symbolic_section.progression_changed,
            symbolic_event_stream_changed: !symbolic_section.all_events.exact_event_stream_match,
            source_symbolic,
            contextual_symbolic,
            source_acoustic,
            contextual_acoustic,
            source_primary_to_acoustic_shape,
            contextual_primary_to_acoustic_shape,
            change_agreement: ProgSuiteTonalChangeVectorAgreementV1 {
                symbolic_change_vector,
                acoustic_change_vector,
                symbolic_change_l1_norm,
                acoustic_change_l1_norm,
                change_vector_l1_mismatch,
                change_vector_cosine_similarity,
            },
            disposition,
        });
    }

    Ok(ProgSuiteSymbolicAcousticTonalEvidenceV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_SYMBOLIC_ACOUSTIC_TONAL_VERSION.into(),
        subject_index: pitch_class.subject_index,
        subject_id: pitch_class.subject_id.clone(),
        primary_weighting: ProgSuiteSymbolicPitchClassWeightingV1::DurationWeighted,
        sensitivity_weightings: vec![
            ProgSuiteSymbolicPitchClassWeightingV1::EventCount,
            ProgSuiteSymbolicPitchClassWeightingV1::DurationTimesVelocity,
        ],
        source_pitch_class_evidence: pitch_class.clone(),
        sections,
        nonclaims: required_nonclaims(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_prog_suite_symbolic_acoustic_tonal_evidence(
    evidence: &ProgSuiteSymbolicAcousticTonalEvidenceV1,
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<(), ProgSuiteSymbolicAcousticTonalErrorV1> {
    let canonical = measure_prog_suite_symbolic_acoustic_tonal_evidence(
        protocol,
        comparison,
        &evidence.source_pitch_class_evidence,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )?;
    if &canonical != evidence {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::EvidenceMismatch);
    }
    Ok(())
}

fn symbolic_reference_set(
    score: &Score,
    start: Duration,
    end: Duration,
    section_index: usize,
) -> Result<ProgSuiteSymbolicPitchClassReferencesV1, ProgSuiteSymbolicAcousticTonalErrorV1> {
    let notes = score
        .notes
        .iter()
        .filter(|note| {
            compare_duration(note.onset, start) != Ordering::Less
                && compare_duration(note.onset, end) == Ordering::Less
        })
        .collect::<Vec<_>>();
    if notes.is_empty() {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::EmptySymbolicSection {
            section_index,
        });
    }

    let mut event_weights = [0.0_f64; 12];
    let mut duration_weights = [0.0_f64; 12];
    let mut duration_velocity_weights = [0.0_f64; 12];
    for note in &notes {
        if compare_duration(note.duration, Duration::zero()) != Ordering::Greater {
            return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicDuration {
                section_index,
            });
        }
        if !note.velocity.is_finite() || !(0.0..=1.0).contains(&note.velocity) {
            return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicVelocity {
                section_index,
            });
        }
        let duration = note.duration.beats();
        if !duration.is_finite() || duration <= 0.0 {
            return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicDuration {
                section_index,
            });
        }
        let pitch_class = note.pitch.pitch_class().value() as usize;
        event_weights[pitch_class] += 1.0;
        duration_weights[pitch_class] += duration;
        duration_velocity_weights[pitch_class] += duration * f64::from(note.velocity);
    }

    Ok(ProgSuiteSymbolicPitchClassReferencesV1 {
        event_count: make_symbolic_profile(
            ProgSuiteSymbolicPitchClassWeightingV1::EventCount,
            notes.len(),
            event_weights,
            section_index,
        )?,
        duration_weighted: make_symbolic_profile(
            ProgSuiteSymbolicPitchClassWeightingV1::DurationWeighted,
            notes.len(),
            duration_weights,
            section_index,
        )?,
        duration_times_velocity: make_symbolic_profile(
            ProgSuiteSymbolicPitchClassWeightingV1::DurationTimesVelocity,
            notes.len(),
            duration_velocity_weights,
            section_index,
        )?,
    })
}

fn make_symbolic_profile(
    weighting: ProgSuiteSymbolicPitchClassWeightingV1,
    note_count: usize,
    weights: [f64; 12],
    section_index: usize,
) -> Result<ProgSuiteSymbolicPitchClassProfileV1, ProgSuiteSymbolicAcousticTonalErrorV1> {
    if weights.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicProfile {
            section_index,
        });
    }
    let total_weight = weights.iter().sum::<f64>();
    if !total_weight.is_finite() || total_weight < 0.0 {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicProfile {
            section_index,
        });
    }
    let mut normalized_profile = [0.0_f64; 12];
    if total_weight > 0.0 {
        for (target, source) in normalized_profile.iter_mut().zip(weights) {
            *target = source / total_weight;
        }
    }
    validate_unit_or_zero_profile(&normalized_profile, section_index)?;
    Ok(ProgSuiteSymbolicPitchClassProfileV1 {
        weighting,
        note_count,
        total_weight,
        normalized_profile,
    })
}

fn acoustic_mean_shape(
    raw_mean_profile: [f64; 12],
    section_index: usize,
) -> Result<ProgSuiteAcousticMeanPitchClassProfileV1, ProgSuiteSymbolicAcousticTonalErrorV1> {
    if raw_mean_profile
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidAcousticProfile {
            section_index,
        });
    }
    let profile_mass = raw_mean_profile.iter().sum::<f64>();
    if !profile_mass.is_finite() || profile_mass < 0.0 || profile_mass > 1.0 + 1.0e-9 {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidAcousticProfile {
            section_index,
        });
    }
    let mut normalized_shape = [0.0_f64; 12];
    if profile_mass > 0.0 {
        for (target, source) in normalized_shape.iter_mut().zip(raw_mean_profile) {
            *target = source / profile_mass;
        }
    }
    validate_unit_or_zero_profile(&normalized_shape, section_index)
        .map_err(|_| ProgSuiteSymbolicAcousticTonalErrorV1::InvalidAcousticProfile {
            section_index,
        })?;
    Ok(ProgSuiteAcousticMeanPitchClassProfileV1 {
        raw_mean_profile,
        profile_mass,
        normalized_shape,
    })
}

fn validate_unit_or_zero_profile(
    profile: &[f64; 12],
    section_index: usize,
) -> Result<(), ProgSuiteSymbolicAcousticTonalErrorV1> {
    if profile.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicProfile {
            section_index,
        });
    }
    let sum = profile.iter().sum::<f64>();
    if !sum.is_finite() || (sum > 0.0 && (sum - 1.0).abs() > 1.0e-9) {
        return Err(ProgSuiteSymbolicAcousticTonalErrorV1::InvalidSymbolicProfile {
            section_index,
        });
    }
    Ok(())
}

fn profile_distance(
    left: &[f64; 12],
    right: &[f64; 12],
) -> Option<ProgSuitePitchClassProfileDistanceV1> {
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return None;
    }
    let mut l1_distance = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut left_squared = 0.0_f64;
    let mut right_squared = 0.0_f64;
    for (&a, &b) in left.iter().zip(right) {
        l1_distance += (a - b).abs();
        dot += a * b;
        left_squared += a * a;
        right_squared += b * b;
    }
    let cosine_distance = if left_squared > 0.0 && right_squared > 0.0 {
        let similarity = (dot / (left_squared.sqrt() * right_squared.sqrt())).clamp(-1.0, 1.0);
        Some(1.0 - similarity)
    } else {
        None
    };
    Some(ProgSuitePitchClassProfileDistanceV1 {
        l1_distance,
        cosine_distance,
    })
}

fn subtract_profiles(left: &[f64; 12], right: &[f64; 12]) -> [f64; 12] {
    let mut result = [0.0_f64; 12];
    for index in 0..12 {
        result[index] = left[index] - right[index];
    }
    result
}

fn l1_norm(vector: &[f64; 12]) -> f64 {
    vector.iter().map(|value| value.abs()).sum()
}

fn cosine_similarity(left: &[f64; 12], right: &[f64; 12]) -> Option<f64> {
    let mut dot = 0.0_f64;
    let mut left_squared = 0.0_f64;
    let mut right_squared = 0.0_f64;
    for (&a, &b) in left.iter().zip(right) {
        dot += a * b;
        left_squared += a * a;
        right_squared += b * b;
    }
    if left_squared == 0.0 || right_squared == 0.0 {
        None
    } else {
        Some((dot / (left_squared.sqrt() * right_squared.sqrt())).clamp(-1.0, 1.0))
    }
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

fn required_nonclaims() -> Vec<ProgSuiteSymbolicAcousticTonalNonClaimV1> {
    vec![
        ProgSuiteSymbolicAcousticTonalNonClaimV1::SymbolicAcousticDistanceDoesNotEstablishHarmonicCorrectness,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::SmallerDistanceDoesNotEstablishBetterHarmony,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::ChangeVectorAlignmentDoesNotEstablishAudibility,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::ChangeVectorAlignmentDoesNotEstablishListenerPreference,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::ChangeVectorAlignmentDoesNotEstablishArtisticQuality,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::DurationWeightingDoesNotModelInstrumentTimbreOrAmplitude,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::VelocitySensitivityDoesNotModelCalibratedAcousticEnergy,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::AcousticPitchClassProjectionStillContainsHarmonicsAndTimbre,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::AcousticMeanRenormalizationRemovesActivityMassFromShapeComparison,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::MeanProfilesCanHideWithinSectionTemporalDifferences,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::SharedPcmRepresentationsDoNotEstablishIndependentReplication,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::SectionEventsAndFramesDoNotEstablishStatisticalIndependence,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::FixedRendererAndRepresentationDoNotEstablishGeneralization,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::LockboxDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteSymbolicAcousticTonalNonClaimV1::EvidenceDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::score::{Emphasis, PartId, ScoreNote, VoiceRole};

    #[test]
    fn exact_duration_ordering_handles_equivalent_rationals() {
        assert_eq!(
            compare_duration(Duration::new(2, 4), Duration::new(1, 2)),
            Ordering::Equal
        );
        assert_eq!(
            compare_duration(Duration::new(3, 4), Duration::new(2, 3)),
            Ordering::Greater
        );
    }

    #[test]
    fn primary_symbolic_profile_is_duration_weighted() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::new(PitchClass::C, 4),
            onset: Duration::zero(),
            duration: Duration::quarter(),
            velocity: 0.5,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::new(PitchClass::G, 4),
            onset: Duration::quarter(),
            duration: Duration::new(3, 1),
            velocity: 1.0,
            role: VoiceRole::Harmony,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        let refs = symbolic_reference_set(&score, Duration::zero(), Duration::whole(), 0).unwrap();
        assert_eq!(
            refs.primary().weighting,
            ProgSuiteSymbolicPitchClassWeightingV1::DurationWeighted
        );
        assert_eq!(refs.event_count.normalized_profile[0], 0.5);
        assert_eq!(refs.event_count.normalized_profile[7], 0.5);
        assert_eq!(refs.duration_weighted.normalized_profile[0], 0.25);
        assert_eq!(refs.duration_weighted.normalized_profile[7], 0.75);
    }

    #[test]
    fn acoustic_mean_shape_separates_mass_from_shape() {
        let mut raw = [0.0_f64; 12];
        raw[0] = 0.125;
        raw[7] = 0.375;
        let acoustic = acoustic_mean_shape(raw, 0).unwrap();
        assert!((acoustic.profile_mass - 0.5).abs() < 1.0e-12);
        assert!((acoustic.normalized_shape[0] - 0.25).abs() < 1.0e-12);
        assert!((acoustic.normalized_shape[7] - 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn identical_profiles_have_zero_distance() {
        let mut profile = [0.0_f64; 12];
        profile[0] = 0.25;
        profile[4] = 0.25;
        profile[7] = 0.5;
        let distance = profile_distance(&profile, &profile).unwrap();
        assert_eq!(distance.l1_distance, 0.0);
        assert!(distance.cosine_distance.unwrap().abs() < 1.0e-12);
    }

    #[test]
    fn change_vector_similarity_preserves_direction_sign() {
        let mut left = [0.0_f64; 12];
        let mut right = [0.0_f64; 12];
        left[0] = -0.5;
        left[7] = 0.5;
        right[0] = 0.5;
        right[7] = -0.5;
        assert!((cosine_similarity(&left, &left).unwrap() - 1.0).abs() < 1.0e-12);
        assert!((cosine_similarity(&left, &right).unwrap() + 1.0).abs() < 1.0e-12);
    }
}
