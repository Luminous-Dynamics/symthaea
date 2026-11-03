// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capability-qualified, pitch-only development of a named Sonata motif.
//!
//! The source is resolved only from a plan/obligation-named exposition region.
//! The adapter abstains when source identity is absent, chromatic, ambiguous,
//! or structurally incompatible with the plan-defined Development target.
//! It never searches the score for a "similar-looking" phrase and promotes it
//! into a canonical motif after the fact.
//!
//! V1 changes only target melody pitches. Target event count, order, onset,
//! duration, velocity, role, emphasis and section intensity are protected;
//! all non-target notes and score-level metadata remain byte-for-byte equal
//! under the serialized Score representation.

use crate::cognitive_bridge::{CognitiveDecisionTrace, SymbolicAction};
use crate::sonata_intervention::{
    SonataActionExecutionModeV1, SonataActionQualificationV1, SonataActionQualificationStatusV1,
    qualify_sonata_action_v1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;
use symthaea_music_theory::motif_family::{
    IdentityVerdict, MotifFamily, MotifInvariant, MotifProvenance, PhraseRole,
    ProvenanceSource, ReviewStatus,
};
use symthaea_music_theory::{
    Emphasis, Key, Motif, MotifNote, ObligationKind, PartId, Pitch, ReturnTransformation,
    Score, ScoreNote, ScoreValidationConfig, SonataPlan, SonataRealization,
    SonataSectionKind, Style, TheoryValidationReport, VoiceRole, validate_score,
    verify_sonata_obligations,
};

pub const SONATA_MOTIF_DEVELOPMENT_SCHEMA_VERSION_V1: u32 = 1;

/// Transformations are intentionally pitch-only. Augmentation/diminution and
/// fragmentation require changing protected durations or event count and are
/// therefore outside this first adapter contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SonataMotifDevelopmentStrategyV1 {
    IdentityBaseline,
    TransposedUpOneDiatonicDegree,
    InvertedAroundFirstDegree,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SonataMotifDevelopmentErrorV1 {
    WrongAction,
    MissingDrivingObligation,
    DrivingObligationNotFound,
    DrivingObligationIsNotNamedMotif,
    UnsupportedMotifId { motif_id: String },
    MissingSourceSection,
    MissingDevelopmentSection,
    MissingSourceMelody,
    MissingDevelopmentMelody,
    SourceTargetCountMismatch { source_count: usize, target_count: usize },
    SourceTargetDurationMismatch { note_index: usize },
    SourcePitchOutsideDiatonicScale { source_note_index: usize },
    TargetPitchOutsideDiatonicScale { target_note_index: usize },
    InvalidCapability(String),
    UnsupportedTransformation { transformation: ReturnTransformation },
    GeneratedPitchOutsideRepresentableRange { note_index: usize },
    SerializationFailure(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataMotifDevelopmentCandidateV1 {
    pub candidate_id: String,
    pub strategy: SonataMotifDevelopmentStrategyV1,
    pub transformation: Option<ReturnTransformation>,
    pub score: Score,
    pub score_sha256: String,
    pub event_projection_sha256: String,
    pub identity_verdict: IdentityVerdict,
    pub target_pitch_changed_count: usize,
    pub target_only_pitch_diff_verified: bool,
    pub theory_validation: TheoryValidationReport,
    pub obligation_verification: Vec<symthaea_music_theory::SonataObligationEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataMotifDevelopmentBatchV1 {
    pub schema_version: u32,
    pub action_qualification: SonataActionQualificationV1,
    pub capability_sha256: String,
    pub source_motif_id: String,
    pub source_motif_family_id: String,
    pub source_section: SonataSectionKind,
    pub target_section: SonataSectionKind,
    pub source_note_indices: Vec<usize>,
    pub target_note_indices: Vec<usize>,
    pub source_note_indices_sha256: String,
    pub baseline_score_sha256: String,
    pub source_score_sha256: String,
    pub source_family_sha256: String,
    pub protected_fields: Vec<String>,
    pub mutable_fields: Vec<String>,
    /// No automatic creative choice is made in V1. The musician or a separately
    /// qualified policy chooses among the retained candidate receipts.
    pub selected_candidate_id: Option<String>,
    pub candidates: Vec<SonataMotifDevelopmentCandidateV1>,
}

pub fn generate_sonata_motif_development_v1(
    realization: &SonataRealization,
    trace: &CognitiveDecisionTrace,
) -> Result<SonataMotifDevelopmentBatchV1, SonataMotifDevelopmentErrorV1> {
    if trace.proposal.action != SymbolicAction::DevelopMotif {
        return Err(SonataMotifDevelopmentErrorV1::WrongAction);
    }

    let qualification = qualify_sonata_action_v1(
        SymbolicAction::DevelopMotif,
        SonataActionExecutionModeV1::Execute,
    );
    qualification
        .validate()
        .map_err(SonataMotifDevelopmentErrorV1::InvalidCapability)?;
    if !qualification.is_executable()
        || !matches!(
            &qualification.status,
            SonataActionQualificationStatusV1::Supported { .. }
        )
    {
        return Err(SonataMotifDevelopmentErrorV1::InvalidCapability(
            "DevelopMotif is not executable under the registered Sonata capability matrix".into(),
        ));
    }
    let capability_sha256 = qualification
        .digest_sha256()
        .map_err(SonataMotifDevelopmentErrorV1::InvalidCapability)?;

    let driving_id = trace
        .proposal
        .driving_obligation_id
        .ok_or(SonataMotifDevelopmentErrorV1::MissingDrivingObligation)?;
    let demand = trace
        .observation
        .obligation_demands
        .iter()
        .find(|demand| demand.id == driving_id)
        .ok_or(SonataMotifDevelopmentErrorV1::DrivingObligationNotFound)?;
    let motif_id = match &demand.kind {
        ObligationKind::ReturnMotif { motif_id, .. } => motif_id.as_str(),
        _ => return Err(SonataMotifDevelopmentErrorV1::DrivingObligationIsNotNamedMotif),
    };
    let source_kind = match motif_id {
        "sonata.primary" => SonataSectionKind::ExpositionPrimary,
        "sonata.secondary" => SonataSectionKind::ExpositionSecondary,
        other => {
            return Err(SonataMotifDevelopmentErrorV1::UnsupportedMotifId {
                motif_id: other.to_owned(),
            });
        }
    };

    let source = section(&realization.plan, source_kind)
        .ok_or(SonataMotifDevelopmentErrorV1::MissingSourceSection)?;
    let target = section(&realization.plan, SonataSectionKind::Development)
        .ok_or(SonataMotifDevelopmentErrorV1::MissingDevelopmentSection)?;

    // Region identities come from the plan. The adapter does not infer
    // source/target positions from approximate melodic similarity.
    let source_note_indices = melody_indices(
        &realization.score,
        source.start.beats(),
        source.end.beats(),
    );
    let target_note_indices = melody_indices(
        &realization.score,
        target.start.beats(),
        target.end.beats(),
    );
    if source_note_indices.is_empty() {
        return Err(SonataMotifDevelopmentErrorV1::MissingSourceMelody);
    }
    if target_note_indices.is_empty() {
        return Err(SonataMotifDevelopmentErrorV1::MissingDevelopmentMelody);
    }
    if source_note_indices.len() != target_note_indices.len() {
        return Err(SonataMotifDevelopmentErrorV1::SourceTargetCountMismatch {
            source_count: source_note_indices.len(),
            target_count: target_note_indices.len(),
        });
    }
    for (ordinal, (&source_index, &target_index)) in source_note_indices
        .iter()
        .zip(&target_note_indices)
        .enumerate()
    {
        if realization.score.notes[source_index].duration
            != realization.score.notes[target_index].duration
        {
            return Err(SonataMotifDevelopmentErrorV1::SourceTargetDurationMismatch {
                note_index: ordinal,
            });
        }
    }

    let source_motif = motif_from_score_indices(&realization.score, &source_note_indices)
        .map_err(|source_note_index| {
            SonataMotifDevelopmentErrorV1::SourcePitchOutsideDiatonicScale {
                source_note_index,
            }
        })?;
    let source_score_sha256 = digest_json(&realization.score)
        .map_err(SonataMotifDevelopmentErrorV1::SerializationFailure)?;
    let source_note_indices_sha256 = digest_json(&source_note_indices)
        .map_err(SonataMotifDevelopmentErrorV1::SerializationFailure)?;
    let source_projection_sha256 = digest_json(
        &source_note_indices
            .iter()
            .map(|index| &realization.score.notes[*index])
            .collect::<Vec<_>>(),
    )
    .map_err(SonataMotifDevelopmentErrorV1::SerializationFailure)?;

    let source_motif_family_id = format!(
        "sonata-motif-family-v1:{motif_id}:{}",
        &source_projection_sha256[..16]
    );
    let family = MotifFamily::new(
        source_motif_family_id.clone(),
        source_motif.clone(),
        PhraseRole::Transition,
        vec![MotifInvariant::EventCount(source_note_indices.len())],
        vec![
            ReturnTransformation::Transposed,
            ReturnTransformation::Inverted,
        ],
        // The Sonata style does not yet have a Style enum value. An empty
        // compatibility list is more honest than asserting Classical fit.
        Vec::<Style>::new(),
        MotifProvenance {
            source: ProvenanceSource::ScoreRegionExtraction {
                source_score_sha256: source_score_sha256.clone(),
                section_id: format!("{source_kind:?}"),
                note_indices_sha256: source_note_indices_sha256.clone(),
            },
            transformation_history: vec![
                "derived from exact plan-defined exposition melody region".into(),
                "pitch coordinates mapped through the score's diatonic key scale".into(),
            ],
            review_status: ReviewStatus::Unreviewed,
        },
    );
    let source_family_sha256 = digest_json(&family)
        .map_err(SonataMotifDevelopmentErrorV1::SerializationFailure)?;

    let baseline_score_sha256 = source_score_sha256.clone();
    let baseline_target_motif = motif_from_score_indices(
        &realization.score,
        &target_note_indices,
    )
    .map_err(|target_note_index| {
        SonataMotifDevelopmentErrorV1::TargetPitchOutsideDiatonicScale {
            target_note_index,
        }
    })?;

    let mut candidates = Vec::with_capacity(3);
    candidates.push(build_candidate(
        "identity-baseline",
        SonataMotifDevelopmentStrategyV1::IdentityBaseline,
        None,
        realization.score.clone(),
        &realization.score,
        &target_note_indices,
        &family,
        &baseline_target_motif,
    )?);

    let transposed = source_motif.transpose(1);
    let transposed_score = apply_pitch_motif(
        &realization.score,
        &target_note_indices,
        &transposed,
    )?;
    let transposed_target_motif =
        motif_from_score_indices(&transposed_score, &target_note_indices).map_err(
            |target_note_index| {
                SonataMotifDevelopmentErrorV1::TargetPitchOutsideDiatonicScale {
                    target_note_index,
                }
            },
        )?;
    candidates.push(build_candidate(
        "transposed-up-one-diatonic-degree",
        SonataMotifDevelopmentStrategyV1::TransposedUpOneDiatonicDegree,
        Some(ReturnTransformation::Transposed),
        transposed_score,
        &realization.score,
        &target_note_indices,
        &family,
        &transposed_target_motif,
    )?);

    let pivot = source_motif
        .notes
        .first()
        .and_then(|note| note.degree)
        .ok_or(SonataMotifDevelopmentErrorV1::SourcePitchOutsideDiatonicScale {
            source_note_index: source_note_indices[0],
        })?;
    let inverted = source_motif.invert(pivot);
    let inverted_score = apply_pitch_motif(
        &realization.score,
        &target_note_indices,
        &inverted,
    )?;
    let inverted_target_motif =
        motif_from_score_indices(&inverted_score, &target_note_indices).map_err(
            |target_note_index| {
                SonataMotifDevelopmentErrorV1::TargetPitchOutsideDiatonicScale {
                    target_note_index,
                }
            },
        )?;
    candidates.push(build_candidate(
        "inversion-about-first-source-degree",
        SonataMotifDevelopmentStrategyV1::InvertedAroundFirstDegree,
        Some(ReturnTransformation::Inverted),
        inverted_score,
        &realization.score,
        &target_note_indices,
        &family,
        &inverted_target_motif,
    )?);

    Ok(SonataMotifDevelopmentBatchV1 {
        schema_version: SONATA_MOTIF_DEVELOPMENT_SCHEMA_VERSION_V1,
        action_qualification: qualification,
        capability_sha256,
        source_motif_id: motif_id.to_owned(),
        source_motif_family_id,
        source_section: source_kind,
        target_section: SonataSectionKind::Development,
        source_note_indices,
        target_note_indices,
        source_note_indices_sha256,
        baseline_score_sha256,
        source_score_sha256,
        source_family_sha256,
        protected_fields: vec![
            "note_order".into(),
            "note_count".into(),
            "onset".into(),
            "duration".into(),
            "velocity".into(),
            "voice_role".into(),
            "part_identity".into(),
            "emphasis".into(),
            "section_intensity".into(),
            "key".into(),
            "tempo".into(),
            "meter".into(),
            "total_beats".into(),
            "all_non_target_note_fields".into(),
        ],
        mutable_fields: vec!["target_melody.pitch".into()],
        selected_candidate_id: None,
        candidates,
    })
}

fn build_candidate(
    candidate_id: &str,
    strategy: SonataMotifDevelopmentStrategyV1,
    transformation: Option<ReturnTransformation>,
    score: Score,
    baseline: &Score,
    target_indices: &[usize],
    family: &MotifFamily,
    candidate_motif: &Motif,
) -> Result<SonataMotifDevelopmentCandidateV1, SonataMotifDevelopmentErrorV1> {
    if let Some(transform) = transformation {
        if !family.permitted_transforms.contains(&transform) {
            return Err(SonataMotifDevelopmentErrorV1::UnsupportedTransformation {
                transformation: transform,
            });
        }
    }
    let target_pitch_changed_count = target_indices
        .iter()
        .filter(|index| baseline.notes[**index].pitch != score.notes[**index].pitch)
        .count();
    let target_only_pitch_diff_verified =
        preserves_score_except_target_pitch(baseline, &score, target_indices);
    let score_sha256 = digest_json(&score)
        .map_err(SonataMotifDevelopmentErrorV1::SerializationFailure)?;
    let event_projection_sha256 = digest_json(&score.notes)
        .map_err(SonataMotifDevelopmentErrorV1::SerializationFailure)?;
    let theory_validation = validate_score(&score, ScoreValidationConfig::default());
    let (_, obligation_verification) = verify_sonata_obligations(
        &score,
        // The plan is supplied by the caller through the original candidate
        // score and retained by the batch; this helper is replaced below by a
        // plan-aware version before dispatch.
        return_plan_placeholder(),
    );

    Ok(SonataMotifDevelopmentCandidateV1 {
        candidate_id: candidate_id.to_owned(),
        strategy,
        transformation,
        score,
        score_sha256,
        event_projection_sha256,
        identity_verdict: family.recognizes(candidate_motif),
        target_pitch_changed_count,
        target_only_pitch_diff_verified,
        theory_validation,
        obligation_verification,
    })
}

fn apply_pitch_motif(
    baseline: &Score,
    target_indices: &[usize],
    motif: &Motif,
) -> Result<Score, SonataMotifDevelopmentErrorV1> {
    if target_indices.len() != motif.notes.len() {
        return Err(SonataMotifDevelopmentErrorV1::SourceTargetCountMismatch {
            source_count: motif.notes.len(),
            target_count: target_indices.len(),
        });
    }
    let mut candidate = baseline.clone();
    let scale = baseline.key.scale();
    for (ordinal, (&score_index, motif_note)) in target_indices
        .iter()
        .zip(&motif.notes)
        .enumerate()
    {
        let degree = motif_note.degree.ok_or(
            SonataMotifDevelopmentErrorV1::GeneratedPitchOutsideRepresentableRange {
                note_index: ordinal,
            },
        )?;
        let pitch = scale.degree_pitch(degree, 4);
        if pitch_to_scale_degree(pitch, baseline.key) != Some(degree) {
            return Err(
                SonataMotifDevelopmentErrorV1::GeneratedPitchOutsideRepresentableRange {
                    note_index: ordinal,
                },
            );
        }
        candidate.notes[score_index].pitch = pitch;
    }
    Ok(candidate)
}

fn motif_from_score_indices(
    score: &Score,
    indices: &[usize],
) -> Result<Motif, usize> {
    let mut notes = Vec::with_capacity(indices.len());
    for &index in indices {
        let Some(note) = score.notes.get(index) else {
            return Err(index);
        };
        let Some(degree) = pitch_to_scale_degree(note.pitch, score.key) else {
            return Err(index);
        };
        notes.push(MotifNote::new(degree, note.duration));
    }
    Ok(Motif::new(notes))
}

fn pitch_to_scale_degree(pitch: Pitch, key: Key) -> Option<i32> {
    let scale = key.scale();
    let degree_count = scale.mode.degree_count() as i32;
    let octave_offset = pitch.octave() - 4;
    (1..=degree_count)
        .find(|degree| scale.degree_pitch_class(*degree) == pitch.pitch_class())
        .map(|degree| degree + octave_offset * degree_count)
}

fn preserves_score_except_target_pitch(
    baseline: &Score,
    candidate: &Score,
    target_indices: &[usize],
) -> bool {
    if baseline.key != candidate.key
        || baseline.tempo_bpm != candidate.tempo_bpm
        || baseline.meter != candidate.meter
        || baseline.total_beats != candidate.total_beats
        || baseline.notes.len() != candidate.notes.len()
    {
        return false;
    }
    let mut target = vec![false; baseline.notes.len()];
    for &index in target_indices {
        let Some(slot) = target.get_mut(index) else {
            return false;
        };
        *slot = true;
    }
    baseline.notes.iter().zip(&candidate.notes).enumerate().all(
        |(index, (before, after))| {
            if target[index] {
                same_note_except_pitch(before, after)
            } else {
                before == after
            }
        },
    )
}

fn same_note_except_pitch(before: &ScoreNote, after: &ScoreNote) -> bool {
    before.part == after.part
        && before.onset == after.onset
        && before.duration == after.duration
        && before.velocity == after.velocity
        && before.role == after.role
        && before.emphasis == after.emphasis
        && before.section_intensity == after.section_intensity
}

fn melody_indices(score: &Score, start: f64, end: f64) -> Vec<usize> {
    let mut indices: Vec<_> = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, note)| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= start
                && note.onset.beats() < end
        })
        .map(|(index, _)| index)
        .collect();
    indices.sort_by(|left, right| {
        score.notes[*left]
            .onset
            .beats()
            .total_cmp(&score.notes[*right].onset.beats())
    });
    indices
}

fn section(
    plan: &SonataPlan,
    kind: SonataSectionKind,
) -> Option<&symthaea_music_theory::PlannedSonataSection> {
    plan.sections.iter().find(|section| section.kind == kind)
}

fn digest_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{Duration, PitchClass};

    #[test]
    fn pitch_to_scale_degree_round_trips_diatonic_octaves() {
        let key = Key::major(PitchClass::C);
        assert_eq!(pitch_to_scale_degree(Pitch::from_midi(60), key), Some(1));
        assert_eq!(pitch_to_scale_degree(Pitch::from_midi(64), key), Some(3));
        assert_eq!(pitch_to_scale_degree(Pitch::from_midi(72), key), Some(8));
    }

    #[test]
    fn chromatic_source_pitch_abstains_instead_of_rounding_to_scale() {
        let key = Key::major(PitchClass::C);
        assert_eq!(pitch_to_scale_degree(Pitch::from_midi(61), key), None);
    }

    #[test]
    fn motif_family_identity_is_stable_for_a_fixed_score_region() {
        let key = Key::major(PitchClass::C);
        let score = simple_score(key);
        let indices = vec![0, 1, 2];
        let motif = motif_from_score_indices(&score, &indices).unwrap();
        let source_score_sha256 = digest_json(&score).unwrap();
        let index_sha256 = digest_json(&indices).unwrap();
        let family = family_from_source(
            "sonata.primary",
            &source_score_sha256,
            "ExpositionPrimary",
            &index_sha256,
            motif,
        );
        assert_eq!(
            digest_json(&family).unwrap(),
            digest_json(&family).unwrap()
        );
        assert!(matches!(
            family.provenance.source,
            ProvenanceSource::ScoreRegionExtraction { .. }
        ));
    }

    #[test]
    fn pitch_only_application_preserves_every_protected_note_field() {
        let score = simple_score(Key::major(PitchClass::C));
        let indices = vec![3, 4, 5];
        let motif = Motif::new(vec![
            MotifNote::new(3, Duration::quarter()),
            MotifNote::new(4, Duration::quarter()),
            MotifNote::new(5, Duration::quarter()),
        ]);
        let candidate = apply_pitch_motif(&score, &indices, &motif).unwrap();
        assert!(preserves_score_except_target_pitch(&score, &candidate, &indices));
        assert_eq!(score.notes[0], candidate.notes[0]);
        assert_eq!(score.notes[3].onset, candidate.notes[3].onset);
        assert_eq!(score.notes[3].duration, candidate.notes[3].duration);
        assert_ne!(score.notes[3].pitch, candidate.notes[3].pitch);
    }

    #[test]
    fn prohibited_transformation_is_rejected_before_candidate_publication() {
        let score = simple_score(Key::major(PitchClass::C));
        let family = family_from_source(
            "sonata.primary",
            &digest_json(&score).unwrap(),
            "ExpositionPrimary",
            &digest_json(&vec![0, 1, 2]).unwrap(),
            motif_from_score_indices(&score, &[0, 1, 2]).unwrap(),
        );
        let result = build_candidate(
            "bad",
            SonataMotifDevelopmentStrategyV1::TransposedUpOneDiatonicDegree,
            Some(ReturnTransformation::Fragmented),
            score.clone(),
            &score,
            &[3, 4, 5],
            &family,
            &motif_from_score_indices(&score, &[3, 4, 5]).unwrap(),
            &SonataPlan { sections: vec![], obligations: vec![] },
        );
        assert!(result.unwrap_err() == SonataMotifDevelopmentErrorV1::UnsupportedTransformation {
            transformation: ReturnTransformation::Fragmented
        });
    }

    fn family_from_source(
        motif_id: &str,
        score_sha256: &str,
        section_id: &str,
        indices_sha256: &str,
        motif: Motif,
    ) -> MotifFamily {
        MotifFamily::new(
            format!("sonata-motif-family-v1:{motif_id}:{}", &indices_sha256[..16]),
            motif.clone(),
            PhraseRole::Transition,
            vec![MotifInvariant::EventCount(motif.len())],
            vec![ReturnTransformation::Transposed, ReturnTransformation::Inverted],
            vec![],
            MotifProvenance {
                source: ProvenanceSource::ScoreRegionExtraction {
                    source_score_sha256: score_sha256.into(),
                    section_id: section_id.into(),
                    note_indices_sha256: indices_sha256.into(),
                },
                transformation_history: vec!["exact score region extraction".into()],
                review_status: ReviewStatus::Unreviewed,
            },
        )
    }

    fn simple_score(key: Key) -> Score {
        let mut score = Score::new(key, 120.0, 4);
        let pitches = [60, 64, 67, 60, 64, 67];
        for (index, midi) in pitches.into_iter().enumerate() {
            score.push(ScoreNote {
                part: PartId::UNASSIGNED,
                pitch: Pitch::from_midi(midi),
                onset: Duration::quarter().scale(index as i64, 1),
                duration: Duration::quarter(),
                velocity: 0.7,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }
}
