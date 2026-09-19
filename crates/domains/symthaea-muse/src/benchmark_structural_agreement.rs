// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent agreement evidence between declared sonata-form promises and
//! generic score-surface recurrence.
//!
//! The formal source of truth is prospective: `SonataPlan::obligations` says
//! which motif must return and under which transformation. This benchmark does
//! **not** consume `SonataRealization::verification` or its previously-computed
//! motif score. It independently measures the completed score and asks whether
//! the formally declared target is also distinguished from non-target sections
//! by transformation-aware motif evidence and generic chroma recurrence.
//!
//! No channel in this module is an artistic-quality or listener-recognition
//! score. Disagreement is retained as evidence rather than repaired or hidden.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{
    MotifReturnEvidence, ObligationKind, PlannedSonataSection, ReturnTransformation, Score,
    SonataPlan, SonataSectionKind, VoiceRole, compare_melodic_regions,
};

pub const STRUCTURAL_AGREEMENT_VERSION: &str = "melothaea-structural-agreement-v1";
const TIE_EPSILON: f64 = 1.0e-12;

const SECTION_ORDER: [SonataSectionKind; 5] = [
    SonataSectionKind::ExpositionPrimary,
    SonataSectionKind::ExpositionSecondary,
    SonataSectionKind::Development,
    SonataSectionKind::RecapitulationPrimary,
    SonataSectionKind::RecapitulationSecondary,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralAgreementNonClaimV1 {
    SurfaceRecurrenceDoesNotEstablishFormalCorrectness,
    FormalPromiseDoesNotEstablishListenerRecognition,
    SymbolicSimilarityDoesNotEstablishHumanAudibility,
    TopRankDoesNotEstablishMusicalQuality,
    StructuralAgreementDoesNotGrantProductAuthority,
}

pub const REQUIRED_STRUCTURAL_NONCLAIMS_V1: [StructuralAgreementNonClaimV1; 5] = [
    StructuralAgreementNonClaimV1::SurfaceRecurrenceDoesNotEstablishFormalCorrectness,
    StructuralAgreementNonClaimV1::FormalPromiseDoesNotEstablishListenerRecognition,
    StructuralAgreementNonClaimV1::SymbolicSimilarityDoesNotEstablishHumanAudibility,
    StructuralAgreementNonClaimV1::TopRankDoesNotEstablishMusicalQuality,
    StructuralAgreementNonClaimV1::StructuralAgreementDoesNotGrantProductAuthority,
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionSurfaceEvidenceV1 {
    pub section: SonataSectionKind,
    pub start_num: i64,
    pub start_den: i64,
    pub end_num: i64,
    pub end_den: i64,
    pub all_note_count: u64,
    pub melody_note_count: u64,
    pub boundary_crossing_note_count: u64,
    pub all_note_pitch_class_counts: [u64; 12],
    pub melody_pitch_class_counts: [u64; 12],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChromaRecurrenceV1 {
    /// Cosine similarity without transposition compensation.
    pub literal_cosine: f64,
    /// Maximum cosine similarity over all 12 circular pitch-class shifts.
    pub transposition_invariant_cosine: f64,
    /// Semitone shift applied to the left histogram to best match the right.
    /// Ties retain the smallest shift deterministically.
    pub best_left_to_right_shift: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionPairRecurrenceV1 {
    pub left: SonataSectionKind,
    pub right: SonataSectionKind,
    pub all_notes: Option<ChromaRecurrenceV1>,
    pub melody_only: Option<ChromaRecurrenceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormalReturnAgreementV1 {
    pub obligation_id: u64,
    pub motif_id: String,
    pub source: SonataSectionKind,
    pub target: SonataSectionKind,
    pub expected_transformation: ReturnTransformation,
    /// Freshly recomputed from score + plan boundaries; never copied from the
    /// sonata verifier's stored verification record.
    pub target_motif_evidence: MotifReturnEvidence,
    pub target_all_note_chroma: Option<ChromaRecurrenceV1>,
    pub target_melody_chroma: Option<ChromaRecurrenceV1>,
    pub distractor_count: usize,
    pub best_non_target_motif_similarity: Option<f32>,
    pub motif_target_margin: Option<f32>,
    pub target_is_top_tied_motif: bool,
    pub best_non_target_melody_transposition_invariant: Option<f64>,
    pub melody_chroma_target_margin: Option<f64>,
    pub target_is_top_tied_melody_chroma: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataStructureAgreementV1 {
    pub evidence_version: String,
    pub score_sha256: String,
    pub plan_sha256: String,
    pub sections: Vec<SectionSurfaceEvidenceV1>,
    /// Upper triangle in canonical section order, never a hidden aggregate.
    pub pairwise_recurrence: Vec<SectionPairRecurrenceV1>,
    pub formal_returns: Vec<FormalReturnAgreementV1>,
    pub required_nonclaims: Vec<StructuralAgreementNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralAgreementIssueV1 {
    WrongSectionCount { found: usize, required: usize },
    SectionsNotCanonical,
    DuplicateSection { section: SonataSectionKind },
    MissingSection { section: SonataSectionKind },
    InvalidSectionBounds { section: SonataSectionKind },
    NonContiguousSections {
        left: SonataSectionKind,
        right: SonataSectionKind,
    },
    NegativeScoreOnset { note_index: usize },
    NonFiniteScoreScalar { note_index: usize, field: String },
    ScoreNoteOutsidePlan { note_index: usize },
    NoReturnObligations,
    DuplicateReturnMotif { motif_id: String },
    UnknownReturnMotif { motif_id: String },
    ReturnDuePointMismatch { motif_id: String },
    SerializationFailed { field: String },
}

impl SonataStructureAgreementV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

/// Measure agreement between a prospective sonata plan and independent score
/// observations. Stored verifier results are intentionally not an input.
pub fn measure_sonata_structure_agreement(
    score: &Score,
    plan: &SonataPlan,
) -> Result<SonataStructureAgreementV1, Vec<StructuralAgreementIssueV1>> {
    let mut issues = validate_sections(plan);
    validate_score_domain(score, plan, &mut issues);
    let claims = extract_formal_returns(plan, &mut issues);
    if !issues.is_empty() {
        return Err(issues);
    }

    let score_sha256 = match canonical_json_sha256(score) {
        Ok(value) => value,
        Err(_) => {
            return Err(vec![StructuralAgreementIssueV1::SerializationFailed {
                field: "score".into(),
            }]);
        }
    };
    let plan_sha256 = match canonical_json_sha256(plan) {
        Ok(value) => value,
        Err(_) => {
            return Err(vec![StructuralAgreementIssueV1::SerializationFailed {
                field: "plan".into(),
            }]);
        }
    };

    let sections: Vec<_> = SECTION_ORDER
        .into_iter()
        .map(|kind| section_surface(score, planned_section(plan, kind).expect("validated section")))
        .collect();

    let mut pairwise_recurrence = Vec::new();
    for left_index in 0..sections.len() {
        for right_index in left_index + 1..sections.len() {
            let left = &sections[left_index];
            let right = &sections[right_index];
            pairwise_recurrence.push(SectionPairRecurrenceV1 {
                left: left.section,
                right: right.section,
                all_notes: chroma_recurrence(
                    &left.all_note_pitch_class_counts,
                    &right.all_note_pitch_class_counts,
                ),
                melody_only: chroma_recurrence(
                    &left.melody_pitch_class_counts,
                    &right.melody_pitch_class_counts,
                ),
            });
        }
    }

    let formal_returns = claims
        .into_iter()
        .map(|claim| measure_formal_return(score, plan, &sections, claim))
        .collect();

    Ok(SonataStructureAgreementV1 {
        evidence_version: STRUCTURAL_AGREEMENT_VERSION.into(),
        score_sha256,
        plan_sha256,
        sections,
        pairwise_recurrence,
        formal_returns,
        required_nonclaims: REQUIRED_STRUCTURAL_NONCLAIMS_V1.to_vec(),
    })
}

#[derive(Debug, Clone)]
struct FormalReturnClaimV1 {
    obligation_id: u64,
    motif_id: String,
    source: SonataSectionKind,
    target: SonataSectionKind,
    expected_transformation: ReturnTransformation,
}

fn validate_sections(plan: &SonataPlan) -> Vec<StructuralAgreementIssueV1> {
    let mut issues = Vec::new();
    if plan.sections.len() != SECTION_ORDER.len() {
        issues.push(StructuralAgreementIssueV1::WrongSectionCount {
            found: plan.sections.len(),
            required: SECTION_ORDER.len(),
        });
    }

    let found_order: Vec<_> = plan.sections.iter().map(|section| section.kind).collect();
    if found_order != SECTION_ORDER {
        issues.push(StructuralAgreementIssueV1::SectionsNotCanonical);
    }
    for kind in SECTION_ORDER {
        let count = plan.sections.iter().filter(|section| section.kind == kind).count();
        if count == 0 {
            issues.push(StructuralAgreementIssueV1::MissingSection { section: kind });
        } else if count > 1 {
            issues.push(StructuralAgreementIssueV1::DuplicateSection { section: kind });
        }
    }

    for section in &plan.sections {
        let start = section.start.beats();
        let end = section.end.beats();
        if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
            issues.push(StructuralAgreementIssueV1::InvalidSectionBounds {
                section: section.kind,
            });
        }
    }
    for pair in plan.sections.windows(2) {
        if pair[0].end != pair[1].start {
            issues.push(StructuralAgreementIssueV1::NonContiguousSections {
                left: pair[0].kind,
                right: pair[1].kind,
            });
        }
    }
    if plan
        .sections
        .first()
        .is_some_and(|section| section.start != symthaea_music_theory::Duration::zero())
    {
        issues.push(StructuralAgreementIssueV1::InvalidSectionBounds {
            section: plan.sections[0].kind,
        });
    }
    issues
}

fn validate_score_domain(
    score: &Score,
    plan: &SonataPlan,
    issues: &mut Vec<StructuralAgreementIssueV1>,
) {
    let plan_end = plan.sections.last().map(|section| section.end.beats());
    for (note_index, note) in score.notes.iter().enumerate() {
        let onset = note.onset.beats();
        if !onset.is_finite() || onset < 0.0 {
            issues.push(StructuralAgreementIssueV1::NegativeScoreOnset { note_index });
        }
        for (field, value) in [
            ("velocity", note.velocity),
            ("section_intensity", note.section_intensity),
        ] {
            if !value.is_finite() {
                issues.push(StructuralAgreementIssueV1::NonFiniteScoreScalar {
                    note_index,
                    field: field.into(),
                });
            }
        }
        if plan_end.is_none_or(|end| onset < 0.0 || onset >= end) {
            issues.push(StructuralAgreementIssueV1::ScoreNoteOutsidePlan { note_index });
        }
    }
}

fn extract_formal_returns(
    plan: &SonataPlan,
    issues: &mut Vec<StructuralAgreementIssueV1>,
) -> Vec<FormalReturnClaimV1> {
    let mut claims = Vec::new();
    let mut seen_ids = Vec::<String>::new();
    for obligation in plan.obligations.items() {
        let ObligationKind::ReturnMotif {
            motif_id,
            transformation,
        } = &obligation.kind
        else {
            continue;
        };
        if seen_ids.contains(motif_id) {
            issues.push(StructuralAgreementIssueV1::DuplicateReturnMotif {
                motif_id: motif_id.clone(),
            });
            continue;
        }
        seen_ids.push(motif_id.clone());
        let Some((source, target)) = sonata_relation(motif_id) else {
            issues.push(StructuralAgreementIssueV1::UnknownReturnMotif {
                motif_id: motif_id.clone(),
            });
            continue;
        };
        let Some(target_section) = planned_section(plan, target) else {
            continue;
        };
        if obligation.due_by != target_section.end {
            issues.push(StructuralAgreementIssueV1::ReturnDuePointMismatch {
                motif_id: motif_id.clone(),
            });
        }
        claims.push(FormalReturnClaimV1 {
            obligation_id: obligation.id,
            motif_id: motif_id.clone(),
            source,
            target,
            expected_transformation: *transformation,
        });
    }
    if claims.is_empty() {
        issues.push(StructuralAgreementIssueV1::NoReturnObligations);
    }
    claims.sort_by_key(|claim| claim.obligation_id);
    claims
}

fn sonata_relation(motif_id: &str) -> Option<(SonataSectionKind, SonataSectionKind)> {
    match motif_id {
        "sonata.primary" => Some((
            SonataSectionKind::ExpositionPrimary,
            SonataSectionKind::RecapitulationPrimary,
        )),
        "sonata.secondary" => Some((
            SonataSectionKind::ExpositionSecondary,
            SonataSectionKind::RecapitulationSecondary,
        )),
        _ => None,
    }
}

fn planned_section(plan: &SonataPlan, kind: SonataSectionKind) -> Option<&PlannedSonataSection> {
    plan.sections.iter().find(|section| section.kind == kind)
}

fn section_surface(score: &Score, section: &PlannedSonataSection) -> SectionSurfaceEvidenceV1 {
    let mut all_counts = [0u64; 12];
    let mut melody_counts = [0u64; 12];
    let mut all_note_count = 0u64;
    let mut melody_note_count = 0u64;
    let mut boundary_crossing_note_count = 0u64;
    let start = section.start.beats();
    let end = section.end.beats();

    for note in &score.notes {
        let onset = note.onset.beats();
        if onset < start || onset >= end {
            continue;
        }
        let pitch_class = usize::from(note.pitch.midi() % 12);
        all_counts[pitch_class] += 1;
        all_note_count += 1;
        if note.role == VoiceRole::Melody {
            melody_counts[pitch_class] += 1;
            melody_note_count += 1;
        }
        if (note.onset + note.duration).beats() > end {
            boundary_crossing_note_count += 1;
        }
    }

    SectionSurfaceEvidenceV1 {
        section: section.kind,
        start_num: section.start.num(),
        start_den: section.start.den(),
        end_num: section.end.num(),
        end_den: section.end.den(),
        all_note_count,
        melody_note_count,
        boundary_crossing_note_count,
        all_note_pitch_class_counts: all_counts,
        melody_pitch_class_counts: melody_counts,
    }
}

fn measure_formal_return(
    score: &Score,
    plan: &SonataPlan,
    surfaces: &[SectionSurfaceEvidenceV1],
    claim: FormalReturnClaimV1,
) -> FormalReturnAgreementV1 {
    let source = planned_section(plan, claim.source).expect("validated source section");
    let target = planned_section(plan, claim.target).expect("validated target section");
    let source_surface = surface_for(surfaces, claim.source);
    let target_surface = surface_for(surfaces, claim.target);

    let target_motif_evidence = compare_melodic_regions(
        score,
        source.start,
        source.end,
        target.start,
        target.end,
        claim.expected_transformation,
    );
    let target_all_note_chroma = chroma_recurrence(
        &source_surface.all_note_pitch_class_counts,
        &target_surface.all_note_pitch_class_counts,
    );
    let target_melody_chroma = chroma_recurrence(
        &source_surface.melody_pitch_class_counts,
        &target_surface.melody_pitch_class_counts,
    );

    let mut distractor_motif = Vec::new();
    let mut distractor_chroma = Vec::new();
    for candidate_kind in SECTION_ORDER {
        if candidate_kind == claim.source || candidate_kind == claim.target {
            continue;
        }
        let candidate = planned_section(plan, candidate_kind).expect("validated candidate section");
        let candidate_surface = surface_for(surfaces, candidate_kind);
        distractor_motif.push(
            compare_melodic_regions(
                score,
                source.start,
                source.end,
                candidate.start,
                candidate.end,
                claim.expected_transformation,
            )
            .overall_similarity,
        );
        if let Some(recurrence) = chroma_recurrence(
            &source_surface.melody_pitch_class_counts,
            &candidate_surface.melody_pitch_class_counts,
        ) {
            distractor_chroma.push(recurrence.transposition_invariant_cosine);
        }
    }

    let best_non_target_motif_similarity = distractor_motif.into_iter().max_by(f32::total_cmp);
    let motif_target_margin = best_non_target_motif_similarity
        .map(|best| target_motif_evidence.overall_similarity - best);
    let target_is_top_tied_motif = best_non_target_motif_similarity.is_none_or(|best| {
        f64::from(target_motif_evidence.overall_similarity) + TIE_EPSILON >= f64::from(best)
    });

    let best_non_target_melody_transposition_invariant =
        distractor_chroma.into_iter().max_by(f64::total_cmp);
    let target_chroma_value = target_melody_chroma
        .as_ref()
        .map(|value| value.transposition_invariant_cosine);
    let melody_chroma_target_margin = target_chroma_value.zip(
        best_non_target_melody_transposition_invariant,
    )
    .map(|(target, best)| target - best);
    let target_is_top_tied_melody_chroma = target_chroma_value.is_some_and(|target| {
        best_non_target_melody_transposition_invariant
            .is_none_or(|best| target + TIE_EPSILON >= best)
    });

    FormalReturnAgreementV1 {
        obligation_id: claim.obligation_id,
        motif_id: claim.motif_id,
        source: claim.source,
        target: claim.target,
        expected_transformation: claim.expected_transformation,
        target_motif_evidence,
        target_all_note_chroma,
        target_melody_chroma,
        distractor_count: SECTION_ORDER.len() - 2,
        best_non_target_motif_similarity,
        motif_target_margin,
        target_is_top_tied_motif,
        best_non_target_melody_transposition_invariant,
        melody_chroma_target_margin,
        target_is_top_tied_melody_chroma,
    }
}

fn surface_for(
    surfaces: &[SectionSurfaceEvidenceV1],
    kind: SonataSectionKind,
) -> &SectionSurfaceEvidenceV1 {
    surfaces
        .iter()
        .find(|surface| surface.section == kind)
        .expect("validated surface")
}

fn chroma_recurrence(left: &[u64; 12], right: &[u64; 12]) -> Option<ChromaRecurrenceV1> {
    if left.iter().sum::<u64>() == 0 || right.iter().sum::<u64>() == 0 {
        return None;
    }
    let literal_cosine = cosine(left, right);
    let mut best_shift = 0u8;
    let mut best = literal_cosine;
    for shift in 1u8..12 {
        let shifted = rotate_chroma(left, shift);
        let value = cosine(&shifted, right);
        if value > best {
            best = value;
            best_shift = shift;
        }
    }
    Some(ChromaRecurrenceV1 {
        literal_cosine,
        transposition_invariant_cosine: best,
        best_left_to_right_shift: best_shift,
    })
}

fn rotate_chroma(source: &[u64; 12], shift: u8) -> [u64; 12] {
    let mut shifted = [0u64; 12];
    for (pitch_class, count) in source.iter().copied().enumerate() {
        shifted[(pitch_class + usize::from(shift)) % 12] = count;
    }
    shifted
}

fn cosine(left: &[u64; 12], right: &[u64; 12]) -> f64 {
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(left, right)| *left as f64 * *right as f64)
        .sum();
    let left_norm = left
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    dot / (left_norm * right_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{
        CompositionalObligation, Duration, Emphasis, Key, Motif, PartId, Pitch, PitchClass,
        ScoreNote, plan_sonata,
    };

    fn note(onset: Duration, midi: u8) -> ScoreNote {
        ScoreNote {
            part: PartId(1),
            pitch: Pitch::from_midi(midi),
            onset,
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn fixture() -> (Score, SonataPlan) {
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (4, Duration::quarter()),
            (3, Duration::quarter()),
        ]);
        let plan = plan_sonata(Key::major(PitchClass::C), 4.0, &motif, 0x5eed);
        let mut score = Score::new(plan.home_key, 120.0, 4);
        let material = [
            (
                SonataSectionKind::ExpositionPrimary,
                [60u8, 62, 65, 64],
            ),
            (
                SonataSectionKind::ExpositionSecondary,
                [61u8, 63, 66, 68],
            ),
            (SonataSectionKind::Development, [48u8, 49, 55, 58]),
            (
                SonataSectionKind::RecapitulationPrimary,
                [60u8, 62, 65, 64],
            ),
            (
                SonataSectionKind::RecapitulationSecondary,
                [66u8, 68, 71, 73],
            ),
        ];
        for (kind, pitches) in material {
            let section = planned_section(&plan, kind).unwrap();
            for (offset, pitch) in pitches.into_iter().enumerate() {
                score.push(note(section.start + Duration::new(offset as i64, 1), pitch));
            }
        }
        score.total_beats = plan.sections.last().unwrap().end;
        (score, plan)
    }

    #[test]
    fn transposition_invariant_chroma_recovers_shifted_relation() {
        let mut left = [0u64; 12];
        left[0] = 1;
        left[2] = 1;
        left[5] = 1;
        let mut right = [0u64; 12];
        right[5] = 1;
        right[7] = 1;
        right[10] = 1;
        let evidence = chroma_recurrence(&left, &right).unwrap();
        assert!(evidence.literal_cosine < 0.5);
        assert!((evidence.transposition_invariant_cosine - 1.0).abs() < 1.0e-12);
        assert_eq!(evidence.best_left_to_right_shift, 5);
    }

    #[test]
    fn formal_returns_are_measured_independently_against_distractors() {
        let (score, plan) = fixture();
        let evidence = measure_sonata_structure_agreement(&score, &plan).unwrap();
        assert_eq!(evidence.sections.len(), 5);
        assert_eq!(evidence.pairwise_recurrence.len(), 10);
        assert_eq!(evidence.formal_returns.len(), 2);
        assert_eq!(
            evidence.required_nonclaims,
            REQUIRED_STRUCTURAL_NONCLAIMS_V1
        );

        let primary = evidence
            .formal_returns
            .iter()
            .find(|return_| return_.motif_id == "sonata.primary")
            .unwrap();
        assert_eq!(primary.source, SonataSectionKind::ExpositionPrimary);
        assert_eq!(primary.target, SonataSectionKind::RecapitulationPrimary);
        assert_eq!(primary.expected_transformation, ReturnTransformation::Literal);
        assert!(primary.target_motif_evidence.overall_similarity > 0.99);
        assert!(primary.target_is_top_tied_motif);
        assert!(primary.target_is_top_tied_melody_chroma);

        let secondary = evidence
            .formal_returns
            .iter()
            .find(|return_| return_.motif_id == "sonata.secondary")
            .unwrap();
        assert_eq!(secondary.expected_transformation, ReturnTransformation::Transposed);
        let chroma = secondary.target_melody_chroma.unwrap();
        assert!((chroma.transposition_invariant_cosine - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn generic_disagreement_is_retained_instead_of_repaired() {
        let (mut score, plan) = fixture();
        let development = planned_section(&plan, SonataSectionKind::Development).unwrap();
        let recap = planned_section(&plan, SonataSectionKind::RecapitulationPrimary).unwrap();
        let source_pattern = [60u8, 62, 65, 64];
        let weak_target = [60u8, 61, 67, 70];
        let mut development_index = 0usize;
        let mut recap_index = 0usize;
        for note in &mut score.notes {
            let onset = note.onset.beats();
            if onset >= development.start.beats() && onset < development.end.beats() {
                note.pitch = Pitch::from_midi(source_pattern[development_index]);
                development_index += 1;
            } else if onset >= recap.start.beats() && onset < recap.end.beats() {
                note.pitch = Pitch::from_midi(weak_target[recap_index]);
                recap_index += 1;
            }
        }
        let evidence = measure_sonata_structure_agreement(&score, &plan).unwrap();
        let primary = evidence
            .formal_returns
            .iter()
            .find(|return_| return_.motif_id == "sonata.primary")
            .unwrap();
        assert!(!primary.target_is_top_tied_motif);
        assert!(primary.motif_target_margin.is_some_and(|margin| margin < 0.0));
    }

    #[test]
    fn unknown_formal_return_mapping_fails_closed() {
        let (score, mut plan) = fixture();
        let due = plan.sections.last().unwrap().end;
        plan.obligations.add(CompositionalObligation::new(
            999,
            Duration::zero(),
            due,
            1.0,
            ObligationKind::ReturnMotif {
                motif_id: "sonata.unknown".into(),
                transformation: ReturnTransformation::Literal,
            },
        ));
        let issues = measure_sonata_structure_agreement(&score, &plan).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            StructuralAgreementIssueV1::UnknownReturnMotif { motif_id }
                if motif_id == "sonata.unknown"
        )));
    }

    #[test]
    fn evidence_identity_changes_when_surface_music_changes() {
        let (score, plan) = fixture();
        let first = measure_sonata_structure_agreement(&score, &plan).unwrap();
        let mut changed = score.clone();
        changed.notes[0].pitch = Pitch::from_midi(61);
        let second = measure_sonata_structure_agreement(&changed, &plan).unwrap();
        assert_ne!(first.score_sha256, second.score_sha256);
        assert_ne!(
            first.canonical_sha256().unwrap(),
            second.canonical_sha256().unwrap()
        );
    }
}
