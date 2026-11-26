// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact deterministic replay evidence for subject-bound native ProgSuite.
//!
//! This module answers a deliberately narrow causal/reproducibility question:
//! given the exact frozen native plan, exact independent source motif, exact
//! ordered DevelopmentProgram projection, and exact `MusicalIntent`, does the
//! current music-theory engine regenerate the stored symbolic score exactly?
//!
//! The answer is kept separate from observed-relation and perceptual evidence.
//! A successful replay establishes that the current deterministic generator maps
//! these declared inputs to these score bytes and that the ordered program is
//! the canonical projection of the native plan. It does NOT independently prove
//! that the final harmonized melody still exhibits an exact inversion/retrograde
//! relation, that a listener recognizes the relation, or that the result is good
//! music. Phrase construction deliberately performs additional chord-root
//! transposition, strong/weak-beat correction, and cadence steering after the
//! native motif transform, so collapsing those authorities would be incorrect.

use crate::development_program::{
    DevelopmentEvidenceRequirementV1, DevelopmentGraphProjectionV1,
};
use crate::harmony::Key;
use crate::prog_suite::{
    ProgSuitePlanErrorV1, ProgSuiteRealizationV1, ProgSuiteTransformV1,
    realize_prog_suite_with_plan,
};
use crate::prog_suite_subject_bound::{
    ProgSuiteSubjectBoundErrorV1, ProgSuiteSubjectBoundRealizationV1,
};
use crate::rhythm::Duration;
use crate::score::{Score, ScoreNote};
use crate::thematic_identity::ThematicTransformationClassV1;
use crate::MUSIC_THEORY_ENGINE_VERSION;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_REPLAY_EVIDENCE_VERSION: &str =
    "melothaea-prog-suite-replay-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteReplayAuthorityV1 {
    /// Under this exact crate/package implementation, the frozen plan + source
    /// motif + retained intent regenerate the stored symbolic score exactly.
    ExactGeneratorReplayUnderCurrentEngine,
    /// Structural binding remains valid, but deterministic regeneration differs
    /// from the stored score. This is evidence of stale/mutated output or input,
    /// not a partial PASS.
    StoredScoreDiffersFromReplay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteReplayNonClaimV1 {
    DoesNotEstablishHistoricalExecutionTrace,
    DoesNotEstablishIndependentObservedTransformationRelation,
    DoesNotEstablishListenerPerception,
    DoesNotEstablishArtisticQuality,
    DoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteStageMechanismReceiptV1 {
    pub stage_id: String,
    pub section_index: usize,
    pub work_node_id: String,
    pub derivation_id: String,
    /// Native transform enum consumed by `realize_prog_suite_with_plan` for the
    /// section corresponding to this canonical DevelopmentProgram stage.
    pub native_transform: ProgSuiteTransformV1,
    /// Exact ordered program operations retained by the canonical adapter.
    pub operation_ids: Vec<String>,
    pub operation_classes: Vec<ThematicTransformationClassV1>,
    pub graph_projection: DevelopmentGraphProjectionV1,
    pub evidence_requirement: DevelopmentEvidenceRequirementV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProgSuiteScoreReplayDifferenceV1 {
    Key {
        stored: Key,
        replayed: Key,
    },
    TempoBits {
        stored: u32,
        replayed: u32,
    },
    OpeningMeter {
        stored: u8,
        replayed: u8,
    },
    TotalBeats {
        stored: Duration,
        replayed: Duration,
    },
    Note {
        index: usize,
        stored: ScoreNote,
        replayed: ScoreNote,
    },
    NoteCount {
        stored: usize,
        replayed: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteScoreReplayReceiptV1 {
    pub exact_match: bool,
    pub stored_note_count: usize,
    pub replayed_note_count: usize,
    pub first_difference: Option<ProgSuiteScoreReplayDifferenceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteReplayEvidenceV1 {
    pub version: String,
    /// Cargo package version of the implementation that performed replay. This
    /// is not a Git/source-revision identity and must not be promoted into one.
    pub engine_version: String,
    pub authority: ProgSuiteReplayAuthorityV1,
    /// Complete self-contained replay input plus the stored score under test.
    pub subject: ProgSuiteSubjectBoundRealizationV1,
    /// Fresh result regenerated from the subject's bound plan + motif + intent.
    pub replayed_realization: ProgSuiteRealizationV1,
    /// Canonical ordered-program ↔ native-section mechanism bindings.
    pub stage_mechanisms: Vec<ProgSuiteStageMechanismReceiptV1>,
    pub score_replay: ProgSuiteScoreReplayReceiptV1,
    pub nonclaims: Vec<ProgSuiteReplayNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteReplayEvidenceErrorV1 {
    WrongVersion { found: String },
    WrongEngineVersion { found: String },
    SubjectBound(ProgSuiteSubjectBoundErrorV1),
    NativeReplay(ProgSuitePlanErrorV1),
    MissingStageSectionBinding { stage_id: String, work_node_id: String },
    MissingNativeSection { stage_id: String, section_index: usize },
    NonCanonicalNonClaims,
    CanonicalEvidenceMismatch,
}

pub fn derive_prog_suite_replay_evidence(
    subject: &ProgSuiteSubjectBoundRealizationV1,
) -> Result<ProgSuiteReplayEvidenceV1, ProgSuiteReplayEvidenceErrorV1> {
    subject
        .validate_structure()
        .map_err(ProgSuiteReplayEvidenceErrorV1::SubjectBound)?;

    // `validate_structure()` proved the declaration is canonical and source
    // material complete. Replay therefore consumes no loose motif/intent input.
    let source_motif = subject
        .declaration
        .source_motif()
        .map_err(ProgSuiteReplayEvidenceErrorV1::SubjectBound)?;
    let replayed_realization = realize_prog_suite_with_plan(
        &subject.declaration.work_binding.native_plan,
        source_motif,
        &subject.intent,
    )
    .map_err(ProgSuiteReplayEvidenceErrorV1::NativeReplay)?;

    let stage_mechanisms = stage_mechanism_receipts(subject)?;
    let score_replay = compare_scores(
        &subject.realization.score,
        &replayed_realization.score,
    );
    let authority = if score_replay.exact_match {
        ProgSuiteReplayAuthorityV1::ExactGeneratorReplayUnderCurrentEngine
    } else {
        ProgSuiteReplayAuthorityV1::StoredScoreDiffersFromReplay
    };

    Ok(ProgSuiteReplayEvidenceV1 {
        version: PROG_SUITE_REPLAY_EVIDENCE_VERSION.into(),
        engine_version: MUSIC_THEORY_ENGINE_VERSION.into(),
        authority,
        subject: subject.clone(),
        replayed_realization,
        stage_mechanisms,
        score_replay,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteReplayEvidenceV1 {
    /// Recompute the entire replay artifact from its retained subject and require
    /// exact equality. This protects serialized evidence from hand-edited
    /// authority/status/receipt fields.
    pub fn validate(&self) -> Result<(), ProgSuiteReplayEvidenceErrorV1> {
        if self.version != PROG_SUITE_REPLAY_EVIDENCE_VERSION {
            return Err(ProgSuiteReplayEvidenceErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.engine_version != MUSIC_THEORY_ENGINE_VERSION {
            return Err(ProgSuiteReplayEvidenceErrorV1::WrongEngineVersion {
                found: self.engine_version.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteReplayEvidenceErrorV1::NonCanonicalNonClaims);
        }
        let canonical = derive_prog_suite_replay_evidence(&self.subject)?;
        if &canonical != self {
            return Err(ProgSuiteReplayEvidenceErrorV1::CanonicalEvidenceMismatch);
        }
        Ok(())
    }
}

fn stage_mechanism_receipts(
    subject: &ProgSuiteSubjectBoundRealizationV1,
) -> Result<Vec<ProgSuiteStageMechanismReceiptV1>, ProgSuiteReplayEvidenceErrorV1> {
    let declaration = &subject.declaration;
    let work_binding = &declaration.work_binding;
    let mut receipts = Vec::with_capacity(declaration.development_program.program.stages.len());

    for stage in &declaration.development_program.program.stages {
        let section_binding = work_binding
            .section_bindings
            .iter()
            .find(|section| {
                section.work_node_id == stage.work_node_id
                    && section.derivation_id.as_deref() == Some(stage.derivation_id.as_str())
            })
            .ok_or_else(|| ProgSuiteReplayEvidenceErrorV1::MissingStageSectionBinding {
                stage_id: stage.stage_id.clone(),
                work_node_id: stage.work_node_id.clone(),
            })?;
        let section = work_binding
            .native_plan
            .sections
            .get(section_binding.section_index)
            .ok_or_else(|| ProgSuiteReplayEvidenceErrorV1::MissingNativeSection {
                stage_id: stage.stage_id.clone(),
                section_index: section_binding.section_index,
            })?;

        receipts.push(ProgSuiteStageMechanismReceiptV1 {
            stage_id: stage.stage_id.clone(),
            section_index: section_binding.section_index,
            work_node_id: stage.work_node_id.clone(),
            derivation_id: stage.derivation_id.clone(),
            native_transform: section.transformation,
            operation_ids: stage
                .operations
                .iter()
                .map(|operation| operation.operation_id.clone())
                .collect(),
            operation_classes: stage
                .operations
                .iter()
                .map(|operation| operation.class.clone())
                .collect(),
            graph_projection: stage.graph_projection.clone(),
            evidence_requirement: stage.evidence_requirement,
        });
    }
    Ok(receipts)
}

fn compare_scores(stored: &Score, replayed: &Score) -> ProgSuiteScoreReplayReceiptV1 {
    let first_difference = if stored.key != replayed.key {
        Some(ProgSuiteScoreReplayDifferenceV1::Key {
            stored: stored.key,
            replayed: replayed.key,
        })
    } else if stored.tempo_bpm.to_bits() != replayed.tempo_bpm.to_bits() {
        Some(ProgSuiteScoreReplayDifferenceV1::TempoBits {
            stored: stored.tempo_bpm.to_bits(),
            replayed: replayed.tempo_bpm.to_bits(),
        })
    } else if stored.meter != replayed.meter {
        Some(ProgSuiteScoreReplayDifferenceV1::OpeningMeter {
            stored: stored.meter,
            replayed: replayed.meter,
        })
    } else if stored.total_beats != replayed.total_beats {
        Some(ProgSuiteScoreReplayDifferenceV1::TotalBeats {
            stored: stored.total_beats,
            replayed: replayed.total_beats,
        })
    } else if let Some((index, (stored_note, replayed_note))) = stored
        .notes
        .iter()
        .zip(&replayed.notes)
        .enumerate()
        .find(|(_, (left, right))| left != right)
    {
        Some(ProgSuiteScoreReplayDifferenceV1::Note {
            index,
            stored: *stored_note,
            replayed: *replayed_note,
        })
    } else if stored.notes.len() != replayed.notes.len() {
        Some(ProgSuiteScoreReplayDifferenceV1::NoteCount {
            stored: stored.notes.len(),
            replayed: replayed.notes.len(),
        })
    } else {
        None
    };

    ProgSuiteScoreReplayReceiptV1 {
        exact_match: first_difference.is_none(),
        stored_note_count: stored.notes.len(),
        replayed_note_count: replayed.notes.len(),
        first_difference,
    }
}

fn required_nonclaims() -> Vec<ProgSuiteReplayNonClaimV1> {
    vec![
        ProgSuiteReplayNonClaimV1::DoesNotEstablishHistoricalExecutionTrace,
        ProgSuiteReplayNonClaimV1::DoesNotEstablishIndependentObservedTransformationRelation,
        ProgSuiteReplayNonClaimV1::DoesNotEstablishListenerPerception,
        ProgSuiteReplayNonClaimV1::DoesNotEstablishArtisticQuality,
        ProgSuiteReplayNonClaimV1::DoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, Pitch, PitchClass, ProgSuiteTransformV1,
        Style, ThematicTransformationClassV1, bind_prog_suite_subject,
        plan_prog_suite, realize_prog_suite_subject_bound,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn plan() -> crate::ProgSuitePlanV1 {
        plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap()
    }

    fn subject() -> ProgSuiteSubjectBoundRealizationV1 {
        let declaration = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        let intent = MusicalIntent {
            energy: 0.73,
            seed: 9182,
            ..MusicalIntent::default()
        };
        realize_prog_suite_subject_bound(&declaration, &intent).unwrap()
    }

    #[test]
    fn canonical_subject_replays_exactly_under_current_engine() {
        let evidence = derive_prog_suite_replay_evidence(&subject()).unwrap();
        assert_eq!(
            evidence.authority,
            ProgSuiteReplayAuthorityV1::ExactGeneratorReplayUnderCurrentEngine
        );
        assert!(evidence.score_replay.exact_match);
        assert!(evidence.score_replay.first_difference.is_none());
        assert_eq!(evidence.subject.realization, evidence.replayed_realization);
        evidence.validate().unwrap();
    }

    #[test]
    fn b_receipt_preserves_native_composite_and_ordered_program_operations() {
        let evidence = derive_prog_suite_replay_evidence(&subject()).unwrap();
        let b = &evidence.stage_mechanisms[0];
        assert_eq!(b.section_index, 1);
        assert_eq!(b.native_transform, ProgSuiteTransformV1::RetrogradeInversion);
        assert_eq!(
            b.operation_classes,
            vec![
                ThematicTransformationClassV1::Inversion,
                ThematicTransformationClassV1::Retrograde,
            ]
        );
        assert_eq!(
            b.operation_ids,
            vec![
                "prog-suite:development:B:op-01-inversion".to_string(),
                "prog-suite:development:B:op-02-retrograde".to_string(),
            ]
        );
        assert_eq!(
            b.evidence_requirement,
            DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured
        );
    }

    #[test]
    fn a_structurally_valid_pitch_mutation_fails_exact_replay() {
        let mut subject = subject();
        let note = subject.realization.score.notes.first_mut().unwrap();
        note.pitch = Pitch::from_midi(note.pitch.midi().saturating_add(1).min(127));
        subject.validate_structure().unwrap();

        let evidence = derive_prog_suite_replay_evidence(&subject).unwrap();
        assert_eq!(
            evidence.authority,
            ProgSuiteReplayAuthorityV1::StoredScoreDiffersFromReplay
        );
        assert!(!evidence.score_replay.exact_match);
        assert!(matches!(
            evidence.score_replay.first_difference,
            Some(ProgSuiteScoreReplayDifferenceV1::Note { .. })
        ));
    }

    #[test]
    fn retained_intent_is_earned_by_replay_not_structural_validation() {
        let mut subject = subject();
        subject.intent.energy = 0.11;
        subject.validate_structure().unwrap();
        let evidence = derive_prog_suite_replay_evidence(&subject).unwrap();
        assert_eq!(
            evidence.authority,
            ProgSuiteReplayAuthorityV1::StoredScoreDiffersFromReplay
        );
    }

    #[test]
    fn alternate_valid_source_material_cannot_reuse_old_score_causal_authority() {
        let mut subject = subject();
        let alternate = Motif::from_degrees(&[
            (7, Duration::quarter()),
            (1, Duration::quarter()),
            (6, Duration::quarter()),
            (2, Duration::quarter()),
        ]);
        subject.declaration = bind_prog_suite_subject(&plan(), &alternate).unwrap();
        subject.validate_structure().unwrap();
        let evidence = derive_prog_suite_replay_evidence(&subject).unwrap();
        assert_eq!(
            evidence.authority,
            ProgSuiteReplayAuthorityV1::StoredScoreDiffersFromReplay
        );
    }

    #[test]
    fn standalone_retrograde_plan_replays_with_single_operation_receipt() {
        let mut plan = plan();
        plan.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        plan.sections[2].transformation = ProgSuiteTransformV1::Inversion;
        plan.validate().unwrap();
        let declaration = bind_prog_suite_subject(&plan, &motif()).unwrap();
        let subject = realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent::default(),
        )
        .unwrap();
        let evidence = derive_prog_suite_replay_evidence(&subject).unwrap();
        assert_eq!(
            evidence.authority,
            ProgSuiteReplayAuthorityV1::ExactGeneratorReplayUnderCurrentEngine
        );
        let b = &evidence.stage_mechanisms[0];
        assert_eq!(b.native_transform, ProgSuiteTransformV1::Retrograde);
        assert_eq!(
            b.operation_classes,
            vec![ThematicTransformationClassV1::Retrograde]
        );
        assert_eq!(
            b.evidence_requirement,
            DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured
        );
    }

    #[test]
    fn replay_artifact_keeps_causal_authority_separate_from_other_claims() {
        let evidence = derive_prog_suite_replay_evidence(&subject()).unwrap();
        assert_eq!(evidence.nonclaims, required_nonclaims());
        assert!(evidence.nonclaims.contains(
            &ProgSuiteReplayNonClaimV1::DoesNotEstablishIndependentObservedTransformationRelation
        ));
        assert!(evidence
            .nonclaims
            .contains(&ProgSuiteReplayNonClaimV1::DoesNotEstablishHistoricalExecutionTrace));
        assert!(evidence
            .nonclaims
            .contains(&ProgSuiteReplayNonClaimV1::DoesNotGrantProductAuthority));
    }

    #[test]
    fn serialized_authority_cannot_be_hand_edited_without_detection() {
        let mut evidence = derive_prog_suite_replay_evidence(&subject()).unwrap();
        evidence.authority = ProgSuiteReplayAuthorityV1::StoredScoreDiffersFromReplay;
        assert_eq!(
            evidence.validate(),
            Err(ProgSuiteReplayEvidenceErrorV1::CanonicalEvidenceMismatch)
        );
    }
}
