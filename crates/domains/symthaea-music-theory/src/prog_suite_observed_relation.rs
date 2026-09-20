// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ordered-development survival evidence for subject-bound native ProgSuite.
//!
//! Generator replay answers whether the current implementation can regenerate a
//! stored score. This module asks what relationship remains observable across
//! the symbolic realization boundaries. V1 deliberately observes the first
//! local-bar statement rather than pretending an entire developed section is an
//! untouched inversion/retrograde.

use crate::development_program::DevelopmentOperationV1;
use crate::harmony::{Key, Tonality};
use crate::motif::Motif;
use crate::phrase::Period;
use crate::pitch::PitchClass;
use crate::prog_suite::{
    ProgSuiteMeterFitReceiptV1, ProgSuitePlanErrorV1, ProgSuiteSectionCarrierV1,
    derive_prog_suite_section_carriers,
};
use crate::prog_suite_subject_bound::{
    ProgSuiteSubjectBoundErrorV1, ProgSuiteSubjectBoundRealizationV1,
};
use crate::rhythm::Duration;
use crate::score::{Score, VoiceRole};
use crate::thematic_identity::ThematicTransformationClassV1;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_OBSERVED_RELATION_VERSION: &str =
    "melothaea-prog-suite-observed-relation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteObservedOperationSemanticsV1 {
    LiteralReturnIdentity,
    InversionAboutCurrentFirstPitchedDegree,
    RetrogradeExactEventOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteOperationObservationV1 {
    pub operation_id: String,
    pub class: ThematicTransformationClassV1,
    pub semantics: ProgSuiteObservedOperationSemanticsV1,
    pub inversion_pivot_degree: Option<i32>,
    pub input: Motif,
    pub output: Motif,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuitePhraseAdaptationObservationV1 {
    pub first_progression_degree: i32,
    pub meter_fit: ProgSuiteMeterFitReceiptV1,
    pub phrase_statement: Motif,
    pub source_event_count: usize,
    pub phrase_event_count: usize,
    pub duration_matches: usize,
    pub rest_pattern_matches: usize,
    pub degree_matches_after_reanchor: usize,
    pub event_alignment_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteObservedMelodyEventV1 {
    pub relative_onset: Duration,
    pub duration: Duration,
    pub pitch_midi: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteScoreProjectionObservationV1 {
    pub expected: Vec<ProgSuiteObservedMelodyEventV1>,
    pub observed: Vec<ProgSuiteObservedMelodyEventV1>,
    /// Only this pitch class may use the profile's +1-semitone allowance in the
    /// first measure. `None` means no chromatic raise is admissible there.
    pub eligible_raised_pitch_class: Option<PitchClass>,
    pub aligned_event_count: usize,
    pub onset_matches: usize,
    pub duration_matches: usize,
    pub exact_pitch_matches: usize,
    pub octave_equivalent_pitch_matches: usize,
    pub eligible_raised_semitone_matches: usize,
    pub event_alignment_exact: bool,
    pub all_pitch_differences_compatible_with_render_profile: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteObservedRelationStatusV1 {
    ObservedWithDeclaredAdaptation,
    InsufficientMaterial,
    OperationMechanismMismatch,
    PhraseEventAlignmentFailed,
    ScoreProjectionFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteObservedRelationNonClaimV1 {
    DoesNotEstablishHistoricalExecutionTrace,
    DoesNotEstablishWholeSectionExactTransformation,
    DoesNotEstablishListenerPerception,
    DoesNotEstablishArtisticQuality,
    DoesNotGrantProductAuthority,
    OperationInterpreterReusesCanonicalMotifPrimitives,
    CanonicalPhraseReconstructionIsNotIndependentComposerImplementation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteStageObservedRelationV1 {
    pub stage_id: String,
    pub section_index: usize,
    pub work_node_id: String,
    pub derivation_id: String,
    pub operations: Vec<ProgSuiteOperationObservationV1>,
    pub independently_interpreted_transform: Motif,
    pub native_transformed_carrier: Motif,
    pub ordered_operation_matches_native_carrier: bool,
    pub meter_fitted_carrier: Motif,
    pub phrase_adaptation: ProgSuitePhraseAdaptationObservationV1,
    pub score_projection: ProgSuiteScoreProjectionObservationV1,
    pub status: ProgSuiteObservedRelationStatusV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteObservedRelationEvidenceV1 {
    pub version: String,
    pub subject: ProgSuiteSubjectBoundRealizationV1,
    pub stages: Vec<ProgSuiteStageObservedRelationV1>,
    pub nonclaims: Vec<ProgSuiteObservedRelationNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteObservedRelationErrorV1 {
    WrongVersion { found: String },
    SubjectBound(ProgSuiteSubjectBoundErrorV1),
    Carrier(ProgSuitePlanErrorV1),
    MissingSectionBinding { stage_id: String, work_node_id: String },
    MissingCarrier { stage_id: String, section_index: usize },
    MissingNativeSection { stage_id: String, section_index: usize },
    StageInputIsNotProgramSource { stage_id: String, input_identity_id: String },
    UnsupportedOperation {
        stage_id: String,
        operation_id: String,
        class: ThematicTransformationClassV1,
    },
    MissingInversionPivot { stage_id: String, operation_id: String },
    EmptyProgression { stage_id: String },
    PhraseStatementDurationMismatch {
        stage_id: String,
        expected: Duration,
        found: Duration,
    },
    NonCanonicalNonClaims,
    CanonicalEvidenceMismatch,
}

pub fn measure_prog_suite_observed_relations(
    subject: &ProgSuiteSubjectBoundRealizationV1,
) -> Result<ProgSuiteObservedRelationEvidenceV1, ProgSuiteObservedRelationErrorV1> {
    subject
        .validate_structure()
        .map_err(ProgSuiteObservedRelationErrorV1::SubjectBound)?;

    let source_motif = subject
        .declaration
        .source_motif()
        .map_err(ProgSuiteObservedRelationErrorV1::SubjectBound)?;
    let plan = &subject.declaration.work_binding.native_plan;
    let carriers = derive_prog_suite_section_carriers(plan, source_motif)
        .map_err(ProgSuiteObservedRelationErrorV1::Carrier)?;
    let program = &subject.declaration.development_program.program;

    let mut stages = Vec::with_capacity(program.stages.len());
    for stage in &program.stages {
        if stage.input_identity_id != program.source_identity_id {
            return Err(ProgSuiteObservedRelationErrorV1::StageInputIsNotProgramSource {
                stage_id: stage.stage_id.clone(),
                input_identity_id: stage.input_identity_id.clone(),
            });
        }
        let binding = subject
            .declaration
            .work_binding
            .section_bindings
            .iter()
            .find(|binding| {
                binding.work_node_id == stage.work_node_id
                    && binding.derivation_id.as_deref() == Some(stage.derivation_id.as_str())
            })
            .ok_or_else(|| ProgSuiteObservedRelationErrorV1::MissingSectionBinding {
                stage_id: stage.stage_id.clone(),
                work_node_id: stage.work_node_id.clone(),
            })?;
        let section_index = binding.section_index;
        let section = plan.sections.get(section_index).ok_or_else(|| {
            ProgSuiteObservedRelationErrorV1::MissingNativeSection {
                stage_id: stage.stage_id.clone(),
                section_index,
            }
        })?;
        let carrier = carriers.sections.get(section_index).ok_or_else(|| {
            ProgSuiteObservedRelationErrorV1::MissingCarrier {
                stage_id: stage.stage_id.clone(),
                section_index,
            }
        })?;

        let (interpreted, operations) =
            interpret_operations(&stage.stage_id, source_motif, &stage.operations)?;
        let operation_matches = interpreted == carrier.transformed_motif;
        let first_progression_degree = *section
            .progression_degrees
            .first()
            .ok_or_else(|| ProgSuiteObservedRelationErrorV1::EmptyProgression {
                stage_id: stage.stage_id.clone(),
            })?;

        let period = Period::parallel_in(
            &carrier.meter_fitted_motif,
            &section.progression_degrees,
            f64::from(section.meter),
            section.key.cadence_dominant_degree(),
        );
        let phrase_statement = period
            .antecedent
            .line
            .fragment(0, carrier.meter_fitted_motif.len());
        if phrase_statement.total_duration() != carrier.meter_fit.target_duration {
            return Err(ProgSuiteObservedRelationErrorV1::PhraseStatementDurationMismatch {
                stage_id: stage.stage_id.clone(),
                expected: carrier.meter_fit.target_duration,
                found: phrase_statement.total_duration(),
            });
        }

        let phrase_adaptation =
            observe_phrase_adaptation(carrier, &phrase_statement, first_progression_degree);
        let score_projection = observe_score_projection(
            &subject.realization.score,
            section.start,
            section.key,
            &section.progression_degrees,
            subject.intent.seed,
            &phrase_statement,
            carrier.meter_fit.target_duration,
        );
        let pitched = phrase_statement
            .notes
            .iter()
            .filter(|note| note.degree.is_some())
            .count();
        let status = if pitched < 2 || score_projection.observed.len() < 2 {
            ProgSuiteObservedRelationStatusV1::InsufficientMaterial
        } else if !operation_matches {
            ProgSuiteObservedRelationStatusV1::OperationMechanismMismatch
        } else if !phrase_adaptation.event_alignment_exact {
            ProgSuiteObservedRelationStatusV1::PhraseEventAlignmentFailed
        } else if !score_projection.event_alignment_exact
            || !score_projection.all_pitch_differences_compatible_with_render_profile
        {
            ProgSuiteObservedRelationStatusV1::ScoreProjectionFailed
        } else {
            ProgSuiteObservedRelationStatusV1::ObservedWithDeclaredAdaptation
        };

        stages.push(ProgSuiteStageObservedRelationV1 {
            stage_id: stage.stage_id.clone(),
            section_index,
            work_node_id: stage.work_node_id.clone(),
            derivation_id: stage.derivation_id.clone(),
            operations,
            independently_interpreted_transform: interpreted,
            native_transformed_carrier: carrier.transformed_motif.clone(),
            ordered_operation_matches_native_carrier: operation_matches,
            meter_fitted_carrier: carrier.meter_fitted_motif.clone(),
            phrase_adaptation,
            score_projection,
            status,
        });
    }

    Ok(ProgSuiteObservedRelationEvidenceV1 {
        version: PROG_SUITE_OBSERVED_RELATION_VERSION.into(),
        subject: subject.clone(),
        stages,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteObservedRelationEvidenceV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteObservedRelationErrorV1> {
        if self.version != PROG_SUITE_OBSERVED_RELATION_VERSION {
            return Err(ProgSuiteObservedRelationErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteObservedRelationErrorV1::NonCanonicalNonClaims);
        }
        let canonical = measure_prog_suite_observed_relations(&self.subject)?;
        if &canonical != self {
            return Err(ProgSuiteObservedRelationErrorV1::CanonicalEvidenceMismatch);
        }
        Ok(())
    }
}

fn interpret_operations(
    stage_id: &str,
    source: &Motif,
    operations: &[DevelopmentOperationV1],
) -> Result<(Motif, Vec<ProgSuiteOperationObservationV1>), ProgSuiteObservedRelationErrorV1> {
    let mut current = source.clone();
    let mut receipts = Vec::with_capacity(operations.len());
    for operation in operations {
        let input = current.clone();
        let (semantics, pivot, output) = match &operation.class {
            ThematicTransformationClassV1::LiteralReturn => (
                ProgSuiteObservedOperationSemanticsV1::LiteralReturnIdentity,
                None,
                current.clone(),
            ),
            ThematicTransformationClassV1::Inversion => {
                let pivot = current
                    .notes
                    .iter()
                    .find_map(|note| note.degree)
                    .ok_or_else(|| ProgSuiteObservedRelationErrorV1::MissingInversionPivot {
                        stage_id: stage_id.into(),
                        operation_id: operation.operation_id.clone(),
                    })?;
                (
                    ProgSuiteObservedOperationSemanticsV1::InversionAboutCurrentFirstPitchedDegree,
                    Some(pivot),
                    current.invert(pivot),
                )
            }
            ThematicTransformationClassV1::Retrograde => (
                ProgSuiteObservedOperationSemanticsV1::RetrogradeExactEventOrder,
                None,
                current.retrograde(),
            ),
            unsupported => {
                return Err(ProgSuiteObservedRelationErrorV1::UnsupportedOperation {
                    stage_id: stage_id.into(),
                    operation_id: operation.operation_id.clone(),
                    class: unsupported.clone(),
                });
            }
        };
        receipts.push(ProgSuiteOperationObservationV1 {
            operation_id: operation.operation_id.clone(),
            class: operation.class.clone(),
            semantics,
            inversion_pivot_degree: pivot,
            input,
            output: output.clone(),
        });
        current = output;
    }
    Ok((current, receipts))
}

fn observe_phrase_adaptation(
    carrier: &ProgSuiteSectionCarrierV1,
    phrase_statement: &Motif,
    first_progression_degree: i32,
) -> ProgSuitePhraseAdaptationObservationV1 {
    let source_count = carrier.meter_fitted_motif.notes.len();
    let phrase_count = phrase_statement.notes.len();
    let aligned = source_count.min(phrase_count);
    let reanchor = first_progression_degree - 1;
    let duration_matches = carrier
        .meter_fitted_motif
        .notes
        .iter()
        .zip(&phrase_statement.notes)
        .filter(|(left, right)| left.duration == right.duration)
        .count();
    let rest_matches = carrier
        .meter_fitted_motif
        .notes
        .iter()
        .zip(&phrase_statement.notes)
        .filter(|(left, right)| left.degree.is_none() == right.degree.is_none())
        .count();
    let degree_matches = carrier
        .meter_fitted_motif
        .notes
        .iter()
        .zip(&phrase_statement.notes)
        .filter(|(source, phrase)| match (source.degree, phrase.degree) {
            (None, None) => true,
            (Some(source), Some(phrase)) => source + reanchor == phrase,
            _ => false,
        })
        .count();

    ProgSuitePhraseAdaptationObservationV1 {
        first_progression_degree,
        meter_fit: carrier.meter_fit.clone(),
        phrase_statement: phrase_statement.clone(),
        source_event_count: source_count,
        phrase_event_count: phrase_count,
        duration_matches,
        rest_pattern_matches: rest_matches,
        degree_matches_after_reanchor: degree_matches,
        event_alignment_exact: source_count == phrase_count
            && duration_matches == aligned
            && rest_matches == aligned,
    }
}

fn observe_score_projection(
    score: &Score,
    section_start: Duration,
    key: Key,
    progression: &[i32],
    seed: u64,
    phrase_statement: &Motif,
    statement_duration: Duration,
) -> ProgSuiteScoreProjectionObservationV1 {
    let expected = rendered_statement_events(phrase_statement, key);
    let eligible_raise = eligible_applied_dominant_raise_pc(key, progression, 0, seed);
    let section_end = section_start + statement_duration;
    let mut observed: Vec<_> = score
        .notes
        .iter()
        .filter(|note| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= section_start.beats() - 1e-9
                && note.onset.beats() < section_end.beats() - 1e-9
        })
        .map(|note| ProgSuiteObservedMelodyEventV1 {
            relative_onset: note.onset.saturating_sub(section_start),
            duration: note.duration,
            pitch_midi: note.pitch.midi(),
        })
        .collect();
    observed.sort_by(|left, right| {
        left.relative_onset
            .beats()
            .total_cmp(&right.relative_onset.beats())
    });

    let expected_len = expected.len();
    let observed_len = observed.len();
    let aligned = expected_len.min(observed_len);
    let onset_matches = expected
        .iter()
        .zip(&observed)
        .filter(|(left, right)| left.relative_onset == right.relative_onset)
        .count();
    let duration_matches = expected
        .iter()
        .zip(&observed)
        .filter(|(left, right)| left.duration == right.duration)
        .count();
    let exact_pitch_matches = expected
        .iter()
        .zip(&observed)
        .filter(|(left, right)| left.pitch_midi == right.pitch_midi)
        .count();
    let octave_matches = expected
        .iter()
        .zip(&observed)
        .filter(|(left, right)| {
            left.pitch_midi != right.pitch_midi && left.pitch_midi % 12 == right.pitch_midi % 12
        })
        .count();
    let raised_matches = expected
        .iter()
        .zip(&observed)
        .filter(|(left, right)| {
            let Some(eligible) = eligible_raise else {
                return false;
            };
            PitchClass::new(i32::from(left.pitch_midi)) == eligible
                && PitchClass::new(i32::from(right.pitch_midi)) == eligible.transpose(1)
        })
        .count();
    let compatible = exact_pitch_matches + octave_matches + raised_matches;
    let same_cardinality = aligned > 0 && aligned == expected_len && aligned == observed_len;

    ProgSuiteScoreProjectionObservationV1 {
        expected,
        observed,
        eligible_raised_pitch_class: eligible_raise,
        aligned_event_count: aligned,
        onset_matches,
        duration_matches,
        exact_pitch_matches,
        octave_equivalent_pitch_matches: octave_matches,
        eligible_raised_semitone_matches: raised_matches,
        event_alignment_exact: same_cardinality
            && onset_matches == aligned
            && duration_matches == aligned,
        all_pitch_differences_compatible_with_render_profile: same_cardinality
            && compatible == aligned,
    }
}

fn rendered_statement_events(statement: &Motif, key: Key) -> Vec<ProgSuiteObservedMelodyEventV1> {
    let mut cursor = Duration::zero();
    let mut events = Vec::new();
    for (pitch, duration) in statement.render(key.scale(), 5) {
        if let Some(pitch) = pitch {
            events.push(ProgSuiteObservedMelodyEventV1 {
                relative_onset: cursor,
                duration,
                pitch_midi: pitch.midi(),
            });
        }
        cursor = cursor + duration;
    }
    events
}

/// Evidence-side transcription of the composer's eligibility rule for the
/// first-measure applied-dominant raised pitch class. It does not regenerate
/// any score note and grants no causal attribution by itself.
fn eligible_applied_dominant_raise_pc(
    key: Key,
    progression: &[i32],
    measure_index: usize,
    seed: u64,
) -> Option<PitchClass> {
    if key.tonality != Tonality::Major {
        return None;
    }
    let current = *progression.get(measure_index)?;
    let target = *progression.get(measure_index + 1)?;
    let expected_current = ((target - 1 + 4).rem_euclid(7)) + 1;
    if current != expected_current || current == 7 || target == 1 {
        return None;
    }
    if current == 1 && (seed & 4) == 0 {
        return None;
    }
    // V7/V on I is a harmonic-substitution case; its raised melody pitch is
    // intentionally not modeled by `applied_dominant_raised_pc` upstream.
    if current == 1 {
        return None;
    }
    let third_degree = ((current - 1 + 2).rem_euclid(7)) + 1;
    Some(key.scale().degree_pitch_class(third_degree))
}

fn required_nonclaims() -> Vec<ProgSuiteObservedRelationNonClaimV1> {
    vec![
        ProgSuiteObservedRelationNonClaimV1::DoesNotEstablishHistoricalExecutionTrace,
        ProgSuiteObservedRelationNonClaimV1::DoesNotEstablishWholeSectionExactTransformation,
        ProgSuiteObservedRelationNonClaimV1::DoesNotEstablishListenerPerception,
        ProgSuiteObservedRelationNonClaimV1::DoesNotEstablishArtisticQuality,
        ProgSuiteObservedRelationNonClaimV1::DoesNotGrantProductAuthority,
        ProgSuiteObservedRelationNonClaimV1::OperationInterpreterReusesCanonicalMotifPrimitives,
        ProgSuiteObservedRelationNonClaimV1::CanonicalPhraseReconstructionIsNotIndependentComposerImplementation,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, Pitch, PitchClass, Style,
        bind_prog_suite_subject, plan_prog_suite, realize_prog_suite_subject_bound,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn subject() -> ProgSuiteSubjectBoundRealizationV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        let declaration = bind_prog_suite_subject(&plan, &motif()).unwrap();
        realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent {
                energy: 0.73,
                seed: 9182,
                ..MusicalIntent::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn seed_five_b_preserves_inversion_then_retrograde_order() {
        let evidence = measure_prog_suite_observed_relations(&subject()).unwrap();
        let b = &evidence.stages[0];
        assert_eq!(b.operations.len(), 2);
        assert_eq!(b.operations[0].inversion_pivot_degree, Some(1));
        assert_eq!(b.independently_interpreted_transform.degrees(), vec![-3, -1, 0, 1]);
        assert!(b.ordered_operation_matches_native_carrier);
    }

    #[test]
    fn b_preserves_meter_fit_and_first_statement_alignment() {
        let evidence = measure_prog_suite_observed_relations(&subject()).unwrap();
        let b = &evidence.stages[0];
        assert_eq!(
            (
                b.phrase_adaptation.meter_fit.rhythm_scale_numerator,
                b.phrase_adaptation.meter_fit.rhythm_scale_denominator,
            ),
            (7, 4)
        );
        assert!(b.phrase_adaptation.event_alignment_exact);
        assert!(b.score_projection.event_alignment_exact);
        assert!(b.score_projection.all_pitch_differences_compatible_with_render_profile);
        assert_eq!(
            b.status,
            ProgSuiteObservedRelationStatusV1::ObservedWithDeclaredAdaptation
        );
    }

    #[test]
    fn arbitrary_semitone_mutation_is_not_admitted_as_chromatic_adaptation() {
        let mut subject = subject();
        let b_start = subject.declaration.work_binding.native_plan.sections[1].start;
        let note = subject
            .realization
            .score
            .notes
            .iter_mut()
            .find(|note| note.role == VoiceRole::Melody && note.onset.beats() >= b_start.beats())
            .unwrap();
        note.pitch = Pitch::from_midi(note.pitch.midi().saturating_add(1).min(127));
        subject.validate_structure().unwrap();
        let evidence = measure_prog_suite_observed_relations(&subject).unwrap();
        assert_eq!(evidence.stages[0].score_projection.eligible_raised_pitch_class, None);
        assert_eq!(
            evidence.stages[0].status,
            ProgSuiteObservedRelationStatusV1::ScoreProjectionFailed
        );
    }

    #[test]
    fn reversed_composite_order_is_distinguishable_for_this_subject() {
        let reversed = vec![
            DevelopmentOperationV1 {
                operation_id: "retrograde-first".into(),
                class: ThematicTransformationClassV1::Retrograde,
            },
            DevelopmentOperationV1 {
                operation_id: "invert-second".into(),
                class: ThematicTransformationClassV1::Inversion,
            },
        ];
        let (wrong_order, _) = interpret_operations("test", &motif(), &reversed).unwrap();
        let evidence = measure_prog_suite_observed_relations(&subject()).unwrap();
        assert_ne!(wrong_order, evidence.stages[0].native_transformed_carrier);
    }

    #[test]
    fn canonical_rederivation_rejects_status_tampering() {
        let mut evidence = measure_prog_suite_observed_relations(&subject()).unwrap();
        evidence.stages[0].status = ProgSuiteObservedRelationStatusV1::ScoreProjectionFailed;
        assert_eq!(
            evidence.validate(),
            Err(ProgSuiteObservedRelationErrorV1::CanonicalEvidenceMismatch)
        );
    }
}
