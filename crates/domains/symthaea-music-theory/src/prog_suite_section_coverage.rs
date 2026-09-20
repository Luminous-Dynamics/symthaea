// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Whole-section symbolic projection coverage for ProgSuite development stages.
//!
//! The first-statement relation evidence proves that a declared transformed
//! carrier survives into an audible symbolic statement. This module extends the
//! projection check across the complete development section without claiming
//! that every section event is itself an untouched inversion/retrograde.
//!
//! Instead, it reconstructs the canonical `Period` from the exact meter-fitted
//! carrier and checks whether every completed-score Melody event is explained by
//! the declared phrase/harmonic realization profile: exact slot timing, the
//! phrase-breath duration rule, octave/register migration, and only the exact
//! applied-dominant chromatic raises independently eligible in that measure.

use crate::harmony::{Key, Tonality};
use crate::phrase::{Period, Phrase};
use crate::pitch::PitchClass;
use crate::prog_suite::derive_prog_suite_section_carriers;
use crate::prog_suite_observed_relation::{
    ProgSuiteObservedMelodyEventV1, ProgSuiteObservedRelationErrorV1,
    ProgSuiteObservedRelationEvidenceV1, ProgSuiteObservedRelationStatusV1,
    measure_prog_suite_observed_relations,
};
use crate::rhythm::Duration;
use crate::score::{Score, VoiceRole};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const PROG_SUITE_SECTION_COVERAGE_VERSION: &str =
    "melothaea-prog-suite-section-coverage-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionExpectedMelodyEventV1 {
    pub relative_onset: Duration,
    /// Phrase-written slot duration before performance breath shortening.
    pub written_duration: Duration,
    /// Exact duration expected in the completed score after the disclosed
    /// phrase-breath rule.
    pub sounded_duration: Duration,
    pub pitch_midi: u8,
    /// Pitch class eligible for the composer's +1 applied-dominant correction
    /// at this exact event position, if any.
    pub eligible_raised_pitch_class: Option<PitchClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSectionCoverageStatusV1 {
    /// First-statement relation evidence is positive and the complete section
    /// projection is explained by the declared adaptation profile.
    CoveredUnderDeclaredAdaptation,
    /// The source relation profile had too little material to establish identity.
    IndeterminateInsufficientMaterial,
    /// First-statement relation evidence rejected the declared relationship.
    SourceRelationRejected,
    /// The source relation was admitted, but some later event in the completed
    /// section is not explained by the declared projection profile.
    SectionProjectionFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSectionCoverageNonClaimV1 {
    WholeSectionProjectionDoesNotMeanWholeSectionExactTransformation,
    DoesNotEstablishHistoricalExecutionTrace,
    DoesNotEstablishListenerPerception,
    DoesNotEstablishArtisticQuality,
    DoesNotGrantProductAuthority,
    CanonicalPhraseReconstructionIsNotIndependentComposerImplementation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionCoverageRecordV1 {
    pub stage_id: String,
    pub section_index: usize,
    pub section_start: Duration,
    pub section_end: Duration,
    pub source_relation_status: ProgSuiteObservedRelationStatusV1,
    pub expected: Vec<ProgSuiteSectionExpectedMelodyEventV1>,
    pub observed: Vec<ProgSuiteObservedMelodyEventV1>,
    pub aligned_event_count: usize,
    pub onset_matches: usize,
    pub sounded_duration_matches: usize,
    pub exact_pitch_matches: usize,
    pub octave_equivalent_pitch_matches: usize,
    pub eligible_raised_semitone_matches: usize,
    pub event_alignment_exact: bool,
    pub all_pitch_differences_compatible_with_projection_profile: bool,
    pub status: ProgSuiteSectionCoverageStatusV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionCoverageEvidenceV1 {
    pub version: String,
    /// Canonical first-statement relation substrate. This layer strengthens
    /// coverage but does not replace its ordered-operation authority.
    pub observed_relations: ProgSuiteObservedRelationEvidenceV1,
    pub records: Vec<ProgSuiteSectionCoverageRecordV1>,
    pub nonclaims: Vec<ProgSuiteSectionCoverageNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteSectionCoverageErrorV1 {
    WrongVersion { found: String },
    ObservedRelation(ProgSuiteObservedRelationErrorV1),
    SourceObservedRelationMismatch,
    MissingSourceMotif,
    Carrier(crate::prog_suite::ProgSuitePlanErrorV1),
    MissingCarrier { stage_id: String, section_index: usize },
    MissingNativeSection { stage_id: String, section_index: usize },
    NonCanonicalNonClaims,
    CanonicalCoverageMismatch,
}

pub fn measure_prog_suite_section_coverage(
    observed_relations: &ProgSuiteObservedRelationEvidenceV1,
) -> Result<ProgSuiteSectionCoverageEvidenceV1, ProgSuiteSectionCoverageErrorV1> {
    let canonical_observed = measure_prog_suite_observed_relations(&observed_relations.subject)
        .map_err(ProgSuiteSectionCoverageErrorV1::ObservedRelation)?;
    if &canonical_observed != observed_relations {
        return Err(ProgSuiteSectionCoverageErrorV1::SourceObservedRelationMismatch);
    }

    let source_motif = observed_relations
        .subject
        .declaration
        .source_motif()
        .map_err(|_| ProgSuiteSectionCoverageErrorV1::MissingSourceMotif)?;
    let plan = &observed_relations.subject.declaration.work_binding.native_plan;
    let carriers = derive_prog_suite_section_carriers(plan, source_motif)
        .map_err(ProgSuiteSectionCoverageErrorV1::Carrier)?;

    let mut records = Vec::with_capacity(observed_relations.stages.len());
    for source in &observed_relations.stages {
        let section = plan.sections.get(source.section_index).ok_or_else(|| {
            ProgSuiteSectionCoverageErrorV1::MissingNativeSection {
                stage_id: source.stage_id.clone(),
                section_index: source.section_index,
            }
        })?;
        let carrier = carriers.sections.get(source.section_index).ok_or_else(|| {
            ProgSuiteSectionCoverageErrorV1::MissingCarrier {
                stage_id: source.stage_id.clone(),
                section_index: source.section_index,
            }
        })?;

        let period = Period::parallel_in(
            &carrier.meter_fitted_motif,
            &section.progression_degrees,
            f64::from(section.meter),
            section.key.cadence_dominant_degree(),
        );
        let expected = expected_section_events(
            &period,
            section.key,
            section.meter,
            observed_relations.subject.intent.seed,
        );
        let observed = observed_section_events(
            &observed_relations.subject.realization.score,
            section.start,
            section.end,
        );

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
            .filter(|(left, right)| left.sounded_duration == right.duration)
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
                let Some(eligible) = left.eligible_raised_pitch_class else {
                    return false;
                };
                PitchClass::new(i32::from(left.pitch_midi)) == eligible
                    && PitchClass::new(i32::from(right.pitch_midi)) == eligible.transpose(1)
            })
            .count();
        let same_cardinality = aligned > 0 && aligned == expected_len && aligned == observed_len;
        let event_alignment_exact = same_cardinality
            && onset_matches == aligned
            && duration_matches == aligned;
        let pitch_profile_exact = same_cardinality
            && exact_pitch_matches + octave_matches + raised_matches == aligned;

        let status = match source.status {
            ProgSuiteObservedRelationStatusV1::InsufficientMaterial => {
                ProgSuiteSectionCoverageStatusV1::IndeterminateInsufficientMaterial
            }
            ProgSuiteObservedRelationStatusV1::OperationMechanismMismatch
            | ProgSuiteObservedRelationStatusV1::PhraseEventAlignmentFailed
            | ProgSuiteObservedRelationStatusV1::ScoreProjectionFailed => {
                ProgSuiteSectionCoverageStatusV1::SourceRelationRejected
            }
            ProgSuiteObservedRelationStatusV1::ObservedWithDeclaredAdaptation
                if event_alignment_exact && pitch_profile_exact =>
            {
                ProgSuiteSectionCoverageStatusV1::CoveredUnderDeclaredAdaptation
            }
            ProgSuiteObservedRelationStatusV1::ObservedWithDeclaredAdaptation => {
                ProgSuiteSectionCoverageStatusV1::SectionProjectionFailed
            }
        };

        records.push(ProgSuiteSectionCoverageRecordV1 {
            stage_id: source.stage_id.clone(),
            section_index: source.section_index,
            section_start: section.start,
            section_end: section.end,
            source_relation_status: source.status,
            expected,
            observed,
            aligned_event_count: aligned,
            onset_matches,
            sounded_duration_matches: duration_matches,
            exact_pitch_matches,
            octave_equivalent_pitch_matches: octave_matches,
            eligible_raised_semitone_matches: raised_matches,
            event_alignment_exact,
            all_pitch_differences_compatible_with_projection_profile: pitch_profile_exact,
            status,
        });
    }

    Ok(ProgSuiteSectionCoverageEvidenceV1 {
        version: PROG_SUITE_SECTION_COVERAGE_VERSION.into(),
        observed_relations: observed_relations.clone(),
        records,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteSectionCoverageEvidenceV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteSectionCoverageErrorV1> {
        if self.version != PROG_SUITE_SECTION_COVERAGE_VERSION {
            return Err(ProgSuiteSectionCoverageErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteSectionCoverageErrorV1::NonCanonicalNonClaims);
        }
        let canonical = measure_prog_suite_section_coverage(&self.observed_relations)?;
        if &canonical != self {
            return Err(ProgSuiteSectionCoverageErrorV1::CanonicalCoverageMismatch);
        }
        Ok(())
    }

    pub fn all_stages_covered(&self) -> bool {
        self.records.iter().all(|record| {
            record.status == ProgSuiteSectionCoverageStatusV1::CoveredUnderDeclaredAdaptation
        })
    }
}

fn expected_section_events(
    period: &Period,
    key: Key,
    meter: u8,
    seed: u64,
) -> Vec<ProgSuiteSectionExpectedMelodyEventV1> {
    let mut events = Vec::new();
    let mut section_cursor = Duration::zero();
    append_phrase_events(
        &mut events,
        &period.antecedent,
        key,
        meter,
        seed,
        section_cursor,
        true,
    );
    section_cursor = section_cursor + period.antecedent.total_duration();
    append_phrase_events(
        &mut events,
        &period.consequent,
        key,
        meter,
        seed,
        section_cursor,
        false,
    );
    events
}

#[allow(clippy::too_many_arguments)]
fn append_phrase_events(
    events: &mut Vec<ProgSuiteSectionExpectedMelodyEventV1>,
    phrase: &Phrase,
    key: Key,
    meter: u8,
    seed: u64,
    phrase_start: Duration,
    apply_phrase_breath: bool,
) {
    let rendered = phrase.render(key, 5);
    let last_pitched = rendered.iter().rposition(|(pitch, _)| pitch.is_some());
    let mut cursor = Duration::zero();
    for (event_index, (pitch, duration)) in rendered.into_iter().enumerate() {
        if let Some(pitch) = pitch {
            let sounded_duration = if apply_phrase_breath && Some(event_index) == last_pitched {
                if duration.beats() > 1.0 {
                    duration.saturating_sub(Duration::eighth())
                } else {
                    duration.scale(2, 3)
                }
            } else {
                duration
            };
            let measure_index = measure_index_for_offset(cursor, meter);
            let eligible = eligible_applied_dominant_raise_pc(
                key,
                &phrase.progression,
                measure_index,
                seed,
            );
            events.push(ProgSuiteSectionExpectedMelodyEventV1 {
                relative_onset: phrase_start + cursor,
                written_duration: duration,
                sounded_duration,
                pitch_midi: pitch.midi(),
                eligible_raised_pitch_class: eligible,
            });
        }
        cursor = cursor + duration;
    }
}

fn observed_section_events(
    score: &Score,
    section_start: Duration,
    section_end: Duration,
) -> Vec<ProgSuiteObservedMelodyEventV1> {
    let mut observed: Vec<_> = score
        .notes
        .iter()
        .filter(|note| {
            note.role == VoiceRole::Melody
                && compare_duration(note.onset, section_start) != Ordering::Less
                && compare_duration(note.onset, section_end) == Ordering::Less
        })
        .map(|note| ProgSuiteObservedMelodyEventV1 {
            relative_onset: note.onset.saturating_sub(section_start),
            duration: note.duration,
            pitch_midi: note.pitch.midi(),
        })
        .collect();
    observed.sort_by(|left, right| compare_duration(left.relative_onset, right.relative_onset));
    observed
}

fn measure_index_for_offset(offset: Duration, meter: u8) -> usize {
    let numerator = i128::from(offset.num());
    let denominator = i128::from(offset.den()) * i128::from(meter);
    if denominator <= 0 || numerator <= 0 {
        0
    } else {
        (numerator / denominator) as usize
    }
}

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
    if current == 1 {
        return None;
    }
    let third_degree = ((current - 1 + 2).rem_euclid(7)) + 1;
    Some(key.scale().degree_pitch_class(third_degree))
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

fn required_nonclaims() -> Vec<ProgSuiteSectionCoverageNonClaimV1> {
    vec![
        ProgSuiteSectionCoverageNonClaimV1::WholeSectionProjectionDoesNotMeanWholeSectionExactTransformation,
        ProgSuiteSectionCoverageNonClaimV1::DoesNotEstablishHistoricalExecutionTrace,
        ProgSuiteSectionCoverageNonClaimV1::DoesNotEstablishListenerPerception,
        ProgSuiteSectionCoverageNonClaimV1::DoesNotEstablishArtisticQuality,
        ProgSuiteSectionCoverageNonClaimV1::DoesNotGrantProductAuthority,
        ProgSuiteSectionCoverageNonClaimV1::CanonicalPhraseReconstructionIsNotIndependentComposerImplementation,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, Pitch, PitchClass, Style, bind_prog_suite_subject,
        measure_prog_suite_observed_relations, plan_prog_suite, realize_prog_suite_subject_bound,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn observed() -> ProgSuiteObservedRelationEvidenceV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        let declaration = bind_prog_suite_subject(&plan, &motif()).unwrap();
        let subject = realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent {
                energy: 0.73,
                seed: 9182,
                ..MusicalIntent::default()
            },
        )
        .unwrap();
        measure_prog_suite_observed_relations(&subject).unwrap()
    }

    #[test]
    fn canonical_prog_suite_has_complete_section_projection_coverage() {
        let coverage = measure_prog_suite_section_coverage(&observed()).unwrap();
        assert_eq!(coverage.records.len(), 3);
        assert!(coverage.all_stages_covered());
        for record in &coverage.records {
            assert!(record.event_alignment_exact);
            assert!(record.all_pitch_differences_compatible_with_projection_profile);
            assert_eq!(record.expected.len(), record.observed.len());
        }
        coverage.validate().unwrap();
    }

    #[test]
    fn later_section_damage_is_detected_even_when_first_statement_survives() {
        let mut observed = observed();
        let section = observed.subject.declaration.work_binding.native_plan.sections[1].clone();
        let first_bar_end = section.start + Duration::new(i64::from(section.meter), 1);
        let note = observed
            .subject
            .realization
            .score
            .notes
            .iter_mut()
            .find(|note| {
                note.role == VoiceRole::Melody
                    && compare_duration(note.onset, first_bar_end) != Ordering::Less
                    && compare_duration(note.onset, section.end) == Ordering::Less
            })
            .unwrap();
        note.pitch = Pitch::from_midi(note.pitch.midi().saturating_add(6).min(127));

        // Re-derive first-statement evidence from the damaged subject: the first
        // bar remains intact, so B is still admitted there.
        observed = measure_prog_suite_observed_relations(&observed.subject).unwrap();
        assert_eq!(
            observed.stages[0].status,
            ProgSuiteObservedRelationStatusV1::ObservedWithDeclaredAdaptation
        );

        let coverage = measure_prog_suite_section_coverage(&observed).unwrap();
        assert_eq!(
            coverage.records[0].status,
            ProgSuiteSectionCoverageStatusV1::SectionProjectionFailed
        );
        assert!(!coverage.all_stages_covered());
    }

    #[test]
    fn section_projection_records_phrase_breath_exactly() {
        let coverage = measure_prog_suite_section_coverage(&observed()).unwrap();
        let b = &coverage.records[0];
        assert!(b.expected.iter().any(|event| event.written_duration != event.sounded_duration));
        assert_eq!(b.sounded_duration_matches, b.aligned_event_count);
    }

    #[test]
    fn serialized_coverage_status_cannot_be_promoted() {
        let mut coverage = measure_prog_suite_section_coverage(&observed()).unwrap();
        coverage.records[0].status = ProgSuiteSectionCoverageStatusV1::SectionProjectionFailed;
        assert_eq!(
            coverage.validate(),
            Err(ProgSuiteSectionCoverageErrorV1::CanonicalCoverageMismatch)
        );
    }

    #[test]
    fn whole_section_coverage_retains_exact_nonclaim_boundary() {
        let coverage = measure_prog_suite_section_coverage(&observed()).unwrap();
        assert_eq!(coverage.nonclaims, required_nonclaims());
        assert!(coverage.nonclaims.contains(
            &ProgSuiteSectionCoverageNonClaimV1::WholeSectionProjectionDoesNotMeanWholeSectionExactTransformation
        ));
    }
}
