// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Controlled legacy-vs-contextual harmony comparison for native ProgSuite.
//!
//! This module turns the optional contextual harmony profile into an exact
//! paired symbolic subject. The source motif, MusicalIntent, native plan,
//! long-range context, section keys/meters/spans, thematic transformations,
//! tempo, and generator implementation are held constant. The only admitted
//! plan intervention is the progression-degree rewrite already frozen by
//! [`crate::prog_suite_contextual_harmony`].
//!
//! Observing a symbolic difference under that controlled intervention supports
//! a narrow causal statement about the current deterministic symbolic engine.
//! It does not establish audibility, listener preference, artistic quality,
//! historical correctness, or generalization to other motifs/seeds/styles.

use crate::MUSIC_THEORY_ENGINE_VERSION;
use crate::composer::MusicalIntent;
use crate::motif::Motif;
use crate::prog_suite::{
    ProgSuitePlanErrorV1, ProgSuiteRealizationV1, realize_prog_suite_with_plan,
};
use crate::prog_suite_contextual_harmony::{
    ProgSuiteContextualHarmonyErrorV1, ProgSuiteContextualHarmonyPlanV1,
    ProgSuiteContextualHarmonyProfileV1, derive_prog_suite_contextual_harmony,
};
use crate::prog_suite_development_context::{
    ProgSuiteDevelopmentContextErrorV1, ProgSuiteDevelopmentContextV1,
};
use crate::rhythm::Duration;
use crate::score::{Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_COMPARISON_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-comparison-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyComparisonAuthorityV1 {
    /// The contextual policy changed at least one progression vector and the
    /// current deterministic symbolic engine produced a different Score.
    ControlledProgressionInterventionWithSymbolicDifference,
    /// The policy changed at least one progression vector, but the realized
    /// symbolic Score stayed exactly equal for this retained subject.
    ControlledProgressionInterventionWithoutSymbolicDifference,
    /// For this subject the source plan already matched the profile in every
    /// section, so there was no actual progression-vector intervention.
    NoProgressionDifferenceForSubject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyComparisonNonClaimV1 {
    SymbolicDifferenceDoesNotEstablishAudibleDifference,
    SymbolicDifferenceDoesNotEstablishListenerPreference,
    SymbolicDifferenceDoesNotEstablishArtisticQuality,
    SubjectResultDoesNotEstablishGeneralization,
    PolicyDoesNotEstablishHistoricalStyle,
    ComparisonDoesNotGrantProductAuthority,
}

/// Pairwise field differences for events aligned by deterministic generation
/// order inside one exact work span. This is deliberately not an edit distance:
/// unmatched tails are reported separately instead of being guessed into an
/// alignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteEventDifferenceSummaryV1 {
    pub source_note_count: usize,
    pub contextual_note_count: usize,
    pub aligned_event_count: usize,
    pub aligned_event_difference_count: usize,
    pub pitch_difference_count: usize,
    pub onset_difference_count: usize,
    pub duration_difference_count: usize,
    pub velocity_bits_difference_count: usize,
    pub part_difference_count: usize,
    pub role_difference_count: usize,
    pub emphasis_difference_count: usize,
    pub section_intensity_bits_difference_count: usize,
    pub unmatched_source_note_count: usize,
    pub unmatched_contextual_note_count: usize,
    pub exact_event_stream_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteVoiceComparisonV1 {
    pub role: VoiceRole,
    pub differences: ProgSuiteEventDifferenceSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionHarmonyComparisonV1 {
    pub section_index: usize,
    pub start: Duration,
    pub end: Duration,
    pub source_progression_degrees: Vec<i32>,
    pub contextual_progression_degrees: Vec<i32>,
    pub progression_changed: bool,
    pub all_events: ProgSuiteEventDifferenceSummaryV1,
    pub voices: Vec<ProgSuiteVoiceComparisonV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyComparisonV1 {
    pub version: String,
    /// Cargo package version of the implementation that performed both paired
    /// realizations. This is not a Git/source-revision identity.
    pub engine_version: String,
    pub profile: ProgSuiteContextualHarmonyProfileV1,
    /// Complete retained long-range declaration/context authority.
    pub context: ProgSuiteDevelopmentContextV1,
    /// Exact common source material supplied to both realizations.
    pub source_motif: Motif,
    /// Exact common expressive input supplied to both realizations.
    pub intent: MusicalIntent,
    /// Canonical progression-only intervention plan.
    pub intervention: ProgSuiteContextualHarmonyPlanV1,
    pub source_realization: ProgSuiteRealizationV1,
    pub contextual_realization: ProgSuiteRealizationV1,
    pub changed_progression_section_indices: Vec<usize>,
    pub whole_score: ProgSuiteEventDifferenceSummaryV1,
    pub sections: Vec<ProgSuiteSectionHarmonyComparisonV1>,
    pub authority: ProgSuiteHarmonyComparisonAuthorityV1,
    pub nonclaims: Vec<ProgSuiteHarmonyComparisonNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteContextualHarmonyComparisonErrorV1 {
    WrongVersion { found: String },
    WrongEngineVersion { found: String },
    SourceContext(ProgSuiteDevelopmentContextErrorV1),
    Intervention(ProgSuiteContextualHarmonyErrorV1),
    SourceRealization(ProgSuitePlanErrorV1),
    ContextualRealization(ProgSuitePlanErrorV1),
    NonCanonicalNonClaims,
    CanonicalComparisonMismatch,
}

pub fn derive_prog_suite_contextual_harmony_comparison(
    context: &ProgSuiteDevelopmentContextV1,
    source_motif: &Motif,
    intent: &MusicalIntent,
    profile: ProgSuiteContextualHarmonyProfileV1,
) -> Result<ProgSuiteContextualHarmonyComparisonV1, ProgSuiteContextualHarmonyComparisonErrorV1>
{
    context
        .validate()
        .map_err(ProgSuiteContextualHarmonyComparisonErrorV1::SourceContext)?;
    let intervention = derive_prog_suite_contextual_harmony(context, profile)
        .map_err(ProgSuiteContextualHarmonyComparisonErrorV1::Intervention)?;
    intervention
        .validate(context)
        .map_err(ProgSuiteContextualHarmonyComparisonErrorV1::Intervention)?;

    let source_realization = realize_prog_suite_with_plan(
        &intervention.source_plan,
        source_motif,
        intent,
    )
    .map_err(ProgSuiteContextualHarmonyComparisonErrorV1::SourceRealization)?;
    let contextual_realization = realize_prog_suite_with_plan(
        &intervention.contextual_plan,
        source_motif,
        intent,
    )
    .map_err(ProgSuiteContextualHarmonyComparisonErrorV1::ContextualRealization)?;

    let changed_progression_section_indices = intervention
        .receipts
        .iter()
        .filter_map(|receipt| {
            (receipt.source_progression_degrees != receipt.contextual_progression_degrees)
                .then_some(receipt.section_index)
        })
        .collect::<Vec<_>>();

    let whole_score = compare_note_streams(
        &source_realization.score.notes,
        &contextual_realization.score.notes,
    );
    let sections = intervention
        .source_plan
        .sections
        .iter()
        .enumerate()
        .map(|(section_index, section)| {
            let source_notes = notes_in_span(
                &source_realization.score,
                section.start,
                section.end,
                None,
            );
            let contextual_notes = notes_in_span(
                &contextual_realization.score,
                section.start,
                section.end,
                None,
            );
            let voices = [
                VoiceRole::Melody,
                VoiceRole::Harmony,
                VoiceRole::Bass,
                VoiceRole::CounterMelody,
            ]
            .into_iter()
            .map(|role| ProgSuiteVoiceComparisonV1 {
                role,
                differences: compare_note_streams(
                    &notes_in_span(
                        &source_realization.score,
                        section.start,
                        section.end,
                        Some(role),
                    ),
                    &notes_in_span(
                        &contextual_realization.score,
                        section.start,
                        section.end,
                        Some(role),
                    ),
                ),
            })
            .collect();
            ProgSuiteSectionHarmonyComparisonV1 {
                section_index,
                start: section.start,
                end: section.end,
                source_progression_degrees: section.progression_degrees.clone(),
                contextual_progression_degrees: intervention.contextual_plan.sections
                    [section_index]
                    .progression_degrees
                    .clone(),
                progression_changed: section.progression_degrees
                    != intervention.contextual_plan.sections[section_index].progression_degrees,
                all_events: compare_note_streams(&source_notes, &contextual_notes),
                voices,
            }
        })
        .collect::<Vec<_>>();

    let score_exact_match = scores_match_exactly(
        &source_realization.score,
        &contextual_realization.score,
    );
    let authority = if changed_progression_section_indices.is_empty() {
        ProgSuiteHarmonyComparisonAuthorityV1::NoProgressionDifferenceForSubject
    } else if score_exact_match {
        ProgSuiteHarmonyComparisonAuthorityV1::ControlledProgressionInterventionWithoutSymbolicDifference
    } else {
        ProgSuiteHarmonyComparisonAuthorityV1::ControlledProgressionInterventionWithSymbolicDifference
    };

    Ok(ProgSuiteContextualHarmonyComparisonV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_COMPARISON_VERSION.into(),
        engine_version: MUSIC_THEORY_ENGINE_VERSION.into(),
        profile,
        context: context.clone(),
        source_motif: source_motif.clone(),
        intent: *intent,
        intervention,
        source_realization,
        contextual_realization,
        changed_progression_section_indices,
        whole_score,
        sections,
        authority,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteContextualHarmonyComparisonV1 {
    /// Recompute both plans, both realizations, every difference summary, and
    /// the authority state from retained inputs. Serialized summaries cannot be
    /// promoted to evidence by editing them in place.
    pub fn validate(
        &self,
    ) -> Result<(), ProgSuiteContextualHarmonyComparisonErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_COMPARISON_VERSION {
            return Err(ProgSuiteContextualHarmonyComparisonErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.engine_version != MUSIC_THEORY_ENGINE_VERSION {
            return Err(
                ProgSuiteContextualHarmonyComparisonErrorV1::WrongEngineVersion {
                    found: self.engine_version.clone(),
                },
            );
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteContextualHarmonyComparisonErrorV1::NonCanonicalNonClaims);
        }
        let canonical = derive_prog_suite_contextual_harmony_comparison(
            &self.context,
            &self.source_motif,
            &self.intent,
            self.profile,
        )?;
        if &canonical != self {
            return Err(
                ProgSuiteContextualHarmonyComparisonErrorV1::CanonicalComparisonMismatch,
            );
        }
        Ok(())
    }
}

fn notes_in_span(
    score: &Score,
    start: Duration,
    end: Duration,
    role: Option<VoiceRole>,
) -> Vec<ScoreNote> {
    score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            compare_duration(note.onset, start) != Ordering::Less
                && compare_duration(note.onset, end) == Ordering::Less
                && role.is_none_or(|expected| note.role == expected)
        })
        .collect()
}

fn compare_note_streams(
    source: &[ScoreNote],
    contextual: &[ScoreNote],
) -> ProgSuiteEventDifferenceSummaryV1 {
    let aligned_event_count = source.len().min(contextual.len());
    let mut aligned_event_difference_count = 0usize;
    let mut pitch_difference_count = 0usize;
    let mut onset_difference_count = 0usize;
    let mut duration_difference_count = 0usize;
    let mut velocity_bits_difference_count = 0usize;
    let mut part_difference_count = 0usize;
    let mut role_difference_count = 0usize;
    let mut emphasis_difference_count = 0usize;
    let mut section_intensity_bits_difference_count = 0usize;

    for (left, right) in source.iter().zip(contextual) {
        if left != right {
            aligned_event_difference_count += 1;
        }
        pitch_difference_count += usize::from(left.pitch != right.pitch);
        onset_difference_count += usize::from(left.onset != right.onset);
        duration_difference_count += usize::from(left.duration != right.duration);
        velocity_bits_difference_count +=
            usize::from(left.velocity.to_bits() != right.velocity.to_bits());
        part_difference_count += usize::from(left.part != right.part);
        role_difference_count += usize::from(left.role != right.role);
        emphasis_difference_count += usize::from(left.emphasis != right.emphasis);
        section_intensity_bits_difference_count += usize::from(
            left.section_intensity.to_bits() != right.section_intensity.to_bits(),
        );
    }

    let unmatched_source_note_count = source.len().saturating_sub(contextual.len());
    let unmatched_contextual_note_count = contextual.len().saturating_sub(source.len());
    ProgSuiteEventDifferenceSummaryV1 {
        source_note_count: source.len(),
        contextual_note_count: contextual.len(),
        aligned_event_count,
        aligned_event_difference_count,
        pitch_difference_count,
        onset_difference_count,
        duration_difference_count,
        velocity_bits_difference_count,
        part_difference_count,
        role_difference_count,
        emphasis_difference_count,
        section_intensity_bits_difference_count,
        unmatched_source_note_count,
        unmatched_contextual_note_count,
        exact_event_stream_match: source == contextual,
    }
}

fn scores_match_exactly(source: &Score, contextual: &Score) -> bool {
    source.key == contextual.key
        && source.tempo_bpm.to_bits() == contextual.tempo_bpm.to_bits()
        && source.meter == contextual.meter
        && source.total_beats == contextual.total_beats
        && source.notes == contextual.notes
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

fn required_nonclaims() -> Vec<ProgSuiteHarmonyComparisonNonClaimV1> {
    vec![
        ProgSuiteHarmonyComparisonNonClaimV1::SymbolicDifferenceDoesNotEstablishAudibleDifference,
        ProgSuiteHarmonyComparisonNonClaimV1::SymbolicDifferenceDoesNotEstablishListenerPreference,
        ProgSuiteHarmonyComparisonNonClaimV1::SymbolicDifferenceDoesNotEstablishArtisticQuality,
        ProgSuiteHarmonyComparisonNonClaimV1::SubjectResultDoesNotEstablishGeneralization,
        ProgSuiteHarmonyComparisonNonClaimV1::PolicyDoesNotEstablishHistoricalStyle,
        ProgSuiteHarmonyComparisonNonClaimV1::ComparisonDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Key, PitchClass, Style, derive_prog_suite_development_context, plan_prog_suite,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn context(seed: u64) -> ProgSuiteDevelopmentContextV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            seed,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        derive_prog_suite_development_context(&plan).unwrap()
    }

    fn comparison() -> ProgSuiteContextualHarmonyComparisonV1 {
        let context = context(5);
        let intent = MusicalIntent {
            energy: 0.73,
            seed: 9182,
            ..MusicalIntent::default()
        };
        derive_prog_suite_contextual_harmony_comparison(
            &context,
            &motif(),
            &intent,
            ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
        )
        .unwrap()
    }

    #[test]
    fn seed_five_is_a_controlled_progression_intervention_with_symbolic_effect() {
        let comparison = comparison();
        assert!(!comparison.changed_progression_section_indices.is_empty());
        assert_eq!(
            comparison.authority,
            ProgSuiteHarmonyComparisonAuthorityV1::ControlledProgressionInterventionWithSymbolicDifference
        );
        assert!(!comparison.whole_score.exact_event_stream_match);
        assert_ne!(
            comparison.source_realization.score,
            comparison.contextual_realization.score
        );
        comparison.validate().unwrap();
    }

    #[test]
    fn paired_realizations_hold_work_identity_and_duration_constant() {
        let comparison = comparison();
        let source = &comparison.source_realization;
        let contextual = &comparison.contextual_realization;
        assert_eq!(source.plan.home_key, contextual.plan.home_key);
        assert_eq!(source.plan.source_seed, contextual.plan.source_seed);
        assert_eq!(source.plan.total_beats, contextual.plan.total_beats);
        assert_eq!(source.score.total_beats, contextual.score.total_beats);
        assert_eq!(source.score.key, contextual.score.key);
        assert_eq!(source.score.tempo_bpm.to_bits(), contextual.score.tempo_bpm.to_bits());
        assert_eq!(source.score.meter, contextual.score.meter);
    }

    #[test]
    fn section_and_voice_panels_localize_symbolic_changes_without_ranking_them() {
        let comparison = comparison();
        assert_eq!(comparison.sections.len(), 4);
        assert!(
            comparison
                .sections
                .iter()
                .any(|section| !section.all_events.exact_event_stream_match)
        );
        for section in &comparison.sections {
            assert_eq!(section.voices.len(), 4);
            assert_eq!(
                section.progression_changed,
                section.source_progression_degrees != section.contextual_progression_degrees
            );
        }
    }

    #[test]
    fn serialized_difference_tampering_fails_canonical_validation() {
        let mut comparison = comparison();
        comparison.whole_score.pitch_difference_count = comparison
            .whole_score
            .pitch_difference_count
            .saturating_add(1);
        assert!(comparison.validate().is_err());
    }

    #[test]
    fn comparison_keeps_perceptual_and_quality_nonclaims_explicit() {
        let comparison = comparison();
        assert_eq!(comparison.nonclaims, required_nonclaims());
        assert!(comparison.nonclaims.contains(
            &ProgSuiteHarmonyComparisonNonClaimV1::SymbolicDifferenceDoesNotEstablishListenerPreference
        ));
        assert!(comparison.nonclaims.contains(
            &ProgSuiteHarmonyComparisonNonClaimV1::SymbolicDifferenceDoesNotEstablishArtisticQuality
        ));
    }
}
