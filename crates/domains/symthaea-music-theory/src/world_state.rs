// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed musical world-state observation for the Melothaea program.
//!
//! This is the MEL-005A boundary. Score-derived facts stay explicit, while
//! context supplied by a planner or evidence ledger remains separately typed.
//! Nothing here grants the cognitive layer authority to bypass theory rules,
//! obligations, or performance constraints.

use crate::cognitive_analysis::{ScoreCognitiveProfile, profile_score_region};
use crate::form::SectionRole;
use crate::grammar::PerformanceDialect;
use crate::harmony::{HarmonicFunction, Key, Tonality};
use crate::motif_return::{MotifReturnEvidence, compare_melodic_regions};
use crate::obligation::ObligationPressure;
use crate::pitch::PitchClass;
use crate::rhythm::Duration;
use crate::score::{Score, VoiceRole};
use crate::state_space::MusicalStateFrame;
use serde::{Deserialize, Serialize};

/// Schema identifier for the typed Melothaea musical world-state boundary.
pub const MUSICAL_WORLD_STATE_V1: &str = "musical-world-state-v1";

/// Explicit context that is not inferred from raw score observations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MusicalWorldStateContext {
    /// Renderer-level performance intent, when a grammar has declared one.
    pub performance_dialect: Option<PerformanceDialect>,
    /// Section role supplied by the formal plan, not guessed from note statistics.
    pub section_role: Option<SectionRole>,
    /// Local formal tonal region from the declared plan; distinct from the
    /// score-level key and never inferred from pitch-class frequency alone.
    pub tonal_region: Option<Key>,
    /// Harmonic function supplied by formal harmonic analysis, not inferred
    /// from pitch-class frequency alone.
    pub harmonic_function: Option<HarmonicFunction>,
    /// Thematic relations bound to exact source/target score regions.
    /// These are symbolic measurements, not listener-recognition judgments.
    pub motif_relations: Vec<MotifRelationObservationV1>,
    /// Current prospective-memory pressure, when an obligation ledger is
    /// active for this composition.
    pub obligation_pressure: Option<ObligationPressure>,
}

/// A motif-return measurement bound to exact half-open source and target regions.
///
/// The envelope preserves where the comparison was measured; the nested
/// evidence records symbolic similarity channels. This still does not claim
/// that a listener recognizes the return.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotifRelationObservationV1 {
    pub source_start: Duration,
    pub source_end: Duration,
    pub target_start: Duration,
    pub target_end: Duration,
    pub evidence: MotifReturnEvidence,
}

impl MotifRelationObservationV1 {
    /// Measure a relation from the score instead of accepting an unlocalized
    /// similarity value. Returns None when either region is invalid, outside
    /// the score, or contains no melody attacks.
    pub fn from_score(
        score: &Score,
        source_start: Duration,
        source_end: Duration,
        target_start: Duration,
        target_end: Duration,
        expected: crate::obligation::ReturnTransformation,
    ) -> Option<Self> {
        let total = score.total_beats.beats();
        let valid_region = |start: Duration, end: Duration| {
            start.beats() >= 0.0
                && end.beats() > start.beats()
                && end.beats() <= total + 1e-9
        };
        if !valid_region(source_start, source_end) || !valid_region(target_start, target_end) {
            return None;
        }
        let evidence = compare_melodic_regions(
            score,
            source_start,
            source_end,
            target_start,
            target_end,
            expected,
        );
        if evidence.source_note_count == 0 || evidence.target_note_count == 0 {
            return None;
        }
        Some(Self {
            source_start,
            source_end,
            target_start,
            target_end,
            evidence,
        })
    }
}

/// Typed, renderer-independent symbolic observation of a musical region.
///
/// Score-derived observations and externally supplied compositional context
/// are separate fields by construction. This makes it possible for the
/// cognitive bridge to preserve provenance rather than flattening everything
/// into one embedding or one quality score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalWorldStateV1 {
    pub schema_version: String,
    pub start: Duration,
    pub end: Duration,
    /// Midpoint location normalized against the complete score length.
    pub normalized_position: f64,
    /// Score-level declared key. This is not a local modulation estimate.
    pub declared_tonic: PitchClass,
    pub declared_tonality: Tonality,
    /// Attack-based temporal state, absent for a silent region. The existing
    /// cognitive profile separately handles notes carried into a region.
    pub temporal_state: Option<MusicalStateFrame>,
    pub active_voice_roles: Vec<VoiceRole>,
    /// True only when every score note carries an assigned PartId.
    pub part_identity_available: bool,
    /// Existing deterministic symbolic profile; deliberately retained as
    /// one measurement layer rather than replaced by the world-state schema.
    pub cognitive_profile: ScoreCognitiveProfile,
    /// Explicit context supplied by the caller, never inferred silently.
    pub context: MusicalWorldStateContext,
}

impl MusicalWorldStateV1 {
    /// Observe a region using exact rational beat boundaries.
    ///
    /// Returns None for negative, empty, reversed, or out-of-score regions.
    /// Carry-in notes are handled by profile_score_region exactly as in the
    /// existing cognitive analysis contract; temporal_state uses onsets in the
    /// region only and is None when the region has no attacks.
    pub fn observe_region(
        score: &Score,
        start: Duration,
        end: Duration,
        context: MusicalWorldStateContext,
    ) -> Option<Self> {
        let start_beats = start.beats();
        let end_beats = end.beats();
        let score_end = score.total_beats.beats();
        if start_beats < 0.0 || end_beats <= start_beats || end_beats > score_end + 1e-9 {
            return None;
        }
        let profile = profile_score_region(score, start, end)?;
        let total = score_end.max(1e-9);
        let midpoint = ((start_beats + end_beats) * 0.5 / total).clamp(0.0, 1.0);

        let active_voice_roles = active_voice_roles(score, start_beats, end_beats);
        let overlapping_notes: Vec<_> = score
            .notes
            .iter()
            .filter(|note| {
                let onset = note.onset.beats();
                let note_end = (note.onset + note.duration).beats();
                note_end > start_beats && onset < end_beats
            })
            .collect();
        let part_identity_available = !overlapping_notes.is_empty()
            && overlapping_notes.iter().all(|note| note.part.is_assigned());
        let temporal_state = MusicalStateFrame::from_region(score, start, end);

        Some(Self {
            schema_version: MUSICAL_WORLD_STATE_V1.into(),
            start,
            end,
            normalized_position: midpoint,
            declared_tonic: score.key.tonic,
            declared_tonality: score.key.tonality,
            temporal_state,
            active_voice_roles,
            part_identity_available,
            cognitive_profile: profile,
            context,
        })
    }
}

fn active_voice_roles(score: &Score, start: f64, end: f64) -> Vec<VoiceRole> {
    let order = [
        VoiceRole::Melody,
        VoiceRole::Harmony,
        VoiceRole::Bass,
        VoiceRole::CounterMelody,
    ];
    order
        .into_iter()
        .filter(|role| {
            score.notes.iter().any(|note| {
                let onset = note.onset.beats();
                let note_end = (note.onset + note.duration).beats();
                note.role == *role && note_end > start && onset < end
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::Pitch;
    use crate::score::{Emphasis, PartId, ScoreNote};

    fn note(
        pc: PitchClass,
        octave: i32,
        onset: i64,
        role: VoiceRole,
        part: PartId,
    ) -> ScoreNote {
        ScoreNote {
            part,
            pitch: Pitch::new(pc, octave),
            onset: Duration::new(onset, 1),
            duration: Duration::quarter(),
            velocity: 0.7,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(
            PitchClass::C,
            4,
            0,
            VoiceRole::Melody,
            PartId(1),
        ));
        score.push(note(
            PitchClass::E,
            4,
            1,
            VoiceRole::Harmony,
            PartId(2),
        ));
        score.push(note(
            PitchClass::G,
            2,
            2,
            VoiceRole::Bass,
            PartId(3),
        ));
        score
    }

    #[test]
    fn motif_relation_is_bound_to_measured_source_and_target_regions() {
        let mut piece = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (pc, onset) in [(0, 0), (2, 1), (4, 2), (0, 3), (2, 4), (4, 5)] {
            piece.push(note(
                PitchClass::new(pc),
                4,
                onset,
                VoiceRole::Melody,
                PartId(1),
            ));
        }

        let relation = MotifRelationObservationV1::from_score(
            &piece,
            Duration::new(0, 1),
            Duration::new(3, 1),
            Duration::new(3, 1),
            Duration::new(6, 1),
            crate::obligation::ReturnTransformation::Literal,
        )
        .expect("both score regions contain melody");

        assert_eq!(relation.source_start, Duration::zero());
        assert_eq!(relation.source_end, Duration::new(3, 1));
        assert_eq!(relation.target_start, Duration::new(3, 1));
        assert_eq!(relation.target_end, Duration::new(6, 1));
        assert_eq!(relation.evidence.source_note_count, 3);
        assert_eq!(relation.evidence.target_note_count, 3);
        assert!(relation.evidence.meets_threshold(0.9));
    }

    #[test]
    fn motif_relation_rejects_unmeasurable_regions() {
        let piece = score();
        assert!(MotifRelationObservationV1::from_score(
            &piece,
            Duration::new(0, 1),
            Duration::new(1, 1),
            Duration::new(1, 1),
            Duration::new(2, 1),
            crate::obligation::ReturnTransformation::Literal,
        ).is_none());
        assert!(MotifRelationObservationV1::from_score(
            &piece,
            Duration::new(1, 1),
            Duration::new(2, 1),
            Duration::new(2, 1),
            Duration::new(3, 1),
            crate::obligation::ReturnTransformation::Literal,
        ).is_none());
    }

    #[test]
    fn observation_preserves_score_facts_and_separates_context() {
        let state = MusicalWorldStateV1::observe_region(
            &score(),
            Duration::zero(),
            Duration::new(3, 1),
            MusicalWorldStateContext {
                performance_dialect: Some(PerformanceDialect::ClassicalRubato),
                ..Default::default()
            },
        )
        .expect("valid region");

        assert_eq!(state.schema_version, MUSICAL_WORLD_STATE_V1);
        assert_eq!(state.declared_tonic, PitchClass::C);
        assert!(state.temporal_state.is_some());
        assert_eq!(state.active_voice_roles.len(), 3);
        assert!(state.part_identity_available);
        assert!(state.cognitive_profile.note_count > 0);
        assert_eq!(
            state.context.performance_dialect,
            Some(PerformanceDialect::ClassicalRubato)
        );
    }

    #[test]
    fn empty_or_reversed_regions_fail_closed() {
        let score = score();
        assert!(
            MusicalWorldStateV1::observe_region(
                &score,
                Duration::new(4, 1),
                Duration::new(4, 1),
                Default::default()
            )
            .is_none()
        );
        assert!(
            MusicalWorldStateV1::observe_region(
                &score,
                Duration::new(3, 1),
                Duration::new(2, 1),
                Default::default()
            )
            .is_none()
        );
    }

    #[test]
    fn score_level_key_is_not_reported_for_out_of_bounds_regions() {
        let score = score();
        assert!(MusicalWorldStateV1::observe_region(
            &score,
            Duration::new(-1, 1),
            Duration::new(1, 1),
            Default::default()
        ).is_none());
        assert!(MusicalWorldStateV1::observe_region(
            &score,
            Duration::new(2, 1),
            Duration::new(4, 1),
            Default::default()
        ).is_none());
    }

    #[test]
    fn silent_region_keeps_world_state_but_does_not_claim_part_identity() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(PitchClass::C, 4, 0, VoiceRole::Melody, PartId(1)));
        score.push(note(PitchClass::G, 4, 4, VoiceRole::Melody, PartId(1)));

        let state = MusicalWorldStateV1::observe_region(
            &score,
            Duration::new(2, 1),
            Duration::new(3, 1),
            Default::default()
        ).expect("silent in-score region is a valid observation");

        assert!(state.temporal_state.is_none());
        assert!(!state.part_identity_available);
        assert!(state.active_voice_roles.is_empty());
    }

    #[test]
    fn partial_region_is_not_labeled_as_having_part_identity_when_unassigned_notes_overlap_it() {
        let mut score = score();
        score.push(note(
            PitchClass::D,
            4,
            1,
            VoiceRole::Melody,
            PartId::UNASSIGNED,
        ));

        let state = MusicalWorldStateV1::observe_region(
            &score,
            Duration::new(1, 1),
            Duration::new(2, 1),
            Default::default(),
        )
        .expect("valid region");

        assert!(!state.part_identity_available);
    }
}
