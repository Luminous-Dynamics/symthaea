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
use crate::motif_return::{MotifReturnEvidence, compare_melodic_regions, melodic_notes_in_region};
use crate::obligation::ObligationPressure;
use crate::pitch::PitchClass;
use crate::rhythm::Duration;
use crate::score::{PartId, Score, VoiceRole};
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
    pub source_part: PartId,
    pub target_part: PartId,
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
            start.beats() >= 0.0 && end.beats() > start.beats() && end.beats() <= total + 1e-9
        };
        if !valid_region(source_start, source_end) || !valid_region(target_start, target_end) {
            return None;
        }
        let source_notes = melodic_notes_in_region(score, source_start, source_end);
        let target_notes = melodic_notes_in_region(score, target_start, target_end);
        let source_part = source_notes.first()?.part;
        let target_part = target_notes.first()?.part;
        // VoiceRole is a function, not persistent line identity. Never merge
        // multiple/unassigned parts into one inferred motif sequence.
        let ambiguous_onsets = |notes: &[crate::score::ScoreNote]| {
            notes
                .windows(2)
                .any(|pair| (pair[1].onset.beats() - pair[0].onset.beats()).abs() <= 1e-9)
        };
        if !source_part.is_assigned()
            || !target_part.is_assigned()
            || source_notes.iter().any(|note| note.part != source_part)
            || target_notes.iter().any(|note| note.part != target_part)
            || ambiguous_onsets(&source_notes)
            || ambiguous_onsets(&target_notes)
        {
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
            source_part,
            target_part,
            source_start,
            source_end,
            target_start,
            target_end,
            evidence,
        })
    }

    /// Re-measure this serialized or caller-supplied claim against the current
    /// score. Public fields and Serde make the V1 record a wire format, not
    /// proof that its embedded similarity values were actually measured.
    pub fn validates_against_score(&self, score: &Score) -> bool {
        Self::from_score(
            score,
            self.source_start,
            self.source_end,
            self.target_start,
            self.target_end,
            self.evidence.expected_transformation.clone(),
        )
        .as_ref()
            == Some(self)
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
    /// Temporal structural state, absent only for a truly silent region.
    /// Sustains carried into a region are represented with `event_count = 0`.
    pub temporal_state: Option<MusicalStateFrame>,
    pub active_voice_roles: Vec<VoiceRole>,
    /// True only when every note overlapping this observed region has an assigned PartId.
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
    /// The cognitive profile and temporal state both preserve carry-in notes,
    /// while `onset_count` / `event_count` remain zero when the region contains
    /// no new attacks. `temporal_state` is None only when the interval is silent.
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
        // Context can cross a Serde/API boundary. Do not treat a structured
        // motif claim as measured evidence until it reproduces from this score.
        if context
            .motif_relations
            .iter()
            .any(|relation| !relation.validates_against_score(score))
        {
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
                let Some(note_end) = note.onset.checked_add(note.duration) else {
                    return false;
                };
                let note_end = note_end.beats();
                note_end > onset && note_end > start_beats && onset < end_beats
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
                let Some(note_end) = note.onset.checked_add(note.duration) else {
                    return false;
                };
                let note_end = note_end.beats();
                note.role == *role && note_end > onset && note_end > start && onset < end
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

    fn note(pc: PitchClass, octave: i32, onset: i64, role: VoiceRole, part: PartId) -> ScoreNote {
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
        score.push(note(PitchClass::C, 4, 0, VoiceRole::Melody, PartId(1)));
        score.push(note(PitchClass::E, 4, 1, VoiceRole::Harmony, PartId(2)));
        score.push(note(PitchClass::G, 2, 2, VoiceRole::Bass, PartId(3)));
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

        assert_eq!(relation.source_part, PartId(1));
        assert_eq!(relation.target_part, PartId(1));
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
        assert!(
            MotifRelationObservationV1::from_score(
                &piece,
                Duration::new(0, 1),
                Duration::new(1, 1),
                Duration::new(1, 1),
                Duration::new(2, 1),
                crate::obligation::ReturnTransformation::Literal,
            )
            .is_none()
        );
        assert!(
            MotifRelationObservationV1::from_score(
                &piece,
                Duration::new(1, 1),
                Duration::new(2, 1),
                Duration::new(2, 1),
                Duration::new(3, 1),
                crate::obligation::ReturnTransformation::Literal,
            )
            .is_none()
        );
    }

    #[test]
    fn motif_relation_rejects_simultaneous_attacks_within_a_line() {
        let mut piece = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (pc, onset) in [(0, 0), (2, 0), (4, 1), (0, 2), (2, 3), (4, 4)] {
            piece.push(note(
                PitchClass::new(pc),
                4,
                onset,
                VoiceRole::Melody,
                PartId(1),
            ));
        }
        assert!(
            MotifRelationObservationV1::from_score(
                &piece,
                Duration::new(0, 1),
                Duration::new(2, 1),
                Duration::new(2, 1),
                Duration::new(5, 1),
                crate::obligation::ReturnTransformation::Literal,
            )
            .is_none()
        );
    }

    #[test]
    fn motif_relation_wire_data_must_reproduce_from_the_score() {
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
        .expect("both source regions contain one assigned melody part");
        assert!(relation.validates_against_score(&piece));

        let mut forged = relation;
        forged.evidence.overall_similarity = 0.0;
        assert!(!forged.validates_against_score(&piece));

        let context = MusicalWorldStateContext {
            motif_relations: vec![forged],
            ..Default::default()
        };
        assert!(
            MusicalWorldStateV1::observe_region(
                &piece,
                Duration::new(0, 1),
                Duration::new(3, 1),
                context,
            )
            .is_none()
        );
    }

    #[test]
    fn motif_relation_rejects_unassigned_or_multi_part_melody_regions() {
        let mut unassigned = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (pc, onset) in [(0, 0), (2, 1), (0, 2), (2, 3)] {
            unassigned.push(note(
                PitchClass::new(pc),
                4,
                onset,
                VoiceRole::Melody,
                PartId::UNASSIGNED,
            ));
        }
        assert!(
            MotifRelationObservationV1::from_score(
                &unassigned,
                Duration::new(0, 1),
                Duration::new(2, 1),
                Duration::new(2, 1),
                Duration::new(4, 1),
                crate::obligation::ReturnTransformation::Literal,
            )
            .is_none()
        );

        let mut ambiguous = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (pc, onset, part) in [
            (0, 0, PartId(1)),
            (2, 1, PartId(1)),
            (4, 0, PartId(2)),
            (5, 1, PartId(2)),
            (0, 2, PartId(1)),
            (2, 3, PartId(1)),
        ] {
            ambiguous.push(note(PitchClass::new(pc), 4, onset, VoiceRole::Melody, part));
        }
        assert!(
            MotifRelationObservationV1::from_score(
                &ambiguous,
                Duration::new(0, 2),
                Duration::new(2, 1),
                Duration::new(2, 1),
                Duration::new(4, 1),
                crate::obligation::ReturnTransformation::Literal,
            )
            .is_none()
        );
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
    fn sustain_only_region_keeps_world_state_without_fabricating_attacks() {
        let mut held = note(PitchClass::C, 4, 0, VoiceRole::Melody, PartId(7));
        held.duration = Duration::new(4, 1);
        let mut piece = Score::new(Key::major(PitchClass::C), 120.0, 4);
        piece.push(held);

        let state = MusicalWorldStateV1::observe_region(
            &piece,
            Duration::new(1, 1),
            Duration::new(2, 1),
            MusicalWorldStateContext::default(),
        )
        .expect("the held note overlaps the observation region");

        let temporal = state.temporal_state.expect("a sounding sustain is a state");
        assert_eq!(temporal.event_count, 0);
        assert_eq!(temporal.onset_density, 0.0);
        assert!((temporal.pitch_class_hist[0] - 1.0).abs() < 1e-9);
        assert_eq!(state.cognitive_profile.note_count, 1);
        assert_eq!(state.cognitive_profile.onset_count, 0);
        assert!(state.part_identity_available);
        assert_eq!(state.active_voice_roles, vec![VoiceRole::Melody]);
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
        assert!(
            MusicalWorldStateV1::observe_region(
                &score,
                Duration::new(-1, 1),
                Duration::new(1, 1),
                Default::default()
            )
            .is_none()
        );
        assert!(
            MusicalWorldStateV1::observe_region(
                &score,
                Duration::new(2, 1),
                Duration::new(4, 1),
                Default::default()
            )
            .is_none()
        );
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
            Default::default(),
        )
        .expect("silent in-score region is a valid observation");

        assert!(state.temporal_state.is_none());
        assert!(!state.part_identity_available);
        assert!(state.active_voice_roles.is_empty());
    }

    #[test]
    fn zero_and_negative_duration_notes_do_not_create_world_state_activity() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(PitchClass::C, 4, 2, VoiceRole::Melody, PartId(1)));
        score.notes[0].duration = Duration::zero();
        let mut negative = note(PitchClass::G, 4, 2, VoiceRole::Bass, PartId(2));
        negative.onset = Duration::new(5, 2);
        negative.duration = Duration::new(-1, 2);
        score.push(negative);
        // Extend the score past the observed interval without adding sound there.
        score.push(note(PitchClass::C, 3, 4, VoiceRole::Bass, PartId(3)));

        let state = MusicalWorldStateV1::observe_region(
            &score,
            Duration::new(2, 1),
            Duration::new(3, 1),
            Default::default(),
        )
        .expect("a silent interval inside the score remains observable");

        assert!(state.temporal_state.is_none());
        assert_eq!(state.cognitive_profile.note_count, 0);
        assert_eq!(state.cognitive_profile.onset_count, 0);
        assert!(state.active_voice_roles.is_empty());
        assert!(!state.part_identity_available);
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
