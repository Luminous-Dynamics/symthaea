// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed musical world-state observation for the Melothaea program.
//!
//! This is the MEL-005A boundary. Score-derived facts stay explicit, while
//! context supplied by a planner or evidence ledger remains separately typed.
//! Nothing here grants the cognitive layer authority to bypass theory rules,
//! obligations, or performance constraints.

use crate::cognitive_analysis::{ScoreCognitiveProfile, profile_score_region};
use crate::grammar::PerformanceDialect;
use crate::harmony::Tonality;
use crate::motif_return::MotifReturnEvidence;
use crate::obligation::ObligationPressure;
use crate::pitch::PitchClass;
use crate::rhythm::Duration;
use crate::score::{Score, VoiceRole};
use serde::{Deserialize, Serialize};

/// Schema identifier for the typed Melothaea musical world-state boundary.
pub const MUSICAL_WORLD_STATE_V1: &str = "musical-world-state-v1";

/// Explicit context that is not inferred from raw score observations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MusicalWorldStateContext {
    /// Renderer-level performance intent, when a grammar has declared one.
    pub performance_dialect: Option<PerformanceDialect>,
    /// Independently computed thematic relations. These are observations,
    /// not claims that a listener recognizes the relation.
    pub motif_relations: Vec<MotifReturnEvidence>,
    /// Current prospective-memory pressure, when an obligation ledger is
    /// active for this composition.
    pub obligation_pressure: Option<ObligationPressure>,
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
    pub tonic: PitchClass,
    pub tonality: Tonality,
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
    /// Returns None for an empty or reversed region. Carry-in notes are
    /// handled by profile_score_region exactly as in the existing cognitive
    /// analysis contract.
    pub fn observe_region(
        score: &Score,
        start: Duration,
        end: Duration,
        context: MusicalWorldStateContext,
    ) -> Option<Self> {
        let profile = profile_score_region(score, start, end)?;

        let start_beats = start.beats();
        let end_beats = end.beats();
        let total = score.total_beats.beats().max(1e-9);
        let midpoint = ((start_beats + end_beats) * 0.5 / total).clamp(0.0, 1.0);

        let active_voice_roles = active_voice_roles(score, start_beats, end_beats);
        let part_identity_available = score
            .notes
            .iter()
            .filter(|note| {
                let onset = note.onset.beats();
                let note_end = (note.onset + note.duration).beats();
                note_end > start_beats && onset < end_beats
            })
            .all(|note| note.part.is_assigned());

        Some(Self {
            schema_version: MUSICAL_WORLD_STATE_V1.into(),
            start,
            end,
            normalized_position: midpoint,
            tonic: score.key.tonic,
            tonality: score.key.tonality,
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
        assert_eq!(state.tonic, PitchClass::C);
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
