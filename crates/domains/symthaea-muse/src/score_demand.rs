// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Renderer-independent score-demand measurements for future orchestration.
//!
//! This module records what the score asks of each part/role. It deliberately
//! does not infer instrument range, tessitura, articulation, timbral suitability,
//! sample availability or renderer support. Those require separate qualified
//! instrument/asset evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use symthaea_music_theory::score::{Score, ScoreNote, VoiceRole};

pub const SCORE_DEMAND_PROFILE_SCHEMA: &str = "muse-score-demand-profile-v1";
pub const SCORE_DEMAND_INPUT_DIGEST_SCHEMA: &str = "muse-score-serde-json-sha256-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DemandIdentityBasis {
    StablePartId,
    RoleAggregateSummary,
    RoleAggregateBecausePartIdentityIsIncomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceDemand {
    /// Absent when the input score does not assign a real PartId to every note.
    pub part_id: Option<u16>,
    /// A stable part can perform more than one function over the score. Keep
    /// those roles attached to the part instead of splitting one instrument
    /// line into multiple independently assigned groups.
    pub roles: Vec<VoiceRole>,
    pub identity_basis: DemandIdentityBasis,
    pub note_count: usize,
    pub min_midi: u8,
    pub max_midi: u8,
    pub mean_midi: f64,
    pub pitch_span_semitones: u8,
    pub distinct_onset_count: usize,
    pub mean_note_duration_beats: f64,
    pub peak_simultaneous_notes: usize,
    pub first_onset_beat: f64,
    pub last_note_end_beat: f64,
    pub active_span_beats: f64,
    pub notes_per_score_beat: f64,
    pub notes_per_bar: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreDemandProfile {
    pub schema: String,
    pub input_digest_schema: String,
    /// SHA-256 over a domain separator and the complete serialized symbolic
    /// Score (including notes, key, tempo, meter and declared total span).
    pub score_sha256: String,
    pub meter_beats: u8,
    pub tempo_bpm: f32,
    pub declared_score_span_beats: f64,
    /// Maximum of the declared span and the latest analyzed note end.
    pub analyzed_span_beats: f64,
    pub has_complete_part_identity: bool,
    pub assigned_part_note_fraction: f64,
    /// With complete PartIds this includes both independent part rows and
    /// role rollups. RoleAggregateSummary rows are summaries, not extra
    /// parts, and must not be counted as additional performers. Without
    /// complete identity, only role aggregates are emitted.
    pub voice_demands: Vec<VoiceDemand>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreDemandError {
    EmptyScore,
    NonFiniteTempo,
    NonPositiveTempo,
    NonFiniteDeclaredSpan,
    NegativeDeclaredSpan,
    NonFiniteOnset { note_index: usize },
    NegativeOnset { note_index: usize },
    NonFiniteDuration { note_index: usize },
    NonPositiveDuration { note_index: usize },
    NonFiniteNoteEnd { note_index: usize },
    NonFiniteVelocity { note_index: usize },
    VelocityOutOfRange { note_index: usize },
    NonFiniteSectionIntensity { note_index: usize },
    NegativeSectionIntensity { note_index: usize },
    ScoreSerialization(String),
}

#[derive(Debug, Clone, Copy)]
struct TimingEvent {
    beat: f64,
    delta: i8,
}

pub fn analyze_score_demand(
    score: &Score,
) -> Result<ScoreDemandProfile, ScoreDemandError> {
    if score.notes.is_empty() {
        return Err(ScoreDemandError::EmptyScore);
    }

    let has_complete_part_identity = score.has_part_identity();
    let assigned_part_notes = score
        .notes
        .iter()
        .filter(|note| note.part.is_assigned())
        .count();
    let assigned_part_note_fraction =
        assigned_part_notes as f64 / score.notes.len() as f64;

    // If any note lacks an assigned PartId, do not infer missing identities
    // from register, temporal overlap or VoiceRole. Aggregate the whole score
    // by role and label that loss of identity explicitly.
    let mut part_groups = BTreeMap::<u16, Vec<&ScoreNote>>::new();
    let mut role_groups = BTreeMap::<u8, Vec<&ScoreNote>>::new();
    if !score.tempo_bpm.is_finite() {
        return Err(ScoreDemandError::NonFiniteTempo);
    }
    if score.tempo_bpm <= 0.0 {
        return Err(ScoreDemandError::NonPositiveTempo);
    }
    let declared_span = score.total_beats.beats();
    if !declared_span.is_finite() {
        return Err(ScoreDemandError::NonFiniteDeclaredSpan);
    }
    if declared_span < 0.0 {
        return Err(ScoreDemandError::NegativeDeclaredSpan);
    }
    let mut analyzed_span = declared_span;

    for (index, note) in score.notes.iter().enumerate() {
        let onset = note.onset.beats();
        let duration = note.duration.beats();
        if !onset.is_finite() {
            return Err(ScoreDemandError::NonFiniteOnset { note_index: index });
        }
        if onset < 0.0 {
            return Err(ScoreDemandError::NegativeOnset { note_index: index });
        }
        if !duration.is_finite() {
            return Err(ScoreDemandError::NonFiniteDuration { note_index: index });
        }
        if duration <= 0.0 {
            return Err(ScoreDemandError::NonPositiveDuration { note_index: index });
        }
        let end = onset + duration;
        if !end.is_finite() {
            return Err(ScoreDemandError::NonFiniteNoteEnd { note_index: index });
        }
        if !note.velocity.is_finite() {
            return Err(ScoreDemandError::NonFiniteVelocity { note_index: index });
        }
        if !(0.0..=1.0).contains(&note.velocity) {
            return Err(ScoreDemandError::VelocityOutOfRange { note_index: index });
        }
        if !note.section_intensity.is_finite() {
            return Err(ScoreDemandError::NonFiniteSectionIntensity { note_index: index });
        }
        if note.section_intensity < 0.0 {
            return Err(ScoreDemandError::NegativeSectionIntensity { note_index: index });
        }
        analyzed_span = analyzed_span.max(end);

        role_groups
            .entry(role_rank(note.role))
            .or_default()
            .push(note);
        if has_complete_part_identity {
            // Group by stable PartId only. One continuing line remains one
            // instrument demand even if its functional role changes.
            part_groups.entry(note.part.0).or_default().push(note);
        }
    }

    let score_sha256 = score_input_sha256(score)?;
    let score_span = analyzed_span.max(f64::EPSILON);
    let mut voice_demands = Vec::with_capacity(
        part_groups.len() + role_groups.len(),
    );
    if has_complete_part_identity {
        // Part-level demands preserve identity across role changes.
        voice_demands.extend(part_groups.into_iter().map(|(part_id, notes)| {
            build_voice_demand(
                Some(part_id),
                notes,
                DemandIdentityBasis::StablePartId,
                score_span,
                score.meter,
            )
        }));
        // Role summaries retain workload information useful to lead/bass/
        // harmony balancing, but are not additional independent parts.
        voice_demands.extend(role_groups.into_iter().map(|(_, notes)| {
            build_voice_demand(
                None,
                notes,
                DemandIdentityBasis::RoleAggregateSummary,
                score_span,
                score.meter,
            )
        }));
    } else {
        // No complete part identity: role aggregates are the strongest
        // claim supported by the source score. Do not reconstruct parts.
        voice_demands.extend(role_groups.into_iter().map(|(_, notes)| {
            build_voice_demand(
                None,
                notes,
                DemandIdentityBasis::RoleAggregateBecausePartIdentityIsIncomplete,
                score_span,
                score.meter,
            )
        }));
    }

    Ok(ScoreDemandProfile {
        schema: SCORE_DEMAND_PROFILE_SCHEMA.into(),
        input_digest_schema: SCORE_DEMAND_INPUT_DIGEST_SCHEMA.into(),
        score_sha256,
        meter_beats: score.meter,
        tempo_bpm: score.tempo_bpm,
        declared_score_span_beats: declared_span,
        analyzed_span_beats: analyzed_span,
        has_complete_part_identity,
        assigned_part_note_fraction,
        voice_demands,
    })
}

fn score_input_sha256(score: &Score) -> Result<String, ScoreDemandError> {
    // serde_json serializes this struct in declaration order; the domain tag
    // prevents the same bytes from being confused with another artifact type.
    let bytes = serde_json::to_vec(score)
        .map_err(|error| ScoreDemandError::ScoreSerialization(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:muse-score-demand-input:v1\n");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn role_rank(role: VoiceRole) -> u8 {
    match role {
        VoiceRole::Melody => 0,
        VoiceRole::CounterMelody => 1,
        VoiceRole::Harmony => 2,
        VoiceRole::Bass => 3,
    }
}

fn build_voice_demand(
    part_id: Option<u16>,
    notes: Vec<&ScoreNote>,
    identity_basis: DemandIdentityBasis,
    score_span: f64,
    meter: u8,
) -> VoiceDemand {
    let mut roles = Vec::new();
    for note in &notes {
        if !roles.contains(&note.role) {
            roles.push(note.role);
        }
    }
    roles.sort_by_key(|role| role_rank(*role));
    let mut min_midi = u8::MAX;
    let mut max_midi = 0u8;
    let mut midi_sum = 0u64;
    let mut duration_sum = 0.0;
    let mut first_onset = f64::INFINITY;
    let mut last_end = f64::NEG_INFINITY;
    let mut onsets = Vec::with_capacity(notes.len());
    let mut timing_events = Vec::with_capacity(notes.len() * 2);

    for note in &notes {
        let midi = note.pitch.midi();
        let onset = note.onset.beats();
        let end = onset + note.duration.beats();
        min_midi = min_midi.min(midi);
        max_midi = max_midi.max(midi);
        midi_sum += u64::from(midi);
        duration_sum += note.duration.beats();
        first_onset = first_onset.min(onset);
        last_end = last_end.max(end);
        onsets.push(onset);
        timing_events.push(TimingEvent { beat: onset, delta: 1 });
        timing_events.push(TimingEvent { beat: end, delta: -1 });
    }

    onsets.sort_by(f64::total_cmp);
    let mut distinct_onset_count = 0usize;
    let mut previous_onset = None::<f64>;
    for onset in &onsets {
        if previous_onset.is_none_or(|previous| (*onset - previous).abs() > 1.0e-9) {
            distinct_onset_count += 1;
            previous_onset = Some(*onset);
        }
    }

    // At a shared boundary, note-off must be processed before note-on:
    // adjacent notes in one monophonic part do not count as overlapping.
    timing_events.sort_by(|left, right| {
        left.beat
            .total_cmp(&right.beat)
            .then_with(|| left.delta.cmp(&right.delta))
    });
    let mut active = 0usize;
    let mut peak_simultaneous_notes = 0usize;
    for event in timing_events {
        if event.delta < 0 {
            active = active.saturating_sub(1);
        } else {
            active += 1;
            peak_simultaneous_notes = peak_simultaneous_notes.max(active);
        }
    }

    let note_count = notes.len();
    let active_span_beats = (last_end - first_onset).max(0.0);
    let notes_per_score_beat = note_count as f64 / score_span;
    let notes_per_bar = (meter > 0).then(|| {
        let bar_count = score_span / f64::from(meter);
        if bar_count > f64::EPSILON {
            note_count as f64 / bar_count
        } else {
            0.0
        }
    });

    VoiceDemand {
        part_id,
        roles,
        identity_basis,
        note_count,
        min_midi,
        max_midi,
        mean_midi: midi_sum as f64 / note_count as f64,
        pitch_span_semitones: max_midi.saturating_sub(min_midi),
        distinct_onset_count,
        mean_note_duration_beats: duration_sum / note_count as f64,
        peak_simultaneous_notes,
        first_onset_beat: first_onset,
        last_note_end_beat: last_end,
        active_span_beats,
        notes_per_score_beat,
        notes_per_bar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::rhythm::Duration;
    use symthaea_music_theory::score::{Emphasis, PartId, ScoreNote};

    fn score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (part, role, midi, onset, duration) in [
            (PartId(1), VoiceRole::Melody, 60, 0, 2),
            (PartId(1), VoiceRole::Melody, 64, 1, 1),
            (PartId(1), VoiceRole::Melody, 65, 2, 1),
            (PartId(2), VoiceRole::Bass, 36, 0, 1),
            (PartId(2), VoiceRole::Bass, 43, 1, 1),
            (PartId(2), VoiceRole::Bass, 40, 2, 1),
        ] {
            score.push(ScoreNote {
                part,
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(duration, 1),
                velocity: 0.7,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    #[test]
    fn captures_range_density_duration_and_polyphony_by_stable_part() {
        let profile = analyze_score_demand(&score()).unwrap();
        assert!(profile.has_complete_part_identity);
        assert_eq!(profile.assigned_part_note_fraction, 1.0);
        // Two part demands plus two role summaries.
        assert_eq!(profile.voice_demands.len(), 4);

        let melody = profile
            .voice_demands
            .iter()
            .find(|voice| {
                voice.part_id == Some(1)
                    && voice.identity_basis == DemandIdentityBasis::StablePartId
            })
            .unwrap();
        assert_eq!(melody.part_id, Some(1));
        assert_eq!(melody.note_count, 3);
        assert_eq!(melody.min_midi, 60);
        assert_eq!(melody.max_midi, 65);
        assert_eq!(melody.distinct_onset_count, 3);
        assert_eq!(melody.peak_simultaneous_notes, 2);
        assert!((melody.mean_note_duration_beats - 4.0 / 3.0).abs() < 1.0e-12);
        assert!((melody.notes_per_score_beat - 1.0).abs() < 1.0e-12);

        let bass = profile
            .voice_demands
            .iter()
            .find(|voice| {
                voice.part_id == Some(2)
                    && voice.identity_basis == DemandIdentityBasis::StablePartId
            })
            .unwrap();
        assert_eq!(bass.part_id, Some(2));
        assert_eq!(bass.peak_simultaneous_notes, 1);
    }

    #[test]
    fn stable_part_remains_one_demand_when_its_role_changes() {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (role, midi, onset) in [
            (VoiceRole::Melody, 67, 0),
            (VoiceRole::Harmony, 64, 1),
        ] {
            score.push(ScoreNote {
                part: PartId(9),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity: 0.6,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        let profile = analyze_score_demand(&score).unwrap();
        assert_eq!(profile.voice_demands.len(), 3);
        let part = profile.voice_demands.iter()
            .find(|demand| demand.part_id == Some(9))
            .unwrap();
        assert_eq!(
            part.roles,
            vec![VoiceRole::Melody, VoiceRole::Harmony]
        );
        assert_eq!(
            profile.voice_demands.iter()
                .filter(|demand| demand.identity_basis == DemandIdentityBasis::RoleAggregateSummary)
                .count(),
            2
        );
    }

    #[test]
    fn note_off_precedes_note_on_at_a_shared_boundary() {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (midi, onset) in [(60, 0), (62, 1)] {
            score.push(ScoreNote {
                part: PartId(1),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity: 0.6,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        let profile = analyze_score_demand(&score).unwrap();
        assert_eq!(profile.voice_demands[0].peak_simultaneous_notes, 1);
    }

    #[test]
    fn missing_part_identity_uses_role_aggregate_without_guessing() {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (part, midi, onset) in [
            (PartId(5), 60, 0),
            (PartId::UNASSIGNED, 64, 1),
        ] {
            score.push(ScoreNote {
                part,
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity: 0.6,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        let profile = analyze_score_demand(&score).unwrap();
        assert!(!profile.has_complete_part_identity);
        assert_eq!(profile.assigned_part_note_fraction, 0.5);
        assert_eq!(profile.voice_demands.len(), 1);
        assert_eq!(profile.voice_demands[0].part_id, None);
        assert_eq!(
            profile.voice_demands[0].identity_basis,
            DemandIdentityBasis::RoleAggregateBecausePartIdentityIsIncomplete
        );
        assert_eq!(profile.voice_demands[0].note_count, 2);
        assert_eq!(profile.voice_demands[0].roles, vec![VoiceRole::Melody]);
    }

    #[test]
    fn input_digest_is_deterministic_and_changes_with_score_content() {
        let original = score();
        let first = analyze_score_demand(&original).unwrap();
        let second = analyze_score_demand(&score()).unwrap();
        assert_eq!(first.score_sha256, second.score_sha256);
        assert_eq!(first.input_digest_schema, SCORE_DEMAND_INPUT_DIGEST_SCHEMA);

        let mut changed = original;
        changed.notes[0].velocity = 0.8;
        let changed_profile = analyze_score_demand(&changed).unwrap();
        assert_ne!(first.score_sha256, changed_profile.score_sha256);
    }

    #[test]
    fn rejects_nonfinite_and_out_of_range_dynamics_before_digesting() {
        let mut invalid = score();
        invalid.notes[0].velocity = f32::NAN;
        assert_eq!(
            analyze_score_demand(&invalid),
            Err(ScoreDemandError::NonFiniteVelocity { note_index: 0 })
        );

        let mut invalid = score();
        invalid.notes[0].velocity = 1.1;
        assert_eq!(
            analyze_score_demand(&invalid),
            Err(ScoreDemandError::VelocityOutOfRange { note_index: 0 })
        );

        let mut invalid = score();
        invalid.notes[0].section_intensity = f32::INFINITY;
        assert_eq!(
            analyze_score_demand(&invalid),
            Err(ScoreDemandError::NonFiniteSectionIntensity { note_index: 0 })
        );
    }

    #[test]
    fn identical_scores_produce_identical_profiles() {
        let first = analyze_score_demand(&score()).unwrap();
        let second = analyze_score_demand(&score()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_empty_and_nonpositive_notes_instead_of_dropping_them() {
        let empty = Score::new(Key::major(PitchClass::C), 100.0, 4);
        assert_eq!(analyze_score_demand(&empty), Err(ScoreDemandError::EmptyScore));

        let mut invalid = Score::new(Key::major(PitchClass::C), 100.0, 4);
        invalid.push(ScoreNote {
            part: PartId(1),
            pitch: Pitch::from_midi(60),
            onset: Duration::zero(),
            duration: Duration::zero(),
            velocity: 0.6,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        assert_eq!(
            analyze_score_demand(&invalid),
            Err(ScoreDemandError::NonPositiveDuration { note_index: 0 })
        );
    }
}
