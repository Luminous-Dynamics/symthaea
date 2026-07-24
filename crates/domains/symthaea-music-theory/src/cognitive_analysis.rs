// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interpretable score-side observations for the Symthaea–Muse cognitive loop.
//!
//! These measurements are deterministic *proxies*, not claims about musical
//! quality or listener experience. They give the cognitive bridge four
//! auditable channels that correspond to its first prediction vocabulary:
//! tension, density, familiarity, and tonal displacement.
//!
//! The functions operate only on the symbolic [`Score`]. Renderer-side and
//! listener-side evidence should be recorded separately rather than folded
//! into these values.

use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Deterministic symbolic observations used by the first cognitive loop.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoreCognitiveProfile {
    /// Composite symbolic tension proxy in [0, 1].
    pub tension: f32,
    /// Note-onset density proxy in [0, 1].
    pub density: f32,
    /// Recurrence of melodic interval trigrams in [0, 1].
    pub familiarity: f32,
    /// Mean circular pitch-class distance from the declared tonic in [0, 1].
    pub tonal_displacement: f32,
    /// Number of notes sounding at any point in the observed region.
    pub note_count: usize,
    /// Number of note attacks whose onset lies inside the observed region.
    #[serde(default)]
    pub onset_count: usize,
    /// Raw note onsets per beat retained for auditability.
    pub notes_per_beat: f32,
    /// Number of voice roles active in the observed region.
    pub active_voice_count: usize,
    /// Observed span in beats.
    pub span_beats: f64,
}

impl Default for ScoreCognitiveProfile {
    fn default() -> Self {
        Self {
            tension: 0.0,
            density: 0.0,
            familiarity: 0.0,
            tonal_displacement: 0.0,
            note_count: 0,
            onset_count: 0,
            notes_per_beat: 0.0,
            active_voice_count: 0,
            span_beats: 0.0,
        }
    }
}

/// Directional change from one symbolic score observation to another.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoreCognitiveDelta {
    pub tension_delta: f32,
    pub density_delta: f32,
    pub familiarity_delta: f32,
    pub tonal_displacement_delta: f32,
}

impl ScoreCognitiveProfile {
    /// Compute the directional change from `self` to `other`.
    pub fn delta_to(self, other: Self) -> ScoreCognitiveDelta {
        ScoreCognitiveDelta {
            tension_delta: other.tension - self.tension,
            density_delta: other.density - self.density,
            familiarity_delta: other.familiarity - self.familiarity,
            tonal_displacement_delta: other.tonal_displacement - self.tonal_displacement,
        }
    }
}

/// Observe the complete symbolic score.
pub fn profile_score(score: &Score) -> ScoreCognitiveProfile {
    profile_notes(
        score,
        &score.notes,
        score.notes.len(),
        0.0,
        score.total_beats.beats(),
    )
}

/// Observe all notes sounding within `[start, end)`.
///
/// Carry-in notes that began before `start` are clipped to the region so they
/// remain available to vertical-tension and tonal-duration measurements. They
/// are not counted as new attacks: `onset_count` and `notes_per_beat` include
/// only notes whose onset lies inside the selected region. Returns `None` when
/// the region is empty or reversed.
pub fn profile_score_region(
    score: &Score,
    start: Duration,
    end: Duration,
) -> Option<ScoreCognitiveProfile> {
    let start_beats = start.beats();
    let end_beats = end.beats();
    if end_beats <= start_beats {
        return None;
    }

    let mut notes = Vec::new();
    let mut onset_count = 0usize;
    for note in &score.notes {
        let onset = note.onset.beats();
        let note_end = (note.onset + note.duration).beats();
        if note_end <= start_beats || onset >= end_beats {
            continue;
        }
        if onset >= start_beats {
            onset_count += 1;
        }

        let clipped_start = if onset < start_beats {
            start
        } else {
            note.onset
        };
        let original_end = note.onset + note.duration;
        let clipped_end = if note_end > end_beats {
            end
        } else {
            original_end
        };
        let mut clipped = *note;
        clipped.onset = clipped_start;
        clipped.duration = clipped_end.saturating_sub(clipped_start);
        notes.push(clipped);
    }
    Some(profile_notes(
        score,
        &notes,
        onset_count,
        start_beats,
        end_beats,
    ))
}

fn profile_notes(
    score: &Score,
    notes: &[ScoreNote],
    onset_count: usize,
    start_beats: f64,
    end_beats: f64,
) -> ScoreCognitiveProfile {
    let span_beats = (end_beats - start_beats).max(0.0);
    if notes.is_empty() || span_beats <= f64::EPSILON {
        return ScoreCognitiveProfile {
            span_beats,
            ..ScoreCognitiveProfile::default()
        };
    }

    let notes_per_beat = onset_count as f32 / span_beats as f32;
    // Four simultaneous note onsets per beat is already a very dense symbolic
    // texture for the current Muse score vocabulary. Keep the raw value too.
    let density = (notes_per_beat / 4.0).clamp(0.0, 1.0);
    let tonal_displacement = mean_tonal_displacement(score, notes);
    let vertical_tension = mean_vertical_dissonance(notes);
    let melodic_tension = mean_melodic_leap(notes);
    let structural_tension = mean_structural_emphasis(notes);
    // Tonal displacement remains its own prediction channel. Keeping it out
    // of the tension composite prevents one pitch-distance proxy from being
    // counted twice in the closed-loop error.
    let tension = (0.55 * vertical_tension + 0.35 * melodic_tension + 0.10 * structural_tension)
        .clamp(0.0, 1.0);

    ScoreCognitiveProfile {
        tension,
        density,
        familiarity: melodic_interval_familiarity(notes),
        tonal_displacement,
        note_count: notes.len(),
        onset_count,
        notes_per_beat,
        active_voice_count: active_voice_count(notes),
        span_beats,
    }
}

fn active_voice_count(notes: &[ScoreNote]) -> usize {
    let mut active = [false; 4];
    for note in notes {
        let index = match note.role {
            VoiceRole::Melody => 0,
            VoiceRole::Harmony => 1,
            VoiceRole::Bass => 2,
            VoiceRole::CounterMelody => 3,
        };
        active[index] = true;
    }
    active.into_iter().filter(|value| *value).count()
}

fn mean_tonal_displacement(score: &Score, notes: &[ScoreNote]) -> f32 {
    let tonic = score.key.tonic;
    let mut weighted_sum = 0.0_f64;
    let mut total_weight = 0.0_f64;
    for note in notes {
        let directed = tonic.interval_to(note.pitch.pitch_class()) as f64;
        let circular = directed.min(12.0 - directed) / 6.0;
        let weight = note.duration.beats().max(0.0);
        weighted_sum += circular * weight;
        total_weight += weight;
    }
    if total_weight <= f64::EPSILON {
        0.0
    } else {
        (weighted_sum / total_weight).clamp(0.0, 1.0) as f32
    }
}

fn mean_vertical_dissonance(notes: &[ScoreNote]) -> f32 {
    let mut event_times: Vec<Duration> = notes.iter().map(|note| note.onset).collect();
    event_times.sort_by(|left, right| left.beats().total_cmp(&right.beats()));
    event_times.dedup();

    let mut sum = 0.0_f32;
    let mut pairs = 0usize;
    for event in event_times {
        let active: Vec<&ScoreNote> = notes
            .iter()
            .filter(|note| {
                note.onset.beats() <= event.beats()
                    && (note.onset + note.duration).beats() > event.beats()
            })
            .collect();
        for left in 0..active.len() {
            for right in (left + 1)..active.len() {
                let semitones = active[left]
                    .pitch
                    .semitones_to(active[right].pitch)
                    .unsigned_abs() as u8
                    % 12;
                sum += interval_tension(semitones);
                pairs += 1;
            }
        }
    }

    if pairs == 0 { 0.0 } else { sum / pairs as f32 }
}

fn interval_tension(semitones: u8) -> f32 {
    match semitones {
        0 => 0.0,
        1 | 11 => 0.90,
        2 | 10 => 0.72,
        3 | 4 | 8 | 9 => 0.18,
        5 | 7 => 0.10,
        6 => 1.0,
        _ => 0.0,
    }
}

fn mean_melodic_leap(notes: &[ScoreNote]) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for role in [VoiceRole::Melody, VoiceRole::CounterMelody, VoiceRole::Bass] {
        let mut voice: Vec<ScoreNote> = notes
            .iter()
            .copied()
            .filter(|note| note.role == role)
            .collect();
        voice.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        for pair in voice.windows(2) {
            let semitones = pair[0]
                .pitch
                .semitones_to(pair[1].pitch)
                .unsigned_abs()
                .min(12) as f32;
            sum += semitones / 12.0;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn mean_structural_emphasis(notes: &[ScoreNote]) -> f32 {
    let sum: f32 = notes
        .iter()
        .map(|note| match note.emphasis {
            Emphasis::Normal => 0.0,
            Emphasis::PhraseStart => 0.15,
            Emphasis::Cadential => 0.55,
            Emphasis::Climax => 1.0,
        })
        .sum();
    sum / notes.len() as f32
}

fn melodic_interval_familiarity(notes: &[ScoreNote]) -> f32 {
    let mut counts: BTreeMap<[i16; 3], usize> = BTreeMap::new();
    let mut windows = 0usize;
    for role in [VoiceRole::Melody, VoiceRole::CounterMelody] {
        let mut voice: Vec<ScoreNote> = notes
            .iter()
            .copied()
            .filter(|note| note.role == role)
            .collect();
        voice.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let intervals: Vec<i16> = voice
            .windows(2)
            .map(|pair| pair[0].pitch.semitones_to(pair[1].pitch) as i16)
            .collect();
        for window in intervals.windows(3) {
            *counts.entry([window[0], window[1], window[2]]).or_default() += 1;
            windows += 1;
        }
    }
    if windows < 2 {
        return 0.0;
    }
    let matching_pairs: usize = counts
        .values()
        .map(|count| count.saturating_mul(count.saturating_sub(1)))
        .sum();
    let possible_pairs = windows * (windows - 1);
    matching_pairs as f32 / possible_pairs as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};

    fn note(pitch: Pitch, onset: Duration, duration: Duration, role: VoiceRole) -> ScoreNote {
        ScoreNote {
            pitch,
            onset,
            duration,
            velocity: 0.7,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    #[test]
    fn density_rises_with_more_onsets_in_the_same_span() {
        let mut sparse = Score::new(Key::major(PitchClass::C), 120.0, 4);
        sparse.push(note(
            Pitch::new(PitchClass::C, 4),
            Duration::zero(),
            Duration::whole(),
            VoiceRole::Melody,
        ));

        let mut dense = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for beat in 0..4 {
            dense.push(note(
                Pitch::new(PitchClass::C, 4),
                Duration::new(beat, 1),
                Duration::quarter(),
                VoiceRole::Melody,
            ));
        }

        assert!(profile_score(&dense).density > profile_score(&sparse).density);
        assert!(profile_score(&dense).notes_per_beat > profile_score(&sparse).notes_per_beat);
    }

    #[test]
    fn simultaneous_tritone_has_more_tension_than_a_perfect_fifth() {
        let mut fifth = Score::new(Key::major(PitchClass::C), 120.0, 4);
        fifth.push(note(
            Pitch::new(PitchClass::C, 4),
            Duration::zero(),
            Duration::whole(),
            VoiceRole::Bass,
        ));
        fifth.push(note(
            Pitch::new(PitchClass::G, 4),
            Duration::zero(),
            Duration::whole(),
            VoiceRole::Harmony,
        ));

        let mut tritone = fifth.clone();
        tritone.notes[1].pitch = Pitch::new(PitchClass::FSHARP, 4);

        assert!(profile_score(&tritone).tension > profile_score(&fifth).tension);
    }

    #[test]
    fn sustained_bass_is_measured_against_later_harmony_onset() {
        let mut fifth = Score::new(Key::major(PitchClass::C), 120.0, 4);
        fifth.push(note(
            Pitch::new(PitchClass::C, 3),
            Duration::zero(),
            Duration::whole(),
            VoiceRole::Bass,
        ));
        fifth.push(note(
            Pitch::new(PitchClass::G, 4),
            Duration::quarter(),
            Duration::quarter(),
            VoiceRole::Harmony,
        ));

        let mut tritone = fifth.clone();
        tritone.notes[1].pitch = Pitch::new(PitchClass::FSHARP, 4);

        assert!(profile_score(&tritone).tension > profile_score(&fifth).tension);
    }

    #[test]
    fn repeated_interval_grammar_has_greater_familiarity() {
        let mut repeated = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (index, midi) in [60, 62, 64, 66, 68, 70, 72].into_iter().enumerate() {
            repeated.push(note(
                Pitch::from_midi(midi),
                Duration::new(index as i64, 1),
                Duration::quarter(),
                VoiceRole::Melody,
            ));
        }

        let mut varied = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (index, midi) in [60, 61, 63, 66, 70, 75, 81].into_iter().enumerate() {
            varied.push(note(
                Pitch::from_midi(midi),
                Duration::new(index as i64, 1),
                Duration::quarter(),
                VoiceRole::Melody,
            ));
        }

        assert!(profile_score(&repeated).familiarity > profile_score(&varied).familiarity);
    }

    #[test]
    fn tritone_is_farther_from_tonic_than_tonic() {
        let mut tonic = Score::new(Key::major(PitchClass::C), 120.0, 4);
        tonic.push(note(
            Pitch::new(PitchClass::C, 4),
            Duration::zero(),
            Duration::whole(),
            VoiceRole::Melody,
        ));
        let mut displaced = tonic.clone();
        displaced.notes[0].pitch = Pitch::new(PitchClass::FSHARP, 4);

        assert!(
            profile_score(&displaced).tonal_displacement > profile_score(&tonic).tonal_displacement
        );
    }

    #[test]
    fn region_profiles_include_carry_in_notes_without_counting_new_attacks() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(
            Pitch::new(PitchClass::C, 4),
            Duration::zero(),
            Duration::new(6, 1),
            VoiceRole::Bass,
        ));
        score.push(note(
            Pitch::new(PitchClass::G, 4),
            Duration::new(4, 1),
            Duration::quarter(),
            VoiceRole::Melody,
        ));

        let first = profile_score_region(&score, Duration::zero(), Duration::new(4, 1)).unwrap();
        let second =
            profile_score_region(&score, Duration::new(4, 1), Duration::new(8, 1)).unwrap();
        assert_eq!(first.note_count, 1);
        assert_eq!(first.onset_count, 1);
        assert_eq!(second.note_count, 2);
        assert_eq!(second.onset_count, 1);
        assert!(second.tonal_displacement > first.tonal_displacement);
    }
}
