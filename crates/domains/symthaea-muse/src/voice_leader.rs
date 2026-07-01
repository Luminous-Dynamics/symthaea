// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Voice leading: independent voices with counterpoint rules.
//!
//! Each voice (lead, bass, harmony, ostinato) follows its own melodic logic.
//! Bass walks by step or arpeggiation. Harmony fills chord tones between
//! lead and bass. Lead carries the melody. Ostinato repeats a pattern.
//!
//! Call-and-response: the lead plays a phrase, then the harmony echoes it
//! (transposed down a third or fifth) in the next phrase.

use crate::Note;
use crate::melodic_grammar::MelodicContext;

/// Voice roles with independent behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceRole {
    Lead,     // carries melody, highest priority
    Bass,     // root movement, walking pattern
    Harmony,  // fills between lead and bass, echoes lead
    Ostinato, // repeating pattern
}

/// State for each voice.
pub struct VoiceState {
    pub role: VoiceRole,
    pub prev_freq: Option<f32>,
    pub prev_direction: i32,
    pub prev_interval: f32,
    pub phrase_notes: Vec<Note>,
    /// For call-and-response: queued phrase from another voice to echo.
    pub echo_queue: Vec<Note>,
    pub notes_generated: usize,
}

impl VoiceState {
    pub fn new(role: VoiceRole) -> Self {
        Self {
            role,
            prev_freq: None,
            prev_direction: 0,
            prev_interval: 0.0,
            phrase_notes: Vec::new(),
            echo_queue: Vec::new(),
            notes_generated: 0,
        }
    }

    pub fn melodic_context(&self, phrase_pos: f32, beat_pos: f32) -> MelodicContext {
        MelodicContext {
            prev_freq: self.prev_freq,
            last_direction: self.prev_direction,
            last_interval: self.prev_interval,
            phrase_position: phrase_pos,
            beat_position: beat_pos,
            prev_was_leap: self.prev_interval.abs() > 4.0,
        }
    }

    pub fn record(&mut self, note: Note) {
        let freq = note.frequency;
        if let Some(prev) = self.prev_freq {
            self.prev_interval = (freq / prev).log2() * 12.0;
            self.prev_direction = if freq > prev {
                1
            } else if freq < prev {
                -1
            } else {
                0
            };
        }
        self.prev_freq = Some(freq);
        self.phrase_notes.push(note);
        self.notes_generated += 1;
    }

    pub fn reset(&mut self) {
        self.prev_freq = None;
        self.prev_direction = 0;
        self.prev_interval = 0.0;
        self.phrase_notes.clear();
        self.echo_queue.clear();
        self.notes_generated = 0;
    }
}

/// Generate a bass note for the current chord.
///
/// Bass voice follows these rules:
/// 1. On beat 1: play chord root
/// 2. On beat 3: play chord 5th
/// 3. On beats 2, 4: walk by step toward next chord root
/// 4. Octave: one octave below lead range
pub fn bass_note(
    chord_root_freq: f32,
    chord_fifth_freq: f32,
    beat_position: f32,
    state: &VoiceState,
    scale_tones: &[f32],
) -> f32 {
    let bass_octave = 0.5; // one octave below middle

    let beat = beat_position % 4.0;
    if beat < 0.5 {
        // Beat 1: root
        chord_root_freq * bass_octave
    } else if (beat - 2.0).abs() < 0.5 {
        // Beat 3: fifth
        chord_fifth_freq * bass_octave
    } else {
        // Walking: step from previous toward next target
        if let Some(prev) = state.prev_freq {
            // Find nearest scale tone above or below previous
            let bass_scale: Vec<f32> = scale_tones.iter().map(|&f| f * bass_octave).collect();
            let idx = bass_scale
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((**a - prev).abs())
                        .partial_cmp(&((**b - prev).abs()))
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            // Walk toward root
            let target = chord_root_freq * bass_octave;
            let direction = if target > prev { 1i32 } else { -1 };
            let next_idx = (idx as i32 + direction).clamp(0, bass_scale.len() as i32 - 1) as usize;
            bass_scale.get(next_idx).copied().unwrap_or(prev)
        } else {
            chord_root_freq * bass_octave
        }
    }
}

/// Generate a harmony note: fill between lead and bass.
///
/// Prefers chord tones that aren't the root or the lead note.
/// Creates call-and-response by echoing the lead's last phrase
/// transposed down a third.
pub fn harmony_note(lead_freq: f32, chord_tones: &[f32], state: &mut VoiceState) -> f32 {
    // Call-and-response: if echo queue has notes, play from it
    if let Some(echo) = state.echo_queue.first().copied() {
        state.echo_queue.remove(0);
        return echo.frequency;
    }

    // Pick a chord tone that's below the lead but not the root
    let candidates: Vec<f32> = chord_tones
        .iter()
        .copied()
        .filter(|&f| f < lead_freq * 0.95 && f > lead_freq * 0.4) // below lead, above bass
        .collect();

    if let Some(&best) = candidates.iter().min_by(|a, b| {
        ((**a - lead_freq * 0.75).abs())
            .partial_cmp(&((**b - lead_freq * 0.75).abs()))
            .unwrap()
    }) {
        best
    } else {
        // Fallback: lead frequency down a major third
        lead_freq * 2.0f32.powf(-4.0 / 12.0)
    }
}

/// Set up call-and-response: queue a transposed copy of the lead's phrase
/// into the harmony voice.
pub fn setup_call_response(lead_phrase: &[Note], harmony_state: &mut VoiceState) {
    // Transpose down a third (4 semitones)
    let ratio = 2.0f32.powf(-4.0 / 12.0);
    harmony_state.echo_queue = lead_phrase
        .iter()
        .map(|n| Note {
            frequency: n.frequency * ratio,
            start_time: n.start_time,
            duration: n.duration,
            velocity: n.velocity * 0.8, // slightly softer echo
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bass_plays_root_on_beat_one() {
        let state = VoiceState::new(VoiceRole::Bass);
        let result = bass_note(
            261.63,
            392.00,
            0.0,
            &state,
            &[261.63, 293.66, 329.63, 392.00],
        );
        assert!(
            (result - 261.63 * 0.5).abs() < 1.0,
            "bass should play root: {result}"
        );
    }

    #[test]
    fn bass_plays_fifth_on_beat_three() {
        let state = VoiceState::new(VoiceRole::Bass);
        let result = bass_note(
            261.63,
            392.00,
            2.0,
            &state,
            &[261.63, 293.66, 329.63, 392.00],
        );
        assert!(
            (result - 392.00 * 0.5).abs() < 1.0,
            "bass should play fifth: {result}"
        );
    }

    #[test]
    fn harmony_below_lead() {
        let mut state = VoiceState::new(VoiceRole::Harmony);
        let chord = vec![261.63, 329.63, 392.00, 523.25];
        let result = harmony_note(440.0, &chord, &mut state);
        assert!(result < 440.0, "harmony should be below lead: {result}");
    }

    #[test]
    fn call_response_transposes() {
        let mut state = VoiceState::new(VoiceRole::Harmony);
        let phrase = vec![
            Note {
                frequency: 440.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 493.88,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        setup_call_response(&phrase, &mut state);
        assert_eq!(state.echo_queue.len(), 2);
        assert!(
            state.echo_queue[0].frequency < 440.0,
            "echo should be transposed down"
        );
    }
}
