// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Polyphonic voice system for consciousness-driven compositions.
//!
//! Generates multiple simultaneous melodic lines (voices), each with a distinct
//! role and pitch range. Voice count is consciousness-gated: higher Psi unlocks
//! richer polyphony.
//!
//! # Voice Roles
//!
//! | Role | Pitch | Behavior | Consciousness |
//! |------|-------|----------|---------------|
//! | Lead | Mid | Melody from generator | Always |
//! | Bass | Low | Root motion, octave below | Psi > 0.4 |
//! | Harmony | Mid-High | Parallel 3rds/5ths above lead | Psi > 0.6 |
//! | Ostinato | Variable | Repeating rhythmic figure | Psi > 0.7 |

use crate::instruments::Instrument;
use crate::{MusicalState, Note};
use serde::{Deserialize, Serialize};

/// Role of a voice in the polyphonic texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceRole {
    /// Primary melody line (always active).
    Lead,
    /// Bass voice: root motion an octave below the lead.
    Bass,
    /// Harmony voice: parallel intervals above the lead.
    Harmony,
    /// Ostinato: short repeating rhythmic figure.
    Ostinato,
}

/// A single voice in the polyphonic texture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    /// Voice role.
    pub role: VoiceRole,
    /// Note sequence for this voice.
    pub notes: Vec<Note>,
    /// Pitch range (min_hz, max_hz).
    pub pitch_range: (f32, f32),
    /// Volume multiplier [0, 1].
    pub volume: f32,
    /// Stereo pan position [-1.0 left, 0.0 center, 1.0 right].
    pub pan: f32,
    /// Which instrument timbre renders this voice. `None` preserves the
    /// legacy behavior: `synth::render_arrangement` falls back to its
    /// single mood-derived (valence/arousal quadrant) timbre shared by
    /// every voice — the sound this field exists to move callers away
    /// from, since one shared patch across melody/harmony/bass is exactly
    /// what reads as "one chiptune voice in three registers."
    pub instrument: Option<Instrument>,
}

/// Polyphonic arrangement: multiple voices derived from a lead melody.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arrangement {
    /// All voices in the arrangement.
    pub voices: Vec<Voice>,
}

/// Derive polyphonic voices from a lead melody and cognitive state.
///
/// The lead melody is always included. Additional voices are consciousness-gated:
/// - Bass: Psi > 0.4
/// - Harmony: Psi > 0.6
/// - Ostinato: Psi > 0.7
pub fn arrange(lead_notes: &[Note], state: &MusicalState) -> Arrangement {
    let psi = state.consciousness_level;
    let mut voices = Vec::new();

    // Lead voice (always present, center)
    voices.push(Voice {
        role: VoiceRole::Lead,
        notes: lead_notes.to_vec(),
        pitch_range: (130.0, 1000.0),
        volume: 1.0,
        pan: 0.0,
        instrument: None,
    });

    // Bass voice: root motion, octave below lead
    if psi > 0.4 && !lead_notes.is_empty() {
        voices.push(derive_bass(lead_notes, state));
    }

    // Harmony voice: parallel intervals above lead
    if psi > 0.6 && !lead_notes.is_empty() {
        voices.push(derive_harmony(lead_notes, state));
    }

    // Ostinato: repeating figure from first few notes
    if psi > 0.7 && lead_notes.len() >= 2 {
        voices.push(derive_ostinato(lead_notes, state));
    }

    Arrangement { voices }
}

/// Derive bass voice: take every Nth lead note and drop an octave.
fn derive_bass(lead: &[Note], state: &MusicalState) -> Voice {
    let step = if state.arousal > 0.5 { 1 } else { 2 };
    let notes: Vec<Note> = lead
        .iter()
        .step_by(step)
        .map(|n| Note {
            frequency: (n.frequency * 0.5).max(55.0),
            start_time: n.start_time,
            duration: n.duration * 1.5,
            velocity: n.velocity * 0.8,
        })
        .collect();

    Voice {
        role: VoiceRole::Bass,
        notes,
        pitch_range: (55.0, 500.0),
        volume: 0.7,
        pan: -0.2, // slight left
        instrument: None,
    }
}

/// Derive harmony voice: parallel interval above the lead.
///
/// The interval is driven by the dominant harmony activation:
/// - PanSentientFlourishing (harmony 1) > 0.4 → major 3rd (ratio 5/4)
/// - IntegralWisdom (harmony 2) > 0.4 → perfect 5th (ratio 3/2)
/// - Otherwise → major 3rd
fn derive_harmony(lead: &[Note], state: &MusicalState) -> Voice {
    let ratio = if state.harmony_activations[2] > state.harmony_activations[1] {
        1.5 // perfect 5th
    } else {
        1.2599 // major 3rd (2^(4/12))
    };

    let notes: Vec<Note> = lead
        .iter()
        .map(|n| {
            let freq = (n.frequency * ratio).min(2000.0);
            Note {
                frequency: freq,
                start_time: n.start_time,
                duration: n.duration,
                velocity: n.velocity * 0.6,
            }
        })
        .collect();

    Voice {
        role: VoiceRole::Harmony,
        notes,
        pitch_range: (200.0, 2000.0),
        volume: 0.5,
        pan: 0.4, // right
        instrument: None,
    }
}

/// Derive ostinato: loop a short motif from the lead throughout the piece.
fn derive_ostinato(lead: &[Note], state: &MusicalState) -> Voice {
    let motif_len = if state.harmony_activations[3] > 0.5 {
        3 // playful: short motif
    } else {
        2 // simpler
    };
    let motif: Vec<Note> = lead.iter().take(motif_len).cloned().collect();
    if motif.is_empty() {
        return Voice {
            role: VoiceRole::Ostinato,
            notes: vec![],
            pitch_range: (100.0, 800.0),
            volume: 0.3,
            pan: -0.3,
            instrument: None,
        };
    }

    // Calculate motif duration
    let motif_duration: f32 = motif.iter().map(|n| n.duration).sum::<f32>().max(0.1);

    // Total piece duration from the last lead note
    let total_duration = lead
        .last()
        .map(|n| n.start_time + n.duration)
        .unwrap_or(1.0);

    // Repeat motif throughout
    let mut notes = Vec::new();
    let mut time = 0.0;
    while time < total_duration {
        for m in &motif {
            if time >= total_duration {
                break;
            }
            notes.push(Note {
                frequency: m.frequency,
                start_time: time,
                duration: m.duration * 0.8,
                velocity: m.velocity * 0.4,
            });
            time += m.duration;
        }
        // Small gap between repetitions
        time += motif_duration * 0.1;
    }

    Voice {
        role: VoiceRole::Ostinato,
        notes,
        pitch_range: (100.0, 800.0),
        volume: 0.3,
        pan: -0.3, // left
        instrument: None,
    }
}

/// Flatten all voices into a single sorted note sequence for rendering.
pub fn flatten_voices(arrangement: &Arrangement) -> Vec<Note> {
    let mut all_notes: Vec<Note> = arrangement
        .voices
        .iter()
        .flat_map(|v| {
            v.notes.iter().map(move |n| Note {
                frequency: n.frequency,
                start_time: n.start_time,
                duration: n.duration,
                velocity: n.velocity * v.volume,
            })
        })
        .collect();
    all_notes.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_notes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lead() -> Vec<Note> {
        vec![
            Note {
                frequency: 261.63,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 329.63,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 392.00,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.9,
            },
            Note {
                frequency: 523.25,
                start_time: 1.5,
                duration: 0.5,
                velocity: 0.6,
            },
        ]
    }

    #[test]
    fn lead_always_present() {
        let state = MusicalState {
            consciousness_level: 0.1,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        assert_eq!(arr.voices.len(), 1);
        assert_eq!(arr.voices[0].role, VoiceRole::Lead);
    }

    #[test]
    fn bass_at_medium_consciousness() {
        let state = MusicalState {
            consciousness_level: 0.5,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        assert!(arr.voices.iter().any(|v| v.role == VoiceRole::Bass));
    }

    #[test]
    fn harmony_at_high_consciousness() {
        let state = MusicalState {
            consciousness_level: 0.7,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        assert!(arr.voices.iter().any(|v| v.role == VoiceRole::Harmony));
    }

    #[test]
    fn all_voices_at_peak_consciousness() {
        let state = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        assert_eq!(
            arr.voices.len(),
            4,
            "should have Lead + Bass + Harmony + Ostinato"
        );
    }

    #[test]
    fn bass_notes_lower_than_lead() {
        let state = MusicalState {
            consciousness_level: 0.5,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        let bass = arr
            .voices
            .iter()
            .find(|v| v.role == VoiceRole::Bass)
            .unwrap();
        let lead = arr
            .voices
            .iter()
            .find(|v| v.role == VoiceRole::Lead)
            .unwrap();
        for (b, l) in bass.notes.iter().zip(lead.notes.iter().step_by(2)) {
            assert!(
                b.frequency < l.frequency,
                "bass {} should be below lead {}",
                b.frequency,
                l.frequency
            );
        }
    }

    #[test]
    fn harmony_notes_higher_than_lead() {
        let state = MusicalState {
            consciousness_level: 0.7,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        let harmony = arr
            .voices
            .iter()
            .find(|v| v.role == VoiceRole::Harmony)
            .unwrap();
        let lead = arr
            .voices
            .iter()
            .find(|v| v.role == VoiceRole::Lead)
            .unwrap();
        for (h, l) in harmony.notes.iter().zip(lead.notes.iter()) {
            assert!(
                h.frequency > l.frequency,
                "harmony {} should be above lead {}",
                h.frequency,
                l.frequency
            );
        }
    }

    #[test]
    fn flatten_preserves_all_notes() {
        let state = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        let total_voice_notes: usize = arr.voices.iter().map(|v| v.notes.len()).sum();
        let flat = flatten_voices(&arr);
        assert_eq!(flat.len(), total_voice_notes);
    }

    #[test]
    fn flatten_sorted_by_time() {
        let state = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        let arr = arrange(&test_lead(), &state);
        let flat = flatten_voices(&arr);
        for w in flat.windows(2) {
            assert!(
                w[0].start_time <= w[1].start_time,
                "not sorted: {} > {}",
                w[0].start_time,
                w[1].start_time
            );
        }
    }

    #[test]
    fn empty_lead_produces_solo() {
        let state = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        let arr = arrange(&[], &state);
        assert_eq!(arr.voices.len(), 1); // only empty lead
    }
}
