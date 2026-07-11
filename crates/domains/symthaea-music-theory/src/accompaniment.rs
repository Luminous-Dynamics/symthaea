// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Accompaniment patterns: how a voiced chord is laid out IN TIME within its
//! measure.
//!
//! Until this existed the harmony voice had exactly one texture: every voiced
//! tone struck at the bar line and held for the whole measure — a static pad
//! under the melody, whatever the style. A pattern changes only the RHYTHM of
//! the accompaniment: the pitches always come from the same voice-led chord
//! (`voicing::lead_upper`), so functional harmony, voice leading, and the
//! melody's chord-tone snapping are untouched. That containment is the safety
//! property (tested): a pattern can never introduce a wrong note, only a new
//! placement of right ones.

use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};

/// A one-measure accompaniment figure. All variants realize the SAME voiced
/// chord; they differ only in when (and in which grouping) its tones sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Accompaniment {
    /// The original texture: every tone at the bar line, held for the bar.
    Block,
    /// Ascending eighth-note arpeggio cycling through the chord tones —
    /// the guitar-fingerpicking / broken-chord texture.
    Arpeggio,
    /// Alberti figure (low, high, middle, high in eighths) — the classical
    /// keyboard accompaniment idiom.
    Alberti,
    /// Oom-pah-pah: beat 1 belongs to the bass alone; the full chord answers
    /// as a quarter-note stab on every later beat. THE waltz texture (and it
    /// generalizes to 4/4 as oom-pah-pah-pah).
    OomPah,
    /// Syncopated comping: two short full-chord stabs on offbeats — a
    /// playful, rhythmically active texture.
    Comp,
}

impl Accompaniment {
    /// Realize one measure of this pattern from an already voice-led chord.
    /// `voiced` comes straight from [`crate::voicing::lead_upper`]; the
    /// pattern never alters which pitches exist, only their onsets/durations.
    #[allow(clippy::too_many_arguments)] // mirrors realize_harmony_measures' state set
    pub(crate) fn realize_measure(
        self,
        score: &mut Score,
        voiced: &[Pitch],
        bar_onset: Duration,
        meter_beats: f64,
        velocity: f32,
        section_intensity: f32,
    ) {
        if voiced.is_empty() {
            return;
        }
        let bar = Duration::new(meter_beats as i64, 1);
        let push = |score: &mut Score, pitch: Pitch, onset: Duration, dur: Duration, vel: f32| {
            score.push(ScoreNote {
                pitch,
                onset,
                duration: dur,
                velocity: vel,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
        };
        // Patterns that walk single tones want them in register order; the
        // voice-led ordering (which optimizes movement, not register) stays
        // authoritative for which pitches exist.
        let mut ascending: Vec<Pitch> = voiced.to_vec();
        ascending.sort_by_key(|p| p.midi());

        match self {
            Accompaniment::Block => {
                for &pitch in voiced {
                    push(score, pitch, bar_onset, bar, velocity);
                }
            }
            Accompaniment::Arpeggio => {
                let eighth = Duration::eighth();
                let steps = (meter_beats * 2.0) as i64; // eighths per bar
                for i in 0..steps {
                    let pitch = ascending[(i as usize) % ascending.len()];
                    // Light metric shape: beat-start eighths slightly stronger.
                    let vel = if i % 2 == 0 {
                        velocity
                    } else {
                        velocity * 0.85
                    };
                    push(score, pitch, bar_onset + eighth.scale(i, 1), eighth, vel);
                }
            }
            Accompaniment::Alberti => {
                let eighth = Duration::eighth();
                let n = ascending.len();
                // low – high – middle – high (n=3: 0,2,1,2; n=4: 0,3,1,3).
                let figure = [0usize, n - 1, (n - 1) / 2, n - 1];
                let steps = (meter_beats * 2.0) as i64;
                for i in 0..steps {
                    let pitch = ascending[figure[(i as usize) % 4].min(n - 1)];
                    let vel = if i % 4 == 0 {
                        velocity
                    } else {
                        velocity * 0.85
                    };
                    push(score, pitch, bar_onset + eighth.scale(i, 1), eighth, vel);
                }
            }
            Accompaniment::OomPah => {
                let beat = Duration::new(1, 1);
                for b in 1..(meter_beats as i64) {
                    for &pitch in voiced {
                        push(
                            score,
                            pitch,
                            bar_onset + beat.scale(b, 1),
                            beat,
                            velocity * 0.9,
                        );
                    }
                }
            }
            Accompaniment::Comp => {
                let eighth = Duration::eighth();
                let m = meter_beats as i64;
                // Offbeat stabs: the "and" before each half-bar boundary
                // (4/4 → beats 1.5 and 3.5; 3/4 → 1.0 and 2.5).
                let stabs = [Duration::new(m - 1, 2), Duration::new(2 * m - 1, 2)];
                for onset in stabs {
                    for &pitch in voiced {
                        push(score, pitch, bar_onset + onset, eighth, velocity);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn triad() -> Vec<Pitch> {
        // C4, E4, G4 (deliberately unsorted to prove register sorting).
        vec![
            Pitch::new(PitchClass::E, 4),
            Pitch::new(PitchClass::C, 4),
            Pitch::new(PitchClass::G, 4),
        ]
    }

    fn realize(pattern: Accompaniment, meter: f64) -> Vec<ScoreNote> {
        let mut score = Score::new(
            crate::harmony::Key::major(PitchClass::C),
            100.0,
            meter as u8,
        );
        pattern.realize_measure(&mut score, &triad(), Duration::zero(), meter, 0.5, 1.0);
        score.notes
    }

    #[test]
    fn block_is_the_original_texture() {
        let notes = realize(Accompaniment::Block, 4.0);
        assert_eq!(notes.len(), 3);
        for n in &notes {
            assert_eq!(n.onset, Duration::zero());
            assert_eq!(n.duration, Duration::new(4, 1));
        }
    }

    #[test]
    fn arpeggio_fills_the_bar_exactly_with_eighths() {
        let notes = realize(Accompaniment::Arpeggio, 4.0);
        assert_eq!(notes.len(), 8);
        // Contiguous eighths from bar start to bar end.
        for (i, n) in notes.iter().enumerate() {
            assert_eq!(n.onset, Duration::eighth().scale(i as i64, 1));
            assert_eq!(n.duration, Duration::eighth());
        }
        let last = notes.last().unwrap();
        assert_eq!(last.onset + last.duration, Duration::new(4, 1));
        // Ascends in register from the lowest tone.
        assert_eq!(notes[0].pitch.midi(), 60); // C4
        assert!(notes[1].pitch.midi() > notes[0].pitch.midi());
    }

    #[test]
    fn alberti_walks_low_high_middle_high() {
        let notes = realize(Accompaniment::Alberti, 4.0);
        let midis: Vec<u8> = notes.iter().map(|n| n.pitch.midi()).collect();
        // C4 E4 G4 sorted → low=60, mid=64, high=67.
        assert_eq!(&midis[..4], &[60, 67, 64, 67]);
        assert_eq!(&midis[4..8], &[60, 67, 64, 67]);
    }

    #[test]
    fn oom_pah_leaves_beat_one_to_the_bass() {
        let notes = realize(Accompaniment::OomPah, 3.0);
        // 3/4: chords on beats 2 and 3 only (onsets 1 and 2), quarter each.
        assert_eq!(notes.len(), 6);
        assert!(notes.iter().all(|n| n.onset != Duration::zero()));
        let onsets: std::collections::BTreeSet<i64> = notes
            .iter()
            .map(|n| (n.onset.beats() * 2.0) as i64)
            .collect();
        assert_eq!(onsets, [2i64, 4].into_iter().collect()); // beats 1.0, 2.0 in half-beat units
        assert!(notes.iter().all(|n| n.duration == Duration::new(1, 1)));
    }

    #[test]
    fn comp_stabs_land_on_offbeats() {
        let notes = realize(Accompaniment::Comp, 4.0);
        assert_eq!(notes.len(), 6); // 2 stabs × 3 tones
        for n in &notes {
            let b = n.onset.beats();
            assert!(
                (b - 1.5).abs() < 1e-9 || (b - 3.5).abs() < 1e-9,
                "stab off the expected offbeats: {b}"
            );
            assert_eq!(n.duration, Duration::eighth());
        }
    }

    #[test]
    fn every_pattern_only_uses_the_voiced_pitches() {
        // The safety property: a pattern may re-time chord tones but can
        // NEVER introduce a pitch the voice leading didn't produce.
        let allowed: std::collections::BTreeSet<u8> = triad().iter().map(|p| p.midi()).collect();
        for pattern in [
            Accompaniment::Block,
            Accompaniment::Arpeggio,
            Accompaniment::Alberti,
            Accompaniment::OomPah,
            Accompaniment::Comp,
        ] {
            for meter in [3.0, 4.0] {
                for n in realize(pattern, meter) {
                    assert!(
                        allowed.contains(&n.pitch.midi()),
                        "{pattern:?} produced a non-chord pitch {}",
                        n.pitch.midi()
                    );
                }
            }
        }
    }

    #[test]
    fn every_pattern_stays_within_the_bar() {
        for pattern in [
            Accompaniment::Block,
            Accompaniment::Arpeggio,
            Accompaniment::Alberti,
            Accompaniment::OomPah,
            Accompaniment::Comp,
        ] {
            for meter in [3.0, 4.0] {
                for n in realize(pattern, meter) {
                    assert!((n.onset + n.duration).beats() <= meter + 1e-9);
                }
            }
        }
    }
}
