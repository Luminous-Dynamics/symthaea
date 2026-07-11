// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chords: a root pitch class plus a quality, expanded into pitch classes and
//! absolute voicings. The symbolic basis for Layer 1 functional harmony.

use crate::pitch::{Pitch, PitchClass};
use serde::{Deserialize, Serialize};

/// Chord quality, defined by the semitone offsets of its chord tones from the
/// root (root = 0, implicit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChordQuality {
    Major,           // 0 4 7
    Minor,           // 0 3 7
    Diminished,      // 0 3 6
    Augmented,       // 0 4 8
    Major7,          // 0 4 7 11
    Minor7,          // 0 3 7 10
    Dominant7,       // 0 4 7 10
    MinorMajor7,     // 0 3 7 11
    HalfDiminished7, // 0 3 6 10 (ø7)
    Diminished7,     // 0 3 6 9
    Sus2,            // 0 2 7
    Sus4,            // 0 5 7
}

impl ChordQuality {
    /// Semitone offsets of the chord tones from the root.
    pub fn intervals(self) -> &'static [u8] {
        match self {
            ChordQuality::Major => &[0, 4, 7],
            ChordQuality::Minor => &[0, 3, 7],
            ChordQuality::Diminished => &[0, 3, 6],
            ChordQuality::Augmented => &[0, 4, 8],
            ChordQuality::Major7 => &[0, 4, 7, 11],
            ChordQuality::Minor7 => &[0, 3, 7, 10],
            ChordQuality::Dominant7 => &[0, 4, 7, 10],
            ChordQuality::MinorMajor7 => &[0, 3, 7, 11],
            ChordQuality::HalfDiminished7 => &[0, 3, 6, 10],
            ChordQuality::Diminished7 => &[0, 3, 6, 9],
            ChordQuality::Sus2 => &[0, 2, 7],
            ChordQuality::Sus4 => &[0, 5, 7],
        }
    }

    /// True for a seventh chord (four tones) — needed to know a chord has a
    /// dissonant seventh that wants resolution.
    pub fn is_seventh(self) -> bool {
        self.intervals().len() == 4
    }

    /// A crude harmonic-tension weight in [0, 1] — consonant triads low,
    /// dominant/diminished sevenths high. Used by Layer 1's tension curve.
    pub fn tension(self) -> f32 {
        match self {
            ChordQuality::Major | ChordQuality::Minor => 0.1,
            ChordQuality::Sus2 | ChordQuality::Sus4 => 0.3,
            ChordQuality::Major7 | ChordQuality::Minor7 | ChordQuality::MinorMajor7 => 0.4,
            ChordQuality::Augmented => 0.6,
            ChordQuality::Dominant7 => 0.7,
            ChordQuality::HalfDiminished7 => 0.75,
            ChordQuality::Diminished => 0.8,
            ChordQuality::Diminished7 => 0.9,
        }
    }
}

/// A chord: a root pitch class and a quality. Octave-free (a pitch-class set);
/// call [`Chord::voice`] to realize it as absolute pitches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chord {
    pub root: PitchClass,
    pub quality: ChordQuality,
}

impl Chord {
    pub fn new(root: PitchClass, quality: ChordQuality) -> Self {
        Chord { root, quality }
    }

    /// The chord tones as pitch classes, in root-position order.
    pub fn pitch_classes(self) -> Vec<PitchClass> {
        self.quality
            .intervals()
            .iter()
            .map(|&i| self.root.transpose(i as i32))
            .collect()
    }

    /// True if `pc` is a chord tone (as opposed to a non-chord tone).
    pub fn contains(self, pc: PitchClass) -> bool {
        let rel = self.root.interval_to(pc);
        self.quality.intervals().contains(&rel)
    }

    /// Voice the chord in close position, root in `root_octave` and each
    /// successive tone at or above the previous (so it always ascends).
    pub fn voice(self, root_octave: i32) -> Vec<Pitch> {
        let root = Pitch::new(self.root, root_octave);
        self.quality
            .intervals()
            .iter()
            .map(|&i| root.transpose(i as i32))
            .collect()
    }

    /// Apply an inversion: rotate the lowest `inversion` tones up an octave.
    /// Inversion 0 = root position, 1 = first inversion, etc.
    pub fn voice_inverted(self, root_octave: i32, inversion: usize) -> Vec<Pitch> {
        let mut voices = self.voice(root_octave);
        let n = voices.len();
        for v in voices.iter_mut().take(inversion.min(n)) {
            *v = v.transpose(12);
        }
        voices.sort();
        voices
    }

    /// The bass pitch class if this chord is used with the given inversion
    /// (root, third, fifth, seventh, …).
    pub fn bass_pitch_class(self, inversion: usize) -> PitchClass {
        let intervals = self.quality.intervals();
        let i = inversion % intervals.len();
        self.root.transpose(intervals[i] as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn major_triad_ground_truth() {
        // C major = C E G.
        let c = Chord::new(PitchClass::C, ChordQuality::Major);
        assert_eq!(
            c.pitch_classes(),
            vec![PitchClass::C, PitchClass::E, PitchClass::G]
        );
    }

    #[test]
    fn minor_triad_ground_truth() {
        // A minor = A C E.
        let a = Chord::new(PitchClass::A, ChordQuality::Minor);
        assert_eq!(
            a.pitch_classes(),
            vec![PitchClass::A, PitchClass::C, PitchClass::E]
        );
    }

    #[test]
    fn g_dominant_seventh_ground_truth() {
        // G7 = G B D F (the dominant of C major).
        let g7 = Chord::new(PitchClass::G, ChordQuality::Dominant7);
        assert_eq!(
            g7.pitch_classes(),
            vec![PitchClass::G, PitchClass::B, PitchClass::D, PitchClass::F]
        );
        assert!(g7.quality.is_seventh());
    }

    #[test]
    fn diminished_seventh_is_stacked_minor_thirds() {
        // B°7 = B D F Ab — all minor thirds.
        let dim7 = Chord::new(PitchClass::B, ChordQuality::Diminished7);
        assert_eq!(
            dim7.pitch_classes(),
            vec![
                PitchClass::B,
                PitchClass::D,
                PitchClass::F,
                PitchClass::GSHARP
            ]
        );
    }

    #[test]
    fn chord_tone_membership() {
        let c = Chord::new(PitchClass::C, ChordQuality::Major);
        assert!(c.contains(PitchClass::E)); // chord tone
        assert!(!c.contains(PitchClass::D)); // non-chord tone (a 2nd)
    }

    #[test]
    fn voicing_ascends_from_root_octave() {
        let c = Chord::new(PitchClass::C, ChordQuality::Major);
        let v = c.voice(4);
        assert_eq!(v[0], Pitch::new(PitchClass::C, 4)); // C4 = 60
        assert_eq!(v[0].midi(), 60);
        assert_eq!(v[1].midi(), 64); // E4
        assert_eq!(v[2].midi(), 67); // G4
    }

    #[test]
    fn first_inversion_puts_third_in_bass() {
        // C/E: first inversion of C major has E in the bass.
        let c = Chord::new(PitchClass::C, ChordQuality::Major);
        assert_eq!(c.bass_pitch_class(0), PitchClass::C);
        assert_eq!(c.bass_pitch_class(1), PitchClass::E);
        assert_eq!(c.bass_pitch_class(2), PitchClass::G);
        let inv = c.voice_inverted(4, 1);
        assert_eq!(inv[0].pitch_class(), PitchClass::E); // lowest is E
    }

    #[test]
    fn tension_orders_consonance_to_dissonance() {
        assert!(ChordQuality::Major.tension() < ChordQuality::Dominant7.tension());
        assert!(ChordQuality::Dominant7.tension() < ChordQuality::Diminished7.tension());
    }
}
