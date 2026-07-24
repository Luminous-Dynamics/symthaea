// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scales and modes: a tonic pitch class plus an interval pattern, and the
//! map from scale degree to pitch class.

use crate::pitch::{Pitch, PitchClass};
use serde::{Deserialize, Serialize};

/// A scale/mode family, defined by its ascending semitone offsets from the
/// tonic (the tonic itself is offset 0 and is implicit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Mode {
    Ionian,     // major
    Dorian,     // minor with raised 6
    Phrygian,   // minor with lowered 2 (Spanish/dark)
    Lydian,     // major with raised 4 (bright/floating)
    Mixolydian, // major with lowered 7 (bluesy/folk)
    Aeolian,    // natural minor
    Locrian,    // diminished (unstable)
    HarmonicMinor,
    MelodicMinor, // ascending form
    MajorPentatonic,
    MinorPentatonic,
    WholeTone, // dreamlike/ambiguous
}

impl Mode {
    /// Semitone offsets from the tonic, excluding the octave.
    pub fn offsets(self) -> &'static [u8] {
        match self {
            Mode::Ionian => &[0, 2, 4, 5, 7, 9, 11],
            Mode::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            Mode::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
            Mode::Lydian => &[0, 2, 4, 6, 7, 9, 11],
            Mode::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            Mode::Aeolian => &[0, 2, 3, 5, 7, 8, 10],
            Mode::Locrian => &[0, 1, 3, 5, 6, 8, 10],
            Mode::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            Mode::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
            Mode::MajorPentatonic => &[0, 2, 4, 7, 9],
            Mode::MinorPentatonic => &[0, 3, 5, 7, 10],
            Mode::WholeTone => &[0, 2, 4, 6, 8, 10],
        }
    }

    /// How many degrees before the octave repeats (7 for diatonic, 5 for
    /// pentatonic, 6 for whole-tone).
    pub fn degree_count(self) -> usize {
        self.offsets().len()
    }

    /// True for the "minor-flavoured" modes (lowered 3rd) — used by the
    /// consciousness→mode mapping (valence < 0 → darker modes).
    pub fn is_minor_flavored(self) -> bool {
        matches!(
            self,
            Mode::Dorian
                | Mode::Phrygian
                | Mode::Aeolian
                | Mode::Locrian
                | Mode::HarmonicMinor
                | Mode::MelodicMinor
                | Mode::MinorPentatonic
        )
    }
}

/// A concrete scale: a tonic pitch class and a mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scale {
    pub tonic: PitchClass,
    pub mode: Mode,
}

impl Scale {
    pub fn new(tonic: PitchClass, mode: Mode) -> Self {
        Scale { tonic, mode }
    }

    /// The pitch class at a given scale degree. Degrees are 1-based in the
    /// music-theory sense (degree 1 = tonic); this accepts any integer and
    /// wraps across octaves, so degree 8 == degree 1, degree 0 == the degree
    /// below the tonic. Chromatic offsets are taken mod-12.
    pub fn degree_pitch_class(self, degree: i32) -> PitchClass {
        let offsets = self.mode.offsets();
        let n = offsets.len() as i32;
        // Convert 1-based degree to 0-based index, wrapping across octaves.
        let idx0 = degree - 1;
        let within = idx0.rem_euclid(n) as usize;
        self.tonic.transpose(offsets[within] as i32)
    }

    /// An absolute pitch for a scale degree, anchored so degree 1 lands in the
    /// given octave. Degrees above `degree_count` roll up into higher octaves.
    pub fn degree_pitch(self, degree: i32, tonic_octave: i32) -> Pitch {
        let offsets = self.mode.offsets();
        let n = offsets.len() as i32;
        let idx0 = degree - 1;
        let octave_shift = idx0.div_euclid(n);
        let within = idx0.rem_euclid(n) as usize;
        let semis = offsets[within] as i32 + 12 * octave_shift;
        Pitch::new(self.tonic, tonic_octave).transpose(semis)
    }

    /// True if `pc` belongs to this scale (a chord tone / diatonic pitch).
    pub fn contains(self, pc: PitchClass) -> bool {
        let rel = self.tonic.interval_to(pc);
        self.mode.offsets().contains(&rel)
    }

    /// All pitch classes of the scale, ascending from the tonic.
    pub fn pitch_classes(self) -> Vec<PitchClass> {
        self.mode
            .offsets()
            .iter()
            .map(|&o| self.tonic.transpose(o as i32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_major_scale_ground_truth() {
        // C major = C D E F G A B.
        let s = Scale::new(PitchClass::C, Mode::Ionian);
        let pcs = s.pitch_classes();
        let expect = [
            PitchClass::C,
            PitchClass::D,
            PitchClass::E,
            PitchClass::F,
            PitchClass::G,
            PitchClass::A,
            PitchClass::B,
        ];
        assert_eq!(pcs, expect);
    }

    #[test]
    fn a_natural_minor_has_no_sharps() {
        // A aeolian = A B C D E F G (relative minor of C major).
        let s = Scale::new(PitchClass::A, Mode::Aeolian);
        let pcs = s.pitch_classes();
        let expect = [
            PitchClass::A,
            PitchClass::B,
            PitchClass::C,
            PitchClass::D,
            PitchClass::E,
            PitchClass::F,
            PitchClass::G,
        ];
        assert_eq!(pcs, expect);
    }

    #[test]
    fn degrees_are_one_based_and_wrap() {
        let s = Scale::new(PitchClass::C, Mode::Ionian);
        assert_eq!(s.degree_pitch_class(1), PitchClass::C); // tonic
        assert_eq!(s.degree_pitch_class(5), PitchClass::G); // dominant
        assert_eq!(s.degree_pitch_class(8), PitchClass::C); // octave = degree 1
        assert_eq!(s.degree_pitch_class(9), PitchClass::D); // = degree 2
    }

    #[test]
    fn degree_pitch_climbs_octaves() {
        let s = Scale::new(PitchClass::C, Mode::Ionian);
        assert_eq!(s.degree_pitch(1, 4), Pitch::new(PitchClass::C, 4)); // C4
        assert_eq!(s.degree_pitch(8, 4), Pitch::new(PitchClass::C, 5)); // C5
        assert_eq!(s.degree_pitch(5, 4).midi(), 67); // G4
    }

    #[test]
    fn contains_membership() {
        let s = Scale::new(PitchClass::C, Mode::Ionian);
        assert!(s.contains(PitchClass::E));
        assert!(!s.contains(PitchClass::CSHARP));
    }

    #[test]
    fn harmonic_minor_raises_the_seventh() {
        // A harmonic minor has G#, not G (the leading tone).
        let s = Scale::new(PitchClass::A, Mode::HarmonicMinor);
        assert!(s.contains(PitchClass::GSHARP));
        assert!(!s.contains(PitchClass::G));
    }

    #[test]
    fn pentatonic_has_five_degrees() {
        assert_eq!(Mode::MajorPentatonic.degree_count(), 5);
        assert_eq!(Mode::WholeTone.degree_count(), 6);
        assert_eq!(Mode::Ionian.degree_count(), 7);
    }
}
