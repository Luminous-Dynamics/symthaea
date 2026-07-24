// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Notation-grade pitch spelling and altered scale degrees.
//!
//! [`PitchClass`](crate::pitch::PitchClass) intentionally collapses enharmonic
//! spellings so that C-sharp and D-flat are the same sounding pitch class.
//! Composition and analysis also need the *written* identity of a note,
//! because spelling carries harmonic meaning. This module adds that parallel
//! representation without changing the compact MIDI-oriented [`Pitch`] model.

use crate::harmony::Key;
use crate::pitch::PitchClass;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The seven diatonic letter names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LetterName {
    C,
    D,
    E,
    F,
    G,
    A,
    B,
}

impl LetterName {
    /// The natural pitch class of this letter.
    pub const fn natural_pitch_class(self) -> PitchClass {
        match self {
            Self::C => PitchClass::C,
            Self::D => PitchClass::D,
            Self::E => PitchClass::E,
            Self::F => PitchClass::F,
            Self::G => PitchClass::G,
            Self::A => PitchClass::A,
            Self::B => PitchClass::B,
        }
    }

    /// Zero-based diatonic index (C=0 ... B=6).
    pub const fn index(self) -> u8 {
        match self {
            Self::C => 0,
            Self::D => 1,
            Self::E => 2,
            Self::F => 3,
            Self::G => 4,
            Self::A => 5,
            Self::B => 6,
        }
    }

    /// ASCII letter name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::C => "C",
            Self::D => "D",
            Self::E => "E",
            Self::F => "F",
            Self::G => "G",
            Self::A => "A",
            Self::B => "B",
        }
    }
}

/// A written accidental, measured in semitones from the natural letter.
///
/// The common values are -2 (double-flat), -1 (flat), 0 (natural),
/// 1 (sharp), and 2 (double-sharp). Wider values are retained because imported
/// notation may contain them, but [`Accidental::is_standard`] exposes whether
/// the spelling is within the ordinary double-accidental range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Accidental(i8);

impl Accidental {
    pub const DOUBLE_FLAT: Self = Self(-2);
    pub const FLAT: Self = Self(-1);
    pub const NATURAL: Self = Self(0);
    pub const SHARP: Self = Self(1);
    pub const DOUBLE_SHARP: Self = Self(2);

    pub const fn new(semitones: i8) -> Self {
        Self(semitones)
    }

    pub const fn semitones(self) -> i8 {
        self.0
    }

    pub const fn is_standard(self) -> bool {
        self.0 >= -2 && self.0 <= 2
    }
}

impl fmt::Display for Accidental {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            -2 => f.write_str("bb"),
            -1 => f.write_str("b"),
            0 => Ok(()),
            1 => f.write_str("#"),
            2 => f.write_str("x"),
            n => write!(f, "({n:+})"),
        }
    }
}

/// A pitch class whose written letter and accidental are preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpelledPitchClass {
    pub letter: LetterName,
    pub accidental: Accidental,
}

impl SpelledPitchClass {
    pub const fn new(letter: LetterName, accidental: Accidental) -> Self {
        Self { letter, accidental }
    }

    /// Collapse the spelling to its sounding enharmonic pitch class.
    pub fn pitch_class(self) -> PitchClass {
        self.letter
            .natural_pitch_class()
            .transpose(self.accidental.semitones() as i32)
    }

    /// True when two spellings sound at the same pitch class.
    pub fn is_enharmonic_with(self, other: Self) -> bool {
        self.pitch_class() == other.pitch_class()
    }
}

impl fmt::Display for SpelledPitchClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.letter.name(), self.accidental)
    }
}

/// A diatonic scale degree plus a chromatic alteration.
///
/// `degree` is always 1..=7. Examples: `4/#` is the raised fourth, `6/b` is
/// the lowered sixth. This is the representation needed for applied harmony,
/// modal mixture, chromatic tendency tones, and notation-aware analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AlteredDegree {
    degree: u8,
    alteration: i8,
}

impl AlteredDegree {
    /// Construct an altered degree. Returns `None` outside 1..=7.
    pub const fn new(degree: u8, alteration: i8) -> Option<Self> {
        if degree >= 1 && degree <= 7 {
            Some(Self { degree, alteration })
        } else {
            None
        }
    }

    /// Construct an unaltered degree. Panics only for a programmer error.
    pub const fn diatonic(degree: u8) -> Self {
        assert!(degree >= 1 && degree <= 7, "scale degree must be 1..=7");
        Self {
            degree,
            alteration: 0,
        }
    }

    pub const fn degree(self) -> u8 {
        self.degree
    }

    pub const fn alteration(self) -> i8 {
        self.alteration
    }

    /// Realize this degree as a sounding pitch class in `key`.
    pub fn pitch_class_in(self, key: Key) -> PitchClass {
        key.scale()
            .degree_pitch_class(self.degree as i32)
            .transpose(self.alteration as i32)
    }
}

impl fmt::Display for AlteredDegree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", Accidental::new(self.alteration), self.degree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enharmonic_pitch_classes_keep_distinct_spelling() {
        let c_sharp = SpelledPitchClass::new(LetterName::C, Accidental::SHARP);
        let d_flat = SpelledPitchClass::new(LetterName::D, Accidental::FLAT);

        assert_ne!(c_sharp, d_flat);
        assert!(c_sharp.is_enharmonic_with(d_flat));
        assert_eq!(c_sharp.pitch_class(), PitchClass::CSHARP);
        assert_eq!(c_sharp.to_string(), "C#");
        assert_eq!(d_flat.to_string(), "Db");
    }

    #[test]
    fn altered_degree_realizes_in_key() {
        let c_major = Key::major(PitchClass::C);
        let raised_fourth = AlteredDegree::new(4, 1).unwrap();
        let lowered_sixth = AlteredDegree::new(6, -1).unwrap();

        assert_eq!(raised_fourth.pitch_class_in(c_major), PitchClass::FSHARP);
        assert_eq!(lowered_sixth.pitch_class_in(c_major), PitchClass::GSHARP);
        assert_eq!(raised_fourth.to_string(), "#4");
        assert_eq!(lowered_sixth.to_string(), "b6");
    }

    #[test]
    fn degree_range_is_explicit() {
        assert!(AlteredDegree::new(0, 0).is_none());
        assert!(AlteredDegree::new(8, 0).is_none());
        assert_eq!(AlteredDegree::diatonic(5).degree(), 5);
    }
}
