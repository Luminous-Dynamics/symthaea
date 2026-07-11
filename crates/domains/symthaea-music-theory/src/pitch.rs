// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pitch primitives: pitch class, absolute pitch, and interval.
//!
//! The atom of this whole crate is [`PitchClass`] (0–11), NOT a frequency.
//! Frequencies belong to `symthaea-muse`, which realizes a symbolic score.

use serde::{Deserialize, Serialize};

/// One of the 12 pitch classes, C = 0 … B = 11 (enharmonic; C# == Db).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PitchClass(u8);

impl PitchClass {
    pub const C: PitchClass = PitchClass(0);
    pub const CSHARP: PitchClass = PitchClass(1);
    pub const D: PitchClass = PitchClass(2);
    pub const DSHARP: PitchClass = PitchClass(3);
    pub const E: PitchClass = PitchClass(4);
    pub const F: PitchClass = PitchClass(5);
    pub const FSHARP: PitchClass = PitchClass(6);
    pub const G: PitchClass = PitchClass(7);
    pub const GSHARP: PitchClass = PitchClass(8);
    pub const A: PitchClass = PitchClass(9);
    pub const ASHARP: PitchClass = PitchClass(10);
    pub const B: PitchClass = PitchClass(11);

    /// Build from any integer, wrapping mod 12 (so `new(-1)` == B, `new(12)` == C).
    pub fn new(value: i32) -> Self {
        PitchClass(value.rem_euclid(12) as u8)
    }

    /// The raw value, 0–11.
    pub fn value(self) -> u8 {
        self.0
    }

    /// Transpose by a signed number of semitones (wraps mod 12).
    pub fn transpose(self, semitones: i32) -> Self {
        PitchClass::new(self.0 as i32 + semitones)
    }

    /// Directed interval up to `other` in [0, 12) semitones.
    pub fn interval_to(self, other: PitchClass) -> u8 {
        ((other.0 as i32 - self.0 as i32).rem_euclid(12)) as u8
    }

    /// Standard sharp-spelling name (C, C#, D, …).
    pub fn name(self) -> &'static str {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        NAMES[self.0 as usize]
    }
}

/// An absolute pitch: a pitch class plus an octave (scientific pitch notation,
/// where octave 4 contains middle C). Represented internally as a MIDI number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Pitch {
    /// MIDI note number: 60 = middle C (C4), 69 = A4 (440 Hz).
    midi: u8,
}

impl Pitch {
    /// From a MIDI note number (0–127).
    pub fn from_midi(midi: u8) -> Self {
        Pitch { midi }
    }

    /// From a pitch class and octave (octave 4 = middle-C octave).
    /// MIDI = (octave + 1) * 12 + class, so C4 = 60.
    pub fn new(class: PitchClass, octave: i32) -> Self {
        let midi = ((octave + 1) * 12 + class.value() as i32).clamp(0, 127) as u8;
        Pitch { midi }
    }

    pub fn midi(self) -> u8 {
        self.midi
    }

    /// The pitch class (octave discarded).
    pub fn pitch_class(self) -> PitchClass {
        PitchClass::new(self.midi as i32)
    }

    /// Scientific octave (C4 = octave 4).
    pub fn octave(self) -> i32 {
        self.midi as i32 / 12 - 1
    }

    /// Transpose by semitones (clamped to the MIDI range).
    pub fn transpose(self, semitones: i32) -> Self {
        Pitch {
            midi: (self.midi as i32 + semitones).clamp(0, 127) as u8,
        }
    }

    /// Signed semitone distance to `other` (positive = `other` is higher).
    pub fn semitones_to(self, other: Pitch) -> i32 {
        other.midi as i32 - self.midi as i32
    }

    /// The undirected interval up from this pitch.
    pub fn interval_to(self, other: Pitch) -> Interval {
        Interval::from_semitones(self.semitones_to(other).unsigned_abs() as i32)
    }
}

/// Consonance/quality classification of an interval, by semitone count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntervalQuality {
    /// Unison, octave, P4, P5 — perfect consonances.
    PerfectConsonance,
    /// Major/minor 3rds and 6ths — imperfect consonances.
    ImperfectConsonance,
    /// Major/minor 2nds and 7ths — dissonances.
    Dissonance,
    /// The tritone — the defining unstable interval.
    Tritone,
}

/// An interval measured in semitones (compound intervals reduced mod 12 for
/// quality classification, but the raw semitone count is preserved).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Interval {
    semitones: i32,
}

impl Interval {
    pub const UNISON: Interval = Interval { semitones: 0 };
    pub const MINOR_SECOND: Interval = Interval { semitones: 1 };
    pub const MAJOR_SECOND: Interval = Interval { semitones: 2 };
    pub const MINOR_THIRD: Interval = Interval { semitones: 3 };
    pub const MAJOR_THIRD: Interval = Interval { semitones: 4 };
    pub const PERFECT_FOURTH: Interval = Interval { semitones: 5 };
    pub const TRITONE: Interval = Interval { semitones: 6 };
    pub const PERFECT_FIFTH: Interval = Interval { semitones: 7 };
    pub const MINOR_SIXTH: Interval = Interval { semitones: 8 };
    pub const MAJOR_SIXTH: Interval = Interval { semitones: 9 };
    pub const MINOR_SEVENTH: Interval = Interval { semitones: 10 };
    pub const MAJOR_SEVENTH: Interval = Interval { semitones: 11 };
    pub const OCTAVE: Interval = Interval { semitones: 12 };

    pub fn from_semitones(semitones: i32) -> Self {
        Interval { semitones }
    }

    pub fn semitones(self) -> i32 {
        self.semitones
    }

    /// Classify by simple-interval quality (reduced to within an octave).
    pub fn quality(self) -> IntervalQuality {
        match self.semitones.rem_euclid(12) {
            0 | 5 | 7 => IntervalQuality::PerfectConsonance,
            3 | 4 | 8 | 9 => IntervalQuality::ImperfectConsonance,
            6 => IntervalQuality::Tritone,
            _ => IntervalQuality::Dissonance, // 1, 2, 10, 11
        }
    }

    /// True for a step (minor or major 2nd) — the unit of conjunct motion.
    pub fn is_step(self) -> bool {
        matches!(self.semitones.unsigned_abs(), 1 | 2)
    }

    /// True for a leap (a 3rd or wider).
    pub fn is_leap(self) -> bool {
        self.semitones.unsigned_abs() >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_class_wraps() {
        assert_eq!(PitchClass::new(-1), PitchClass::B);
        assert_eq!(PitchClass::new(12), PitchClass::C);
        assert_eq!(PitchClass::new(13), PitchClass::CSHARP);
    }

    #[test]
    fn middle_c_is_midi_60() {
        // Ground truth: middle C (C4) is MIDI 60; A4 is 69.
        assert_eq!(Pitch::new(PitchClass::C, 4).midi(), 60);
        assert_eq!(Pitch::new(PitchClass::A, 4).midi(), 69);
        assert_eq!(Pitch::from_midi(60).pitch_class(), PitchClass::C);
        assert_eq!(Pitch::from_midi(60).octave(), 4);
    }

    #[test]
    fn interval_to_directed_pitch_class() {
        // C up to G is a perfect fifth (7 semitones); G up to C is a fourth (5).
        assert_eq!(PitchClass::C.interval_to(PitchClass::G), 7);
        assert_eq!(PitchClass::G.interval_to(PitchClass::C), 5);
    }

    #[test]
    fn interval_quality_ground_truth() {
        assert_eq!(
            Interval::PERFECT_FIFTH.quality(),
            IntervalQuality::PerfectConsonance
        );
        assert_eq!(
            Interval::PERFECT_FOURTH.quality(),
            IntervalQuality::PerfectConsonance
        );
        assert_eq!(
            Interval::OCTAVE.quality(),
            IntervalQuality::PerfectConsonance
        );
        assert_eq!(
            Interval::MAJOR_THIRD.quality(),
            IntervalQuality::ImperfectConsonance
        );
        assert_eq!(
            Interval::MINOR_SIXTH.quality(),
            IntervalQuality::ImperfectConsonance
        );
        assert_eq!(
            Interval::MINOR_SECOND.quality(),
            IntervalQuality::Dissonance
        );
        assert_eq!(
            Interval::MAJOR_SEVENTH.quality(),
            IntervalQuality::Dissonance
        );
        assert_eq!(Interval::TRITONE.quality(), IntervalQuality::Tritone);
    }

    #[test]
    fn step_vs_leap() {
        assert!(Interval::MAJOR_SECOND.is_step());
        assert!(!Interval::MAJOR_SECOND.is_leap());
        assert!(Interval::MINOR_THIRD.is_leap());
        assert!(!Interval::MINOR_THIRD.is_step());
        assert!(Interval::PERFECT_FIFTH.is_leap());
    }

    #[test]
    fn transpose_pitch_clamps_and_moves() {
        let c4 = Pitch::new(PitchClass::C, 4);
        assert_eq!(c4.transpose(12), Pitch::new(PitchClass::C, 5));
        assert_eq!(c4.transpose(7).pitch_class(), PitchClass::G);
        // Clamp at the bottom of the MIDI range.
        assert_eq!(Pitch::from_midi(2).transpose(-10).midi(), 0);
    }

    #[test]
    fn octave_is_twelve_semitones() {
        let c4 = Pitch::new(PitchClass::C, 4);
        let c5 = Pitch::new(PitchClass::C, 5);
        assert_eq!(c4.semitones_to(c5), 12);
        assert_eq!(c4.interval_to(c5), Interval::OCTAVE);
    }
}
