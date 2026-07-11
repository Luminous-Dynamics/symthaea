// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Motif: a short symbolic melodic germ, and the classical transformations
//! that DEVELOP it into a phrase.
//!
//! This is the core of the fix for "aimless" melody. Instead of drawing each
//! note at random, we generate ONE idea (3–6 notes) and grow a line by
//! transforming it — transposition, inversion, retrograde, augmentation,
//! sequence. A listener recognizes the returning-and-growing idea and perceives
//! *intention*; that recognition is what "understanding" and structure sound
//! like (Schoenberg, *Fundamentals of Musical Composition*).
//!
//! A motif is scale-degree-relative (degree 1 = tonic), so transposition and
//! inversion stay diatonic (in key) by construction. It carries rhythm
//! ([`Duration`]) so augmentation/diminution are real. Rests are `degree:
//! None` and pass through pitch transforms untouched.

use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::scale::Scale;
use serde::{Deserialize, Serialize};

/// One event in a motif: a scale degree (1 = tonic; may be negative or > 7 for
/// other octaves) or a rest, with a duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotifNote {
    /// Scale degree, or `None` for a rest.
    pub degree: Option<i32>,
    pub duration: Duration,
}

impl MotifNote {
    pub fn new(degree: i32, duration: Duration) -> Self {
        MotifNote {
            degree: Some(degree),
            duration,
        }
    }
    pub fn rest(duration: Duration) -> Self {
        MotifNote {
            degree: None,
            duration,
        }
    }
    pub fn is_rest(self) -> bool {
        self.degree.is_none()
    }
}

/// Direction of melodic motion between two consecutive pitched notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Contour {
    Up,
    Down,
    Same,
}

/// A motif: an ordered sequence of degree/rest events. Immutable
/// transformations return new motifs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Motif {
    pub notes: Vec<MotifNote>,
}

impl Motif {
    pub fn new(notes: Vec<MotifNote>) -> Self {
        Motif { notes }
    }

    /// Convenience constructor from `(degree, duration)` pairs (no rests).
    pub fn from_degrees(pairs: &[(i32, Duration)]) -> Self {
        Motif {
            notes: pairs
                .iter()
                .map(|&(d, dur)| MotifNote::new(d, dur))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.notes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    /// Total rhythmic length in beats.
    pub fn total_duration(&self) -> Duration {
        self.notes
            .iter()
            .fold(Duration::zero(), |acc, n| acc + n.duration)
    }

    // ── Classical transformations (each a pure function) ──────────────────

    /// Diatonic transposition: shift every degree by `steps` scale degrees.
    /// `transpose(a).transpose(b) == transpose(a + b)`.
    pub fn transpose(&self, steps: i32) -> Motif {
        self.map_degrees(|d| d + steps)
    }

    /// Diatonic inversion about a pivot degree: `d -> 2*pivot - d`, mirroring
    /// the contour. `invert(p).invert(p) == self` (an involution).
    pub fn invert(&self, pivot: i32) -> Motif {
        self.map_degrees(|d| 2 * pivot - d)
    }

    /// Retrograde: reverse the event order (pitch AND rhythm run backwards).
    /// `retrograde().retrograde() == self`.
    pub fn retrograde(&self) -> Motif {
        let mut notes = self.notes.clone();
        notes.reverse();
        Motif { notes }
    }

    /// Rhythmic augmentation/diminution: scale every duration by `n/d`
    /// (augment `2,1`; diminish `1,2`). Pitches unchanged; exact and
    /// reversible.
    pub fn scale_rhythm(&self, n: i64, d: i64) -> Motif {
        Motif {
            notes: self
                .notes
                .iter()
                .map(|note| MotifNote {
                    degree: note.degree,
                    duration: note.duration.scale(n, d),
                })
                .collect(),
        }
    }

    pub fn augment(&self) -> Motif {
        self.scale_rhythm(2, 1)
    }
    pub fn diminish(&self) -> Motif {
        self.scale_rhythm(1, 2)
    }

    /// Melodic sequence: repeat the motif `count` times, each successive copy
    /// transposed by `step` scale degrees (e.g. a descending step-sequence).
    /// Length is `count * self.len()`.
    pub fn sequence(&self, count: usize, step: i32) -> Motif {
        let mut notes = Vec::with_capacity(self.len() * count);
        for i in 0..count {
            notes.extend(self.transpose(step * i as i32).notes);
        }
        Motif { notes }
    }

    /// A contiguous fragment `[start, start+len)` of the motif (motivic
    /// fragmentation — developing by using just a piece of the idea).
    pub fn fragment(&self, start: usize, len: usize) -> Motif {
        let end = (start + len).min(self.notes.len());
        let start = start.min(end);
        Motif {
            notes: self.notes[start..end].to_vec(),
        }
    }

    /// Concatenate another motif after this one.
    pub fn then(&self, other: &Motif) -> Motif {
        let mut notes = self.notes.clone();
        notes.extend(other.notes.iter().copied());
        Motif { notes }
    }

    // ── Analysis ──────────────────────────────────────────────────────────

    /// The degree of each pitched note, rests skipped.
    pub fn degrees(&self) -> Vec<i32> {
        self.notes.iter().filter_map(|n| n.degree).collect()
    }

    /// Contour: direction between consecutive pitched notes.
    pub fn contour(&self) -> Vec<Contour> {
        let ds = self.degrees();
        ds.windows(2)
            .map(|w| match w[1].cmp(&w[0]) {
                std::cmp::Ordering::Greater => Contour::Up,
                std::cmp::Ordering::Less => Contour::Down,
                std::cmp::Ordering::Equal => Contour::Same,
            })
            .collect()
    }

    // ── Realization ───────────────────────────────────────────────────────

    /// Realize into absolute pitches over a scale, anchoring degree 1 in
    /// `tonic_octave`. Rests become `None`. This is the boundary the muse
    /// crate consumes — still symbolic pitch, no Hz.
    pub fn render(&self, scale: Scale, tonic_octave: i32) -> Vec<(Option<Pitch>, Duration)> {
        self.notes
            .iter()
            .map(|n| {
                let p = n.degree.map(|d| scale.degree_pitch(d, tonic_octave));
                (p, n.duration)
            })
            .collect()
    }

    fn map_degrees(&self, f: impl Fn(i32) -> i32) -> Motif {
        Motif {
            notes: self
                .notes
                .iter()
                .map(|note| MotifNote {
                    degree: note.degree.map(&f),
                    duration: note.duration,
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;
    use crate::scale::Mode;

    fn germ() -> Motif {
        // A little 4-note idea: 1 2 3 5 (do re mi sol), quarter notes.
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    #[test]
    fn transpose_is_additive() {
        let m = germ();
        assert_eq!(m.transpose(2).transpose(3), m.transpose(5));
        assert_eq!(m.transpose(0), m);
        // Shifting up two degrees: 1 2 3 5 -> 3 4 5 7
        assert_eq!(m.transpose(2).degrees(), vec![3, 4, 5, 7]);
    }

    #[test]
    fn inversion_is_an_involution() {
        let m = germ();
        assert_eq!(m.invert(3).invert(3), m);
        // Invert 1 2 3 5 about degree 3: d -> 6-d => 5 4 3 1 (contour mirrored)
        assert_eq!(m.invert(3).degrees(), vec![5, 4, 3, 1]);
    }

    #[test]
    fn retrograde_is_an_involution() {
        let m = germ();
        assert_eq!(m.retrograde().retrograde(), m);
        assert_eq!(m.retrograde().degrees(), vec![5, 3, 2, 1]);
    }

    #[test]
    fn augment_doubles_total_duration_and_diminish_undoes_it() {
        let m = germ();
        assert_eq!(m.augment().total_duration(), m.total_duration().scale(2, 1));
        assert_eq!(m.augment().diminish(), m); // exact round-trip
    }

    #[test]
    fn sequence_repeats_and_transposes() {
        let m = germ();
        let seq = m.sequence(3, -1); // three copies, each a step lower
        assert_eq!(seq.len(), m.len() * 3);
        // Second copy is transposed down one degree.
        assert_eq!(seq.notes[4].degree, Some(0)); // first note of copy 2: 1 - 1 = 0
        // Total duration is three motifs.
        assert_eq!(seq.total_duration(), m.total_duration().scale(3, 1));
    }

    #[test]
    fn fragment_takes_a_subslice() {
        let m = germ();
        let head = m.fragment(0, 2);
        assert_eq!(head.degrees(), vec![1, 2]);
        // Out-of-range length is clamped, not a panic.
        assert_eq!(m.fragment(2, 99).degrees(), vec![3, 5]);
    }

    #[test]
    fn contour_reads_direction() {
        // 1 2 3 5 is all ascending.
        assert_eq!(
            germ().contour(),
            vec![Contour::Up, Contour::Up, Contour::Up]
        );
        assert_eq!(
            germ().retrograde().contour(),
            vec![Contour::Down, Contour::Down, Contour::Down]
        );
    }

    #[test]
    fn rests_survive_pitch_transforms() {
        let m = Motif::new(vec![
            MotifNote::new(1, Duration::quarter()),
            MotifNote::rest(Duration::eighth()),
            MotifNote::new(3, Duration::quarter()),
        ]);
        let t = m.transpose(2);
        assert!(t.notes[1].is_rest());
        assert_eq!(t.degrees(), vec![3, 5]); // rests skipped in degrees()
        assert_eq!(t.total_duration(), m.total_duration());
    }

    #[test]
    fn render_maps_degrees_to_pitches() {
        // Over C major from C4: degree 1 -> C4 (60), degree 5 -> G4 (67).
        let scale = Scale::new(PitchClass::C, Mode::Ionian);
        let realized = germ().render(scale, 4);
        assert_eq!(realized.len(), 4);
        assert_eq!(realized[0].0.unwrap().midi(), 60); // C4
        assert_eq!(realized[3].0.unwrap().midi(), 67); // G4
    }
}
