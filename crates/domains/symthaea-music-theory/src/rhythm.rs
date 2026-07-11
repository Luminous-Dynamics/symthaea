// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rhythm: note durations as exact rationals in beats (quarter note = 1 beat).
//!
//! Durations are rational, not floating point, so that motivic augmentation
//! and diminution (×2, ÷3, dotting) are EXACT and reversible — a property the
//! motif-transformation tests depend on (augment then diminish == identity).

use serde::{Deserialize, Serialize};

/// A duration in beats, as a reduced rational (quarter note = 1/1 beat).
/// Always stored with `den > 0` and `gcd(num, den) == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Duration {
    num: i64,
    den: i64,
}

impl std::ops::Add for Duration {
    type Output = Duration;
    fn add(self, other: Duration) -> Duration {
        Duration::new(
            self.num * other.den + other.num * self.den,
            self.den * other.den,
        )
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

impl Duration {
    /// Construct from a rational `num/den` beats (reduced; `den` must be > 0).
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "duration denominator must be non-zero");
        let sign = if den < 0 { -1 } else { 1 };
        let (num, den) = (num * sign, den * sign);
        let g = gcd(num, den);
        Duration {
            num: num / g,
            den: den / g,
        }
    }

    pub fn whole() -> Self {
        Duration::new(4, 1)
    }
    pub fn half() -> Self {
        Duration::new(2, 1)
    }
    pub fn quarter() -> Self {
        Duration::new(1, 1)
    }
    pub fn eighth() -> Self {
        Duration::new(1, 2)
    }
    pub fn sixteenth() -> Self {
        Duration::new(1, 4)
    }
    pub fn triplet_eighth() -> Self {
        Duration::new(1, 3)
    }
    pub fn zero() -> Self {
        Duration::new(0, 1)
    }

    pub fn num(self) -> i64 {
        self.num
    }
    pub fn den(self) -> i64 {
        self.den
    }

    /// Duration in beats as a float (quarter note = 1.0). For realization only.
    pub fn beats(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Dotted value: 1.5× (adds half its own length). Exact.
    pub fn dotted(self) -> Self {
        self.scale(3, 2)
    }

    /// Scale by a rational factor `n/d` (augmentation `2/1`, diminution `1/2`,
    /// triplet `2/3`). Exact and reversible: `scale(a,b).scale(b,a) == self`.
    pub fn scale(self, n: i64, d: i64) -> Self {
        Duration::new(self.num * n, self.den * d)
    }

    /// Convert to seconds at a given tempo (beats per minute).
    pub fn seconds(self, tempo_bpm: f64) -> f64 {
        self.beats() * 60.0 / tempo_bpm
    }

    /// Exact rational subtraction, floored at zero (a Duration is a length —
    /// negative lengths are never meaningful in a score).
    pub fn saturating_sub(self, other: Duration) -> Self {
        let num = self.num * other.den - other.num * self.den;
        if num <= 0 {
            Duration::zero()
        } else {
            Duration::new(num, self.den * other.den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_values_in_beats() {
        assert_eq!(Duration::quarter().beats(), 1.0);
        assert_eq!(Duration::half().beats(), 2.0);
        assert_eq!(Duration::whole().beats(), 4.0);
        assert_eq!(Duration::eighth().beats(), 0.5);
    }

    #[test]
    fn reduces_to_lowest_terms() {
        assert_eq!(Duration::new(2, 4), Duration::eighth());
        assert_eq!(Duration::new(4, 4), Duration::quarter());
    }

    #[test]
    fn dotted_quarter_is_three_eighths() {
        // A dotted quarter = 1.5 beats = 3/2.
        let dq = Duration::quarter().dotted();
        assert_eq!(dq, Duration::new(3, 2));
        assert_eq!(dq.beats(), 1.5);
    }

    #[test]
    fn augment_then_diminish_is_identity() {
        // The property the motif tests rely on — exact because rational.
        let d = Duration::quarter().dotted(); // 3/2, an "awkward" value
        assert_eq!(d.scale(2, 1).scale(1, 2), d); // augment then diminish
        let t = Duration::eighth();
        assert_eq!(t.scale(2, 3).scale(3, 2), t); // triplet then un-triplet, exact
    }

    #[test]
    fn addition_is_exact() {
        // Triplet eighths: three of them make a quarter, exactly.
        let te = Duration::triplet_eighth();
        assert_eq!(te + te + te, Duration::quarter());
    }

    #[test]
    fn seconds_at_tempo() {
        // A quarter note at 120 BPM is 0.5 s.
        assert!((Duration::quarter().seconds(120.0) - 0.5).abs() < 1e-12);
    }
}
