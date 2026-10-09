// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rhythm: note durations as exact rationals in beats (quarter note = 1 beat).
//!
//! Durations are rational, not floating point, so that motivic augmentation
//! and diminution (×2, ÷3, dotting) are EXACT and reversible — a property the
//! motif-transformation tests depend on (augment then diminish == identity).

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

/// A duration in beats, as a reduced rational (quarter note = 1/1 beat).
/// Always stored with `den > 0` and `gcd(num, den) == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Duration {
    num: i64,
    den: i64,
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireDuration {
            num: i64,
            den: i64,
        }

        let wire = WireDuration::deserialize(deserializer)?;
        if wire.den == 0 {
            return Err(de::Error::custom("duration denominator must be non-zero"));
        }
        // Normalize rather than reject older/non-canonical-but-valid rationals.
        // This preserves their mathematical value while restoring the type's
        // positive-denominator/reduced-fraction invariant at the wire boundary.
        from_i128_rational(i128::from(wire.num), i128::from(wire.den))
            .ok_or_else(|| de::Error::custom("duration cannot be represented canonically"))
    }
}

impl std::ops::Add for Duration {
    type Output = Duration;

    /// Exact addition for representable results.
    ///
    /// The operator is retained for ergonomic use with trusted, ordinary
    /// musical values. Call `checked_add` when inputs can come from external
    /// scores or other untrusted sources.
    fn add(self, other: Duration) -> Duration {
        self.checked_add(other)
            .expect("duration addition is not representable; use checked_add for untrusted values")
    }
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

/// Reduce an exact rational in wide arithmetic, then return it only if the
/// canonical numerator and positive denominator both fit the wire type.
fn from_i128_rational(num: i128, den: i128) -> Option<Duration> {
    if den == 0 {
        return None;
    }
    let (num, den) = if den < 0 {
        (num.checked_neg()?, den.checked_neg()?)
    } else {
        (num, den)
    };
    let divisor = i128::try_from(gcd_u128(num.unsigned_abs(), den as u128)).ok()?;
    let reduced_num = num / divisor;
    let reduced_den = den / divisor;
    Some(Duration {
        num: i64::try_from(reduced_num).ok()?,
        den: i64::try_from(reduced_den).ok()?,
    })
}

impl Duration {
    /// Construct from a rational `num/den` beats and reduce it exactly.
    ///
    /// The input denominator must be non-zero. The stored denominator is always
    /// positive, and the reduced result must fit the i64 wire representation.
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "duration denominator must be non-zero");
        from_i128_rational(num as i128, den as i128)
            .expect("normalized duration cannot be represented by i64 numerator/denominator")
    }

    /// Add two exact rationals without intermediate i64 overflow.
    ///
    /// Returns None when an input violates the canonical representation or
    /// the reduced result cannot fit this type's i64 numerator/denominator.
    pub fn checked_add(self, other: Duration) -> Option<Self> {
        if self.den <= 0 || other.den <= 0 {
            return None;
        }
        let left = i128::from(self.num).checked_mul(i128::from(other.den))?;
        let right = i128::from(other.num).checked_mul(i128::from(self.den))?;
        let numerator = left.checked_add(right)?;
        let denominator = i128::from(self.den).checked_mul(i128::from(other.den))?;
        from_i128_rational(numerator, denominator)
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
    pub fn checked_scale(self, n: i64, d: i64) -> Option<Self> {
        if self.den <= 0 || d == 0 {
            return None;
        }
        let numerator = i128::from(self.num).checked_mul(i128::from(n))?;
        let denominator = i128::from(self.den).checked_mul(i128::from(d))?;
        from_i128_rational(numerator, denominator)
    }

    pub fn scale(self, n: i64, d: i64) -> Self {
        self.checked_scale(n, d)
            .expect("duration scaling is not representable; use checked_scale for untrusted values")
    }

    /// Convert to seconds at a given tempo (beats per minute).
    pub fn seconds(self, tempo_bpm: f64) -> f64 {
        self.beats() * 60.0 / tempo_bpm
    }

    /// Exact rational subtraction, floored at zero (a Duration is a length —
    /// negative lengths are never meaningful in a score).
    pub fn saturating_sub(self, other: Duration) -> Self {
        if self.den <= 0 || other.den <= 0 {
            return Duration::zero();
        }
        let Some(left) = i128::from(self.num).checked_mul(i128::from(other.den)) else {
            return Duration::zero();
        };
        let Some(right) = i128::from(other.num).checked_mul(i128::from(self.den)) else {
            return Duration::zero();
        };
        let Some(num) = left.checked_sub(right) else {
            return Duration::zero();
        };
        if num <= 0 {
            return Duration::zero();
        }
        let Some(den) = i128::from(self.den).checked_mul(i128::from(other.den)) else {
            return Duration::zero();
        };
        from_i128_rational(num, den).unwrap_or_else(|| {
            let quotient = num / den;
            let exceeds_max = quotient > i128::from(i64::MAX)
                || (quotient == i128::from(i64::MAX) && num % den > 0);
            if exceeds_max {
                Duration::new(i64::MAX, 1)
            } else {
                // A positive but unrepresentable sub-beat fraction cannot be
                // preserved exactly; do not turn it into a fabricated maximum.
                Duration::zero()
            }
        })
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
    fn deserialization_normalizes_legacy_rationals_and_rejects_zero_denominator() {
        let canonical = Duration::new(3, 2);
        let encoded = serde_json::to_string(&canonical).unwrap();
        assert_eq!(serde_json::from_str::<Duration>(&encoded).unwrap(), canonical);

        // Previously accepted wire values keep their mathematical meaning,
        // but are brought back into the canonical in-memory representation.
        assert_eq!(
            serde_json::from_str::<Duration>(r#"{"num":2,"den":4}"#).unwrap(),
            Duration::new(1, 2)
        );
        assert_eq!(
            serde_json::from_str::<Duration>(r#"{"num":1,"den":-2}"#).unwrap(),
            Duration::new(-1, 2)
        );
        assert_eq!(
            serde_json::from_str::<Duration>(r#"{"num":0,"den":2}"#).unwrap(),
            Duration::zero()
        );
        assert_eq!(
            serde_json::from_str::<Duration>(r#"{"num":-2,"den":-4}"#).unwrap(),
            Duration::new(1, 2)
        );
        assert!(serde_json::from_str::<Duration>(r#"{"num":1,"den":0}"#).is_err());
    }

    #[test]
    fn checked_add_and_scale_fail_without_integer_wraparound() {
        let largest = Duration::new(i64::MAX, 1);
        assert_eq!(largest.checked_add(Duration::quarter()), None);
        assert_eq!(largest.checked_scale(2, 1), None);
        assert_eq!(
            Duration::new(1, 2).checked_add(Duration::new(1, 3)),
            Some(Duration::new(5, 6))
        );
        assert_eq!(
            Duration::new(i64::MAX, 1).saturating_sub(Duration::new(-i64::MAX, 1)),
            Duration::new(i64::MAX, 1)
        );
        assert_eq!(
            Duration::new(1, i64::MAX - 1).saturating_sub(Duration::new(1, i64::MAX)),
            Duration::zero()
        );
    }

    #[test]
    fn constructor_handles_signed_extremes_when_representable() {
        assert_eq!(
            Duration::new(2, i64::MIN),
            Duration::new(-1, 1_i64 << 62)
        );
        assert_eq!(Duration::new(i64::MIN, 1).num(), i64::MIN);
    }

    #[test]
    fn seconds_at_tempo() {
        // A quarter note at 120 BPM is 0.5 s.
        assert!((Duration::quarter().seconds(120.0) - 0.5).abs() < 1e-12);
    }
}
