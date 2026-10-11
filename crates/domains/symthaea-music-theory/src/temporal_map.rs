// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact work-scale temporal authority.
//!
//! `Score` historically carries one opening tempo and one quarter-note meter.
//! That is sufficient for short fixed-meter pieces, but it cannot faithfully
//! represent a work whose authoritative tempo or written meter changes later.
//! This module provides an additive migration path without changing `Score` yet:
//! exact beat offsets map to optional tempo and meter changes, with a required
//! complete initial state at beat zero.
//!
//! V1 deliberately supports piecewise-constant tempo only. Continuous tempo
//! ramps and metric modulation belong to a later contract so their integration
//! and export semantics can be qualified independently.

use crate::meter::TimeSignature;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const TEMPORAL_MAP_VERSION: &str = "melothaea-temporal-map-v1";

/// Exact quarter-note tempo expressed as a reduced rational BPM value.
///
/// Avoiding floating-point storage makes timeline identity deterministic while
/// still allowing ordinary values (`120/1`) and precise fractional tempi
/// (`243/2 == 121.5 BPM`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TempoV1 {
    numerator_bpm: u32,
    denominator: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalPointV1 {
    /// Exact quarter-note beats from the beginning of the work.
    pub at: Duration,
    /// `None` means retain the previously authoritative meter.
    pub meter: Option<TimeSignature>,
    /// `None` means retain the previously authoritative tempo.
    pub tempo: Option<TempoV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalMapV1 {
    pub version: String,
    /// Canonical strictly increasing points. Point zero must occur at beat zero
    /// and define both tempo and meter.
    pub points: Vec<TemporalPointV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalMapErrorV1 {
    ZeroTempoNumerator,
    ZeroTempoDenominator,
    InvalidFloatTempo,
    FloatTempoUnrepresentable,
    WrongVersion { found: String },
    EmptyMap,
    InitialPointNotAtZero,
    InitialMeterMissing,
    InitialTempoMissing,
    NegativePoint { point_index: usize },
    EmptyPoint { point_index: usize },
    NonIncreasingPoint { point_index: usize },
    InvalidTempo { point_index: usize },
    InvalidMeter { point_index: usize },
    QueryBeforeZero,
}

impl TempoV1 {
    pub fn new(numerator_bpm: u32, denominator: u32) -> Result<Self, TemporalMapErrorV1> {
        if numerator_bpm == 0 {
            return Err(TemporalMapErrorV1::ZeroTempoNumerator);
        }
        if denominator == 0 {
            return Err(TemporalMapErrorV1::ZeroTempoDenominator);
        }
        let gcd = gcd_u32(numerator_bpm, denominator);
        Ok(Self {
            numerator_bpm: numerator_bpm / gcd,
            denominator: denominator / gcd,
        })
    }

    pub fn integer(bpm: u32) -> Result<Self, TemporalMapErrorV1> {
        Self::new(bpm, 1)
    }

    /// Preserve a positive finite legacy `f32` BPM value as an exact reduced
    /// rational whenever that rational fits V1's `u32/u32` representation.
    ///
    /// This is not a decimal approximation: the IEEE-754 significand and
    /// binary exponent are converted exactly. As a result, projecting the
    /// returned rational back through `f64 -> f32` reproduces the source tempo
    /// bit-for-bit. Extremely tiny/large finite floats whose reduced rational
    /// cannot fit `u32/u32` fail closed instead of being rounded silently.
    pub fn from_f32_exact(bpm: f32) -> Result<Self, TemporalMapErrorV1> {
        if !bpm.is_finite() || bpm <= 0.0 {
            return Err(TemporalMapErrorV1::InvalidFloatTempo);
        }

        let bits = bpm.to_bits();
        let exponent_bits = ((bits >> 23) & 0xff) as i32;
        let fraction = u64::from(bits & 0x7f_ffff);
        let (mut mantissa, mut exponent_two) = if exponent_bits == 0 {
            // Positive subnormal: fraction * 2^-149.
            (fraction, -149)
        } else {
            // Positive normal: (2^23 + fraction) * 2^(e-127-23).
            ((1_u64 << 23) | fraction, exponent_bits - 127 - 23)
        };

        if mantissa == 0 {
            return Err(TemporalMapErrorV1::InvalidFloatTempo);
        }

        // Reduce powers of two before checking whether the denominator fits.
        if exponent_two < 0 {
            let removable = mantissa
                .trailing_zeros()
                .min((-exponent_two) as u32);
            mantissa >>= removable;
            exponent_two += removable as i32;
        }

        let (numerator, denominator) = if exponent_two >= 0 {
            let shifted = mantissa
                .checked_shl(exponent_two as u32)
                .ok_or(TemporalMapErrorV1::FloatTempoUnrepresentable)?;
            if shifted > u64::from(u32::MAX) {
                return Err(TemporalMapErrorV1::FloatTempoUnrepresentable);
            }
            (shifted as u32, 1)
        } else {
            let denominator_shift = (-exponent_two) as u32;
            if denominator_shift >= u32::BITS || mantissa > u64::from(u32::MAX) {
                return Err(TemporalMapErrorV1::FloatTempoUnrepresentable);
            }
            (mantissa as u32, 1_u32 << denominator_shift)
        };

        let tempo = Self::new(numerator, denominator)?;
        if (tempo.bpm() as f32).to_bits() != bpm.to_bits() {
            return Err(TemporalMapErrorV1::FloatTempoUnrepresentable);
        }
        Ok(tempo)
    }

    pub const fn numerator_bpm(self) -> u32 {
        self.numerator_bpm
    }

    pub const fn denominator(self) -> u32 {
        self.denominator
    }

    pub fn bpm(self) -> f64 {
        f64::from(self.numerator_bpm) / f64::from(self.denominator)
    }

    /// Seconds occupied by an exact quarter-note beat duration at this tempo.
    pub fn seconds_for(self, duration: Duration) -> f64 {
        duration.beats() * 60.0 * f64::from(self.denominator)
            / f64::from(self.numerator_bpm)
    }

    fn validate(self) -> bool {
        self.numerator_bpm > 0
            && self.denominator > 0
            && gcd_u32(self.numerator_bpm, self.denominator) == 1
    }
}

impl TemporalMapV1 {
    pub fn new(initial_tempo: TempoV1, initial_meter: TimeSignature) -> Self {
        Self {
            version: TEMPORAL_MAP_VERSION.into(),
            points: vec![TemporalPointV1 {
                at: Duration::zero(),
                meter: Some(initial_meter),
                tempo: Some(initial_tempo),
            }],
        }
    }

    /// Append one canonical temporal change.
    ///
    /// At least one of `meter`/`tempo` must change and offsets must be strictly
    /// increasing. Simultaneous tempo+meter changes belong in one point rather
    /// than duplicate points at the same beat.
    pub fn push_change(
        &mut self,
        at: Duration,
        meter: Option<TimeSignature>,
        tempo: Option<TempoV1>,
    ) -> Result<(), TemporalMapErrorV1> {
        if at.num() < 0 {
            return Err(TemporalMapErrorV1::NegativePoint {
                point_index: self.points.len(),
            });
        }
        if meter.is_none() && tempo.is_none() {
            return Err(TemporalMapErrorV1::EmptyPoint {
                point_index: self.points.len(),
            });
        }
        if let Some(last) = self.points.last()
            && compare_duration(at, last.at) != Ordering::Greater
        {
            return Err(TemporalMapErrorV1::NonIncreasingPoint {
                point_index: self.points.len(),
            });
        }
        self.points.push(TemporalPointV1 { at, meter, tempo });
        self.validate()?;
        Ok(())
    }

    /// Fail-closed validation for deserialized or externally supplied maps.
    pub fn validate(&self) -> Result<(), TemporalMapErrorV1> {
        if self.version != TEMPORAL_MAP_VERSION {
            return Err(TemporalMapErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        let Some(initial) = self.points.first() else {
            return Err(TemporalMapErrorV1::EmptyMap);
        };
        if initial.at != Duration::zero() {
            return Err(TemporalMapErrorV1::InitialPointNotAtZero);
        }
        if initial.meter.is_none() {
            return Err(TemporalMapErrorV1::InitialMeterMissing);
        }
        if initial.tempo.is_none() {
            return Err(TemporalMapErrorV1::InitialTempoMissing);
        }

        for (index, point) in self.points.iter().enumerate() {
            if point.at.num() < 0 {
                return Err(TemporalMapErrorV1::NegativePoint { point_index: index });
            }
            if point.meter.is_none() && point.tempo.is_none() {
                return Err(TemporalMapErrorV1::EmptyPoint { point_index: index });
            }
            if let Some(tempo) = point.tempo
                && !tempo.validate()
            {
                return Err(TemporalMapErrorV1::InvalidTempo { point_index: index });
            }
            if let Some(meter) = &point.meter
                && TimeSignature::with_grouping(
                    meter.numerator(),
                    meter.denominator(),
                    meter.grouping().to_vec(),
                )
                .is_err()
            {
                return Err(TemporalMapErrorV1::InvalidMeter { point_index: index });
            }
            if index > 0
                && compare_duration(point.at, self.points[index - 1].at) != Ordering::Greater
            {
                return Err(TemporalMapErrorV1::NonIncreasingPoint { point_index: index });
            }
        }
        Ok(())
    }

    pub fn tempo_at(&self, at: Duration) -> Result<TempoV1, TemporalMapErrorV1> {
        self.validate()?;
        if at.num() < 0 {
            return Err(TemporalMapErrorV1::QueryBeforeZero);
        }
        let mut current = self.points[0].tempo.expect("validated initial tempo");
        for point in self.points.iter().skip(1) {
            if compare_duration(point.at, at) == Ordering::Greater {
                break;
            }
            if let Some(tempo) = point.tempo {
                current = tempo;
            }
        }
        Ok(current)
    }

    pub fn meter_at(&self, at: Duration) -> Result<TimeSignature, TemporalMapErrorV1> {
        self.validate()?;
        if at.num() < 0 {
            return Err(TemporalMapErrorV1::QueryBeforeZero);
        }
        let mut current = self.points[0]
            .meter
            .clone()
            .expect("validated initial meter");
        for point in self.points.iter().skip(1) {
            if compare_duration(point.at, at) == Ordering::Greater {
                break;
            }
            if let Some(meter) = &point.meter {
                current = meter.clone();
            }
        }
        Ok(current)
    }

    /// Integrate elapsed wall-clock seconds from beat zero to `at` under the
    /// piecewise-constant tempo map. Meter changes do not affect beat duration.
    pub fn elapsed_seconds(&self, at: Duration) -> Result<f64, TemporalMapErrorV1> {
        self.validate()?;
        if at.num() < 0 {
            return Err(TemporalMapErrorV1::QueryBeforeZero);
        }

        let mut tempo = self.points[0].tempo.expect("validated initial tempo");
        let mut cursor = Duration::zero();
        let mut seconds = 0.0;

        for point in self.points.iter().skip(1) {
            if compare_duration(point.at, at) != Ordering::Less {
                break;
            }
            seconds += beats_between(cursor, point.at) * 60.0 / tempo.bpm();
            cursor = point.at;
            if let Some(next) = point.tempo {
                tempo = next;
            }
        }

        seconds += beats_between(cursor, at) * 60.0 / tempo.bpm();
        Ok(seconds)
    }
}

fn gcd_u32(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left.max(1)
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

/// Positive beat distance from `start` to `end`; callers establish ordering.
fn beats_between(start: Duration, end: Duration) -> f64 {
    let numerator = i128::from(end.num()) * i128::from(start.den())
        - i128::from(start.num()) * i128::from(end.den());
    let denominator = i128::from(end.den()) * i128::from(start.den());
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_four() -> TimeSignature {
        TimeSignature::new(4, 4).unwrap()
    }

    #[test]
    fn tempo_is_reduced_and_exactly_identified() {
        let tempo = TempoV1::new(243, 2).unwrap();
        assert_eq!(tempo.numerator_bpm(), 243);
        assert_eq!(tempo.denominator(), 2);
        assert!((tempo.bpm() - 121.5).abs() < 1e-12);

        assert_eq!(TempoV1::new(240, 2).unwrap(), TempoV1::integer(120).unwrap());
        assert_eq!(TempoV1::new(0, 1), Err(TemporalMapErrorV1::ZeroTempoNumerator));
        assert_eq!(TempoV1::new(120, 0), Err(TemporalMapErrorV1::ZeroTempoDenominator));
    }

    #[test]
    fn legacy_f32_tempo_is_preserved_bit_exactly_when_representable() {
        for bpm in [60.0_f32, 121.5_f32, 93.7_f32, 137.33333_f32] {
            let tempo = TempoV1::from_f32_exact(bpm).unwrap();
            assert_eq!((tempo.bpm() as f32).to_bits(), bpm.to_bits());
        }
        assert_eq!(
            TempoV1::from_f32_exact(120.0).unwrap(),
            TempoV1::integer(120).unwrap()
        );
    }

    #[test]
    fn legacy_f32_tempo_rejects_invalid_or_unrepresentable_values() {
        for bpm in [0.0_f32, -1.0_f32, f32::NAN, f32::INFINITY] {
            assert_eq!(
                TempoV1::from_f32_exact(bpm),
                Err(TemporalMapErrorV1::InvalidFloatTempo)
            );
        }
        assert_eq!(
            TempoV1::from_f32_exact(f32::MIN_POSITIVE),
            Err(TemporalMapErrorV1::FloatTempoUnrepresentable)
        );
    }

    #[test]
    fn mixed_meter_and_tempo_queries_are_exact_at_boundaries() {
        let mut map = TemporalMapV1::new(TempoV1::integer(120).unwrap(), four_four());
        let seven_eight = TimeSignature::with_grouping(7, 8, vec![2, 2, 3]).unwrap();
        map.push_change(Duration::new(8, 1), Some(seven_eight.clone()), None)
            .unwrap();
        map.push_change(
            Duration::new(12, 1),
            None,
            Some(TempoV1::integer(90).unwrap()),
        )
        .unwrap();

        assert_eq!(map.meter_at(Duration::new(7, 1)).unwrap(), four_four());
        assert_eq!(map.meter_at(Duration::new(8, 1)).unwrap(), seven_eight);
        assert_eq!(map.tempo_at(Duration::new(11, 1)).unwrap().bpm(), 120.0);
        assert_eq!(map.tempo_at(Duration::new(12, 1)).unwrap().bpm(), 90.0);
    }

    #[test]
    fn additive_grouping_is_authority_not_display_metadata() {
        let mut left = TemporalMapV1::new(TempoV1::integer(120).unwrap(), four_four());
        let mut right = left.clone();
        left.push_change(
            Duration::new(4, 1),
            Some(TimeSignature::with_grouping(7, 8, vec![2, 2, 3]).unwrap()),
            None,
        )
        .unwrap();
        right
            .push_change(
                Duration::new(4, 1),
                Some(TimeSignature::with_grouping(7, 8, vec![3, 2, 2]).unwrap()),
                None,
            )
            .unwrap();
        assert_ne!(left, right);
    }

    #[test]
    fn elapsed_seconds_integrates_piecewise_tempo() {
        let mut map = TemporalMapV1::new(TempoV1::integer(120).unwrap(), four_four());
        map.push_change(
            Duration::new(8, 1),
            None,
            Some(TempoV1::integer(60).unwrap()),
        )
        .unwrap();

        assert!((map.elapsed_seconds(Duration::new(8, 1)).unwrap() - 4.0).abs() < 1e-12);
        assert!((map.elapsed_seconds(Duration::new(12, 1)).unwrap() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn simultaneous_meter_and_tempo_change_is_one_canonical_point() {
        let mut map = TemporalMapV1::new(TempoV1::integer(100).unwrap(), four_four());
        let six_eight = TimeSignature::new(6, 8).unwrap();
        map.push_change(
            Duration::new(16, 1),
            Some(six_eight.clone()),
            Some(TempoV1::new(225, 2).unwrap()),
        )
        .unwrap();
        assert_eq!(map.points.len(), 2);
        assert_eq!(map.meter_at(Duration::new(16, 1)).unwrap(), six_eight);
        assert!((map.tempo_at(Duration::new(16, 1)).unwrap().bpm() - 112.5).abs() < 1e-12);
    }

    #[test]
    fn append_api_refuses_noncanonical_points() {
        let mut map = TemporalMapV1::new(TempoV1::integer(120).unwrap(), four_four());
        assert!(matches!(
            map.push_change(Duration::new(4, 1), None, None),
            Err(TemporalMapErrorV1::EmptyPoint { .. })
        ));
        map.push_change(
            Duration::new(4, 1),
            None,
            Some(TempoV1::integer(100).unwrap()),
        )
        .unwrap();
        assert!(matches!(
            map.push_change(
                Duration::new(4, 1),
                Some(TimeSignature::new(3, 4).unwrap()),
                None,
            ),
            Err(TemporalMapErrorV1::NonIncreasingPoint { .. })
        ));
    }

    #[test]
    fn deserialized_invalid_state_fails_closed() {
        let mut map = TemporalMapV1::new(TempoV1::integer(120).unwrap(), four_four());
        map.points.push(TemporalPointV1 {
            at: Duration::new(4, 1),
            meter: None,
            tempo: Some(TempoV1 {
                numerator_bpm: 0,
                denominator: 1,
            }),
        });
        assert_eq!(
            map.validate(),
            Err(TemporalMapErrorV1::InvalidTempo { point_index: 1 })
        );
    }

    #[test]
    fn negative_queries_and_points_are_rejected() {
        let mut map = TemporalMapV1::new(TempoV1::integer(120).unwrap(), four_four());
        assert!(matches!(
            map.push_change(
                Duration::new(-1, 1),
                None,
                Some(TempoV1::integer(90).unwrap()),
            ),
            Err(TemporalMapErrorV1::NegativePoint { .. })
        ));
        assert_eq!(
            map.elapsed_seconds(Duration::new(-1, 1)),
            Err(TemporalMapErrorV1::QueryBeforeZero)
        );
    }
}
